#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "DiagnosticDump.h"
#include "ImuService.h"
#include "HandBoneCalibrator.h"
#include "HandPoseModel.h"
#include "HandFusion.h" // CameraFrameResult
#include "HandRoiQuality.h" // LumaFlickerTracker
#include "TrackingRecorder.h"
#include "TrackingTypes.h"

class AppConfig;
class VideoCaptureSystem;
class HandTrackingPipeline;
class LandmarkTo3D;
class OscStreamer;
class CVVideoFrameProcessor;

// Latest processed frame for one camera, published to the main/render thread
// (latest-wins). result is that camera's own (unfused, unsmoothed) tracking.
struct VisionPreviewFrame
{
	cv::Mat bgr; // undistorted when intrinsics are applied
	TrackingFrameResult result;
	bool valid= false;
};

// Owns the inference thread: drains each camera's frames, converts to BGR,
// optionally undistorts, runs that camera's ML pipeline and 3D projection,
// fuses all cameras' world-space results (visibility-weighted softmax blend),
// smooths the fused output, streams it over OSC and publishes previews.
//
// Cameras are processed sequentially on one thread: DirectML serializes on a
// single GPU queue anyway, and the per-camera CameraContext boundary keeps a
// later thread-per-camera upgrade possible.
class VisionThread
{
public:
	VisionThread(VideoCaptureSystem* videoCapture, AppConfig* config);
	~VisionThread();

	// Sizes the per-camera contexts from the config's camera count and spawns
	// the thread. Changing the camera COUNT at runtime requires stop()+start()
	// (the context list is never resized while the thread runs - that keeps
	// main-thread accessors race-free); per-camera settings changes only need
	// requestConfigRefresh().
	void start();
	void stop();

	// Pauses ML tracking for one camera (calibration wizards run their
	// detection on the main thread from that camera's preview frames).
	// Preview frames keep flowing; the other cameras keep tracking.
	void setTrackingEnabled(int cameraIndex, bool bEnabled);
	bool isTrackingEnabled(int cameraIndex) const;

	// Disable to receive raw distorted preview frames for one camera (needed
	// while capturing intrinsics calibration samples)
	void setUndistortEnabled(int cameraIndex, bool bEnabled);
	bool isUndistortEnabled(int cameraIndex) const;

	// Depth preview (RealSense devices): the camera's preview pane shows the
	// colorized depth stream instead of color. Tracking is unaffected - it
	// always consumes the color image.
	void setDepthPreviewEnabled(int cameraIndex, bool bEnabled);
	bool isDepthPreviewEnabled(int cameraIndex) const;

	// Copies the newest preview frame + per-camera result for a camera.
	// Returns false if nothing new arrived since the last call.
	bool fetchPreviewFrame(int cameraIndex, VisionPreviewFrame& outFrame);

	// Copies the newest fused (world-space, smoothed) tracking result.
	// Returns false if nothing new arrived since the last call.
	bool fetchFusedResult(TrackingFrameResult& outResult);

	// Per-side camera index that dominated the last fusion (-1 = untracked)
	int getDominantCamera(eHandSide side) const { return m_dominantCamera[(int)side].load(); }

	// One camera's observation confidence for a fused side in the last fusion
	// (presence x measured stability), or -1 if that camera contributed no
	// observation to that side. For the UI's per-camera quality readout.
	float getObservationConfidence(int cameraIndex, eHandSide side) const;

	// Stereo auto hand-scale: current correction factor over the configured
	// hand scale (1 = unchanged), refined from cross-camera wrist
	// triangulation when enabled and both cameras see the same hand
	float getAutoHandScaleFactor() const { return m_autoScaleFactor.load(); }

	// Re-reads config (cameras/intrinsics/extrinsics/hand scale/tracking/
	// fusion/osc) on the vision thread before the next frame
	void requestConfigRefresh() { m_bConfigRefreshRequested= true; }

	// Rest-pose calibration: captures what EVERY camera currently reports for
	// each tracked hand. Per camera because the model landmarks are
	// view-dependent - one camera's rest angles do not zero another's.
	void requestRestPoseCapture() { m_bRestPoseCaptureRequested= true; }

	// Wrist IMU: solve the mounting rotation for both wrists from the two
	// recorded calibration motions (see imuSolveMountingFromMotions). Record a
	// twist and a curl first; there is no pose to hold.
	void requestImuMountingCapture() { m_bImuMountingCaptureRequested= true; }
	struct ImuMountingCapture
	{
		MountingCaptureResult sides[2];
	};
	bool fetchImuMountingCapture(ImuMountingCapture& outCapture);

	// Live per-wrist IMU status for the UI (main thread safe: plain copies)
	ImuSideStatus getImuSideStatus(eHandSide side) const;
	// Starts recording one of the two calibration motions, discarding whatever
	// that window held before and clearing the live twist history so the
	// progress readout describes the stage being asked for. Pass
	// eMountingMotion::None to stop recording.
	void requestImuMotionRecording(eMountingMotion motion)
	{
		m_requestedImuMotionRecording= (int)motion;
		m_bImuMotionRecordingRequested= true;
	}
	// Static gyro bias calibration: measures each resting controller's bias
	// directly. Progress is reported through getImuSideStatus().
	void requestImuBiasCalibration() { m_bImuBiasCalibrationRequested= true; }
	void cancelImuBiasCalibration() { m_bImuBiasCancelRequested= true; }

	// One camera's captured rest angles for both sides
	struct RestPoseCapture
	{
		std::array<std::array<FingerAngles, FINGER_COUNT>, 2> angles{};
		bool bCaptured[2]= {false, false};
	};
	// Fetches a completed capture (one entry per camera, in camera order,
	// plus the stereo-triangulated fused capture when it was available);
	// returns false if none is pending
	bool fetchRestPoseCapture(std::vector<RestPoseCapture>& outCaptures, RestPoseCapture& outFused);

	// Hand bone calibration: measures the user's own skeleton from the
	// stereo-triangulated landmarks over a sampling window. Unlike the rest
	// pose this needs TIME rather than an instant - it is a median over many
	// frames, and one pose only measures that pose's triangulation luck.
	void requestBoneCalibration(float durationSeconds)
	{
		m_boneCalibrationSeconds= durationSeconds;
		m_bBoneCalibrationRequested= true;
	}
	void cancelBoneCalibration() { m_bBoneCalibrationCancelRequested= true; }
	// Live sample counts while a window is open, so the UI can show progress
	// and refuse a capture that never saw a hand
	void getBoneCalibrationProgress(bool& outActive, int& outLeftSamples, int& outRightSamples) const
	{
		outActive= m_bBoneCalibrationActive.load();
		outLeftSamples= m_boneCalibrationSamples[0].load();
		outRightSamples= m_boneCalibrationSamples[1].load();
	}

	struct BoneCalibrationCapture
	{
		std::array<HandSkeleton, 2> skeleton{};
		std::array<HandBoneCalibrator::Quality, 2> quality{};
		bool bCaptured[2]= {false, false};
	};
	// Fetches a completed window; returns false if none is pending
	bool fetchBoneCalibration(BoneCalibrationCapture& outCapture);

	// Tracking recording: captures every input to the post-MediaPipe stages
	// (per-camera landmarks, depth measurements, timestamps, IMU forearm
	// output) so the tracking pipeline can be re-run offline bit-exactly.
	// Starting a recording RESETS the transient tracking state (filters,
	// jitter EMAs, PnP warm starts, palmar memories) - replay runs on fresh
	// instances, so live must start from the same zero state. A config
	// refresh while recording finalizes the file (the header's config
	// snapshot must describe every frame).
	void requestRecordingStart(const std::string& filePath);
	void requestRecordingStop();
	bool isRecording() const { return m_recorder != nullptr && m_recorder->isRecording(); }
	int64_t getRecordingFrameCount() const { return m_recorder != nullptr ? m_recorder->getFrameCount() : 0; }
	uint64_t getRecordingBytes() const { return m_recorder != nullptr ? m_recorder->getBytesWritten() : 0; }
	bool didLastRecordingAbort() const { return m_recorder != nullptr && m_recorder->wasAborted(); }
	// Path of the last finalized recording ("" until one completes)
	std::string getLastRecordingPath();

	// Diagnostic dump (F9): the vision thread writes its rolling state history,
	// the current camera frames (raw + annotated PNGs) and the live config to
	// dumpDir on its next loop iteration
	void requestDiagnosticDump(const std::string& dumpDir);
	// Directory of the last completed dump ("" until one succeeds)
	std::string getLastDumpPath();

	// Introspection for the UI
	const char* getActiveExecutionProvider(int cameraIndex= 0) const;
	float getLastInferenceMs() const { return m_lastInferenceMs; } // summed across cameras

	// -- Frame-loop hitch watchdog ------------------------------------------
	// Every camera shares this one thread, so any phase that overruns the
	// frame budget starves ALL of them at once: frames keep arriving from the
	// driver, find no free block, and are dropped. The symptom downstream is a
	// synchronized multi-frame tracking gap that looks like a camera or USB
	// fault, so the loop measures itself and names the phase responsible.
	enum class eVisionPhase : int
	{
		None= 0,
		ConfigRefresh, // config refresh / recording start-stop
		Capture,       // frame pop, undistort, ML inference, 3D projection
		Imu,           // wrist IMU sample drain, discovery, forearm publish
		Fusion,        // cross-camera fusion + smoothing
		Osc,
		Diagnostics, // dump history, recording enqueue, dump writes
		Count
	};
	static const char* getPhaseName(eVisionPhase phase);

	int getHitchCount() const { return m_hitchCount; }
	float getLastHitchMs() const { return m_lastHitchMs; }
	eVisionPhase getLastHitchPhase() const { return (eVisionPhase)m_lastHitchPhase.load(); }

private:
	// Everything one camera needs on the vision thread. No shared mutable
	// state between contexts (the fusion step is the only join point).
	struct CameraContext
	{
		int cameraIndex= -1;

		std::unique_ptr<HandTrackingPipeline> pipeline;
		std::unique_ptr<LandmarkTo3D> landmarkTo3D; // smoothing always disabled (post-fusion smoothing)
		std::unique_ptr<CVVideoFrameProcessor> undistorter;

		std::atomic_bool bTrackingEnabled{true};
		std::atomic_bool bUndistortEnabled{true};
		// Preview shows the colorized depth stream instead of color
		// (RealSense devices only; ignored when no depth is available)
		std::atomic_bool bDepthPreview{false};

		// Read by the main thread; written by the vision thread after pipeline
		// startup (never dereference the pipeline from the main thread)
		std::atomic<const char*> activeEp{"none"};

		// Fusion input: this camera's latest processed result
		CameraFrameResult lastResult;

		// Preview handoff (mutex-guarded, latest-wins)
		std::mutex previewMutex;
		VisionPreviewFrame previewFrame;
		bool bPreviewFresh= false;

		// Whole-frame luminance oscillation (flicker / AE hunting) diagnostics
		LumaFlickerTracker flickerTracker;

		cv::Mat bgrScratch;
		cv::Mat undistortedScratch;
		// Points at whichever scratch mat the last processed frame ended up in
		// (vision thread only; stable between iterations for diagnostic dumps)
		const cv::Mat* lastActiveFrame= nullptr;
		double lastFrameTimestampMs= 0.0;
		float captureFps= 0.f;

		// Cross-camera seeding retry throttle (a failed speculative landmark
		// pass costs a few ms - don't pay it every frame)
		int hintCooldownFrames= 0;

		// Recording staging: this camera's inputs for the current iteration,
		// gathered in processCameraFrame and consumed when the frame record
		// is assembled after fusion
		RecordedCameraInput pendingRecordInput;
		bool bPendingRecordFresh= false;
	};

	void threadLoop();
	void refreshConfigOnThread();
	// Processes one newly popped frame for a context; returns true if a new
	// result was produced
	bool processCameraFrame(CameraContext& context);
	// RealSense depth: samples metric depth at each tracked hand's 2D
	// landmarks (mapping undistorted pipeline pixels back to raw sensor
	// pixels first). Returns false when the camera has no depth stream.
	bool sampleHandDepth(CameraContext& context, const struct CameraProfile& profile,
						 const TrackingFrameResult& result,
						 std::array<struct HandDepthMeasurement, 2>& outMeasurements);
	// Colorized depth image for the preview pane (false when the camera has
	// no depth stream)
	bool buildDepthPreview(CameraContext& context, const cv::Size& targetSize, cv::Mat& outBgr);
	// Cross-camera search seeding: hands the fused result tracks but this
	// camera lost get projected into its image as pipeline search hints
	void seedSearchHints(CameraContext& context, const TrackingFrameResult& lastFused);
	// Services a pending requestDiagnosticDump on the vision thread
	void performDiagnosticDump(const TrackingFrameResult& latestOutput);

	VideoCaptureSystem* m_videoCapture= nullptr;
	AppConfig* m_config= nullptr;

	std::vector<std::unique_ptr<CameraContext>> m_cameras;
	HandFusion m_fusion;

	// Wrist IMU service (devices + per-device orientation filters). Lives on
	// the vision thread; the UI reads snapshots through a mutex.
	ImuService m_imuService;
	std::atomic_bool m_bImuMountingCaptureRequested{false};
	std::atomic_bool m_bImuMotionRecordingRequested{false};
	std::atomic_int m_requestedImuMotionRecording{(int)eMountingMotion::None};
	std::atomic_bool m_bImuBiasCalibrationRequested{false};
	std::atomic_bool m_bImuBiasCancelRequested{false};
	mutable std::mutex m_imuMutex;
	ImuMountingCapture m_capturedImuMounting;
	bool m_bImuMountingReady= false;
	ImuSideStatus m_imuStatus[2];
	std::unique_ptr<OscStreamer> m_oscStreamer;

	std::thread m_thread;
	std::atomic_bool m_bRunning{false};
	std::atomic_bool m_bConfigRefreshRequested{true};
	std::atomic<float> m_lastInferenceMs{0.f};
	std::atomic_int m_hitchCount{0};
	std::atomic<float> m_lastHitchMs{0.f};
	std::atomic_int m_lastHitchPhase{(int)eVisionPhase::None};
	std::atomic<int> m_dominantCamera[2]= {-1, -1};
	std::atomic<float> m_autoScaleFactor{1.f};

	// Per (camera, side) confidence from the last fusion, published for the
	// UI readout. Fixed size so it needs no locking; cameras beyond this are
	// simply not reported.
	static constexpr int k_maxReportedCameras= 8;
	std::atomic<float> m_observationConfidence[k_maxReportedCameras * 2];

	// Fused result handoff (mutex-guarded, latest-wins)
	std::mutex m_fusedMutex;
	TrackingFrameResult m_fusedResult;
	bool m_bFusedFresh= false;

	// Rest-pose capture handoff
	std::atomic_bool m_bRestPoseCaptureRequested{false};
	std::mutex m_restPoseMutex;
	std::vector<RestPoseCapture> m_capturedRestPose;
	RestPoseCapture m_capturedFusedRest;
	bool m_bRestPoseReady= false;

	std::atomic_bool m_bBoneCalibrationRequested{false};
	std::atomic_bool m_bBoneCalibrationCancelRequested{false};
	std::atomic_bool m_bBoneCalibrationActive{false};
	std::atomic<float> m_boneCalibrationSeconds{0.f};
	std::atomic_int m_boneCalibrationSamples[2]{};
	// Vision-thread only
	HandBoneCalibrator m_boneCalibrator;
	double m_boneCalibrationEndMs= 0.0;
	std::mutex m_boneCalibrationMutex;
	BoneCalibrationCapture m_capturedBones;
	bool m_bBoneCalibrationReady= false;

	// Logs and publishes a loop iteration that overran the hitch threshold,
	// attributing it to the phase that consumed the most time
	void reportHitchIfSlow(double totalMs, const double* phaseMs);

	// Consumes a pending recording start request on the vision thread (state
	// reset + header snapshot + recorder start)
	void handleRecordingStartOnThread();
	// Finalizes an active recording on the vision thread
	void finalizeRecordingOnThread(bool bAborted, const std::string& reason);

	// Tracking recorder (vision thread owns it; UI reads atomics through the
	// accessors above)
	std::unique_ptr<TrackingRecorder> m_recorder;
	std::atomic_bool m_bRecordingStartRequested{false};
	std::atomic_bool m_bRecordingStopRequested{false};
	std::mutex m_recordingMutex;
	std::string m_requestedRecordingPath;
	std::string m_lastRecordingPath;
	int64_t m_recordingSeq= 0;

	// Diagnostic dump: history lives on the vision thread; the request path
	// and completion path are the only cross-thread strings (mutex-guarded)
	DiagnosticDump m_diagnostics;
	std::atomic_bool m_bDumpRequested{false};
	std::mutex m_dumpMutex;
	std::string m_requestedDumpDir;
	std::string m_lastDumpPath;
};
