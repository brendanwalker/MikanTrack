#pragma once

#include <atomic>
#include <memory>
#include <mutex>

#include "opencv2/core/mat.hpp"

#include "HandFusion.h" // CameraFrameResult
#include "LumaFlickerTracker.h"
#include "TrackingRecording.h" // RecordedCameraInput
#include "TrackingTypes.h"

class AppConfig;
class BodyPoseTracker;
class CVVideoFrameProcessor;
class FrameRecorder;
struct HandSeedStats;
class HandTrackingPipeline;
class LandmarkTo3D;
class VideoCaptureSystem;

// Latest processed frame for one camera, published to the main/render thread
// (latest-wins). result is that camera's own (unfused, unsmoothed) tracking.
struct VisionPreviewFrame
{
	cv::Mat bgr; // undistorted when intrinsics are applied
	TrackingFrameResult result;
	bool valid= false;
};

// Everything one camera needs on the vision thread: the capture and inference
// stage. process() pops that camera's newest frame, converts it to BGR,
// optionally undistorts, runs the ML pipelines and the 3D projection, stages
// the recording inputs, and publishes the preview. No shared mutable state
// between contexts (the fusion step is the only join point), which keeps a
// later thread-per-camera upgrade possible.
//
// The main thread only touches the atomics and the preview handoff. The
// pipeline objects are created and destroyed on the vision thread, where the
// ONNX Runtime sessions inside them have to live and die.
class CameraContext
{
public:
	CameraContext(int cameraIndex, const AppConfig* config);
	~CameraContext();

	int getCameraIndex() const { return m_cameraIndex; }

	// -- Main thread -----

	// Pauses ML tracking for this camera (calibration wizards run their
	// detection on the main thread from the preview frames). Preview frames
	// keep flowing.
	void setTrackingEnabled(bool bEnabled) { m_bTrackingEnabled= bEnabled; }
	bool isTrackingEnabled() const { return m_bTrackingEnabled; }

	// Disable to receive raw distorted preview frames (needed while capturing
	// intrinsics calibration samples)
	void setUndistortEnabled(bool bEnabled) { m_bUndistortEnabled= bEnabled; }
	bool isUndistortEnabled() const { return m_bUndistortEnabled; }

	// Written by the vision thread after pipeline startup; the pipeline itself
	// is never dereferenced from the main thread
	const char* getActiveExecutionProvider() const { return m_activeEp.load(); }

	// Copies the newest preview frame + per-camera result. Returns false if
	// nothing new arrived since the last call.
	bool fetchPreviewFrame(VisionPreviewFrame& outFrame);

	// -- Vision thread -----

	// (Re)builds the pipeline, the 3D projection, the undistorter, and the
	// body-pose stage from this camera's profile. Invalidates the last result
	// so stale calibration state cannot leak through a config change.
	void configure(const AppConfig& config);

	// Resets the transient tracking state and invalidates the fusion input, so
	// a recording starts from the same zero state replay's fresh instances do
	void resetTransientState();

	// Processes this camera's newest frame, if one arrived. Returns true when
	// a new tracking result was produced.
	// seedFromFused: the previous iteration's fused world result, projected
	//   into this image as search hints for hands this camera lost; null with
	//   a single camera
	// autoScaleFactor: the current stereo hand-scale correction
	// bRecordingInputs: stage this frame's inputs for the tracking recording
	// frameRecorder: non-null only while raw frames are being captured
	bool process(VideoCaptureSystem* videoCapture, const TrackingFrameResult* seedFromFused,
				 float autoScaleFactor, bool bRecordingInputs, FrameRecorder* frameRecorder);

	// Fusion input: this camera's latest processed result
	const CameraFrameResult& getLastResult() const { return m_lastResult; }

	// Stereo auto hand-scale: the effective reference length the 3D
	// projection scales its object model by
	void setRefLengthMeters(float refLengthMeters);

	// Hands over the inputs staged by the last process() call, once
	bool consumePendingRecordInput(RecordedCameraInput& outInput);

	// The frame the last process() call ran on (vision thread only; stable
	// between iterations for diagnostic dumps)
	const cv::Mat* getLastActiveFrame() const { return m_lastActiveFrame; }
	// Cross-camera seed accounting; null when the camera has no pipeline
	const HandSeedStats* getSeedStats() const;

private:
	// Cross-camera search seeding: hands the fused result tracks but this
	// camera lost get projected into its image as pipeline search hints
	void seedSearchHints(const TrackingFrameResult& lastFused, float autoScaleFactor);

	int m_cameraIndex= -1;
	const AppConfig* m_config= nullptr;

	std::unique_ptr<HandTrackingPipeline> m_pipeline;
	std::unique_ptr<LandmarkTo3D> m_landmarkTo3D; // smoothing always disabled (post-fusion smoothing)
	std::unique_ptr<CVVideoFrameProcessor> m_undistorter;
	// Opt-in body-pose stage; only allocated for cameras whose profile
	// enables body pose
	std::unique_ptr<BodyPoseTracker> m_bodyPoseTracker;

	std::atomic_bool m_bTrackingEnabled{true};
	std::atomic_bool m_bUndistortEnabled{true};
	std::atomic<const char*> m_activeEp{"none"};

	CameraFrameResult m_lastResult;

	// Preview handoff (mutex-guarded, latest-wins)
	std::mutex m_previewMutex;
	VisionPreviewFrame m_previewFrame;
	bool m_bPreviewFresh= false;

	// Whole-frame luminance oscillation (flicker / AE hunting) diagnostics
	LumaFlickerTracker m_flickerTracker;

	cv::Mat m_bgrScratch;
	cv::Mat m_undistortedScratch;
	// Points at whichever scratch mat the last processed frame ended up in
	const cv::Mat* m_lastActiveFrame= nullptr;
	double m_lastFrameTimestampMs= 0.0;
	float m_captureFps= 0.f;

	// Cross-camera seeding retry throttle (a failed speculative landmark
	// pass costs a few ms - don't pay it every frame)
	int m_hintCooldownFrames= 0;

	// Recording staging: this camera's inputs for the current iteration,
	// gathered in process() and consumed when the frame record is assembled
	// after fusion
	RecordedCameraInput m_pendingRecordInput;
	bool m_bPendingRecordFresh= false;
};
