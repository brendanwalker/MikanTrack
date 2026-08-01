#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "DiagnosticDump.h"
#include "HandFusion.h" // CameraFrameResult
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

	// Copies the newest preview frame + per-camera result for a camera.
	// Returns false if nothing new arrived since the last call.
	bool fetchPreviewFrame(int cameraIndex, VisionPreviewFrame& outFrame);

	// Copies the newest fused (world-space, smoothed) tracking result.
	// Returns false if nothing new arrived since the last call.
	bool fetchFusedResult(TrackingFrameResult& outResult);

	// Per-side camera index that dominated the last fusion (-1 = untracked)
	int getDominantCamera(eHandSide side) const { return m_dominantCamera[(int)side].load(); }

	// Stereo auto hand-scale: current correction factor over the configured
	// hand scale (1 = unchanged), refined from cross-camera wrist
	// triangulation when enabled and both cameras see the same hand
	float getAutoHandScaleFactor() const { return m_autoScaleFactor.load(); }

	// Re-reads config (cameras/intrinsics/extrinsics/hand scale/tracking/
	// fusion/osc) on the vision thread before the next frame
	void requestConfigRefresh() { m_bConfigRefreshRequested= true; }

	// Diagnostic dump (F9): the vision thread writes its rolling state history,
	// the current camera frames (raw + annotated PNGs) and the live config to
	// dumpDir on its next loop iteration
	void requestDiagnosticDump(const std::string& dumpDir);
	// Directory of the last completed dump ("" until one succeeds)
	std::string getLastDumpPath();

	// Introspection for the UI
	const char* getActiveExecutionProvider(int cameraIndex= 0) const;
	float getLastInferenceMs() const { return m_lastInferenceMs; } // summed across cameras

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

		// Read by the main thread; written by the vision thread after pipeline
		// startup (never dereference the pipeline from the main thread)
		std::atomic<const char*> activeEp{"none"};

		// Fusion input: this camera's latest processed result
		CameraFrameResult lastResult;

		// Preview handoff (mutex-guarded, latest-wins)
		std::mutex previewMutex;
		VisionPreviewFrame previewFrame;
		bool bPreviewFresh= false;

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
	};

	void threadLoop();
	void refreshConfigOnThread();
	// Processes one newly popped frame for a context; returns true if a new
	// result was produced
	bool processCameraFrame(CameraContext& context);
	// Cross-camera search seeding: hands the fused result tracks but this
	// camera lost get projected into its image as pipeline search hints
	void seedSearchHints(CameraContext& context, const TrackingFrameResult& lastFused);
	// Services a pending requestDiagnosticDump on the vision thread
	void performDiagnosticDump(const TrackingFrameResult& latestOutput);

	VideoCaptureSystem* m_videoCapture= nullptr;
	AppConfig* m_config= nullptr;

	std::vector<std::unique_ptr<CameraContext>> m_cameras;
	HandFusion m_fusion;
	std::unique_ptr<OscStreamer> m_oscStreamer;

	std::thread m_thread;
	std::atomic_bool m_bRunning{false};
	std::atomic_bool m_bConfigRefreshRequested{true};
	std::atomic<float> m_lastInferenceMs{0.f};
	std::atomic<int> m_dominantCamera[2]= {-1, -1};
	std::atomic<float> m_autoScaleFactor{1.f};

	// Fused result handoff (mutex-guarded, latest-wins)
	std::mutex m_fusedMutex;
	TrackingFrameResult m_fusedResult;
	bool m_bFusedFresh= false;

	// Diagnostic dump: history lives on the vision thread; the request path
	// and completion path are the only cross-thread strings (mutex-guarded)
	DiagnosticDump m_diagnostics;
	std::atomic_bool m_bDumpRequested{false};
	std::mutex m_dumpMutex;
	std::string m_requestedDumpDir;
	std::string m_lastDumpPath;
};
