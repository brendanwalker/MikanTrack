#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>

#include "opencv2/core/mat.hpp"

#include "TrackingTypes.h"

class AppConfig;
class VideoCaptureSystem;
class HandTrackingPipeline;
class LandmarkTo3D;
class OscStreamer;
class CVVideoFrameProcessor;

// Latest processed frame published to the main/render thread (latest-wins)
struct VisionPreviewFrame
{
	cv::Mat bgr; // undistorted when intrinsics are applied
	TrackingFrameResult result;
	bool valid= false;
};

// Owns the inference thread: drains camera frames, converts to BGR,
// optionally undistorts, runs the ML pipeline, projects to 3D, streams OSC,
// and publishes a preview frame for the UI.
class VisionThread
{
public:
	VisionThread(VideoCaptureSystem* videoCapture, AppConfig* config);
	~VisionThread();

	void start();
	void stop();

	// Pauses ML tracking (calibration wizards run their detection on the
	// main thread from the preview frames). Preview frames keep flowing.
	void setTrackingEnabled(bool bEnabled) { m_bTrackingEnabled= bEnabled; }
	bool isTrackingEnabled() const { return m_bTrackingEnabled; }

	// Copies the newest preview frame + result into outFrame.
	// Returns false if nothing new arrived since the last call.
	bool fetchPreviewFrame(VisionPreviewFrame& outFrame);

	// Re-reads config (intrinsics/extrinsics/hand scale/tracking/osc settings)
	// on the vision thread before the next frame.
	void requestConfigRefresh() { m_bConfigRefreshRequested= true; }

	// Introspection for the UI
	const char* getActiveExecutionProvider() const;
	float getLastInferenceMs() const { return m_lastInferenceMs; }

private:
	void threadLoop();
	void refreshConfigOnThread();

	VideoCaptureSystem* m_videoCapture= nullptr;
	AppConfig* m_config= nullptr;

	std::unique_ptr<HandTrackingPipeline> m_pipeline;
	std::unique_ptr<LandmarkTo3D> m_landmarkTo3D;
	std::unique_ptr<OscStreamer> m_oscStreamer;
	std::unique_ptr<CVVideoFrameProcessor> m_undistorter;

	std::thread m_thread;
	std::atomic_bool m_bRunning{false};
	std::atomic_bool m_bTrackingEnabled{true};
	std::atomic_bool m_bConfigRefreshRequested{true};
	std::atomic<float> m_lastInferenceMs{0.f};

	// Preview handoff (mutex-guarded, latest-wins)
	std::mutex m_previewMutex;
	VisionPreviewFrame m_previewFrame;
	bool m_bPreviewFresh= false;

	// Capture fps estimate
	double m_lastFrameTimestampMs= 0.0;
	float m_captureFps= 0.f;
};
