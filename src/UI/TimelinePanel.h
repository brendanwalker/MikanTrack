#pragma once

#include <string>
#include <vector>

#include "Scene3dPanel.h" // SceneCameraView
#include "TrackingReplay.h"
#include "TrackingTypes.h"

class AppConfig;
class VisionThread;

// Timeline panel: tracking-recording control (record/stop), recording loader,
// deterministic replay transport (scrub/step/play), checksum divergence
// markers, and what-if re-simulation with edited fusion parameters. Owns the
// TrackingReplay instance - replay is a main-thread concern; live tracking
// and OSC keep running while a recording is being inspected.
class TimelinePanel
{
public:
	void draw(AppConfig* config, VisionThread* visionThread);

	// When active, MainWindow feeds the 3D scene from the replay instead of
	// the live tracking output
	bool isReplayViewActive() const { return m_bReplayViewActive && m_bLoaded && m_replay.didRun(); }
	// Current frame's fused result for the selected view (Recorded / Replayed
	// / What-if), IMU forearm overlay included
	const TrackingFrameResult& getDisplayFused() const { return m_displayFused; }
	// Camera frustums from the RECORDING's config snapshot (not the live one)
	std::vector<SceneCameraView> getSceneCameras() const;
	// Current frame's replayed per-camera results (stale cameras carry their
	// last fresh result, like the live preview panes). Pointers are valid for
	// the rest of the current UI frame.
	std::vector<const TrackingFrameResult*> getPerCameraResults() const;

private:
	void drawRecordingSection(VisionThread* visionThread);
	void drawLoadSection(VisionThread* visionThread);
	void drawTransportSection();
	void drawWhatIfSection();
	void drawFrameInfoSection();
	void refreshRecordingList();
	void setCurrentFrame(int frameIndex);
	// Rebuilds m_displayFused/m_perCameraCopies for the current frame + view
	void rebuildDisplayFrame();
	// The recorded (live) output reconstructed as a displayable result
	void buildRecordedResult(const RecordedFrame& recorded, TrackingFrameResult& outResult) const;

	TrackingReplay m_replay;
	bool m_bLoaded= false;
	std::string m_loadStatus;

	bool m_bReplayViewActive= false;
	int m_currentFrame= 0;
	int m_viewMode= 1; // 0=Recorded, 1=Replayed, 2=What-if

	bool m_bPlaying= false;
	float m_playSpeed= 1.f;
	double m_playheadMs= 0.0; // recorded-time playhead while playing

	std::vector<std::string> m_recordingFiles;
	int m_selectedRecording= -1;

	TrackingReplay::WhatIfParams m_whatIfParams;
	bool m_bWhatIfSeeded= false;
	std::string m_whatIfStatus;

	// Current-frame display data (rebuilt on frame/view change)
	TrackingFrameResult m_displayFused;
	std::vector<TrackingFrameResult> m_perCameraCopies;
	std::vector<bool> m_perCameraCopyValid;
	int m_builtFrame= -1;
	int m_builtViewMode= -1;
};
