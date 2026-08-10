#pragma once

#include <array>
#include <vector>

#include "glm/ext/vector_double3.hpp"

#include "TrackingTypes.h"

class AppConfig;
class VisionThread;
class VideoPreviewPanel;
class Scene3dPanel;
struct VisionPreviewFrame;

// Frame-to-frame state for the tracking panel (owned by MainWindow, since the
// panel itself is a set of free functions)
struct TrackingPanelState
{
	// Seconds left before the rest-pose capture fires; <= 0 = not counting
	float restPoseCountdown= 0.f;
	// Transient result banner after a capture
	float restPoseResultTimer= 0.f;
	bool bRestPoseResultCaptured[2]= {false, false};
	// Set when the panel wants the wrist IMU mounting wizard opened; the owner
	// consumes and clears it (the mounting calibration is a multi-step guided
	// flow, not something a panel button can drive)
	bool bLaunchMountingWizard= false;

	// UI smoothing for the image-quality readout: the per-frame values are
	// twitchy at camera rate, so the panel shows a ~1s EMA (the dump keeps the
	// raw per-frame series). Indexed [camera][metric row].
	static constexpr int kQualityMetricCount= 8;
	static constexpr int kQualityMaxCameras= 8;
	float qualityEma[kQualityMaxCameras][kQualityMetricCount]= {};
	bool bQualityEmaValid[kQualityMaxCameras][kQualityMetricCount]= {};

	// Hold-still jitter test: countdown, then a sampling window over the
	// fused output. The result is THE number for A/B-ing camera settings and
	// lighting changes - one repeatable measurement per configuration.
	float holdStillCountdown= 0.f;   // seconds before sampling starts; <= 0 idle
	float holdStillSecondsLeft= 0.f; // seconds left in the sampling window
	double holdStillLastTimestampMs= 0.0; // dedupe: fused results arrive slower than UI frames
	struct HoldStillAccum
	{
		int sampleCount= 0;
		std::array<glm::dvec3, HAND_LANDMARK_COUNT> sum{};
		std::array<glm::dvec3, HAND_LANDMARK_COUNT> sumSq{};
	};
	HoldStillAccum holdStillAccum[2];
	bool bHoldStillHasResult= false;
	// Mean per-landmark deviation over the window, mm; -1 = too few samples
	float holdStillDeviationMm[2]= {-1.f, -1.f};
};

// Small side panels: tracking/fusion options and OSC output settings.
// Config edits mark the config dirty and request a vision-thread refresh.
namespace SettingsPanels
{
// latestPreviews: newest per-camera results (image-quality readout);
// fusedResult: the latest fused tracking output (hold-still jitter test)
void drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel,
					   Scene3dPanel* scene3dPanel, TrackingPanelState& panelState,
					   const std::vector<VisionPreviewFrame>& latestPreviews,
					   const TrackingFrameResult& fusedResult);
// fusedResult: the latest fused tracking output (for the live angle readout)
void drawOscPanel(AppConfig* config, VisionThread* visionThread, const TrackingFrameResult& fusedResult);
}
