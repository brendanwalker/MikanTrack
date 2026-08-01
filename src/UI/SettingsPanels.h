#pragma once

class AppConfig;
class VisionThread;
class VideoPreviewPanel;
class Scene3dPanel;
struct TrackingFrameResult;

// Frame-to-frame state for the tracking panel (owned by MainWindow, since the
// panel itself is a set of free functions)
struct TrackingPanelState
{
	// Seconds left before the rest-pose capture fires; <= 0 = not counting
	float restPoseCountdown= 0.f;
	// Transient result banner after a capture
	float restPoseResultTimer= 0.f;
	bool bRestPoseResultCaptured[2]= {false, false};
};

// Small side panels: tracking/fusion options and OSC output settings.
// Config edits mark the config dirty and request a vision-thread refresh.
namespace SettingsPanels
{
void drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel,
					   Scene3dPanel* scene3dPanel, TrackingPanelState& panelState);
// fusedResult: the latest fused tracking output (for the live angle readout)
void drawOscPanel(AppConfig* config, VisionThread* visionThread, const TrackingFrameResult& fusedResult);
}
