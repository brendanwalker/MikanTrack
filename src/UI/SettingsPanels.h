#pragma once

class AppConfig;
class VisionThread;
class VideoPreviewPanel;
class Scene3dPanel;
struct TrackingFrameResult;

// Small side panels: tracking/fusion options and OSC output settings.
// Config edits mark the config dirty and request a vision-thread refresh.
namespace SettingsPanels
{
void drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel,
					   Scene3dPanel* scene3dPanel);
// fusedResult: the latest fused tracking output (for the live angle readout)
void drawOscPanel(AppConfig* config, VisionThread* visionThread, const TrackingFrameResult& fusedResult);
}
