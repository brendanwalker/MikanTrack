#pragma once

class AppConfig;
class VisionThread;
class VideoPreviewPanel;
class Scene3dPanel;

// Small side panels: tracking/fusion options and OSC output settings.
// Config edits mark the config dirty and request a vision-thread refresh.
namespace SettingsPanels
{
void drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel,
					   Scene3dPanel* scene3dPanel);
void drawOscPanel(AppConfig* config, VisionThread* visionThread);
}
