#pragma once

class AppConfig;
class VisionThread;
class VideoPreviewPanel;

// Small side panels: tracking options and OSC output settings.
// Config edits mark the config dirty and request a vision-thread refresh.
namespace SettingsPanels
{
void drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel);
void drawOscPanel(AppConfig* config, VisionThread* visionThread);
}
