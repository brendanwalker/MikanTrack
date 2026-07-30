#pragma once

#include <string>
#include <vector>

class AppConfig;
class VideoCaptureSystem;

// Camera selection panel: device combo, resolution/frame-rate/format combos
// (best-match mode selection), stream start/stop.
class DevicePanel
{
public:
	DevicePanel(VideoCaptureSystem* videoCapture, AppConfig* config);

	void draw();

	// Re-reads the device list (call on hotplug events)
	void refreshDeviceList();
	// Re-reads the mode option lists from the open device
	void refreshModeOptions();

private:
	void applyModeSelection();

	VideoCaptureSystem* m_videoCapture;
	AppConfig* m_config;

	std::vector<std::string> m_devicePaths;
	std::vector<std::string> m_deviceNames;
	int m_selectedDeviceIndex= -1;

	std::vector<std::string> m_resolutionOptions;
	std::vector<std::string> m_frameRateOptions;
	std::vector<std::string> m_formatOptions;
	std::string m_selectedResolution;
	std::string m_selectedFrameRate;
	std::string m_selectedFormat;
};
