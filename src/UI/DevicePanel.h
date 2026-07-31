#pragma once

#include <string>
#include <vector>

class App;
class AppConfig;
class VideoCaptureSystem;

// Camera selection panel: one section per configured camera (device combo,
// resolution/frame-rate/format combos with best-match mode selection, stream
// start/stop), plus add/remove camera controls.
class DevicePanel
{
public:
	DevicePanel(App* app, VideoCaptureSystem* videoCapture, AppConfig* config);

	void draw();

	// Re-reads the device list (call on hotplug events)
	void refreshDeviceList();
	// Re-reads the mode option lists from one camera's open device
	void refreshModeOptions(int cameraIndex);

private:
	struct CameraUiState
	{
		int selectedDeviceIndex= -1;
		std::vector<std::string> resolutionOptions;
		std::vector<std::string> frameRateOptions;
		std::vector<std::string> formatOptions;
		std::string selectedResolution;
		std::string selectedFrameRate;
		std::string selectedFormat;
	};

	void syncCameraStateCount();
	void drawCameraSection(int cameraIndex);
	void applyModeSelection(int cameraIndex);

	App* m_app;
	VideoCaptureSystem* m_videoCapture;
	AppConfig* m_config;

	// Global device enumeration (shared by all camera sections)
	std::vector<std::string> m_devicePaths;
	std::vector<std::string> m_deviceNames;

	std::vector<CameraUiState> m_cameraStates;
};
