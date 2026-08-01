#pragma once

#include <string>
#include <vector>

#include "IVideoDevice.h" // eVideoSettingType, VideoSettingConstraint

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
	struct CameraSettingUi
	{
		eVideoSettingType type= eVideoSettingType::INVALID;
		const char* name= "";
		VideoSettingConstraint constraint;
		int value= 0;
	};

	struct CameraUiState
	{
		int selectedDeviceIndex= -1;
		std::vector<std::string> resolutionOptions;
		std::vector<std::string> frameRateOptions;
		std::vector<std::string> formatOptions;
		std::string selectedResolution;
		std::string selectedFrameRate;
		std::string selectedFormat;
		// Supported ProcAmp/CameraControl settings (cached on open/refresh)
		std::vector<CameraSettingUi> settings;
	};

	void syncCameraStateCount();
	void drawCameraSection(int cameraIndex);
	void drawCameraSettings(int cameraIndex);
	void applyModeSelection(int cameraIndex);
	void refreshCameraSettings(int cameraIndex);

	App* m_app;
	VideoCaptureSystem* m_videoCapture;
	AppConfig* m_config;

	// Global device enumeration (shared by all camera sections)
	std::vector<std::string> m_devicePaths;
	std::vector<std::string> m_deviceNames;

	std::vector<CameraUiState> m_cameraStates;
};
