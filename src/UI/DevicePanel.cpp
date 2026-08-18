#include "DevicePanel.h"

#include "imgui.h"

#include "LocText.h"

#include "App.h"
#include "AppConfig.h"
#include "Logger.h"
#include "VideoCaptureSystem.h"
#include "VideoModeUtils.h"

static constexpr int k_maxCameras= 4;

DevicePanel::DevicePanel(App* app, VideoCaptureSystem* videoCapture, AppConfig* config)
	: m_app(app)
	, m_videoCapture(videoCapture)
	, m_config(config)
{
	refreshDeviceList();
}

void DevicePanel::syncCameraStateCount()
{
	m_cameraStates.resize(m_config->cameraCount());
}

void DevicePanel::refreshDeviceList()
{
	m_devicePaths.clear();
	m_deviceNames.clear();

	const size_t deviceCount= m_videoCapture->getDeviceCount();
	for (size_t i= 0; i < deviceCount; ++i)
	{
		std::string path, name;
		if (m_videoCapture->getDevicePath(i, path) && m_videoCapture->getDeviceFriendlyName(i, name))
		{
			m_devicePaths.push_back(path);
			m_deviceNames.push_back(name);
		}
	}

	// Re-derive each camera's selected device index from its open device
	syncCameraStateCount();
	for (int cameraIndex= 0; cameraIndex < (int)m_cameraStates.size(); ++cameraIndex)
	{
		CameraUiState& state= m_cameraStates[cameraIndex];
		state.selectedDeviceIndex= -1;

		if (m_videoCapture->getIsDeviceOpen(cameraIndex))
		{
			const std::string openPath= m_videoCapture->getCurrentDevicePath(cameraIndex);
			for (int i= 0; i < (int)m_devicePaths.size(); ++i)
			{
				if (m_devicePaths[i] == openPath)
				{
					state.selectedDeviceIndex= i;
					break;
				}
			}
		}
	}
}

void DevicePanel::refreshCameraSettings(int cameraIndex)
{
	CameraUiState& state= m_cameraStates[cameraIndex];
	state.settings.clear();

	const IUsbVideoDevice* device= m_videoCapture->getCurrentDevice(cameraIndex);
	if (device == nullptr)
		return;

	// Curated, in a useful order; Exposure first (it's the one that halves
	// your frame rate in low light)
	static const struct
	{
		eVideoSettingType type;
		const char* name;
	} kSettings[]= {
		{eVideoSettingType::Exposure, "devicePanel.exposureSlider"},
		{eVideoSettingType::Gain, "devicePanel.gainSlider"},
		{eVideoSettingType::Brightness, "devicePanel.brightnessSlider"},
		{eVideoSettingType::Contrast, "devicePanel.contrastSlider"},
		{eVideoSettingType::Saturation, "devicePanel.saturationSlider"},
		{eVideoSettingType::Sharpness, "devicePanel.sharpnessSlider"},
		{eVideoSettingType::WhiteBalance, "devicePanel.whiteBalanceSlider"},
		{eVideoSettingType::Focus, "devicePanel.focusSlider"},
		{eVideoSettingType::Zoom, "devicePanel.zoomSlider"},
	};

	for (const auto& settingInfo : kSettings)
	{
		if (!device->isVideoSettingSupported(settingInfo.type))
			continue;

		CameraSettingUi setting;
		setting.type= settingInfo.type;
		setting.name= settingInfo.name;
		if (!device->getVideoSettingConstraint(settingInfo.type, setting.constraint))
			continue;
		setting.value= device->getVideoSetting(settingInfo.type);
		state.settings.push_back(setting);
	}
}

void DevicePanel::drawCameraSettings(int cameraIndex)
{
	CameraUiState& state= m_cameraStates[cameraIndex];
	if (state.settings.empty())
		return;

	if (!ImGui::TreeNode(locLabel("devicePanel.cameraSettingsHeader")))
		return;

	IUsbVideoDevice* device= m_videoCapture->getCurrentDevice(cameraIndex);
	if (device != nullptr)
	{
		for (CameraSettingUi& setting : state.settings)
		{
			if (ImGui::SliderInt(locLabel(setting.name), &setting.value, setting.constraint.min_value,
								 setting.constraint.max_value))
			{
				device->setVideoSetting(setting.type, setting.value);
			}
			if (setting.type == eVideoSettingType::Exposure)
			{
				ImGui::SetItemTooltip("%s", locText("devicePanel.exposureTooltip"));
			}
		}

		if (ImGui::Button(locLabel("devicePanel.resetDefaultsButton")))
		{
			for (CameraSettingUi& setting : state.settings)
			{
				device->setVideoSetting(setting.type, setting.constraint.default_value);
				setting.value= setting.constraint.default_value;
			}
		}
		ImGui::SameLine();
		if (ImGui::Button(locLabel("devicePanel.rereadButton")))
			refreshCameraSettings(cameraIndex);
	}

	ImGui::TreePop();
}

void DevicePanel::refreshModeOptions(int cameraIndex)
{
	syncCameraStateCount();
	if (cameraIndex < 0 || cameraIndex >= (int)m_cameraStates.size())
		return;

	CameraUiState& state= m_cameraStates[cameraIndex];
	state.resolutionOptions.clear();
	state.frameRateOptions.clear();
	state.formatOptions.clear();

	const IUsbVideoDevice* device= m_videoCapture->getCurrentDevice(cameraIndex);
	if (device == nullptr)
		return;

	VideoModeUtils::getVideoModeOptionLists(device, state.resolutionOptions, state.frameRateOptions,
											state.formatOptions);
	VideoModeUtils::getVideoModeResolutionName(device, state.selectedResolution);
	VideoModeUtils::getVideoModeFrameRateName(device, state.selectedFrameRate);
	VideoModeUtils::getVideoModeFormatName(device, state.selectedFormat);

	refreshCameraSettings(cameraIndex);
}

void DevicePanel::applyModeSelection(int cameraIndex)
{
	CameraUiState& state= m_cameraStates[cameraIndex];
	IUsbVideoDevice* device= m_videoCapture->getCurrentDevice(cameraIndex);
	if (device == nullptr)
		return;

	const std::string newModeName= VideoModeUtils::findBestVideoModeName(
		device, state.selectedResolution, state.selectedFrameRate, state.selectedFormat);
	if (!newModeName.empty() && newModeName != m_videoCapture->getCurrentVideoModeName(cameraIndex))
	{
		m_videoCapture->setVideoModeByName(cameraIndex, newModeName);
		m_config->camera(cameraIndex).video.modeName= m_videoCapture->getCurrentVideoModeName(cameraIndex);
		m_config->markDirty();
	}

	// The device may have snapped to a different combination than requested
	refreshModeOptions(cameraIndex);
}

void DevicePanel::drawCameraSection(int cameraIndex)
{
	CameraUiState& state= m_cameraStates[cameraIndex];

	ImGui::PushID(cameraIndex);

	const std::string headerLabel= locFormat("devicePanel.cameraHeaderFmt", cameraIndex + 1);
	if (ImGui::CollapsingHeader(headerLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
	{
		// Device combo (devices open in another slot are disabled)
		const char* currentDeviceLabel=
			state.selectedDeviceIndex >= 0 ? m_deviceNames[state.selectedDeviceIndex].c_str()
											: locText("devicePanel.selectCameraPlaceholder");
		if (ImGui::BeginCombo(locLabel("devicePanel.deviceCombo"), currentDeviceLabel))
		{
			for (int i= 0; i < (int)m_deviceNames.size(); ++i)
			{
				const bool bSelected= i == state.selectedDeviceIndex;
				const bool bTakenElsewhere= m_videoCapture->isDevicePathOpenElsewhere(m_devicePaths[i], cameraIndex);

				ImGui::BeginDisabled(bTakenElsewhere);
				if (ImGui::Selectable(m_deviceNames[i].c_str(), bSelected) && !bSelected)
				{
					state.selectedDeviceIndex= i;
					if (m_videoCapture->openDeviceByPath(cameraIndex, m_devicePaths[i]))
					{
						CameraProfile& profile= m_config->camera(cameraIndex);
						profile.video.devicePath= m_devicePaths[i];
						profile.video.deviceName= m_deviceNames[i];
						profile.video.modeName= m_videoCapture->getCurrentVideoModeName(cameraIndex);
						m_config->markDirty();
						refreshModeOptions(cameraIndex);
						m_videoCapture->startStream(cameraIndex);
					}
				}
				ImGui::EndDisabled();
				if (bSelected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}

		const bool bDeviceOpen= m_videoCapture->getIsDeviceOpen(cameraIndex);
		ImGui::BeginDisabled(!bDeviceOpen);

		// Mode combos
		auto drawOptionCombo= [](const char* label, std::vector<std::string>& options, std::string& selection) {
			bool bChanged= false;
			if (ImGui::BeginCombo(label, selection.c_str()))
			{
				for (const std::string& option : options)
				{
					const bool bSelected= option == selection;
					if (ImGui::Selectable(option.c_str(), bSelected) && !bSelected)
					{
						selection= option;
						bChanged= true;
					}
					if (bSelected)
						ImGui::SetItemDefaultFocus();
				}
				ImGui::EndCombo();
			}
			return bChanged;
		};

		bool bModeChanged= false;
		bModeChanged|= drawOptionCombo(locLabel("devicePanel.resolutionCombo"), state.resolutionOptions, state.selectedResolution);
		bModeChanged|= drawOptionCombo(locLabel("devicePanel.frameRateCombo"), state.frameRateOptions, state.selectedFrameRate);
		bModeChanged|= drawOptionCombo(locLabel("devicePanel.formatCombo"), state.formatOptions, state.selectedFormat);
		if (bModeChanged)
			applyModeSelection(cameraIndex);

		// Stream control
		if (m_videoCapture->isStreaming(cameraIndex))
		{
			if (ImGui::Button(locLabel("devicePanel.stopStreamButton"), ImVec2(-1, 0)))
				m_videoCapture->stopStream(cameraIndex);
		}
		else
		{
			if (ImGui::Button(locLabel("devicePanel.startStreamButton"), ImVec2(-1, 0)))
				m_videoCapture->startStream(cameraIndex);
		}

		ImGui::EndDisabled();

		if (bDeviceOpen)
		{
			ImGui::TextDisabled(locText("devicePanel.modeFmt"), m_videoCapture->getCurrentVideoModeName(cameraIndex).c_str());

			// Device-side delivery rate: if this reads below the mode's rate,
			// the CAMERA is the bottleneck (usually auto-exposure in dim light
			// or USB bandwidth), not the processing pipeline
			if (m_videoCapture->isStreaming(cameraIndex))
				ImGui::TextDisabled(locText("devicePanel.deviceDeliveringFmt"),
									m_videoCapture->getDeviceFrameRate(cameraIndex));

			const uint64_t droppedFrames= m_videoCapture->getDroppedFrameCount(cameraIndex);
			if (droppedFrames > 0)
				ImGui::TextDisabled(locText("devicePanel.droppedFramesFmt"), (unsigned long long)droppedFrames);

			drawCameraSettings(cameraIndex);
		}

		// Remove (never camera 0)
		if (cameraIndex > 0)
		{
			if (ImGui::Button(locLabel("devicePanel.removeCameraButton"), ImVec2(-1, 0)))
			{
				m_videoCapture->closeDevice(cameraIndex);
				m_config->cameras.erase(m_config->cameras.begin() + cameraIndex);
				m_app->applyCameraCountChange();
				refreshDeviceList();
			}
		}
	}

	ImGui::PopID();
}

void DevicePanel::draw()
{
	if (!ImGui::Begin(locWindowTitle("windows.device")))
	{
		ImGui::End();
		return;
	}

	syncCameraStateCount();

	if (ImGui::SmallButton(locLabel("devicePanel.rescanDevicesButton")))
	{
		m_videoCapture->refreshDeviceList();
		refreshDeviceList();
	}

	for (int cameraIndex= 0; cameraIndex < (int)m_cameraStates.size(); ++cameraIndex)
		drawCameraSection(cameraIndex);

	ImGui::Separator();
	ImGui::BeginDisabled((int)m_config->cameraCount() >= k_maxCameras);
	if (ImGui::Button(locLabel("devicePanel.addCameraButton"), ImVec2(-1, 0)))
	{
		CameraProfile newProfile;
		// New cameras default to 720p - two uncompressed 1080p streams can
		// exceed one USB controller's bandwidth
		newProfile.video.modeName= "1280x720@30fps (NV12)";
		m_config->cameras.push_back(newProfile);
		m_app->applyCameraCountChange();
		refreshDeviceList();
	}
	ImGui::EndDisabled();

	ImGui::End();
}
