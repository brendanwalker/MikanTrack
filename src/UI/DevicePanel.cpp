#include "DevicePanel.h"

#include "imgui.h"

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

	char headerLabel[64];
	snprintf(headerLabel, sizeof(headerLabel), "Camera %d", cameraIndex + 1);
	if (ImGui::CollapsingHeader(headerLabel, ImGuiTreeNodeFlags_DefaultOpen))
	{
		// Device combo (devices open in another slot are disabled)
		const char* currentDeviceLabel=
			state.selectedDeviceIndex >= 0 ? m_deviceNames[state.selectedDeviceIndex].c_str() : "<select camera>";
		if (ImGui::BeginCombo("Device", currentDeviceLabel))
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
		bModeChanged|= drawOptionCombo("Resolution", state.resolutionOptions, state.selectedResolution);
		bModeChanged|= drawOptionCombo("Frame Rate", state.frameRateOptions, state.selectedFrameRate);
		bModeChanged|= drawOptionCombo("Format", state.formatOptions, state.selectedFormat);
		if (bModeChanged)
			applyModeSelection(cameraIndex);

		// Stream control
		if (m_videoCapture->isStreaming(cameraIndex))
		{
			if (ImGui::Button("Stop Stream", ImVec2(-1, 0)))
				m_videoCapture->stopStream(cameraIndex);
		}
		else
		{
			if (ImGui::Button("Start Stream", ImVec2(-1, 0)))
				m_videoCapture->startStream(cameraIndex);
		}

		ImGui::EndDisabled();

		if (bDeviceOpen)
		{
			ImGui::TextDisabled("Mode: %s", m_videoCapture->getCurrentVideoModeName(cameraIndex).c_str());
			const uint64_t droppedFrames= m_videoCapture->getDroppedFrameCount(cameraIndex);
			if (droppedFrames > 0)
				ImGui::TextDisabled("Dropped frames: %llu", (unsigned long long)droppedFrames);
		}

		// Remove (never camera 0)
		if (cameraIndex > 0)
		{
			if (ImGui::Button("Remove Camera", ImVec2(-1, 0)))
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
	if (!ImGui::Begin("Device"))
	{
		ImGui::End();
		return;
	}

	syncCameraStateCount();

	if (ImGui::SmallButton("Rescan Devices"))
	{
		m_videoCapture->refreshDeviceList();
		refreshDeviceList();
	}

	for (int cameraIndex= 0; cameraIndex < (int)m_cameraStates.size(); ++cameraIndex)
		drawCameraSection(cameraIndex);

	ImGui::Separator();
	ImGui::BeginDisabled((int)m_config->cameraCount() >= k_maxCameras);
	if (ImGui::Button("Add Camera", ImVec2(-1, 0)))
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
