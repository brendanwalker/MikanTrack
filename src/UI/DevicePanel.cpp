#include "DevicePanel.h"

#include "imgui.h"

#include "AppConfig.h"
#include "Logger.h"
#include "VideoCaptureSystem.h"
#include "VideoModeUtils.h"

DevicePanel::DevicePanel(VideoCaptureSystem* videoCapture, AppConfig* config)
	: m_videoCapture(videoCapture)
	, m_config(config)
{
	refreshDeviceList();
}

void DevicePanel::refreshDeviceList()
{
	m_devicePaths.clear();
	m_deviceNames.clear();
	m_selectedDeviceIndex= -1;

	const size_t deviceCount= m_videoCapture->getDeviceCount();
	for (size_t i= 0; i < deviceCount; ++i)
	{
		std::string path, name;
		if (m_videoCapture->getDevicePath(i, path) && m_videoCapture->getDeviceFriendlyName(i, name))
		{
			m_devicePaths.push_back(path);
			m_deviceNames.push_back(name);

			if (m_videoCapture->getIsDeviceOpen() && path == m_videoCapture->getCurrentDevicePath())
				m_selectedDeviceIndex= (int)m_devicePaths.size() - 1;
		}
	}
}

void DevicePanel::refreshModeOptions()
{
	m_resolutionOptions.clear();
	m_frameRateOptions.clear();
	m_formatOptions.clear();

	const IUsbVideoDevice* device= m_videoCapture->getCurrentDevice();
	if (device == nullptr)
		return;

	VideoModeUtils::getVideoModeOptionLists(device, m_resolutionOptions, m_frameRateOptions, m_formatOptions);
	VideoModeUtils::getVideoModeResolutionName(device, m_selectedResolution);
	VideoModeUtils::getVideoModeFrameRateName(device, m_selectedFrameRate);
	VideoModeUtils::getVideoModeFormatName(device, m_selectedFormat);
}

void DevicePanel::applyModeSelection()
{
	IUsbVideoDevice* device= m_videoCapture->getCurrentDevice();
	if (device == nullptr)
		return;

	const std::string newModeName=
		VideoModeUtils::findBestVideoModeName(device, m_selectedResolution, m_selectedFrameRate, m_selectedFormat);
	if (!newModeName.empty() && newModeName != m_videoCapture->getCurrentVideoModeName())
	{
		m_videoCapture->setVideoModeByName(newModeName);
		m_config->camera(0).video.modeName= m_videoCapture->getCurrentVideoModeName();
		m_config->markDirty();
	}

	// The device may have snapped to a different combination than requested
	refreshModeOptions();
}

void DevicePanel::draw()
{
	if (!ImGui::Begin("Device"))
	{
		ImGui::End();
		return;
	}

	// Device combo
	const char* currentDeviceLabel=
		m_selectedDeviceIndex >= 0 ? m_deviceNames[m_selectedDeviceIndex].c_str() : "<select camera>";
	if (ImGui::BeginCombo("Camera", currentDeviceLabel))
	{
		for (int i= 0; i < (int)m_deviceNames.size(); ++i)
		{
			const bool bSelected= i == m_selectedDeviceIndex;
			if (ImGui::Selectable(m_deviceNames[i].c_str(), bSelected) && !bSelected)
			{
				m_selectedDeviceIndex= i;
				if (m_videoCapture->openDeviceByPath(m_devicePaths[i]))
				{
					m_config->camera(0).video.devicePath= m_devicePaths[i];
					m_config->camera(0).video.deviceName= m_deviceNames[i];
					m_config->camera(0).video.modeName= m_videoCapture->getCurrentVideoModeName();
					m_config->markDirty();
					refreshModeOptions();
					m_videoCapture->startStream();
				}
			}
			if (bSelected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::SameLine();
	if (ImGui::SmallButton("Rescan"))
	{
		m_videoCapture->refreshDeviceList();
		refreshDeviceList();
	}

	const bool bDeviceOpen= m_videoCapture->getIsDeviceOpen();
	ImGui::BeginDisabled(!bDeviceOpen);

	// Mode combos
	auto drawOptionCombo= [this](const char* label, std::vector<std::string>& options, std::string& selection) {
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
	bModeChanged|= drawOptionCombo("Resolution", m_resolutionOptions, m_selectedResolution);
	bModeChanged|= drawOptionCombo("Frame Rate", m_frameRateOptions, m_selectedFrameRate);
	bModeChanged|= drawOptionCombo("Format", m_formatOptions, m_selectedFormat);
	if (bModeChanged)
		applyModeSelection();

	// Stream control
	if (m_videoCapture->isStreaming())
	{
		if (ImGui::Button("Stop Stream", ImVec2(-1, 0)))
			m_videoCapture->stopStream();
	}
	else
	{
		if (ImGui::Button("Start Stream", ImVec2(-1, 0)))
			m_videoCapture->startStream();
	}

	ImGui::EndDisabled();

	if (bDeviceOpen)
	{
		ImGui::Separator();
		ImGui::TextDisabled("Mode: %s", m_videoCapture->getCurrentVideoModeName().c_str());
		const uint64_t droppedFrames= m_videoCapture->getDroppedFrameCount();
		if (droppedFrames > 0)
			ImGui::TextDisabled("Dropped frames: %llu", (unsigned long long)droppedFrames);
	}

	ImGui::End();
}
