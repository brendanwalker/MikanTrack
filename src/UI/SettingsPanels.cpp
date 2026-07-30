#include "SettingsPanels.h"

#include "imgui.h"

#include "AppConfig.h"
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

void SettingsPanels::drawTrackingPanel(AppConfig* config, VisionThread* visionThread, VideoPreviewPanel* previewPanel)
{
	if (!ImGui::Begin("Tracking"))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	TrackingConfig& tracking= config->tracking;

	bChanged|= ImGui::Checkbox("Flip handedness", &tracking.flipHandedness);
	ImGui::SetItemTooltip("MediaPipe assumes a mirrored selfie view.\nEnable for a normal (non-mirrored) camera.");

	bChanged|= ImGui::SliderInt("Detector interval", &tracking.detectorIntervalFrames, 5, 120);
	ImGui::SetItemTooltip("Palm detector re-runs at least every N frames");

	ImGui::SeparatorText("Smoothing");
	bChanged|= ImGui::Checkbox("Enabled", &tracking.smoothingEnabled);
	bChanged|= ImGui::SliderFloat("Min cutoff", &tracking.smoothingMinCutoff, 0.1f, 5.f, "%.2f Hz");
	bChanged|= ImGui::SliderFloat("Beta", &tracking.smoothingBeta, 0.f, 0.5f, "%.3f");

	ImGui::SeparatorText("Overlay");
	bool bShowOverlay= previewPanel->getShowOverlay();
	if (ImGui::Checkbox("Show skeleton overlay", &bShowOverlay))
		previewPanel->setShowOverlay(bShowOverlay);
	bool bShowBoxes= previewPanel->getShowDetectionBoxes();
	if (ImGui::Checkbox("Show detection boxes", &bShowBoxes))
		previewPanel->setShowDetectionBoxes(bShowBoxes);

	ImGui::SeparatorText("Inference");
	ImGui::Text("Execution provider: %s", visionThread->getActiveExecutionProvider());
	ImGui::Text("Inference: %.1f ms", visionThread->getLastInferenceMs());

	if (bChanged)
	{
		config->markDirty();
		visionThread->requestConfigRefresh();
	}

	ImGui::End();
}

void SettingsPanels::drawOscPanel(AppConfig* config, VisionThread* visionThread)
{
	if (!ImGui::Begin("OSC Output"))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	OscConfig& osc= config->osc;

	bChanged|= ImGui::Checkbox("Enabled", &osc.enabled);

	char ipBuffer[64];
	snprintf(ipBuffer, sizeof(ipBuffer), "%s", osc.targetIp.c_str());
	if (ImGui::InputText("Target IP", ipBuffer, sizeof(ipBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
	{
		osc.targetIp= ipBuffer;
		bChanged= true;
	}

	int port= osc.targetPort;
	if (ImGui::InputInt("Port", &port, 0) && port > 0 && port <= 65535)
	{
		osc.targetPort= port;
		bChanged= true;
	}

	bChanged|= ImGui::SliderInt("Max rate", &osc.maxRateHz, 10, 120, "%d Hz");

	ImGui::Separator();
	ImGui::TextDisabled("Space: marker-anchored, meters,\nright-handed, +Z up from table");
	ImGui::TextDisabled("Addresses: /mikan/hand/{left,right}/...\n/mikan/arm/{left,right}/...");

	if (bChanged)
	{
		config->markDirty();
		visionThread->requestConfigRefresh();
	}

	ImGui::End();
}
