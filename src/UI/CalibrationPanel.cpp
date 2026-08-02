#include "CalibrationPanel.h"

#include "glm/ext/matrix_double4x4.hpp"

#include "imgui.h"

#include "AppConfig.h"

CalibrationPanel::CalibrationPanel(AppConfig* config)
	: m_config(config)
{
}

CalibrationPanel::DrawResult CalibrationPanel::draw(bool bWizardActive)
{
	DrawResult result;

	if (!ImGui::Begin("Calibration"))
	{
		ImGui::End();
		return result;
	}

	for (int cameraIndex= 0; cameraIndex < (int)m_config->cameraCount(); ++cameraIndex)
	{
		const CameraProfile& profile= m_config->camera(cameraIndex);

		ImGui::PushID(cameraIndex);

		char headerLabel[64];
		snprintf(headerLabel, sizeof(headerLabel), "Camera %d", cameraIndex + 1);
		if (ImGui::CollapsingHeader(headerLabel, ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SeparatorText("Intrinsics");
			if (profile.intrinsics.present)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibrated");
				ImGui::Text("%.0fx%.0f  reproj %.3f px",
							profile.intrinsics.intrinsics.pixel_width,
							profile.intrinsics.intrinsics.pixel_height,
							profile.intrinsics.reprojectionError);
				ImGui::Text("hfov %.1f  vfov %.1f",
							profile.intrinsics.intrinsics.hfov,
							profile.intrinsics.intrinsics.vfov);
			}
			else
			{
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Not calibrated");
			}
			ImGui::BeginDisabled(bWizardActive);
			if (ImGui::Button("Calibrate Intrinsics...", ImVec2(-1, 0)))
			{
				result.bLaunchIntrinsicsWizard= true;
				result.cameraIndex= cameraIndex;
			}
			ImGui::EndDisabled();

			ImGui::SeparatorText("Extrinsics");
			if (profile.extrinsics.present)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibrated");
				ImGui::Text("Camera height: %.2f m", profile.extrinsics.markerFromCamera[3].z);
			}
			else
			{
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Not calibrated");
			}
		}

		ImGui::PopID();
	}

	ImGui::SeparatorText("Extrinsics (all cameras)");
	bool bAllIntrinsics= true;
	for (size_t i= 0; i < m_config->cameraCount(); ++i)
		bAllIntrinsics&= m_config->camera(i).intrinsics.present;

	ImGui::TextWrapped(
		"Extrinsics are calibrated for ALL cameras in one session against the "
		"same marker placement - that shared marker defines the common world "
		"space fusion happens in. If the marker ever moves, recalibrate here.");
	ImGui::BeginDisabled(bWizardActive || !bAllIntrinsics);
	if (ImGui::Button("Calibrate Extrinsics (all cameras)...", ImVec2(-1, 0)))
	{
		result.bLaunchExtrinsicsWizard= true;
	}
	ImGui::EndDisabled();
	if (!bAllIntrinsics)
		ImGui::TextDisabled("(every camera needs intrinsics first)");

	ImGui::SeparatorText("Hand Scale (global)");
	ImGui::Text("Seed: %.1f cm (auto-measured live from stereo/depth)",
				m_config->handScale.refLengthMeters * 100.0);

	ImGui::End();
	return result;
}
