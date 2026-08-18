#include "CalibrationPanel.h"

#include "glm/ext/matrix_double4x4.hpp"

#include "imgui.h"

#include "LocText.h"

#include "AppConfig.h"

CalibrationPanel::CalibrationPanel(AppConfig* config)
	: m_config(config)
{
}

CalibrationPanel::DrawResult CalibrationPanel::draw(bool bWizardActive)
{
	DrawResult result;

	if (!ImGui::Begin(locWindowTitle("windows.calibration")))
	{
		ImGui::End();
		return result;
	}

	for (int cameraIndex= 0; cameraIndex < (int)m_config->cameraCount(); ++cameraIndex)
	{
		const CameraProfile& profile= m_config->camera(cameraIndex);

		ImGui::PushID(cameraIndex);

		const std::string headerLabel= locFormat("calibrationPanel.cameraHeaderFmt", cameraIndex + 1);
		if (ImGui::CollapsingHeader(headerLabel.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::SeparatorText(locText("calibrationPanel.intrinsicsHeader"));
			if (profile.intrinsics.present)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s", locText("calibrationPanel.calibratedStatus"));
				ImGui::Text(locText("calibrationPanel.intrinsicsSummaryFmt"),
							profile.intrinsics.intrinsics.pixel_width,
							profile.intrinsics.intrinsics.pixel_height,
							profile.intrinsics.reprojectionError);
				ImGui::Text(locText("calibrationPanel.fovFmt"),
							profile.intrinsics.intrinsics.hfov,
							profile.intrinsics.intrinsics.vfov);
			}
			else
			{
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "%s", locText("calibrationPanel.notCalibratedStatus"));
			}
			ImGui::BeginDisabled(bWizardActive);
			if (ImGui::Button(locLabel("calibrationPanel.calibrateIntrinsicsButton"), ImVec2(-1, 0)))
			{
				result.bLaunchIntrinsicsWizard= true;
				result.cameraIndex= cameraIndex;
			}
			ImGui::EndDisabled();

			ImGui::SeparatorText(locText("calibrationPanel.extrinsicsHeader"));
			if (profile.extrinsics.present)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s", locText("calibrationPanel.calibratedStatus"));
				ImGui::Text(locText("calibrationPanel.cameraHeightFmt"), profile.extrinsics.markerFromCamera[3].z);
			}
			else
			{
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "%s", locText("calibrationPanel.notCalibratedStatus"));
			}
		}

		ImGui::PopID();
	}

	ImGui::SeparatorText(locText("calibrationPanel.extrinsicsAllHeader"));
	bool bAllIntrinsics= true;
	for (size_t i= 0; i < m_config->cameraCount(); ++i)
		bAllIntrinsics&= m_config->camera(i).intrinsics.present;

	ImGui::TextWrapped("%s", locText("calibrationPanel.extrinsicsAllDescription"));
	ImGui::BeginDisabled(bWizardActive || !bAllIntrinsics);
	if (ImGui::Button(locLabel("calibrationPanel.calibrateExtrinsicsButton"), ImVec2(-1, 0)))
	{
		result.bLaunchExtrinsicsWizard= true;
	}
	ImGui::EndDisabled();
	if (!bAllIntrinsics)
		ImGui::TextDisabled("%s", locText("calibrationPanel.needsIntrinsicsFirstText"));

	ImGui::SeparatorText(locText("calibrationPanel.handScaleHeader"));
	ImGui::Text(locText("calibrationPanel.handScaleSeedFmt"),
				m_config->handScale.refLengthMeters * 100.0);

	ImGui::End();
	return result;
}
