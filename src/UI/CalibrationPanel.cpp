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

	ImGui::SeparatorText("Camera Intrinsics");
	if (m_config->intrinsics.present)
	{
		ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibrated");
		ImGui::Text("%.0fx%.0f  reproj %.3f px",
					m_config->intrinsics.intrinsics.pixel_width,
					m_config->intrinsics.intrinsics.pixel_height,
					m_config->intrinsics.reprojectionError);
		ImGui::Text("hfov %.1f  vfov %.1f",
					m_config->intrinsics.intrinsics.hfov,
					m_config->intrinsics.intrinsics.vfov);
	}
	else
	{
		ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Not calibrated");
	}
	ImGui::BeginDisabled(bWizardActive);
	if (ImGui::Button("Calibrate Intrinsics...", ImVec2(-1, 0)))
		result.bLaunchIntrinsicsWizard= true;
	ImGui::EndDisabled();

	ImGui::SeparatorText("Camera Extrinsics + Hand Scale");
	if (m_config->extrinsics.present)
	{
		ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibrated");
		ImGui::Text("Camera height: %.2f m", m_config->extrinsics.markerFromCamera[3].z);
	}
	else
	{
		ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Not calibrated");
	}
	if (m_config->handScale.present)
		ImGui::Text("Hand scale: %.1f cm", m_config->handScale.refLengthMeters * 100.0);
	else
		ImGui::TextDisabled("Hand scale: default %.1f cm", m_config->handScale.refLengthMeters * 100.0);

	ImGui::BeginDisabled(bWizardActive || !m_config->intrinsics.present);
	if (ImGui::Button("Calibrate Extrinsics...", ImVec2(-1, 0)))
		result.bLaunchExtrinsicsWizard= true;
	ImGui::EndDisabled();
	if (!m_config->intrinsics.present)
		ImGui::TextDisabled("(requires intrinsics)");

	ImGui::End();
	return result;
}
