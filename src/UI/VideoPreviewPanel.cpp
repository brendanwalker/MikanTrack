#include "VideoPreviewPanel.h"

#include "GlTexture.h"

VideoPreviewPanel::VideoPreviewPanel()
	: m_texture(std::make_unique<GlTexture>())
{
}

VideoPreviewPanel::~VideoPreviewPanel()= default;

void VideoPreviewPanel::setFrame(const cv::Mat& bgr)
{
	if (!bgr.empty())
		m_texture->uploadBGR(bgr);
}

void VideoPreviewPanel::draw(const TrackingFrameResult& result, const char* executionProvider, bool bHasFrame)
{
	m_lastDrawList= nullptr;

	if (!ImGui::Begin("Video Preview"))
	{
		ImGui::End();
		return;
	}

	const ImVec2 panelSize= ImGui::GetContentRegionAvail();

	if (!m_texture->getIsValid() || !bHasFrame)
	{
		const char* message= "No video stream - select and start a camera in the Device panel";
		const ImVec2 textSize= ImGui::CalcTextSize(message);
		ImGui::SetCursorPos(ImVec2(
			(panelSize.x - textSize.x) * 0.5f,
			(panelSize.y - textSize.y) * 0.5f));
		ImGui::TextDisabled("%s", message);
		ImGui::End();
		return;
	}

	// Letterbox the image inside the panel
	const float imageWidth= (float)m_texture->getWidth();
	const float imageHeight= (float)m_texture->getHeight();
	const float scale= std::min(panelSize.x / imageWidth, panelSize.y / imageHeight);
	const ImVec2 displaySize(imageWidth * scale, imageHeight * scale);

	const ImVec2 cursorStart= ImGui::GetCursorPos();
	ImGui::SetCursorPos(ImVec2(
		cursorStart.x + (panelSize.x - displaySize.x) * 0.5f,
		cursorStart.y + (panelSize.y - displaySize.y) * 0.5f));

	const ImVec2 imageScreenPos= ImGui::GetCursorScreenPos();
	ImGui::Image((ImTextureID)(intptr_t)m_texture->getGlTextureId(), displaySize);

	m_mapping.screenOrigin= imageScreenPos;
	m_mapping.scale= scale;
	m_lastDrawList= ImGui::GetWindowDrawList();

	if (m_bShowOverlay)
		HandOverlay::drawTrackingResult(m_lastDrawList, result, m_mapping, m_bShowDetectionBoxes);

	// HUD: fps / inference / EP badge (top-left corner of the image)
	{
		char hud[128];
		snprintf(hud, sizeof(hud), "%.0f fps  |  %.1f ms  |  %s",
				 result.captureFps, result.inferenceMs, executionProvider);

		const ImVec2 hudPos(imageScreenPos.x + 8.f, imageScreenPos.y + 8.f);
		const ImVec2 hudSize= ImGui::CalcTextSize(hud);
		m_lastDrawList->AddRectFilled(
			ImVec2(hudPos.x - 4.f, hudPos.y - 2.f),
			ImVec2(hudPos.x + hudSize.x + 4.f, hudPos.y + hudSize.y + 2.f),
			IM_COL32(0, 0, 0, 160), 3.f);

		const bool bGpu= strstr(executionProvider, "DirectML") != nullptr;
		m_lastDrawList->AddText(hudPos, bGpu ? IM_COL32(120, 255, 120, 255) : IM_COL32(255, 220, 100, 255), hud);
	}

	ImGui::End();
}
