#include "VideoPreviewPanel.h"

#include "LocText.h"

#include <algorithm>

#include "GlTexture.h"

VideoPreviewPanel::VideoPreviewPanel()
{
	setCameraCount(1);
}

VideoPreviewPanel::~VideoPreviewPanel()= default;

void VideoPreviewPanel::setCameraCount(size_t count)
{
	count= std::max<size_t>(count, 1);

	while (m_panes.size() > count)
		m_panes.pop_back();
	while (m_panes.size() < count)
	{
		auto pane= std::make_unique<CameraPane>();
		pane->texture= std::make_unique<GlTexture>();
		m_panes.push_back(std::move(pane));
	}

	m_activeCamera= std::min(m_activeCamera, (int)m_panes.size() - 1);
}

void VideoPreviewPanel::setFrame(int cameraIndex, const cv::Mat& bgr)
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_panes.size() && !bgr.empty())
	{
		m_panes[cameraIndex]->texture->uploadBGR(bgr);
		m_panes[cameraIndex]->bHasFrame= true;
	}
}

uint32_t VideoPreviewPanel::getCameraTextureId(int cameraIndex) const
{
	if (cameraIndex < 0 || cameraIndex >= (int)m_panes.size())
		return 0;

	const CameraPane& pane= *m_panes[cameraIndex];
	if (!pane.bHasFrame || !pane.texture->getIsValid())
		return 0;
	return pane.texture->getGlTextureId();
}

float VideoPreviewPanel::getCameraAspect(int cameraIndex) const
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_panes.size())
	{
		const CameraPane& pane= *m_panes[cameraIndex];
		if (pane.bHasFrame && pane.texture->getHeight() > 0)
			return (float)pane.texture->getWidth() / (float)pane.texture->getHeight();
	}
	return 16.f / 9.f;
}

const ImageToScreenMapping& VideoPreviewPanel::getImageToScreenMapping(int cameraIndex) const
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_panes.size())
		return m_panes[cameraIndex]->mapping;

	return m_fallbackMapping;
}

void VideoPreviewPanel::draw(const std::vector<const TrackingFrameResult*>& results,
							 const std::vector<const char*>& executionProviders,
							 const std::vector<ForearmOverlay>* forearms)
{
	m_lastDrawList= nullptr;

	if (!ImGui::Begin(locWindowTitle("windows.videoPreview")))
	{
		ImGui::End();
		return;
	}

	const int paneCount= (int)m_panes.size();
	const ImVec2 panelSize= ImGui::GetContentRegionAvail();
	const float paneSpacing= 4.f;
	const float paneWidth= (panelSize.x - paneSpacing * (float)(paneCount - 1)) / (float)paneCount;
	const float paneHeight= panelSize.y;

	m_lastDrawList= ImGui::GetWindowDrawList();
	const ImVec2 panelOrigin= ImGui::GetCursorScreenPos();

	for (int cameraIndex= 0; cameraIndex < paneCount; ++cameraIndex)
	{
		CameraPane& pane= *m_panes[cameraIndex];
		const ImVec2 paneOrigin(panelOrigin.x + (paneWidth + paneSpacing) * (float)cameraIndex, panelOrigin.y);

		// Pane background + active highlight
		m_lastDrawList->AddRectFilled(
			paneOrigin, ImVec2(paneOrigin.x + paneWidth, paneOrigin.y + paneHeight), IM_COL32(12, 12, 14, 255));
		if (paneCount > 1 && cameraIndex == m_activeCamera)
		{
			m_lastDrawList->AddRect(
				paneOrigin, ImVec2(paneOrigin.x + paneWidth, paneOrigin.y + paneHeight),
				IM_COL32(120, 180, 255, 180), 0.f, 0, 2.f);
		}

		if (!pane.bHasFrame || !pane.texture->getIsValid())
		{
			const char* message= paneCount > 1 ? locText("videoPreviewPanel.noStreamText")
												: locText("videoPreviewPanel.noVideoStreamText");
			const ImVec2 textSize= ImGui::CalcTextSize(message);
			m_lastDrawList->AddText(
				ImVec2(paneOrigin.x + (paneWidth - textSize.x) * 0.5f, paneOrigin.y + (paneHeight - textSize.y) * 0.5f),
				IM_COL32(140, 140, 140, 255), message);
			continue;
		}

		// Letterbox the image inside the pane
		const float imageWidth= (float)pane.texture->getWidth();
		const float imageHeight= (float)pane.texture->getHeight();
		const float scale= std::min(paneWidth / imageWidth, paneHeight / imageHeight);
		const ImVec2 displaySize(imageWidth * scale, imageHeight * scale);
		const ImVec2 imagePos(
			paneOrigin.x + (paneWidth - displaySize.x) * 0.5f,
			paneOrigin.y + (paneHeight - displaySize.y) * 0.5f);

		m_lastDrawList->AddImage(
			(ImTextureID)(intptr_t)pane.texture->getGlTextureId(),
			imagePos, ImVec2(imagePos.x + displaySize.x, imagePos.y + displaySize.y));

		pane.mapping.screenOrigin= imagePos;
		pane.mapping.scale= scale;

		// Clicking a pane makes it the active camera
		ImGui::SetCursorScreenPos(paneOrigin);
		ImGui::PushID(cameraIndex);
		if (ImGui::InvisibleButton("pane", ImVec2(paneWidth, paneHeight)))
			m_activeCamera= cameraIndex;
		ImGui::PopID();

		const TrackingFrameResult* result=
			cameraIndex < (int)results.size() ? results[cameraIndex] : nullptr;

		if (m_bShowOverlay)
		{
			if (result != nullptr)
			{
				HandOverlay::drawTrackingResult(m_lastDrawList, *result, pane.mapping, m_bShowDetectionBoxes);
				// Only body-pose cameras carry an observation, so this draws
				// on exactly the camera the stage is enabled for
				if (m_bShowBodyPose)
					HandOverlay::drawBodyPose(m_lastDrawList, result->body, pane.mapping);
			}
			if (forearms != nullptr && cameraIndex < (int)forearms->size())
				HandOverlay::drawForearmOverlay(m_lastDrawList, (*forearms)[cameraIndex], pane.mapping);
		}

		// HUD: camera label + fps / inference / EP badge
		{
			const char* executionProvider=
				cameraIndex < (int)executionProviders.size() ? executionProviders[cameraIndex] : "?";

			std::string hud;
			if (result != nullptr)
			{
				hud= locFormat("videoPreviewPanel.hudWithResultFmt",
							   cameraIndex + 1, result->captureFps, result->inferenceMs, executionProvider);
			}
			else
			{
				hud= locFormat("videoPreviewPanel.hudFmt", cameraIndex + 1, executionProvider);
			}

			const ImVec2 hudPos(imagePos.x + 8.f, imagePos.y + 8.f);
			const ImVec2 hudSize= ImGui::CalcTextSize(hud.c_str());
			m_lastDrawList->AddRectFilled(
				ImVec2(hudPos.x - 4.f, hudPos.y - 2.f),
				ImVec2(hudPos.x + hudSize.x + 4.f, hudPos.y + hudSize.y + 2.f),
				IM_COL32(0, 0, 0, 160), 3.f);

			const bool bGpu= executionProvider != nullptr && strstr(executionProvider, "DirectML") != nullptr;
			m_lastDrawList->AddText(hudPos, bGpu ? IM_COL32(120, 255, 120, 255) : IM_COL32(255, 220, 100, 255), hud.c_str());
		}
	}

	ImGui::End();
}
