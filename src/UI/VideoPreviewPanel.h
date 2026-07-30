#pragma once

#include <memory>

#include "opencv2/core/mat.hpp"

#include "HandOverlay.h"
#include "TrackingTypes.h"

class GlTexture;
class VisionThread;

// Central video preview: streams the latest camera frame into a GlTexture,
// displays it letterboxed inside the panel, and draws the tracking overlay.
class VideoPreviewPanel
{
public:
	VideoPreviewPanel();
	~VideoPreviewPanel();

	// Uploads a new BGR frame (call when the vision thread published one)
	void setFrame(const cv::Mat& bgr);

	// Draws the panel; result is used for the overlay + HUD
	void draw(const TrackingFrameResult& result, const char* executionProvider, bool bHasFrame);

	// Mapping from full-frame image pixels to screen pixels for the most
	// recently drawn image (valid after draw)
	const ImageToScreenMapping& getImageToScreenMapping() const { return m_mapping; }

	bool getShowDetectionBoxes() const { return m_bShowDetectionBoxes; }
	void setShowDetectionBoxes(bool bShow) { m_bShowDetectionBoxes= bShow; }
	bool getShowOverlay() const { return m_bShowOverlay; }
	void setShowOverlay(bool bShow) { m_bShowOverlay= bShow; }

	// Lets wizards draw additional overlay elements over the video image.
	// Returns nullptr when the panel isn't visible.
	ImDrawList* getLastDrawList() const { return m_lastDrawList; }

private:
	std::unique_ptr<GlTexture> m_texture;
	ImageToScreenMapping m_mapping;
	ImDrawList* m_lastDrawList= nullptr;

	bool m_bShowOverlay= true;
	bool m_bShowDetectionBoxes= false;
};
