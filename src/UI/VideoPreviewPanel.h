#pragma once

#include <memory>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "HandOverlay.h"
#include "TrackingTypes.h"

class GlTexture;

// Central video preview: all cameras rendered side by side, each letterboxed
// in its own pane with its own tracking overlay and image->screen mapping.
// The "active" camera (used by the calibration wizards) gets a highlight
// border; wizards pin it via setActiveCamera().
class VideoPreviewPanel
{
public:
	VideoPreviewPanel();
	~VideoPreviewPanel();

	void setCameraCount(size_t count);
	size_t getCameraCount() const { return m_panes.size(); }

	// Uploads a new BGR frame for one camera
	void setFrame(int cameraIndex, const cv::Mat& bgr);

	// Draws the panel. results/executionProviders/dominantSides are indexed
	// by camera; dominantSides marks which sides this camera won in the last
	// fusion (for the "FUSED L/R" badge), pass nullptr to skip.
	void draw(const std::vector<const TrackingFrameResult*>& results,
			  const std::vector<const char*>& executionProviders);

	// Per-camera mapping from full-frame image pixels to screen pixels for the
	// most recently drawn frame (valid after draw)
	const ImageToScreenMapping& getImageToScreenMapping(int cameraIndex) const;

	// Draw list for wizard overlays (nullptr when the panel isn't visible)
	ImDrawList* getLastDrawList() const { return m_lastDrawList; }

	void setActiveCamera(int cameraIndex) { m_activeCamera= cameraIndex; }
	int getActiveCamera() const { return m_activeCamera; }

	bool getShowDetectionBoxes() const { return m_bShowDetectionBoxes; }
	void setShowDetectionBoxes(bool bShow) { m_bShowDetectionBoxes= bShow; }
	bool getShowOverlay() const { return m_bShowOverlay; }
	void setShowOverlay(bool bShow) { m_bShowOverlay= bShow; }

private:
	struct CameraPane
	{
		std::unique_ptr<GlTexture> texture;
		ImageToScreenMapping mapping;
		bool bHasFrame= false;
	};

	std::vector<std::unique_ptr<CameraPane>> m_panes;
	ImageToScreenMapping m_fallbackMapping;
	ImDrawList* m_lastDrawList= nullptr;
	int m_activeCamera= 0;

	bool m_bShowOverlay= true;
	bool m_bShowDetectionBoxes= false;
};
