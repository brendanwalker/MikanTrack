#pragma once

#include "imgui.h"

#include "TrackingTypes.h"

// Draws the tracked-hand wireframes, forearm lines and detection debug boxes
// over the video image using an ImGui draw list.
//
// imageToScreen maps full-frame pixel coordinates to screen coordinates
// (computed by the video preview panel from the displayed image rect).
struct ImageToScreenMapping
{
	ImVec2 screenOrigin{0, 0}; // top-left of the displayed image on screen
	float scale= 1.f;          // screen pixels per image pixel

	ImVec2 toScreen(float imageX, float imageY) const
	{
		return ImVec2(screenOrigin.x + imageX * scale, screenOrigin.y + imageY * scale);
	}
};

namespace HandOverlay
{
void drawTrackingResult(ImDrawList* drawList, const TrackingFrameResult& result, const ImageToScreenMapping& mapping,
						bool bShowDetectionBoxes);
}
