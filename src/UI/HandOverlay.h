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

// One camera's view of the wrist-IMU forearm: the world-space wrist->elbow
// segment projected back into THIS camera's image. Computed by the UI layer
// (which owns the fused result plus every camera's calibration) so the
// vision thread doesn't need to know about per-camera overlays.
struct ForearmOverlay
{
	bool valid[2]= {false, false}; // indexed by eHandSide
	ImVec2 wristPx[2]{};
	ImVec2 elbowPx[2]{};
};

namespace HandOverlay
{
void drawTrackingResult(ImDrawList* drawList, const TrackingFrameResult& result, const ImageToScreenMapping& mapping,
						bool bShowDetectionBoxes);

// Draws the projected forearm (wrist->elbow) with an elbow marker. Lets you
// check the IMU forearm direction against the real arm in the video: if the
// line lies along your forearm the mounting calibration is good, and if the
// marker sits at your elbow the forearm length is right.
void drawForearmOverlay(ImDrawList* drawList, const ForearmOverlay& forearm,
						const ImageToScreenMapping& mapping);

// Draws the BlazePose body skeleton for a body-pose camera: the RAW landmarks
// the solver consumes, dimmed wherever visibility falls under the solver's
// gate, with the visibility of the joints it actually uses (shoulders,
// elbows, wrists) labelled. This is the source data behind the elbow, so a
// bad elbow can be traced to a bad landmark rather than guessed at.
void drawBodyPose(ImDrawList* drawList, const BodyPoseObservation& body,
				  const ImageToScreenMapping& mapping);
}
