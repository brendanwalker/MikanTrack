#include "HandOverlay.h"

static const ImU32 k_leftHandColor= IM_COL32(80, 160, 255, 255);   // blue
static const ImU32 k_rightHandColor= IM_COL32(255, 96, 96, 255);   // red
static const ImU32 k_jointColor= IM_COL32(255, 255, 255, 220);
static const ImU32 k_palmDetectColor= IM_COL32(255, 220, 60, 180); // yellow
static const ImU32 k_bodyColor= IM_COL32(200, 120, 255, 230);      // purple
static const ImU32 k_bodyDimColor= IM_COL32(200, 120, 255, 70);
static const ImU32 k_bodyKeyColor= IM_COL32(120, 255, 180, 255);   // green
// Matches BodyPoseSolver's landmark gate: below this a landmark is treated as
// unseen, so the overlay dims exactly what the solver ignores
static const float k_bodyVisibilityGate= 0.5f;

void HandOverlay::drawTrackingResult(ImDrawList* drawList, const TrackingFrameResult& result,
									 const ImageToScreenMapping& mapping, bool bShowDetectionBoxes)
{
	// Detection debug boxes
	if (bShowDetectionBoxes)
	{
		auto drawBoxes= [&](const std::vector<DetectionBox>& boxes, ImU32 color) {
			for (const DetectionBox& box : boxes)
			{
				for (int i= 0; i < 4; ++i)
				{
					const glm::vec2& c0= box.corners[i];
					const glm::vec2& c1= box.corners[(i + 1) % 4];
					drawList->AddLine(mapping.toScreen(c0.x, c0.y), mapping.toScreen(c1.x, c1.y), color, 1.5f);
				}
			}
		};
		drawBoxes(result.palmDetections, k_palmDetectColor);
	}

	// Hand skeletons
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedHand& hand= result.hands[sideIndex];
		if (!hand.tracked)
			continue;

		const ImU32 boneColor= hand.side == eHandSide::Left ? k_leftHandColor : k_rightHandColor;

		for (int i= 0; i < HAND_CONNECTION_COUNT; ++i)
		{
			const glm::vec3& p0= hand.imagePoints[HAND_CONNECTIONS[i][0]];
			const glm::vec3& p1= hand.imagePoints[HAND_CONNECTIONS[i][1]];
			drawList->AddLine(mapping.toScreen(p0.x, p0.y), mapping.toScreen(p1.x, p1.y), boneColor, 2.f);
		}

		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			const glm::vec3& p= hand.imagePoints[i];
			drawList->AddCircleFilled(mapping.toScreen(p.x, p.y), 3.f, k_jointColor);
		}

		// Label near the wrist
		const glm::vec3& wrist= hand.imagePoints[(int)eHandLandmark::WRIST];
		char label[64];
		snprintf(label, sizeof(label), "%s %.2f", hand.side == eHandSide::Left ? "L" : "R", hand.presence);
		ImVec2 labelPos= mapping.toScreen(wrist.x, wrist.y);
		labelPos.y+= 8.f;
		drawList->AddText(labelPos, boneColor, label);
	}

}

void HandOverlay::drawBodyPose(ImDrawList* drawList, const BodyPoseObservation& body,
							   const ImageToScreenMapping& mapping)
{
	if (!body.valid)
		return;

	auto isVisible= [&](int landmark) { return body.visibility[landmark] >= k_bodyVisibilityGate; };

	for (int i= 0; i < POSE_CONNECTION_COUNT; ++i)
	{
		const int a= POSE_CONNECTIONS[i][0];
		const int b= POSE_CONNECTIONS[i][1];
		// A backend that does not emit a joint leaves its slot at the image
		// origin; drawing it would pin a phantom landmark in the corner
		if (!body.isProvided(a) || !body.isProvided(b))
			continue;
		const bool bBothVisible= isVisible(a) && isVisible(b);
		const glm::vec3& p0= body.imagePoints[a];
		const glm::vec3& p1= body.imagePoints[b];
		drawList->AddLine(
			mapping.toScreen(p0.x, p0.y), mapping.toScreen(p1.x, p1.y),
			bBothVisible ? k_bodyColor : k_bodyDimColor, bBothVisible ? 2.f : 1.f);
	}

	for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
	{
		if (!body.isProvided(landmark))
			continue;
		const glm::vec3& p= body.imagePoints[landmark];
		drawList->AddCircleFilled(
			mapping.toScreen(p.x, p.y), 2.5f, isVisible(landmark) ? k_bodyColor : k_bodyDimColor);
	}

	// The joints the solver actually consumes, with the visibility number it
	// gates them on
	struct KeyJoint
	{
		ePoseLandmark landmark;
		const char* name;
	};
	static const KeyJoint k_keyJoints[]= {
		{ePoseLandmark::LEFT_SHOULDER, "Lsho"}, {ePoseLandmark::RIGHT_SHOULDER, "Rsho"},
		{ePoseLandmark::LEFT_ELBOW, "Lelb"},    {ePoseLandmark::RIGHT_ELBOW, "Relb"},
		{ePoseLandmark::LEFT_WRIST, "Lwri"},    {ePoseLandmark::RIGHT_WRIST, "Rwri"},
	};
	for (const KeyJoint& joint : k_keyJoints)
	{
		const int index= (int)joint.landmark;
		if (!body.isProvided(index))
			continue;
		const glm::vec3& p= body.imagePoints[index];
		const ImVec2 screenPos= mapping.toScreen(p.x, p.y);
		drawList->AddCircle(screenPos, 6.f, isVisible(index) ? k_bodyKeyColor : k_bodyDimColor, 0, 2.f);

		char label[32];
		snprintf(label, sizeof(label), "%s %.2f", joint.name, body.visibility[index]);
		drawList->AddText(ImVec2(screenPos.x + 8.f, screenPos.y - 6.f),
						  isVisible(index) ? k_bodyKeyColor : k_bodyDimColor, label);
	}

	// The box source matters as much as the score: a top-down backend is only
	// as good as the box it was given
	const char* boxSourceName= "?";
	switch (body.boxSource)
	{
	case eBodyBoxSource::Detector: boxSourceName= "detected"; break;
	case eBodyBoxSource::Tracked: boxSourceName= "tracked"; break;
	case eBodyBoxSource::FullFrame: boxSourceName= "full frame"; break;
	case eBodyBoxSource::None: boxSourceName= "none"; break;
	}
	char header[96];
	snprintf(header, sizeof(header), "body pose conf %.2f (box: %s)", body.confidence, boxSourceName);
	drawList->AddText(mapping.toScreen(8.f, 8.f), k_bodyColor, header);
}

void HandOverlay::drawForearmOverlay(ImDrawList* drawList, const ForearmOverlay& forearm,
									 const ImageToScreenMapping& mapping)
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!forearm.valid[sideIndex])
			continue;

		const ImU32 color= sideIndex == (int)eHandSide::Left ? k_leftHandColor : k_rightHandColor;
		const ImVec2 wrist= mapping.toScreen(forearm.wristPx[sideIndex].x, forearm.wristPx[sideIndex].y);
		const ImVec2 elbow= mapping.toScreen(forearm.elbowPx[sideIndex].x, forearm.elbowPx[sideIndex].y);

		// Dashed-looking forearm: thin line plus a hollow elbow ring, so it
		// reads as an ESTIMATE rather than a measured landmark chain
		drawList->AddLine(wrist, elbow, color, 2.f);
		drawList->AddCircle(elbow, 7.f, color, 12, 2.f);
		drawList->AddCircleFilled(elbow, 2.5f, color);
	}
}
