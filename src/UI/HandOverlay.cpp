#include "HandOverlay.h"

static const ImU32 k_leftHandColor= IM_COL32(80, 160, 255, 255);   // blue
static const ImU32 k_rightHandColor= IM_COL32(255, 96, 96, 255);   // red
static const ImU32 k_jointColor= IM_COL32(255, 255, 255, 220);
static const ImU32 k_palmDetectColor= IM_COL32(255, 220, 60, 180); // yellow

static void drawDashedLine(ImDrawList* drawList, const ImVec2& a, const ImVec2& b, ImU32 color, float thickness,
						   float dashLength= 8.f)
{
	const float dx= b.x - a.x;
	const float dy= b.y - a.y;
	const float length= sqrtf(dx * dx + dy * dy);
	if (length < 1.f)
		return;

	const int dashCount= (int)(length / dashLength);
	for (int i= 0; i < dashCount; i+= 2)
	{
		const float t0= (float)i / (float)dashCount;
		const float t1= (float)(i + 1) / (float)dashCount;
		drawList->AddLine(
			ImVec2(a.x + dx * t0, a.y + dy * t0),
			ImVec2(a.x + dx * t1, a.y + dy * t1),
			color, thickness);
	}
}

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

	// Forearms
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedArm& arm= result.arms[sideIndex];
		if (!arm.valid)
			continue;

		const ImU32 color= (eHandSide)sideIndex == eHandSide::Left ? k_leftHandColor : k_rightHandColor;
		const ImVec2 elbow= mapping.toScreen(arm.elbowPixel.x, arm.elbowPixel.y);
		const ImVec2 wrist= mapping.toScreen(arm.wristPixel.x, arm.wristPixel.y);

		if (arm.fromFallback)
			drawDashedLine(drawList, elbow, wrist, color, 3.f);
		else
			drawList->AddLine(elbow, wrist, color, 3.f);

		drawList->AddCircleFilled(elbow, 5.f, color);
	}
}
