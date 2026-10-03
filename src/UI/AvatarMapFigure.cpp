#include "AvatarMapFigure.h"

#include <algorithm>
#include <cmath>

#include "imgui.h"

#include "LocText.h"

static const ImU32 k_silhouetteColor= IM_COL32(130, 130, 130, 255);
static const ImU32 k_limbFillColor= IM_COL32(130, 130, 130, 60);
static const ImU32 k_dotOutlineColor= IM_COL32(20, 20, 20, 255);
static const ImU32 k_selectionRingColor= IM_COL32(255, 255, 255, 255);
static const ImU32 k_markerColor= IM_COL32(180, 180, 180, 255);

static const float k_dotRadius= 6.f;
static const float k_dotPickMargin= 4.f;
static const float k_selectionRingExtra= 3.f;
static const float k_maxFigureHeight= 360.f;

namespace
{
struct MapDot
{
	eHumanoidBone bone;
	ImVec2 pos; // normalized 0..1 within the figure rect
};

// A silhouette piece: an outline, or a thick faded stroke for a limb
struct MapShape
{
	std::vector<ImVec2> points;
	bool bClosed= false;
	float thickness= 1.5f;
	bool bLimb= false;
};

struct MapText
{
	const char* text;
	ImVec2 pos;
};

struct MapLayout
{
	float aspect= 1.f; // width / height
	std::vector<MapDot> dots;
	std::vector<MapShape> shapes;
	std::vector<MapText> markers;
	std::vector<eHumanoidBone> bones;
};

ImVec2 mirrorX(const ImVec2& p)
{
	return ImVec2(1.f - p.x, p.y);
}

std::vector<ImVec2> makeEllipse(float cx, float cy, float rx, float ry, int segments= 28)
{
	std::vector<ImVec2> points;
	for (int i= 0; i < segments; ++i)
	{
		const float a= 2.f * 3.14159265f * (float)i / (float)segments;
		points.push_back(ImVec2(cx + rx * std::cos(a), cy + ry * std::sin(a)));
	}
	return points;
}

void addOutline(MapLayout& layout, std::vector<ImVec2> points, bool bClosed)
{
	MapShape shape;
	shape.points= std::move(points);
	shape.bClosed= bClosed;
	layout.shapes.push_back(std::move(shape));
}

void addLimb(MapLayout& layout, std::vector<ImVec2> points, float thickness)
{
	MapShape shape;
	shape.points= std::move(points);
	shape.thickness= thickness;
	shape.bLimb= true;
	layout.shapes.push_back(std::move(shape));
}

void addDot(MapLayout& layout, eHumanoidBone bone, float x, float y)
{
	layout.dots.push_back({bone, ImVec2(x, y)});
	layout.bones.push_back(bone);
}

// Body: a front view, so the avatar's left is on the image right
MapLayout buildBodyLayout()
{
	MapLayout layout;
	layout.aspect= 0.6f;

	addDot(layout, eHumanoidBone::Hips, 0.50f, 0.50f);
	addDot(layout, eHumanoidBone::Spine, 0.50f, 0.42f);
	addDot(layout, eHumanoidBone::Chest, 0.50f, 0.34f);
	addDot(layout, eHumanoidBone::UpperChest, 0.50f, 0.28f);
	addDot(layout, eHumanoidBone::Neck, 0.50f, 0.21f);
	addDot(layout, eHumanoidBone::Head, 0.50f, 0.13f);

	const ImVec2 armLeft[]= {{0.59f, 0.255f}, {0.72f, 0.27f}, {0.83f, 0.42f}, {0.91f, 0.56f}};
	const eHumanoidBone armLeftBones[]= {
		eHumanoidBone::LeftShoulder, eHumanoidBone::LeftUpperArm, eHumanoidBone::LeftLowerArm, eHumanoidBone::LeftHand};
	const eHumanoidBone armRightBones[]= {
		eHumanoidBone::RightShoulder, eHumanoidBone::RightUpperArm, eHumanoidBone::RightLowerArm, eHumanoidBone::RightHand};
	for (int i= 0; i < 4; ++i)
		addDot(layout, armLeftBones[i], armLeft[i].x, armLeft[i].y);
	for (int i= 0; i < 4; ++i)
		addDot(layout, armRightBones[i], mirrorX(armLeft[i]).x, armLeft[i].y);

	const ImVec2 legLeft[]= {{0.58f, 0.53f}, {0.60f, 0.74f}, {0.60f, 0.92f}, {0.63f, 0.97f}};
	const eHumanoidBone legLeftBones[]= {
		eHumanoidBone::LeftUpperLeg, eHumanoidBone::LeftLowerLeg, eHumanoidBone::LeftFoot, eHumanoidBone::LeftToes};
	const eHumanoidBone legRightBones[]= {
		eHumanoidBone::RightUpperLeg, eHumanoidBone::RightLowerLeg, eHumanoidBone::RightFoot, eHumanoidBone::RightToes};
	for (int i= 0; i < 4; ++i)
		addDot(layout, legLeftBones[i], legLeft[i].x, legLeft[i].y);
	for (int i= 0; i < 4; ++i)
		addDot(layout, legRightBones[i], mirrorX(legLeft[i]).x, legLeft[i].y);

	// Silhouette: head, torso, limbs
	addOutline(layout, makeEllipse(0.50f, 0.09f, 0.075f, 0.065f), true);
	addOutline(layout, {{0.37f, 0.24f}, {0.63f, 0.24f}, {0.60f, 0.52f}, {0.40f, 0.52f}}, true);
	for (int side= 0; side < 2; ++side)
	{
		std::vector<ImVec2> arm;
		for (int i= 0; i < 4; ++i)
			arm.push_back(side == 0 ? armLeft[i] : mirrorX(armLeft[i]));
		addLimb(layout, arm, 12.f);

		std::vector<ImVec2> leg;
		for (int i= 0; i < 3; ++i)
			leg.push_back(side == 0 ? legLeft[i] : mirrorX(legLeft[i]));
		addLimb(layout, leg, 16.f);
	}

	layout.markers= {{"R", {0.03f, 0.03f}}, {"L", {0.93f, 0.03f}}};
	return layout;
}

MapLayout buildHeadLayout()
{
	MapLayout layout;
	layout.aspect= 1.f;

	addDot(layout, eHumanoidBone::Neck, 0.50f, 0.90f);
	addDot(layout, eHumanoidBone::Head, 0.50f, 0.55f);
	// Avatar left eye is on the image right
	addDot(layout, eHumanoidBone::LeftEye, 0.64f, 0.42f);
	addDot(layout, eHumanoidBone::RightEye, 0.36f, 0.42f);
	addDot(layout, eHumanoidBone::Jaw, 0.50f, 0.76f);

	addOutline(layout, makeEllipse(0.50f, 0.46f, 0.28f, 0.36f, 36), true);
	addOutline(layout, {{0.42f, 0.80f}, {0.42f, 1.00f}}, false);
	addOutline(layout, {{0.58f, 0.80f}, {0.58f, 1.00f}}, false);

	layout.markers= {{"R", {0.03f, 0.03f}}, {"L", {0.93f, 0.03f}}};
	return layout;
}

// One hand seen palm down from above, fingers up the image. The left hand
// has its thumb on the image right, the right hand mirrors it.
MapLayout buildHandLayout(int sideIndex)
{
	MapLayout layout;
	layout.aspect= 1.f;
	const bool bRight= (sideIndex == 1);
	auto place= [bRight](float x, float y) { return bRight ? ImVec2(1.f - x, y) : ImVec2(x, y); };

	addDot(layout, sideIndex == 0 ? eHumanoidBone::LeftHand : eHumanoidBone::RightHand, 0.50f, 0.88f);

	// Left hand layout; finger order is thumb, index, middle, ring, little
	const ImVec2 fingers[5][3]= {
		{{0.74f, 0.70f}, {0.84f, 0.58f}, {0.90f, 0.47f}}, // thumb
		{{0.66f, 0.50f}, {0.67f, 0.37f}, {0.68f, 0.27f}}, // index
		{{0.54f, 0.47f}, {0.54f, 0.32f}, {0.54f, 0.20f}}, // middle
		{{0.42f, 0.50f}, {0.41f, 0.36f}, {0.40f, 0.26f}}, // ring
		{{0.30f, 0.54f}, {0.28f, 0.44f}, {0.27f, 0.36f}}, // little
	};
	const eHumanoidBone first= firstHumanoidFingerBone(sideIndex);
	for (int finger= 0; finger < 5; ++finger)
	{
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const ImVec2 p= place(fingers[finger][phalanx].x, fingers[finger][phalanx].y);
			addDot(layout, (eHumanoidBone)((int)first + finger * 3 + phalanx), p.x, p.y);
		}
	}

	// Silhouette: palm outline plus a faded stroke per finger
	std::vector<ImVec2> palm;
	for (const ImVec2& p : std::vector<ImVec2>{{0.40f, 0.94f}, {0.60f, 0.94f}, {0.72f, 0.74f}, {0.70f, 0.52f},
											   {0.28f, 0.57f}, {0.30f, 0.74f}})
		palm.push_back(place(p.x, p.y));
	addOutline(layout, palm, true);
	for (int finger= 0; finger < 5; ++finger)
	{
		std::vector<ImVec2> stroke;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			stroke.push_back(place(fingers[finger][phalanx].x, fingers[finger][phalanx].y));
		addLimb(layout, stroke, 14.f);
	}

	layout.markers= {bRight ? MapText{"R", {0.03f, 0.03f}} : MapText{"L", {0.93f, 0.03f}}};
	return layout;
}

const MapLayout& getLayout(eAvatarMapSection section)
{
	static const MapLayout s_body= buildBodyLayout();
	static const MapLayout s_head= buildHeadLayout();
	static const MapLayout s_leftHand= buildHandLayout(0);
	static const MapLayout s_rightHand= buildHandLayout(1);

	switch (section)
	{
	case eAvatarMapSection::Head:
		return s_head;
	case eAvatarMapSection::LeftHand:
		return s_leftHand;
	case eAvatarMapSection::RightHand:
		return s_rightHand;
	case eAvatarMapSection::Body:
	default:
		return s_body;
	}
}

ImU32 dotColor(eAvatarMapDotState state)
{
	switch (state)
	{
	case eAvatarMapDotState::Mapped:
		return IM_COL32(80, 200, 100, 255);
	case eAvatarMapDotState::RequiredUnmapped:
		return IM_COL32(230, 70, 70, 255);
	case eAvatarMapDotState::Warning:
		return IM_COL32(240, 160, 50, 255);
	case eAvatarMapDotState::OptionalUnmapped:
	default:
		return IM_COL32(140, 140, 140, 255);
	}
}

const MapDot* findDot(const MapLayout& layout, eHumanoidBone bone)
{
	for (const MapDot& dot : layout.dots)
	{
		if (dot.bone == bone)
			return &dot;
	}
	return nullptr;
}
} // namespace

const std::vector<eHumanoidBone>& AvatarMapFigure::getSectionBones(eAvatarMapSection section)
{
	return getLayout(section).bones;
}

bool AvatarMapFigure::draw(
	const std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT>& states,
	eHumanoidBone& ioSelected)
{
	// Section buttons
	const char* sectionLabels[(int)eAvatarMapSection::Count]= {
		locLabel("avatarPanel.sectionBody"),
		locLabel("avatarPanel.sectionHead"),
		locLabel("avatarPanel.sectionLeftHand"),
		locLabel("avatarPanel.sectionRightHand"),
	};
	const float spacing= ImGui::GetStyle().ItemSpacing.x;
	const int sectionCount= (int)eAvatarMapSection::Count;
	const float buttonWidth= (ImGui::GetContentRegionAvail().x - spacing * (float)(sectionCount - 1)) / (float)sectionCount;
	for (int i= 0; i < sectionCount; ++i)
	{
		const bool bActive= ((int)m_section == i);
		if (i > 0)
			ImGui::SameLine();
		if (bActive)
			ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
		if (ImGui::Button(sectionLabels[i], ImVec2(buttonWidth, 0.f)))
			m_section= (eAvatarMapSection)i;
		if (bActive)
			ImGui::PopStyleColor();
	}

	const MapLayout& layout= getLayout(m_section);

	// Figure size: fill the width at the view's aspect ratio, capped in height
	const float availWidth= ImGui::GetContentRegionAvail().x;
	float height= availWidth / layout.aspect;
	float width= availWidth;
	if (height > k_maxFigureHeight)
	{
		height= k_maxFigureHeight;
		width= height * layout.aspect;
	}
	if (width < 1.f || height < 1.f)
		return false;

	// Center the child horizontally when the height cap narrowed it
	ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (availWidth - width) * 0.5f);

	bool bClicked= false;
	const ImGuiWindowFlags childFlags= ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.f, 0.f));
	if (ImGui::BeginChild("##avatarMapFigure", ImVec2(width, height), ImGuiChildFlags_Borders, childFlags))
	{
		const ImVec2 origin= ImGui::GetCursorScreenPos();
		const ImVec2 size= ImGui::GetContentRegionAvail();

		// The invisible button keeps a click on the figure from dragging the parent window
		ImGui::InvisibleButton("##canvas", size);

		auto toScreen= [&](const ImVec2& p) { return ImVec2(origin.x + p.x * size.x, origin.y + p.y * size.y); };

		ImDrawList* drawList= ImGui::GetWindowDrawList();
		const float scale= 0.75f + 0.25f * std::clamp(size.y / k_maxFigureHeight, 0.f, 1.f);
		const float dotRadius= k_dotRadius * scale;

		// Silhouette
		for (const MapShape& shape : layout.shapes)
		{
			std::vector<ImVec2> points;
			for (const ImVec2& p : shape.points)
				points.push_back(toScreen(p));
			if (shape.bLimb)
				drawList->AddPolyline(points.data(), (int)points.size(), k_limbFillColor, ImDrawFlags_None,
									  shape.thickness * scale);
			else
				drawList->AddPolyline(points.data(), (int)points.size(), k_silhouetteColor,
									  shape.bClosed ? ImDrawFlags_Closed : ImDrawFlags_None, shape.thickness);
		}

		// Lines between each dot and its spec parent when both are in this view
		for (const MapDot& dot : layout.dots)
		{
			const MapDot* parent= findDot(layout, humanoidBoneParent(dot.bone));
			if (parent != nullptr)
				drawList->AddLine(toScreen(parent->pos), toScreen(dot.pos), k_silhouetteColor, 1.5f);
		}

		for (const MapText& marker : layout.markers)
			drawList->AddText(toScreen(marker.pos), k_markerColor, marker.text);

		// Nearest dot under the mouse
		const ImVec2 mouse= ImGui::GetIO().MousePos;
		const MapDot* hoveredDot= nullptr;
		if (ImGui::IsWindowHovered())
		{
			const float pickRadius= dotRadius + k_dotPickMargin;
			float bestDistSq= pickRadius * pickRadius;
			for (const MapDot& dot : layout.dots)
			{
				const ImVec2 p= toScreen(dot.pos);
				const float dx= mouse.x - p.x;
				const float dy= mouse.y - p.y;
				const float distSq= dx * dx + dy * dy;
				if (distSq <= bestDistSq)
				{
					bestDistSq= distSq;
					hoveredDot= &dot;
				}
			}
		}

		// Dots
		for (const MapDot& dot : layout.dots)
		{
			const ImVec2 p= toScreen(dot.pos);
			if (dot.bone == ioSelected)
				drawList->AddCircle(p, dotRadius + k_selectionRingExtra, k_selectionRingColor, 0, 2.f);
			drawList->AddCircleFilled(p, dotRadius, dotColor(states[(int)dot.bone]));
			drawList->AddCircle(p, dotRadius, k_dotOutlineColor, 0, 1.5f);
		}

		if (hoveredDot != nullptr)
		{
			ImGui::SetTooltip("%s", humanoidBoneName(hoveredDot->bone));
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Left))
			{
				ioSelected= hoveredDot->bone;
				bClicked= true;
			}
		}
	}
	ImGui::EndChild();
	ImGui::PopStyleVar();

	return bClicked;
}
