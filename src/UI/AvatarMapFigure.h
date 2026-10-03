#pragma once

#include <array>
#include <vector>

#include "AvatarTypes.h"

// Per-bone state the figure colors a dot by
enum class eAvatarMapDotState
{
	Mapped,           // green
	OptionalUnmapped, // grey
	RequiredUnmapped, // red
	Warning,          // orange
};

enum class eAvatarMapSection
{
	Body,
	Head,
	LeftHand,
	RightHand,
	Count,
};

// A schematic humanoid figure drawn with ImGui, in the style of a humanoid
// avatar configuration screen. Four views (body, head, left hand, right hand)
// each show a hand-authored silhouette with one clickable dot per humanoid
// bone. The figure faces the viewer, so the avatar's left side is on the right
// of the image.
class AvatarMapFigure
{
public:
	// Draws the section buttons and the current view. ioSelected is the
	// selected bone (HUMANOID_BONE_NONE for none); a click on a dot replaces
	// it. Returns true when a dot was clicked this frame.
	bool draw(const std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT>& states, eHumanoidBone& ioSelected);

	eAvatarMapSection getSection() const { return m_section; }
	void setSection(eAvatarMapSection section) { m_section= section; }

	// The bones a section shows, in a stable list order (the panel filters
	// its bone list with this)
	static const std::vector<eHumanoidBone>& getSectionBones(eAvatarMapSection section);

private:
	eAvatarMapSection m_section= eAvatarMapSection::Body;
};
