#include "FaceTypes.h"

#include <cstring>
#include <string>

static const char* const k_arkitBlendshapeNames[ARKIT_BLENDSHAPE_COUNT]= {
	"eyeBlinkLeft",      "eyeLookDownLeft",    "eyeLookInLeft",     "eyeLookOutLeft",    "eyeLookUpLeft",
	"eyeSquintLeft",     "eyeWideLeft",        "eyeBlinkRight",     "eyeLookDownRight",  "eyeLookInRight",
	"eyeLookOutRight",   "eyeLookUpRight",     "eyeSquintRight",    "eyeWideRight",      "jawForward",
	"jawLeft",           "jawRight",           "jawOpen",           "mouthClose",        "mouthFunnel",
	"mouthPucker",       "mouthLeft",          "mouthRight",        "mouthSmileLeft",    "mouthSmileRight",
	"mouthFrownLeft",    "mouthFrownRight",    "mouthDimpleLeft",   "mouthDimpleRight",  "mouthStretchLeft",
	"mouthStretchRight", "mouthRollLower",     "mouthRollUpper",    "mouthShrugLower",   "mouthShrugUpper",
	"mouthPressLeft",    "mouthPressRight",    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft",
	"mouthUpperUpRight", "browDownLeft",       "browDownRight",     "browInnerUp",       "browOuterUpLeft",
	"browOuterUpRight",  "cheekPuff",          "cheekSquintLeft",   "cheekSquintRight",  "noseSneerLeft",
	"noseSneerRight",    "tongueOut",
};

const char* arkitBlendshapeName(int index)
{
	if (index < 0 || index >= ARKIT_BLENDSHAPE_COUNT)
		return "";
	return k_arkitBlendshapeNames[index];
}

int arkitBlendshapeFromName(const char* name)
{
	if (name == nullptr)
		return -1;

	std::string spelled= name;

	// The phone suffixes a side as _L/_R where ARKit spells it out
	const size_t length= spelled.size();
	if (length > 2 && spelled[length - 2] == '_' && (spelled[length - 1] == 'L' || spelled[length - 1] == 'R'))
	{
		const bool bLeft= spelled[length - 1] == 'L';
		spelled.resize(length - 2);
		spelled+= bLeft ? "Left" : "Right";
	}

	for (int index= 0; index < ARKIT_BLENDSHAPE_COUNT; ++index)
	{
		if (spelled == k_arkitBlendshapeNames[index])
			return index;
	}
	return -1;
}
