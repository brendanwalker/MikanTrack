#pragma once

#include <array>
#include <cstdint>

#include "glm/vec3.hpp"

// ARKit's 52 face blendshapes, in the order iFacialMocap's table lists them
// (the left/right pairs interleave by region, not alphabetically)
constexpr int ARKIT_BLENDSHAPE_COUNT= 52;

// The ARKit name for a column; "" when out of range
const char* arkitBlendshapeName(int index);

// The column for a name in either spelling: ARKit's ("eyeBlinkLeft") or the
// phone's suffixed form ("eyeBlink_L"). -1 for a name the table lacks.
int arkitBlendshapeFromName(const char* name);

// One face sample as the phone sent it. Rotations are Euler degrees and the
// position is the phone's own units, all in the phone's axis convention: the
// conversion to MikanTrack's spaces happens downstream.
struct FaceSample
{
	// false until the first datagram parses
	bool valid= false;
	// +1 per parsed datagram
	uint64_t sequence= 0;
	// steadyNowMs() at receipt, the same clock camera frames use
	double timestampMs= 0.0;
	// 0..1 (the phone's 0..100 scaled; the v2 format may run slightly negative,
	// kept as sent)
	std::array<float, ARKIT_BLENDSHAPE_COUNT> blendshapes{};
	// The phone's own face-tracked flag (trackingStatus). While it reports the
	// face lost the values are stale, so the face is not applied.
	bool faceTracked= true;
	bool hasHead= false;
	glm::vec3 headEulerDegrees{0.f};
	glm::vec3 headPosition{0.f};
	bool hasEyes= false;
	glm::vec3 leftEyeEulerDegrees{0.f};
	glm::vec3 rightEyeEulerDegrees{0.f};
};
