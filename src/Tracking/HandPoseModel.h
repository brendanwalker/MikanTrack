#pragma once

#include <array>

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/quaternion_float.hpp"

#include "TrackingTypes.h"

// Pure math for the parametric hand representation: palm frame extraction,
// finger bend angles relative to the neutral (straight) pose, skeleton
// geometry, and the forward kinematics to rebuild joint positions from
// pose + angles (3D visualization / client-side reference).
//
// All functions operate on a consistent 21-landmark point set in ANY
// right-handed metric space (the model's local space, camera space or world
// space) - angles and the palm frame are derived from relative geometry only.
namespace HandPoseModel
{
// Which side of the palm plane is the PALMAR one, remembered between frames.
// A hand's palmar side is physically incapable of inverting, so this carries
// the previous answer forward: caller-owned, one per tracked hand, zeroed
// when that hand is lost (a reacquired hand may be anywhere).
//
// Without it the sign is re-decided from scratch every frame, and on frames
// where the geometric evidence is weak - a flat hand with the thumb in plane -
// it falls through to MediaPipe's handedness label, which is view-dependent
// and flips. Measured live: the left palm frame inverting twice in 240 frames
// while the right never did, which poisoned the mounting average with two
// clusters 180 degrees apart (70 deg pose spread against the right's 13).
struct PalmarSideMemory
{
	glm::vec3 palmarNormal{0.f}; // zero = nothing remembered yet

	// Consecutive observations whose evidence contradicted palmarNormal;
	// a flip takes several of them (see computePalmFrame)
	int contradictionCount= 0;
	// The palmar score the count last advanced on: the count moves once per
	// DISTINCT observation, not once per call. One observation is routinely
	// evaluated more than once per frame (the palm frame, then the angles),
	// and that is one piece of evidence, not two. Re-evaluating the same
	// landmarks reproduces the score exactly; different landmarks do not.
	float countedScore= 0.f;

	void reset()
	{
		palmarNormal= glm::vec3(0.f);
		contradictionCount= 0;
		countedScore= 0.f;
	}
};

// Palm frame from the landmarks (see HandPose for the convention).
// Returns a rigid transform: column 0/1/2 = palm X/Y/Z axes, column 3 =
// palm center, expressed in the space of the input points.
//
// ioMemory is optional; pass one per tracked hand to make the palmar sign
// temporally stable. It is updated in place.
glm::mat4 computePalmFrame(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
						   PalmarSideMemory* ioMemory= nullptr);

// The rest pose ("all angles zero") as per-finger proximal-phalanx directions
// in the palm frame. Captured from a real flat hand by captureRestPose, or
// defaulted by makeDefaultNeutralDirections.
using NeutralDirections= std::array<glm::vec3, FINGER_COUNT>;

// Flat-hand default: the four fingers point along palm +X (parallel to the
// middle metacarpal, i.e. a flat hand with the fingers together), the thumb
// along its own metacarpal (it genuinely rests off-axis, so +X would make
// its resting angles large). Derived from base positions alone, so a client
// can reproduce it from the skeleton.
NeutralDirections makeDefaultNeutralDirections(const HandSkeleton& skeleton);

// Rest-pose calibration: the angles this hand reports right now, measured
// against the flat-hand default. Subtracting them from later measurements
// makes THIS pose read all zeros - on all four degrees of freedom, unlike a
// neutral direction, which can only absorb lateral and proximal.
//
// Capture PER CAMERA: MediaPipe's model landmarks are view-dependent, so two
// cameras watching one physical hand disagree about its articulation by tens
// of degrees (measured live 2026-08-01: 41 deg on one finger). Removing each
// camera's own bias is also what makes their angles comparable enough for
// fusion to blend them meaningfully.
void captureRestAngles(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
					   std::array<FingerAngles, FINGER_COUNT>& outRestAngles);

// Finger bend angles (radians) from the landmarks, relative to the rest pose
// given by neutralDirs (see FingerAngles for the sign conventions)
void computeFingerAngles(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
						 const NeutralDirections& neutralDirs,
						 std::array<FingerAngles, FINGER_COUNT>& outAngles,
						 PalmarSideMemory* ioMemory= nullptr);

// Skeleton geometry (finger base positions in the palm frame + phalanx
// lengths) from a metric landmark set. neutralDirInPalm is filled with the
// flat-hand default; overwrite it with a captured rest pose if there is one.
void computeSkeleton(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
					 HandSkeleton& outSkeleton, PalmarSideMemory* ioMemory= nullptr);

// Forward kinematics: joint positions (4 per finger, base -> tip) from a
// palm pose + skeleton + angles, in the space of palmTransform. Exactly
// inverts computeFingerAngles when given the same skeleton.neutralDirInPalm.
void buildFingerJoints(const glm::mat4& palmTransform, const HandSkeleton& skeleton,
					   const std::array<FingerAngles, FINGER_COUNT>& angles,
					   std::array<std::array<glm::vec3, 4>, FINGER_COUNT>& outJoints);
} // namespace HandPoseModel
