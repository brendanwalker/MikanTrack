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
// Palm frame from the landmarks (see HandPose for the convention).
// Returns a rigid transform: column 0/1/2 = palm X/Y/Z axes, column 3 =
// palm center, expressed in the space of the input points.
glm::mat4 computePalmFrame(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side);

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

// Rest-pose calibration: the per-finger proximal directions of THIS hand, in
// its own palm frame. Feeding these back as the neutral makes the captured
// pose read all-zero angles.
NeutralDirections captureRestPose(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side);

// Finger bend angles (radians) from the landmarks, relative to the rest pose
// given by neutralDirs (see FingerAngles for the sign conventions)
void computeFingerAngles(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
						 const NeutralDirections& neutralDirs,
						 std::array<FingerAngles, FINGER_COUNT>& outAngles);

// Skeleton geometry (finger base positions in the palm frame + phalanx
// lengths) from a metric landmark set. neutralDirInPalm is filled with the
// flat-hand default; overwrite it with a captured rest pose if there is one.
void computeSkeleton(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
					 HandSkeleton& outSkeleton);

// Forward kinematics: joint positions (4 per finger, base -> tip) from a
// palm pose + skeleton + angles, in the space of palmTransform. Exactly
// inverts computeFingerAngles when given the same skeleton.neutralDirInPalm.
void buildFingerJoints(const glm::mat4& palmTransform, const HandSkeleton& skeleton,
					   const std::array<FingerAngles, FINGER_COUNT>& angles,
					   std::array<std::array<glm::vec3, 4>, FINGER_COUNT>& outJoints);
} // namespace HandPoseModel
