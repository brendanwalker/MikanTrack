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

// Finger bend angles (radians) from the landmarks, relative to the neutral
// pose (fingers straight along their metacarpal directions)
void computeFingerAngles(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
						 std::array<FingerAngles, FINGER_COUNT>& outAngles);

// Skeleton geometry (finger base positions in the palm frame + phalanx
// lengths) from a metric landmark set
void computeSkeleton(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
					 HandSkeleton& outSkeleton);

// Forward kinematics: joint positions (4 per finger, base -> tip) from a
// palm pose + skeleton + angles, in the space of palmTransform
void buildFingerJoints(const glm::mat4& palmTransform, const HandSkeleton& skeleton,
					   const std::array<FingerAngles, FINGER_COUNT>& angles,
					   std::array<std::array<glm::vec3, 4>, FINGER_COUNT>& outJoints);
} // namespace HandPoseModel
