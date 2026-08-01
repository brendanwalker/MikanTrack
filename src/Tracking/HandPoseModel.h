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
