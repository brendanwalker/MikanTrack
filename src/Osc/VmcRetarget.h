#pragma once

#include <array>

#include "glm/ext/matrix_float3x3.hpp"
#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

#include "TrackingTypes.h"

// Retargets the measured world-space pose onto the VMC protocol's humanoid
// bone stream. Pure math: no socket, no config object, no clock - so the whole
// conversion is exercised by --test-vmc.
//
// WHAT THE RECEIVER EXPECTS. A VMC bone transform is the bone's LOCAL
// (parent-relative) transform in Unity convention, where the identity rotation
// means "this avatar's rest pose". VMC receivers assume a VRM-style rig whose
// humanoid bones sit at identity local rotation in a T-pose, and they REPLACE
// both the rotation and the translation of every bone they receive (a bone
// streamed with a zero translation collapses onto its parent). So every bone
// emitted here carries a real offset, which means the avatar takes the
// measured proportions rather than keeping its own.
//
// FRAMES. The avatar frame is the world frame unchanged: the calibration board
// fixes world +X forward, +Y toward the person's left and +Z up, which is a
// right-handed reading of the same axes a Unity humanoid rests on. The torso
// is deliberately NOT streamed, so it stays at the avatar's rest pose and any
// torso yaw the person performs lands in the clavicle rotations, where a
// receiver can see it.
namespace VmcRetarget
{
// The bones this streams, named exactly as Unity's HumanBodyBones spells them
// (which is what a VMC receiver matches on). Deliberately partial: this rig
// measures an upper body, so legs, spine, neck, eyes and jaw are left to the
// avatar's rest pose rather than being invented.
enum class eVmcBone : int
{
	Head= 0,

	LeftShoulder,
	LeftUpperArm,
	LeftLowerArm,
	LeftHand,

	RightShoulder,
	RightUpperArm,
	RightLowerArm,
	RightHand,

	// 3 bones x 5 fingers x 2 hands, thumb..little, proximal -> distal
	LeftThumbProximal,
	LeftThumbIntermediate,
	LeftThumbDistal,
	LeftIndexProximal,
	LeftIndexIntermediate,
	LeftIndexDistal,
	LeftMiddleProximal,
	LeftMiddleIntermediate,
	LeftMiddleDistal,
	LeftRingProximal,
	LeftRingIntermediate,
	LeftRingDistal,
	LeftLittleProximal,
	LeftLittleIntermediate,
	LeftLittleDistal,

	RightThumbProximal,
	RightThumbIntermediate,
	RightThumbDistal,
	RightIndexProximal,
	RightIndexIntermediate,
	RightIndexDistal,
	RightMiddleProximal,
	RightMiddleIntermediate,
	RightMiddleDistal,
	RightRingProximal,
	RightRingIntermediate,
	RightRingDistal,
	RightLittleProximal,
	RightLittleIntermediate,
	RightLittleDistal,

	Count,
};
constexpr int VMC_BONE_COUNT= (int)eVmcBone::Count;

const char* boneName(eVmcBone bone);

// First finger bone slot of one hand, so a (finger, phalanx) pair indexes as
// firstFingerBone(side) + finger * 3 + phalanx
eVmcBone firstFingerBone(eHandSide side);

struct VmcBone
{
	// False = not measured this frame. The bone is then left out of the
	// bundle entirely, which leaves the receiver holding it at rest.
	bool present= false;
	glm::vec3 localPosition{0.f};              // Unity convention, meters
	glm::quat localRotation{1.f, 0.f, 0.f, 0.f}; // Unity convention
};

struct VmcPose
{
	std::array<VmcBone, VMC_BONE_COUNT> bones{};

	void clear() { bones= {}; }
};

// Bone offsets the emitted skeleton carries. Only lengths: every direction is
// measured. Sourced from the app's BodyDimensions plus the per-hand skeleton.
struct VmcBodyLengths
{
	float shoulderWidthMeters= 0.40f;
	float upperArmLengthMeters= 0.30f;
	float forearmLengthMeters= 0.25f;
	// Neck -> head bone offset. The one length on this rig that nothing
	// measures, so it is a setting rather than a constant hidden in the code.
	float headOffsetMeters= 0.08f;
};

// -- Conventions (public so the self test can pin them down) -----------------

// World (right-handed: +X forward, +Y person's left, +Z up) -> Unity
// (left-handed: +X right, +Y up, +Z forward). The determinant is -1, so this
// is the handedness flip; rotations go through it by conjugation, which keeps
// them proper rotations.
glm::vec3 worldToUnityPosition(const glm::vec3& worldPosition);
glm::quat worldToUnityRotation(const glm::quat& worldRotation);

// The palm frame at the avatar's rest pose (T-pose, palms down), expressed in
// the avatar frame as columns [palm X, palm Y, palm Z]. This is the reference
// the streamed hand rotation is measured against.
glm::mat3 restPalmFrame(eHandSide side);

// Direction the arm bones point at rest: the person's left for a left arm.
glm::vec3 restArmDirection(eHandSide side);

// Rotation carrying `from` onto `to` with no roll about the result. Used
// wherever a bone's own twist is not measured, which is every bone whose only
// evidence is a direction.
glm::quat shortestArc(const glm::vec3& from, const glm::vec3& to);

// -- The retarget ------------------------------------------------------------

// Fills outPose from the resolved per-side hand poses and the head estimate.
// bSideValid selects which sides to stream at all (the caller has already
// applied the confidence gate, the dropout hold and the freeze-on-loss rule).
//
// Degrades one bone at a time: a side with no shoulder skips the clavicle and
// upper arm and still streams a correctly oriented hand, because a bone the
// receiver never hears about stays at the rest pose, which is exactly the
// identity this measures against.
void buildPose(
	const std::array<HandPose, 2>& poses, const bool bSideValid[2],
	const TrackingFrameResult::HeadPose& head, const VmcBodyLengths& lengths, VmcPose& outPose);
} // namespace VmcRetarget
