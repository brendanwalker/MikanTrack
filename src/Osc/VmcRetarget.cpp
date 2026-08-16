#include "VmcRetarget.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"

namespace
{
// Avatar frame -> Unity frame, as columns: world +X (forward) becomes Unity +Z
// (forward), world +Y (the person's left) becomes Unity -X, world +Z (up)
// becomes Unity +Y. Determinant -1: this IS the handedness flip, and writing
// it as a basis change rather than a hand-derived pattern of quaternion sign
// flips is what makes it checkable by inspection.
const glm::mat3 k_worldToUnity(
	glm::vec3(0.f, 0.f, 1.f), glm::vec3(-1.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));

// Rest palm frames, columns [palm X, palm Y, palm Z], in the avatar frame.
//
// The avatar rests in a T-pose with the palms facing down. In the app's palm
// convention (+X toward the fingers, +Z out of the palmar surface, +Y
// completing right-handed) a left hand then reads fingers toward the person's
// left, palm facing down, thumb forward - and a right hand mirrors it. The
// chirality lives entirely in these two matrices; everything downstream is
// side-agnostic.
const glm::mat3 k_restPalmLeft(
	glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f, 0.f, 0.f), glm::vec3(0.f, 0.f, -1.f));
const glm::mat3 k_restPalmRight(
	glm::vec3(0.f, -1.f, 0.f), glm::vec3(-1.f, 0.f, 0.f), glm::vec3(0.f, 0.f, -1.f));

glm::vec3 safeNormalize(const glm::vec3& v)
{
	const float lengthSquared= glm::dot(v, v);
	return lengthSquared > 1e-12f ? v / std::sqrt(lengthSquared) : glm::vec3(0.f);
}

// One bone: its local rotation against the parent's current world rotation,
// and the parent rotation the next bone in the chain hangs off.
struct BoneChain
{
	glm::quat parentWorld{1.f, 0.f, 0.f, 0.f};

	// A bone whose own twist is not measured: swing the rest direction onto
	// the measured one and leave the roll alone.
	glm::quat swingTo(const glm::vec3& restDirection, const glm::vec3& worldDirection)
	{
		const glm::quat local=
			VmcRetarget::shortestArc(restDirection, glm::inverse(parentWorld) * worldDirection);
		parentWorld= parentWorld * local;
		return local;
	}

	// A bone whose full orientation is measured (the forearm, from its roll;
	// the hand, from the palm).
	glm::quat frameTo(const glm::quat& boneWorld)
	{
		const glm::quat local= glm::inverse(parentWorld) * boneWorld;
		parentWorld= boneWorld;
		return local;
	}
};

void emitBone(VmcRetarget::VmcPose& pose, VmcRetarget::eVmcBone bone,
			  const glm::vec3& localPositionWorld, const glm::quat& localRotationWorld)
{
	VmcRetarget::VmcBone& outBone= pose.bones[(int)bone];
	outBone.present= true;
	outBone.localPosition= VmcRetarget::worldToUnityPosition(localPositionWorld);
	outBone.localRotation= VmcRetarget::worldToUnityRotation(localRotationWorld);
}

void buildSide(
	const HandPose& pose, eHandSide side, bool bHasBothShoulders, const glm::vec3& shoulderMidpoint,
	const VmcRetarget::VmcBodyLengths& lengths, VmcRetarget::VmcPose& outPose)
{
	using namespace VmcRetarget;

	const glm::mat3 restPalm= restPalmFrame(side);
	const glm::quat restPalmRotation= glm::quat_cast(restPalm);
	// Rest direction of every arm bone, and the axis their offsets run out
	// along: one signed vector covers both sides
	const glm::vec3 armRest= restArmDirection(side);

	BoneChain chain; // starts at the chest, which is left at the avatar's rest

	// Clavicle. Its direction is the shoulder joint measured against the line
	// between the shoulders, so it needs both of them: with one shoulder there
	// is no reference to measure a shrug or a protraction against.
	if (bHasBothShoulders)
	{
		const glm::vec3 direction= safeNormalize(pose.shoulderPositionWorld - shoulderMidpoint);
		if (glm::dot(direction, direction) > 0.f)
		{
			const eVmcBone bone= side == eHandSide::Left ? eVmcBone::LeftShoulder : eVmcBone::RightShoulder;
			emitBone(outPose, bone, glm::vec3(0.f), chain.swingTo(armRest, direction));
		}
	}

	// Upper arm. The elbow is the same value /mikan/hand/{s}/elbow carries -
	// derived from the measured forearm direction, so no elbow without one.
	const bool bHasElbow= pose.hasForearmPose;
	if (bHasElbow && pose.hasShoulder)
	{
		const glm::vec3 elbow= pose.getElbowPositionWorld(lengths.forearmLengthMeters);
		const glm::vec3 direction= safeNormalize(elbow - pose.shoulderPositionWorld);
		if (glm::dot(direction, direction) > 0.f)
		{
			const eVmcBone bone= side == eHandSide::Left ? eVmcBone::LeftUpperArm : eVmcBone::RightUpperArm;
			emitBone(outPose, bone, armRest * (lengths.shoulderWidthMeters * 0.5f),
					 chain.swingTo(armRest, direction));
		}
	}

	// Forearm. Its full frame is measured, unlike the bones above: the forearm
	// frame is defined as the palm frame at a neutral wrist, so its rest frame
	// is the rest palm frame and its roll (pronation) survives the retarget
	// instead of being dumped into the hand.
	if (bHasElbow)
	{
		const eVmcBone bone= side == eHandSide::Left ? eVmcBone::LeftLowerArm : eVmcBone::RightLowerArm;
		const glm::quat forearmWorld= pose.forearmOrientationWorld * glm::inverse(restPalmRotation);
		emitBone(outPose, bone, armRest * lengths.upperArmLengthMeters, chain.frameTo(forearmWorld));
	}

	// Hand. Its offset puts the bone at the WRIST, which is half a palm back
	// from the palm origin, so the finger offsets below are wrist-relative.
	const glm::quat handWorld= pose.palmOrientationWorld * glm::inverse(restPalmRotation);
	{
		const eVmcBone bone= side == eHandSide::Left ? eVmcBone::LeftHand : eVmcBone::RightHand;
		emitBone(outPose, bone, armRest * lengths.forearmLengthMeters, chain.frameTo(handWorld));
	}

	// Fingers. Rebuilt through the shipping forward kinematics rather than by
	// re-deriving the angle conventions here, so the bones bend exactly the way
	// the 3D view and every other consumer of the same pose do.
	glm::mat4 palmTransform= glm::mat4_cast(pose.palmOrientationWorld);
	palmTransform[3]= glm::vec4(pose.palmPositionWorld, 1.f);

	std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
	HandPoseModel::buildFingerJoints(palmTransform, pose.skeleton, pose.fingers, joints);

	const float halfPalm= pose.skeleton.baseInPalm[(int)eFinger::Middle].x;
	const int firstBone= (int)firstFingerBone(side);

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const glm::vec3& neutralDir= pose.skeleton.neutralDirInPalm[finger];
		// All three phalanges are collinear with the neutral direction at zero
		// angles, so one rest direction serves the whole finger - which is what
		// makes a hand held flat emit identity rotations.
		const glm::vec3 restDirection= restPalm * neutralDir;

		const glm::vec3 localPositions[3]= {
			restPalm * (pose.skeleton.baseInPalm[finger] + glm::vec3(halfPalm, 0.f, 0.f)),
			restPalm * (neutralDir * pose.skeleton.phalanxLengths[finger][0]),
			restPalm * (neutralDir * pose.skeleton.phalanxLengths[finger][1]),
		};

		BoneChain fingerChain;
		fingerChain.parentWorld= handWorld;

		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const glm::vec3 direction= safeNormalize(joints[finger][phalanx + 1] - joints[finger][phalanx]);
			const glm::quat local= glm::dot(direction, direction) > 0.f
									   ? fingerChain.swingTo(restDirection, direction)
									   : glm::quat(1.f, 0.f, 0.f, 0.f);

			emitBone(outPose, (eVmcBone)(firstBone + finger * 3 + phalanx), localPositions[phalanx], local);
		}
	}
}
} // namespace

namespace VmcRetarget
{
const char* boneName(eVmcBone bone)
{
	// Unity HumanBodyBones spelling, which is what a VMC receiver matches on
	static const char* k_names[VMC_BONE_COUNT]= {
		"Head",

		"LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
		"RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",

		"LeftThumbProximal", "LeftThumbIntermediate", "LeftThumbDistal",
		"LeftIndexProximal", "LeftIndexIntermediate", "LeftIndexDistal",
		"LeftMiddleProximal", "LeftMiddleIntermediate", "LeftMiddleDistal",
		"LeftRingProximal", "LeftRingIntermediate", "LeftRingDistal",
		"LeftLittleProximal", "LeftLittleIntermediate", "LeftLittleDistal",

		"RightThumbProximal", "RightThumbIntermediate", "RightThumbDistal",
		"RightIndexProximal", "RightIndexIntermediate", "RightIndexDistal",
		"RightMiddleProximal", "RightMiddleIntermediate", "RightMiddleDistal",
		"RightRingProximal", "RightRingIntermediate", "RightRingDistal",
		"RightLittleProximal", "RightLittleIntermediate", "RightLittleDistal",
	};

	const int index= (int)bone;
	return (index >= 0 && index < VMC_BONE_COUNT) ? k_names[index] : "";
}

eVmcBone firstFingerBone(eHandSide side)
{
	return side == eHandSide::Left ? eVmcBone::LeftThumbProximal : eVmcBone::RightThumbProximal;
}

glm::vec3 worldToUnityPosition(const glm::vec3& worldPosition)
{
	return k_worldToUnity * worldPosition;
}

glm::quat worldToUnityRotation(const glm::quat& worldRotation)
{
	// Conjugation by a determinant -1 basis change: the result is still a
	// proper rotation (the two flips cancel), which is why quat_cast is valid
	// here and no separate axis negation is needed.
	const glm::mat3 rotation= glm::mat3_cast(worldRotation);
	return glm::quat_cast(k_worldToUnity * rotation * glm::transpose(k_worldToUnity));
}

glm::mat3 restPalmFrame(eHandSide side)
{
	return side == eHandSide::Left ? k_restPalmLeft : k_restPalmRight;
}

glm::vec3 restArmDirection(eHandSide side)
{
	// The T-pose: arms out to the sides, so a left arm runs along the world's
	// +Y (the person's left) and a right arm along -Y
	return side == eHandSide::Left ? glm::vec3(0.f, 1.f, 0.f) : glm::vec3(0.f, -1.f, 0.f);
}

glm::quat shortestArc(const glm::vec3& from, const glm::vec3& to)
{
	const glm::vec3 a= safeNormalize(from);
	const glm::vec3 b= safeNormalize(to);
	if (glm::dot(a, a) <= 0.f || glm::dot(b, b) <= 0.f)
		return glm::quat(1.f, 0.f, 0.f, 0.f);

	const float cosAngle= std::clamp(glm::dot(a, b), -1.f, 1.f);
	if (cosAngle > 1.f - 1e-7f)
		return glm::quat(1.f, 0.f, 0.f, 0.f);

	if (cosAngle < -1.f + 1e-7f)
	{
		// Antiparallel: every perpendicular axis is an equally valid half turn,
		// so pick one deterministically rather than letting a near-zero cross
		// product choose
		glm::vec3 axis= glm::cross(a, glm::vec3(1.f, 0.f, 0.f));
		if (glm::dot(axis, axis) < 1e-6f)
			axis= glm::cross(a, glm::vec3(0.f, 1.f, 0.f));
		return glm::angleAxis(glm::pi<float>(), glm::normalize(axis));
	}

	return glm::angleAxis(std::acos(cosAngle), glm::normalize(glm::cross(a, b)));
}

void buildPose(
	const std::array<HandPose, 2>& poses, const bool bSideValid[2],
	const TrackingFrameResult::HeadPose& head, const VmcBodyLengths& lengths, VmcPose& outPose)
{
	outPose.clear();

	// Head. Its measured frame is already +X facing, +Y the person's left, +Z
	// up, which is the avatar frame, so the rest head frame is the identity and
	// the measured orientation IS the local rotation (the neck is not streamed,
	// so it stays at rest).
	if (head.valid)
	{
		emitBone(outPose, eVmcBone::Head, glm::vec3(0.f, 0.f, lengths.headOffsetMeters),
				 head.orientationWorld);
	}

	// The clavicles are measured against the line between the shoulders, so
	// resolve that once for both sides
	const bool bLeftShoulder= bSideValid[0] && poses[0].hasWorldPose && poses[0].hasShoulder;
	const bool bRightShoulder= bSideValid[1] && poses[1].hasWorldPose && poses[1].hasShoulder;
	const bool bHasBothShoulders= bLeftShoulder && bRightShoulder;
	const glm::vec3 shoulderMidpoint=
		bHasBothShoulders
			? (poses[0].shoulderPositionWorld + poses[1].shoulderPositionWorld) * 0.5f
			: glm::vec3(0.f);

	for (int sideIndex= 0; sideIndex < (int)eHandSide::Count; ++sideIndex)
	{
		// VMC bones are a skeleton, not a point cloud: without a world-anchored
		// palm there is nothing to hang an arm off, and a camera-space pose
		// would put the avatar wherever the camera happens to be
		if (!bSideValid[sideIndex] || !poses[sideIndex].hasWorldPose)
			continue;

		buildSide(poses[sideIndex], (eHandSide)sideIndex, bHasBothShoulders, shoulderMidpoint, lengths,
				  outPose);
	}
}
} // namespace VmcRetarget
