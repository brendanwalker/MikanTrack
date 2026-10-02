#include "AvatarRetarget.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "AvatarSkeleton.h"
#include "HandPoseModel.h"
#include "MathGLM.h"

namespace
{
using B= eHumanoidBone;

glm::vec3 safeNormalize(const glm::vec3& v)
{
	const float lengthSquared= glm::dot(v, v);
	return lengthSquared > 1e-12f ? v / std::sqrt(lengthSquared) : glm::vec3(0.f);
}

// Signed angle from a to b about axis (all unit length, a and b perpendicular
// to axis)
float signedAngle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& axis)
{
	return atan2f(glm::dot(glm::cross(a, b), axis), glm::dot(a, b));
}

glm::vec3 projectPerpendicular(const glm::vec3& v, const glm::vec3& unitAxis)
{
	return v - unitAxis * glm::dot(v, unitAxis);
}

float wrapAngle(float radians)
{
	while (radians > glm::pi<float>())
		radians-= glm::two_pi<float>();
	while (radians < -glm::pi<float>())
		radians+= glm::two_pi<float>();
	return radians;
}

eHumanoidBone sideBone(int sideIndex, B left, B right)
{
	return sideIndex == 0 ? left : right;
}

eHumanoidBone fingerBone(int sideIndex, int finger, int phalanx)
{
	return (eHumanoidBone)((int)firstHumanoidFingerBone(sideIndex) + finger * 3 + phalanx);
}

// Rest geometry under the root placement, which every delta is measured from
struct RootedRest
{
	const AvatarSkeleton* skeleton= nullptr;
	glm::vec3 position{0.f};
	glm::quat rotation{1.f, 0.f, 0.f, 0.f};

	glm::vec3 bonePosition(B bone) const { return position + rotation * skeleton->getBone(bone).restPositionWorld; }
	glm::vec3 offset(B bone) const { return rotation * skeleton->getBone(bone).restOffsetFromParentWorld; }
	glm::vec3 direction(B bone) const { return safeNormalize(offset(bone)); }
};

// A bone whose own twist is not measured: swing the parent-posed rest
// direction onto the measured one and chain the result
glm::quat swingTo(const glm::quat& parentDelta, const glm::vec3& rootedRestDirection, const glm::vec3& worldDirection)
{
	return glm_shortest_arc(parentDelta * rootedRestDirection, worldDirection) * parentDelta;
}

// Elbow on the circle where the upper-arm sphere about the shoulder meets
// the forearm sphere about the wrist, at the point nearest the pole. A wrist
// beyond reach straightens the arm; the caller clamps reach beforehand.
glm::vec3 solveElbow(const glm::vec3& shoulder, const glm::vec3& wrist, float upperArm, float forearm,
					 const glm::vec3& polePoint, const glm::vec3& fallbackPoleDirection, const glm::vec3& restDirection)
{
	const glm::vec3 toWrist= wrist - shoulder;
	const float reach= glm::length(toWrist);
	const glm::vec3 axis= reach > 1e-6f ? toWrist / reach : restDirection;

	// At (or within rounding of) full reach the arm is straight; the bend
	// direction is meaningless there and would only jitter
	if (reach >= (upperArm + forearm) * (1.f - 1e-5f))
		return shoulder + axis * upperArm;

	// Law of cosines: distance along the axis from the shoulder to the
	// elbow circle's center, then the circle's radius
	float along= (upperArm * upperArm - forearm * forearm + reach * reach) / std::max(2.f * reach, 1e-6f);
	along= std::clamp(along, -upperArm, upperArm);
	const float radius= std::sqrt(std::max(upperArm * upperArm - along * along, 0.f));
	const glm::vec3 center= shoulder + axis * along;

	glm::vec3 bend= projectPerpendicular(polePoint - center, axis);
	if (glm::dot(bend, bend) < 1e-8f)
		bend= projectPerpendicular(fallbackPoleDirection, axis);
	if (glm::dot(bend, bend) < 1e-8f)
	{
		// The pole sits on the arm axis: any perpendicular will do, chosen
		// deterministically
		bend= glm::cross(axis, glm::vec3(0.f, 0.f, 1.f));
		if (glm::dot(bend, bend) < 1e-8f)
			bend= glm::cross(axis, glm::vec3(1.f, 0.f, 0.f));
	}
	return center + safeNormalize(bend) * radius;
}
} // namespace

void AvatarPose::clear()
{
	valid= false;
	timestampMs= 0.0;
	rootPositionWorld= glm::vec3(0.f);
	rootRotationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
	bones= {};
}

void AvatarRetarget::reset()
{
	m_root= RootState();
}

void AvatarRetarget::solve(const TrackingFrameResult& frame, const BodyDimensions& user,
						   const AvatarSkeleton& skeleton, const AvatarRetargetConfig& config, AvatarPose& outPose)
{
	const bool bSideValid[2]= {
		frame.poses[0].tracked && frame.poses[0].hasWorldPose,
		frame.poses[1].tracked && frame.poses[1].hasWorldPose,
	};
	solve(frame.poses, bSideValid, frame.head, frame.timestampMs, user, skeleton, config, outPose);
}

void AvatarRetarget::resolveRoot(const std::array<HandPose, 2>& poses, const bool bSideValid[2], double timestampMs,
								 const AvatarSkeleton& skeleton, const AvatarRetargetConfig& config,
								 AvatarPose& outPose)
{
	const bool bBothShoulders= bSideValid[0] && bSideValid[1] && poses[0].hasShoulder && poses[1].hasShoulder;

	if (config.followShoulders && bBothShoulders)
	{
		// The shoulder line fixes the yaw (at rest it runs along world +Y,
		// right to left) and the midpoint fixes the translation
		const glm::vec3 midpoint= (poses[0].shoulderPositionWorld + poses[1].shoulderPositionWorld) * 0.5f;
		const glm::vec3 line= poses[0].shoulderPositionWorld - poses[1].shoulderPositionWorld;
		float yaw= m_root.valid ? m_root.yawRadians : glm::radians(config.fixedRootYawDegrees);
		if (line.x * line.x + line.y * line.y > 1e-6f)
			yaw= atan2f(line.y, line.x) - glm::half_pi<float>();

		const glm::quat rotation= glm::angleAxis(yaw, glm::vec3(0.f, 0.f, 1.f));
		const glm::vec3 position= midpoint - rotation * skeleton.getRestShoulderMidpointWorld();

		// First-order follow on the frame clock; a timestamp regression (a
		// replay scrub) snaps rather than filtering across the jump
		const double dtSeconds= m_root.valid ? (timestampMs - m_root.timestampMs) * 0.001 : -1.0;
		if (dtSeconds < 0.0 || config.rootFollowTimeConstantS <= 0.f)
		{
			m_root.position= position;
			m_root.yawRadians= yaw;
		}
		else
		{
			const float alpha= 1.f - expf(-(float)std::min(dtSeconds, 1.0) / config.rootFollowTimeConstantS);
			m_root.position+= (position - m_root.position) * alpha;
			m_root.yawRadians= wrapAngle(m_root.yawRadians + wrapAngle(yaw - m_root.yawRadians) * alpha);
		}
		m_root.valid= true;
		m_root.timestampMs= timestampMs;
	}
	else if (!config.followShoulders)
	{
		m_root.valid= false;
	}

	if (m_root.valid)
	{
		outPose.rootPositionWorld= m_root.position;
		outPose.rootRotationWorld= glm::angleAxis(m_root.yawRadians, glm::vec3(0.f, 0.f, 1.f));
	}
	else
	{
		outPose.rootPositionWorld= config.fixedRootPositionWorld;
		outPose.rootRotationWorld= glm::angleAxis(glm::radians(config.fixedRootYawDegrees), glm::vec3(0.f, 0.f, 1.f));
	}
}

void AvatarRetarget::solve(const std::array<HandPose, 2>& poses, const bool bSideValid[2],
						   const TrackingFrameResult::HeadPose& head, double timestampMs, const BodyDimensions& user,
						   const AvatarSkeleton& skeleton, const AvatarRetargetConfig& config, AvatarPose& outPose)
{
	outPose.clear();
	outPose.valid= true;
	outPose.timestampMs= timestampMs;
	resolveRoot(poses, bSideValid, timestampMs, skeleton, config, outPose);

	RootedRest rest;
	rest.skeleton= &skeleton;
	rest.position= outPose.rootPositionWorld;
	rest.rotation= outPose.rootRotationWorld;

	// The hips carry the root placement into the node hierarchy; everything
	// not posed below follows them at rest
	outPose.bones[(int)B::Hips].present= true;

	// Head: its measured frame is +X facing, +Y left, +Z up, which is the
	// avatar's rest frame, so the measured orientation IS the posed world
	// rotation of the canonical head frame and the delta is what remains
	// after the root yaw
	if (head.valid && skeleton.getBone(B::Head).present)
	{
		AvatarPose::Bone& headBone= outPose.bones[(int)B::Head];
		headBone.present= true;
		headBone.deltaWorld= head.orientationWorld * glm::inverse(rest.rotation);
	}

	const bool bBothShoulders= bSideValid[0] && bSideValid[1] && poses[0].hasShoulder && poses[1].hasShoulder;
	const glm::vec3 measuredShoulderMidpoint=
		bBothShoulders ? (poses[0].shoulderPositionWorld + poses[1].shoulderPositionWorld) * 0.5f : glm::vec3(0.f);
	const glm::vec3 defaultPole= safeNormalize(config.defaultElbowPoleWorld);

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!bSideValid[sideIndex])
			continue;
		const HandPose& pose= poses[sideIndex];
		const eHandSide side= (eHandSide)sideIndex;

		const B shoulderBone= sideBone(sideIndex, B::LeftShoulder, B::RightShoulder);
		const B upperArmBone= sideBone(sideIndex, B::LeftUpperArm, B::RightUpperArm);
		const B lowerArmBone= sideBone(sideIndex, B::LeftLowerArm, B::RightLowerArm);
		const B handBone= sideBone(sideIndex, B::LeftHand, B::RightHand);

		// Clavicle: measured against the line between the shoulders, so it
		// needs both; with one shoulder there is no reference for a shrug
		glm::quat chainDelta(1.f, 0.f, 0.f, 0.f);
		glm::vec3 avatarShoulder;
		if (skeleton.getBone(shoulderBone).present)
		{
			const glm::vec3 clavicleBase= rest.bonePosition(shoulderBone);
			if (bBothShoulders)
			{
				const glm::vec3 direction= safeNormalize(pose.shoulderPositionWorld - measuredShoulderMidpoint);
				if (glm::dot(direction, direction) > 0.f)
				{
					chainDelta= swingTo(chainDelta, rest.direction(upperArmBone), direction);
					AvatarPose::Bone& clavicle= outPose.bones[(int)shoulderBone];
					clavicle.present= true;
					clavicle.deltaWorld= chainDelta;
				}
			}
			avatarShoulder= clavicleBase + chainDelta * rest.offset(upperArmBone);
		}
		else
		{
			avatarShoulder= rest.bonePosition(upperArmBone);
		}

		// The wrist target: the user's reach from their shoulder, scaled onto
		// the avatar's arm. Without a measured shoulder the avatar's own
		// stands in, which keeps the hand where it was seen.
		const float upperArm= skeleton.getUpperArmLength(side);
		const float forearm= skeleton.getForearmLength(side);
		const float avatarReach= upperArm + forearm;
		const float userReach= user.upperArmLengthMeters + user.forearmLengthMeters;
		const float ratio= userReach > 1e-3f ? avatarReach / userReach : 1.f;

		const glm::vec3 userShoulder= pose.hasShoulder ? pose.shoulderPositionWorld : avatarShoulder;
		const glm::vec3 userWrist= pose.getWristPositionWorld();
		glm::vec3 wrist= avatarShoulder + (userWrist - userShoulder) * ratio;
		const glm::vec3 toWrist= wrist - avatarShoulder;
		const float reach= glm::length(toWrist);
		if (reach > avatarReach && reach > 1e-6f)
			wrist= avatarShoulder + toWrist * (avatarReach / reach);

		// Elbow: the measured elbow, scaled the same way, hints which way the
		// arm bends; otherwise the default pole does
		glm::vec3 polePoint;
		if (pose.hasForearmPose)
		{
			const glm::vec3 userElbow= pose.getElbowPositionWorld(user.forearmLengthMeters);
			polePoint= avatarShoulder + (userElbow - userShoulder) * ratio;
		}
		else
		{
			polePoint= (avatarShoulder + wrist) * 0.5f + defaultPole * avatarReach;
		}
		const glm::vec3 elbow=
			solveElbow(avatarShoulder, wrist, upperArm, forearm, polePoint, defaultPole, rest.direction(lowerArmBone));

		// Upper arm: direction only
		const glm::vec3 upperDirection= safeNormalize(elbow - avatarShoulder);
		if (glm::dot(upperDirection, upperDirection) > 0.f)
			chainDelta= swingTo(chainDelta, rest.direction(lowerArmBone), upperDirection);
		{
			AvatarPose::Bone& bone= outPose.bones[(int)upperArmBone];
			bone.present= true;
			bone.deltaWorld= chainDelta;
		}

		// Forearm: direction from the solve, roll from the measured forearm
		// (the IMU or the body solver) or, with none, from the palm, which
		// puts the wrist at zero bend rather than inventing a twist
		const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
		const glm::quat restPalmRotation= glm::quat_cast(glm::mat3(hand.palmFrameWorld));
		{
			glm::vec3 forearmDirection= safeNormalize(wrist - elbow);
			if (glm::dot(forearmDirection, forearmDirection) <= 0.f)
				forearmDirection= upperDirection;
			chainDelta= swingTo(chainDelta, rest.direction(handBone), forearmDirection);

			{
				const glm::quat measuredForearm= pose.hasForearmPose ? pose.forearmOrientationWorld : pose.palmOrientationWorld;
				const glm::vec3 measuredPalmNormal=
					safeNormalize(projectPerpendicular(measuredForearm * glm::vec3(0.f, 0.f, 1.f), forearmDirection));
				const glm::vec3 restPalmNormal= safeNormalize(projectPerpendicular(
					chainDelta * (rest.rotation * glm::vec3(hand.palmFrameWorld[2])), forearmDirection));
				if (glm::dot(measuredPalmNormal, measuredPalmNormal) > 0.f && glm::dot(restPalmNormal, restPalmNormal) > 0.f)
				{
					const float twist= signedAngle(restPalmNormal, measuredPalmNormal, forearmDirection);
					chainDelta= glm::angleAxis(twist, forearmDirection) * chainDelta;
				}
			}
			AvatarPose::Bone& bone= outPose.bones[(int)lowerArmBone];
			bone.present= true;
			bone.deltaWorld= chainDelta;
		}
		const glm::vec3 handPosition= elbow + chainDelta * rest.offset(handBone);

		// Hand: the full measured palm frame against the avatar's rest palm
		const glm::quat handDelta= pose.palmOrientationWorld * glm::inverse(rest.rotation * restPalmRotation);
		{
			AvatarPose::Bone& bone= outPose.bones[(int)handBone];
			bone.present= true;
			bone.deltaWorld= handDelta;
		}

		// Fingers: the measured angles played through the forward kinematics
		// on the AVATAR's hand skeleton, then each phalanx swung from its
		// rest direction relative to the bone before it, so the roll of the
		// hand carries down the chain
		const glm::vec3 restPalmCenter(hand.palmFrameWorld[3]);
		const glm::vec3 restHandPosition= skeleton.getBone(handBone).restPositionWorld;
		const glm::quat posedPalmRotation= handDelta * rest.rotation * restPalmRotation;
		glm::mat4 palmTransform= glm::mat4_cast(posedPalmRotation);
		palmTransform[3]= glm::vec4(handPosition + handDelta * (rest.rotation * (restPalmCenter - restHandPosition)), 1.f);

		// A rig with fewer bones than the measured finger folds the bends
		// past its last bone into that bone, so a fist still closes on a
		// two-bone finger
		std::array<FingerAngles, FINGER_COUNT> angles= pose.fingers;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			FingerAngles& fingerAngles= angles[finger];
			switch (hand.fingerBoneCount[finger])
			{
			case 1:
				fingerAngles.proximal+= fingerAngles.intermediate + fingerAngles.distal;
				fingerAngles.intermediate= 0.f;
				fingerAngles.distal= 0.f;
				break;
			case 2:
				fingerAngles.intermediate+= fingerAngles.distal;
				fingerAngles.distal= 0.f;
				break;
			default:
				break;
			}
		}

		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
		HandPoseModel::buildFingerJoints(palmTransform, hand.skeleton, angles, joints);

		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			glm::quat fingerDelta= handDelta;
			for (int index= 0; index < hand.fingerBoneCount[finger]; ++index)
			{
				const glm::vec3 direction= safeNormalize(joints[finger][index + 1] - joints[finger][index]);
				if (glm::dot(direction, direction) > 0.f)
				{
					fingerDelta=
						swingTo(fingerDelta, rest.rotation * hand.restPhalanxDirWorld[finger][index], direction);
				}
				AvatarPose::Bone& bone=
					outPose.bones[(int)fingerBone(sideIndex, finger, hand.fingerSlots[finger][index])];
				bone.present= true;
				bone.deltaWorld= fingerDelta;
			}
		}
	}
}

void computePosedBones(const AvatarSkeleton& skeleton, const AvatarPose& pose,
					   std::array<AvatarPosedBone, HUMANOID_BONE_COUNT>& outBones)
{
	// Enum order lists every parent before its children, so one pass composes
	// the chain
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const AvatarSkeleton::Bone& restBone= skeleton.getBone((B)index);
		AvatarPosedBone& posed= outBones[index];
		if (!restBone.present)
		{
			posed= AvatarPosedBone();
			continue;
		}

		const glm::quat& delta= pose.bones[index].deltaWorld;
		posed.rotationWorld= delta * pose.rootRotationWorld * restBone.restRotationWorld;
		if (restBone.parent == HUMANOID_BONE_NONE)
		{
			posed.positionWorld= pose.rootPositionWorld + pose.rootRotationWorld * restBone.restPositionWorld;
		}
		else
		{
			const AvatarPosedBone& parent= outBones[(int)restBone.parent];
			const glm::quat& parentDelta= pose.bones[(int)restBone.parent].deltaWorld;
			posed.positionWorld=
				parent.positionWorld + parentDelta * (pose.rootRotationWorld * restBone.restOffsetFromParentWorld);
		}
	}
}

namespace
{
void composePosedGlobals(const AvatarModel& model, const std::vector<glm::mat4>& restLocals, int node,
						 const glm::mat4& parentGlobal, const std::vector<const glm::mat4*>& overrides,
						 std::vector<glm::mat4>& outGlobals)
{
	outGlobals[node]= overrides[node] != nullptr ? *overrides[node] : parentGlobal * restLocals[node];
	for (int child : model.nodes[node].children)
	{
		if (child >= 0 && child < (int)model.nodes.size())
			composePosedGlobals(model, restLocals, child, outGlobals[node], overrides, outGlobals);
	}
}
} // namespace

void computePosedGlobals(const AvatarModel& model, const AvatarSkeleton& skeleton, const AvatarPose& pose,
						 std::vector<glm::mat4>& outGlobals)
{
	if (!pose.valid)
	{
		outGlobals= skeleton.getRestGlobalsAvatar();
		return;
	}

	std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posedBones;
	computePosedBones(skeleton, pose, posedBones);

	// Present bones override their node's global (back in avatar space);
	// everything else composes from its parent as authored
	const glm::mat4 avatarFromWorld= glm::inverse(skeleton.getWorldFromAvatar());
	std::array<glm::mat4, HUMANOID_BONE_COUNT> overrideStorage;
	std::vector<const glm::mat4*> overrides(model.nodes.size(), nullptr);
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const AvatarSkeleton::Bone& restBone= skeleton.getBone((B)index);
		if (!restBone.present || !pose.bones[index].present)
			continue;
		glm::mat4 posedWorld= glm::mat4_cast(posedBones[index].rotationWorld);
		posedWorld[3]= glm::vec4(posedBones[index].positionWorld, 1.f);
		overrideStorage[index]= avatarFromWorld * posedWorld;
		if (restBone.node >= 0 && restBone.node < (int)model.nodes.size())
			overrides[restBone.node]= &overrideStorage[index];
	}

	outGlobals.assign(model.nodes.size(), glm::mat4(1.f));
	for (int root : model.rootNodes)
		composePosedGlobals(model, skeleton.getRestLocalsAvatar(), root, glm::mat4(1.f), overrides, outGlobals);
}
