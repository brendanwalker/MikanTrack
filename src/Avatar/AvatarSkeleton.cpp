#include "AvatarSkeleton.h"

#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"

namespace
{
glm::vec3 safeNormalize(const glm::vec3& v)
{
	const float lengthSquared= glm::dot(v, v);
	return lengthSquared > 1e-12f ? v / std::sqrt(lengthSquared) : glm::vec3(0.f);
}

void composeGlobals(const AvatarModel& model, const std::vector<glm::mat4>& locals, int node,
					const glm::mat4& parentGlobal, std::vector<glm::mat4>& outGlobals)
{
	outGlobals[node]= parentGlobal * locals[node];
	for (int child : model.nodes[node].children)
	{
		if (child >= 0 && child < (int)model.nodes.size())
			composeGlobals(model, locals, child, outGlobals[node], outGlobals);
	}
}

// Rotation part of a transform with any scale divided out
glm::quat rotationOf(const glm::mat4& transform)
{
	glm::mat3 basis(transform);
	for (int column= 0; column < 3; ++column)
		basis[column]= safeNormalize(basis[column]);
	return glm::normalize(glm::quat_cast(basis));
}

// The finger bone slots of one hand in app finger order: thumb, index,
// middle, ring, little, three bones each, proximal first
eHumanoidBone fingerBone(eHandSide side, int finger, int phalanx)
{
	return (eHumanoidBone)((int)firstHumanoidFingerBone((int)side) + finger * 3 + phalanx);
}
} // namespace

void computeNodeGlobals(const AvatarModel& model, const std::vector<glm::mat4>& locals,
						std::vector<glm::mat4>& outGlobals)
{
	outGlobals.assign(model.nodes.size(), glm::mat4(1.f));
	for (int root : model.rootNodes)
		composeGlobals(model, locals, root, glm::mat4(1.f), outGlobals);
}

glm::mat4 makeWorldFromAvatar(eVrmVersion version)
{
	// Columns are the world images of the glTF axes. VRM 1.0: glTF +X (the
	// model's left) -> world +Y, glTF +Y (up) -> world +Z, glTF +Z (facing)
	// -> world +X. VRM 0.x faces -Z instead, which is the same mapping turned
	// half a turn about world +Z.
	if (version == eVrmVersion::Vrm1)
	{
		return glm::mat4(
			glm::vec4(0.f, 1.f, 0.f, 0.f), glm::vec4(0.f, 0.f, 1.f, 0.f), glm::vec4(1.f, 0.f, 0.f, 0.f),
			glm::vec4(0.f, 0.f, 0.f, 1.f));
	}
	return glm::mat4(
		glm::vec4(0.f, -1.f, 0.f, 0.f), glm::vec4(0.f, 0.f, 1.f, 0.f), glm::vec4(-1.f, 0.f, 0.f, 0.f),
		glm::vec4(0.f, 0.f, 0.f, 1.f));
}

AvatarSkeleton::AvatarSkeleton(const AvatarModel& model)
	: m_version(model.version)
	, m_worldFromAvatar(makeWorldFromAvatar(model.version))
{
	m_restLocalsAvatar.reserve(model.nodes.size());
	for (const AvatarNode& node : model.nodes)
		m_restLocalsAvatar.push_back(node.localTransform);
	computeNodeGlobals(model, m_restLocalsAvatar, m_restGlobalsAvatar);

	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		Bone& bone= m_bones[index];
		bone.node= model.humanoidNodes[index];
		bone.present= bone.node >= 0;
		if (!bone.present)
			continue;

		const glm::mat4 restWorld= m_worldFromAvatar * m_restGlobalsAvatar[bone.node];
		bone.restPositionWorld= glm::vec3(restWorld[3]);
		bone.restRotationWorld= rotationOf(restWorld);

		eHumanoidBone parent= humanoidBoneParent((eHumanoidBone)index);
		while (parent != HUMANOID_BONE_NONE && !model.hasBone(parent))
			parent= humanoidBoneParent(parent);
		bone.parent= parent;
	}
	for (Bone& bone : m_bones)
	{
		if (bone.present && bone.parent != HUMANOID_BONE_NONE)
			bone.restOffsetFromParentWorld= bone.restPositionWorld - m_bones[(int)bone.parent].restPositionWorld;
	}

	using B= eHumanoidBone;
	const eHandSide sides[2]= {eHandSide::Left, eHandSide::Right};
	for (eHandSide side : sides)
	{
		const int s= (int)side;
		const Bone& upperArm= getBone(s == 0 ? B::LeftUpperArm : B::RightUpperArm);
		const Bone& lowerArm= getBone(s == 0 ? B::LeftLowerArm : B::RightLowerArm);
		const Bone& hand= getBone(s == 0 ? B::LeftHand : B::RightHand);
		const Bone& middle= getBone(s == 0 ? B::LeftMiddleProximal : B::RightMiddleProximal);
		m_upperArmLength[s]= glm::length(lowerArm.restPositionWorld - upperArm.restPositionWorld);
		m_forearmLength[s]= glm::length(hand.restPositionWorld - lowerArm.restPositionWorld);
		m_handLength[s]= middle.present ? glm::length(middle.restPositionWorld - hand.restPositionWorld) : 0.f;
		buildHand(model, side);
	}
	m_shoulderWidth=
		glm::length(getBone(B::LeftUpperArm).restPositionWorld - getBone(B::RightUpperArm).restPositionWorld);
	m_restShoulderMidpointWorld=
		(getBone(B::LeftUpperArm).restPositionWorld + getBone(B::RightUpperArm).restPositionWorld) * 0.5f;

	const float hipsZ= getBone(B::Hips).restPositionWorld.z;
	m_heightAboveHips= 0.f;
	for (const Bone& bone : m_bones)
	{
		if (bone.present)
			m_heightAboveHips= std::max(m_heightAboveHips, bone.restPositionWorld.z - hipsZ);
	}
}

void AvatarSkeleton::buildHand(const AvatarModel& model, eHandSide side)
{
	HandRest& hand= m_hands[(int)side];
	const Bone& wristBone= getBone(side == eHandSide::Left ? eHumanoidBone::LeftHand : eHumanoidBone::RightHand);

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		hand.fingerPresent[finger]= true;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			hand.fingerPresent[finger]&= getBone(fingerBone(side, finger, phalanx)).present;
	}
	// The palm frame needs the wrist, index, middle and little bases
	hand.valid= hand.fingerPresent[(int)eFinger::Index] && hand.fingerPresent[(int)eFinger::Middle] &&
		hand.fingerPresent[(int)eFinger::Pinky];
	if (!hand.valid)
		return;

	// A 21-landmark hand assembled from the rest joints: the three finger
	// bones are the base, middle and distal joints, and the fingertip (no
	// bone in a VRM) is extrapolated along the distal bone at a typical
	// distal-to-intermediate ratio. A missing optional finger (thumb or
	// ring) gets a stand-in straight along the palm so the shipping skeleton
	// code still runs; its fingerPresent flag keeps it off the output.
	std::array<glm::vec3, HAND_LANDMARK_COUNT> points;
	points[(int)eHandLandmark::WRIST]= wristBone.restPositionWorld;

	const glm::vec3 middleBase= getBone(fingerBone(side, (int)eFinger::Middle, 0)).restPositionWorld;
	const glm::vec3 palmDirection= safeNormalize(middleBase - wristBone.restPositionWorld);
	constexpr float kTipPerDistal= 0.8f;

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const int* joints= FINGER_JOINTS[finger];
		if (hand.fingerPresent[finger])
		{
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				points[joints[phalanx]]= getBone(fingerBone(side, finger, phalanx)).restPositionWorld;
			const glm::vec3 distalBone= points[joints[2]] - points[joints[1]];
			points[joints[3]]= points[joints[2]] + distalBone * kTipPerDistal;
		}
		else
		{
			const float step= glm::length(middleBase - wristBone.restPositionWorld) * 0.3f;
			points[joints[0]]= middleBase;
			for (int joint= 1; joint < 4; ++joint)
				points[joints[joint]]= points[joints[joint - 1]] + palmDirection * step;
		}
	}

	// Palms face down in the T-pose every VRM rests in, so the palmar side is
	// known: seed the palmar memory with world down and the frame code keeps
	// it unless the fingers curl decisively, which rest fingers do not
	HandPoseModel::PalmarSideMemory memory;
	memory.palmarNormal= glm::vec3(0.f, 0.f, -1.f);

	hand.palmFrameWorld= HandPoseModel::computePalmFrame(points, side, &memory);
	HandPoseModel::computeSkeleton(points, side, hand.skeleton, &memory);

	// The avatar's own rest finger directions as the zero pose
	const glm::mat4 palmInverse= glm::inverse(hand.palmFrameWorld);
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const int* joints= FINGER_JOINTS[finger];
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			hand.restPhalanxDirWorld[finger][phalanx]= safeNormalize(points[joints[phalanx + 1]] - points[joints[phalanx]]);
		const glm::vec3 restDirection= glm::mat3(palmInverse) * (points[joints[1]] - points[joints[0]]);
		const glm::vec3 normalized= safeNormalize(restDirection);
		if (glm::dot(normalized, normalized) > 0.f)
			hand.skeleton.neutralDirInPalm[finger]= normalized;
	}
}
