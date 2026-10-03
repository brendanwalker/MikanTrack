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
		m_upperArmLength[s]= glm::length(lowerArm.restPositionWorld - upperArm.restPositionWorld);
		m_forearmLength[s]= glm::length(hand.restPositionWorld - lowerArm.restPositionWorld);
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
	const bool bLeft= side == eHandSide::Left;
	const Bone& wristBone= getBone(bLeft ? eHumanoidBone::LeftHand : eHumanoidBone::RightHand);
	const Bone& lowerArmBone= getBone(bLeft ? eHumanoidBone::LeftLowerArm : eHumanoidBone::RightLowerArm);
	const glm::vec3 wrist= wristBone.restPositionWorld;
	// Palm +Y runs toward the thumb on a left hand and toward the pinky on a right
	const float thumbSide= bLeft ? 1.f : -1.f;

	// The physical bones each finger has, in slot order
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		hand.fingerBoneCount[finger]= 0;
		hand.fingerSlots[finger]= {-1, -1, -1};
		for (int slot= 0; slot < 3; ++slot)
		{
			if (getBone(fingerBone(side, finger, slot)).present)
				hand.fingerSlots[finger][hand.fingerBoneCount[finger]++]= slot;
		}
	}
	auto physicalBonePosition= [&](int finger, int index) {
		return getBone(fingerBone(side, finger, hand.fingerSlots[finger][index])).restPositionWorld;
	};

	// A T-pose palm frame off the hand joint, the stand-in for anything the
	// rig does not say: +X toward the fingers (the middle finger if there is
	// one, else along the forearm), +Z out of the palm, which faces down
	constexpr float kDefaultHandLength= 0.085f;
	const bool bHasMiddle= hand.hasFinger((int)eFinger::Middle);
	glm::vec3 palmX= safeNormalize(bHasMiddle ? physicalBonePosition((int)eFinger::Middle, 0) - wrist
											  : wrist - lowerArmBone.restPositionWorld);
	if (glm::dot(palmX, palmX) <= 0.f)
		palmX= glm::vec3(0.f, bLeft ? 1.f : -1.f, 0.f);
	glm::vec3 palmZ= glm::vec3(0.f, 0.f, -1.f);
	palmZ= safeNormalize(palmZ - palmX * glm::dot(palmZ, palmX));
	if (glm::dot(palmZ, palmZ) <= 0.f)
		palmZ= glm::vec3(1.f, 0.f, 0.f);
	const glm::vec3 palmY= glm::cross(palmZ, palmX);
	const float handLength=
		bHasMiddle ? glm::length(physicalBonePosition((int)eFinger::Middle, 0) - wrist) : kDefaultHandLength;
	const float scale= handLength / kDefaultHandLength;
	m_handLength[(int)side]= bHasMiddle ? handLength : 0.f;

	// A 21-landmark hand assembled from the rest joints: physical bone i is
	// joint i, and the joints the rig lacks (down to the fingertip, which no
	// VRM has) are extrapolated along the last bone at a typical ratio.
	// A finger with no bones gets a stand-in laid out like a flat hand so
	// the shipping skeleton code still runs; its bone count keeps it off the
	// output.
	constexpr float kNextPerPrevious= 0.8f;
	std::array<glm::vec3, HAND_LANDMARK_COUNT> points;
	points[(int)eHandLandmark::WRIST]= wrist;

	struct StandIn
	{
		glm::vec3 base;      // palm-frame offset from the wrist, for a 0.085 m hand
		glm::vec3 direction; // palm-frame rest direction
	};
	const StandIn k_standIns[FINGER_COUNT]= {
		{{0.030f, 0.030f, 0.005f}, glm::normalize(glm::vec3(0.6f, 0.8f, 0.3f))}, // thumb, palmar and toward +Y
		{{0.085f, 0.020f, 0.f}, {1.f, 0.f, 0.f}},
		{{0.085f, 0.f, 0.f}, {1.f, 0.f, 0.f}},
		{{0.080f, -0.020f, 0.f}, {1.f, 0.f, 0.f}},
		{{0.070f, -0.035f, 0.f}, {1.f, 0.f, 0.f}},
	};

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const int* joints= FINGER_JOINTS[finger];
		const int count= hand.fingerBoneCount[finger];
		if (count > 0)
		{
			for (int index= 0; index < count; ++index)
				points[joints[index]]= physicalBonePosition(finger, index);

			glm::vec3 direction;
			float step;
			if (count >= 2)
			{
				const glm::vec3 lastBone= points[joints[count - 1]] - points[joints[count - 2]];
				direction= safeNormalize(lastBone);
				step= glm::length(lastBone) * kNextPerPrevious;
			}
			else
			{
				// One bone: it points the way the finger leaves the wrist
				direction= safeNormalize(points[joints[0]] - wrist);
				step= handLength * 0.35f;
			}
			if (glm::dot(direction, direction) <= 0.f)
				direction= palmX;
			for (int index= count; index < 4; ++index)
			{
				points[joints[index]]= points[joints[index - 1]] + direction * step;
				step*= kNextPerPrevious;
			}
		}
		else
		{
			const StandIn& standIn= k_standIns[finger];
			const glm::vec3 base= wrist + palmX * (standIn.base.x * scale) + palmY * (thumbSide * standIn.base.y * scale) +
				palmZ * (standIn.base.z * scale);
			const glm::vec3 direction= safeNormalize(palmX * standIn.direction.x + palmY * (thumbSide * standIn.direction.y) +
													 palmZ * standIn.direction.z);
			points[joints[0]]= base;
			float step= 0.03f * scale;
			for (int index= 1; index < 4; ++index)
			{
				points[joints[index]]= points[joints[index - 1]] + direction * step;
				step*= kNextPerPrevious;
			}
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
	hand.valid= true;

	// The forward kinematics reads the hand's chirality from which side of
	// the palm the index base sits, so a rig whose index is mapped on the
	// pinky side of its thumb (index and middle swapped, usually) curls its
	// thumb the wrong way and swaps two fingers' angles. Say so.
	if (hand.hasFinger((int)eFinger::Thumb) && hand.hasFinger((int)eFinger::Index))
	{
		const float thumbY= hand.skeleton.baseInPalm[(int)eFinger::Thumb].y;
		const float indexY= hand.skeleton.baseInPalm[(int)eFinger::Index].y;
		if (thumbY * indexY < 0.f)
		{
			m_warnings.push_back(std::string(bLeft ? "Left" : "Right") +
								 " hand: the index finger bone sits on the far side of the middle finger from the "
								 "thumb, so the rig's index and middle fingers look swapped; the thumb will curl "
								 "the wrong way until the mapping is fixed");
		}
	}
}
