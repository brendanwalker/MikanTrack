#include "AvatarTypes.h"

#include <cstring>

const char* humanoidBoneName(eHumanoidBone bone)
{
	static const char* k_names[HUMANOID_BONE_COUNT]= {
		"Hips", "Spine", "Chest", "UpperChest", "Neck", "Head", "LeftEye", "RightEye", "Jaw",

		"LeftShoulder", "LeftUpperArm", "LeftLowerArm", "LeftHand",
		"RightShoulder", "RightUpperArm", "RightLowerArm", "RightHand",

		"LeftUpperLeg", "LeftLowerLeg", "LeftFoot", "LeftToes",
		"RightUpperLeg", "RightLowerLeg", "RightFoot", "RightToes",

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
	return (index >= 0 && index < HUMANOID_BONE_COUNT) ? k_names[index] : "";
}

eHumanoidBone humanoidBoneFromName(const char* name)
{
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		if (std::strcmp(humanoidBoneName((eHumanoidBone)index), name) == 0)
			return (eHumanoidBone)index;
	}
	return eHumanoidBone::Count;
}

eHumanoidBone humanoidBoneParent(eHumanoidBone bone)
{
	using B= eHumanoidBone;
	switch (bone)
	{
	case B::Hips: return HUMANOID_BONE_NONE;
	case B::Spine: return B::Hips;
	case B::Chest: return B::Spine;
	case B::UpperChest: return B::Chest;
	case B::Neck: return B::UpperChest;
	case B::Head: return B::Neck;
	case B::LeftEye:
	case B::RightEye:
	case B::Jaw: return B::Head;

	case B::LeftShoulder: return B::UpperChest;
	case B::LeftUpperArm: return B::LeftShoulder;
	case B::LeftLowerArm: return B::LeftUpperArm;
	case B::LeftHand: return B::LeftLowerArm;
	case B::RightShoulder: return B::UpperChest;
	case B::RightUpperArm: return B::RightShoulder;
	case B::RightLowerArm: return B::RightUpperArm;
	case B::RightHand: return B::RightLowerArm;

	case B::LeftUpperLeg: return B::Hips;
	case B::LeftLowerLeg: return B::LeftUpperLeg;
	case B::LeftFoot: return B::LeftLowerLeg;
	case B::LeftToes: return B::LeftFoot;
	case B::RightUpperLeg: return B::Hips;
	case B::RightLowerLeg: return B::RightUpperLeg;
	case B::RightFoot: return B::RightLowerLeg;
	case B::RightToes: return B::RightFoot;

	default:
		break;
	}

	// Fingers: proximal hangs off the hand, the other two off the bone before
	const int index= (int)bone;
	if (index >= (int)B::LeftThumbProximal && index < HUMANOID_BONE_COUNT)
	{
		const int sideIndex= index >= (int)B::RightThumbProximal ? 1 : 0;
		const int phalanx= (index - (int)firstHumanoidFingerBone(sideIndex)) % 3;
		if (phalanx == 0)
			return sideIndex == 0 ? B::LeftHand : B::RightHand;
		return (B)(index - 1);
	}

	return HUMANOID_BONE_NONE;
}

bool isHumanoidBoneRequired(eHumanoidBone bone)
{
	using B= eHumanoidBone;
	switch (bone)
	{
	case B::Hips:
	case B::Spine:
	case B::Head:
	case B::LeftUpperArm:
	case B::LeftLowerArm:
	case B::LeftHand:
	case B::RightUpperArm:
	case B::RightLowerArm:
	case B::RightHand:
	case B::LeftUpperLeg:
	case B::LeftLowerLeg:
	case B::LeftFoot:
	case B::RightUpperLeg:
	case B::RightLowerLeg:
	case B::RightFoot:
		return true;
	default:
		return false;
	}
}

eHumanoidBone firstHumanoidFingerBone(int sideIndex)
{
	return sideIndex == 0 ? eHumanoidBone::LeftThumbProximal : eHumanoidBone::RightThumbProximal;
}

const AvatarExpression* AvatarModel::findExpressionByPreset(const char* preset) const
{
	if (preset == nullptr || preset[0] == '\0')
		return nullptr;
	for (const AvatarExpression& expression : expressions)
	{
		if (expression.preset == preset)
			return &expression;
	}
	return nullptr;
}

const AvatarExpression* AvatarModel::findExpressionByName(const char* name) const
{
	if (name == nullptr)
		return nullptr;
	for (const AvatarExpression& expression : expressions)
	{
		if (expression.name == name)
			return &expression;
	}
	return nullptr;
}

size_t AvatarModel::triangleCount() const
{
	size_t count= 0;
	for (const AvatarMesh& mesh : meshes)
	{
		for (const AvatarPrimitive& primitive : mesh.primitives)
			count+= primitive.indices.size() / 3;
	}
	return count;
}
