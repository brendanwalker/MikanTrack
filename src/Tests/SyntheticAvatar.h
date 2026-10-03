#pragma once

#include <string>
#include <vector>

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/gtc/quaternion.hpp"

#include "AvatarTypes.h"

// A minimal synthetic humanoid shared by the avatar tests: a T-posed rig with
// authored world rest positions (VRM 1.0 facing, +Z forward, model left +X),
// optional shoulders, no upper chest, both hands with five fingers, and legs.
// The VRM loader test writes it into a GLB; the retarget test builds the
// AvatarModel directly.
namespace SyntheticAvatar
{
struct RigNode
{
	std::string name;          // doubles as the VRM 0.x bone name (lowerCamel)
	const char* bone= nullptr; // null for a helper node
	int parent= -1;
	glm::vec3 world{0.f};      // rest position, VRM 1.0 facing
	glm::quat localRotation{1.f, 0.f, 0.f, 0.f};
};

// Arm lengths and the shoulder geometry the rig is authored with
constexpr float kUpperArmLength= 0.25f;
constexpr float kForearmLength= 0.22f;
constexpr float kShoulderWidth= 0.30f; // between the upper-arm joints
constexpr float kHandLength= 0.085f;   // hand joint to middle finger base

// rotateLeftUpperArmNode puts a 30 degree rest rotation on the left upper arm
// node with its subtree compensated, so the world rest positions stay put (a
// VRM 1.0 rig need not be normalized)
inline std::vector<RigNode> makeRig(bool rotateLeftUpperArmNode)
{
	std::vector<RigNode> nodes;
	auto add= [&](const std::string& bone, int parent, glm::vec3 world) {
		RigNode node;
		node.name= bone;
		node.parent= parent;
		node.world= world;
		nodes.push_back(node);
		nodes.back().bone= nodes.back().name.c_str();
		return (int)nodes.size() - 1;
	};

	const int hips= add("hips", -1, {0.f, 0.9f, 0.f});
	const int spine= add("spine", hips, {0.f, 1.0f, 0.f});
	const int chest= add("chest", spine, {0.f, 1.15f, 0.f});
	const int neck= add("neck", chest, {0.f, 1.35f, 0.f}); // no upperChest: parent resolution falls back
	add("head", neck, {0.f, 1.45f, 0.f});

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const float sx= sideIndex == 0 ? 1.f : -1.f;
		const std::string prefix= sideIndex == 0 ? "left" : "right";
		auto mirror= [&](float x, float y, float z) { return glm::vec3(sx * x, y, z); };

		const int shoulder= add(prefix + "Shoulder", chest, mirror(0.05f, 1.30f, 0.f));
		const int upperArm= add(prefix + "UpperArm", shoulder, mirror(0.15f, 1.30f, 0.f));
		if (rotateLeftUpperArmNode && sideIndex == 0)
			nodes[upperArm].localRotation= glm::angleAxis(glm::radians(30.f), glm::vec3(0.f, 1.f, 0.f));
		const int lowerArm= add(prefix + "LowerArm", upperArm, mirror(0.15f + kUpperArmLength, 1.30f, 0.f));
		const int hand= add(prefix + "Hand", lowerArm, mirror(0.15f + kUpperArmLength + kForearmLength, 1.30f, 0.f));
		const float handX= 0.15f + kUpperArmLength + kForearmLength;

		// Fingers along +X, palm down (-Y), thumb toward the model's front
		// (+Z), slightly palmar and angled down
		struct FingerSpec
		{
			const char* name;
			glm::vec3 base;
			glm::vec3 direction;
		};
		const FingerSpec fingers[5]= {
			{"Thumb", {handX + 0.03f, 1.29f, 0.03f}, glm::normalize(glm::vec3(0.6f, -0.3f, 0.8f))},
			{"Index", {handX + 0.08f, 1.30f, 0.02f}, {1.f, 0.f, 0.f}},
			{"Middle", {handX + kHandLength, 1.30f, 0.f}, {1.f, 0.f, 0.f}},
			{"Ring", {handX + 0.08f, 1.30f, -0.02f}, {1.f, 0.f, 0.f}},
			{"Little", {handX + 0.07f, 1.30f, -0.035f}, {1.f, 0.f, 0.f}},
		};
		for (const FingerSpec& finger : fingers)
		{
			const glm::vec3 base= mirror(finger.base.x, finger.base.y, finger.base.z);
			const glm::vec3 direction= glm::vec3(sx * finger.direction.x, finger.direction.y, finger.direction.z);
			const int proximal= add(prefix + finger.name + "Proximal", hand, base);
			const int intermediate= add(prefix + finger.name + "Intermediate", proximal, base + direction * 0.03f);
			add(prefix + finger.name + "Distal", intermediate, base + direction * 0.05f);
		}

		const int upperLeg= add(prefix + "UpperLeg", hips, mirror(0.08f, 0.85f, 0.f));
		const int lowerLeg= add(prefix + "LowerLeg", upperLeg, mirror(0.08f, 0.45f, 0.f));
		add(prefix + "Foot", lowerLeg, mirror(0.08f, 0.05f, 0.f));
	}
	// The bone pointers alias the names; the vector is done growing
	for (RigNode& node : nodes)
		node.bone= node.name.c_str();
	return nodes;
}

// VRM 0.x files face -Z with the model's left at -X: the same rig turned half
// a turn about Y, so both generations describe one avatar
inline glm::vec3 facingAdjusted(const glm::vec3& v, eVrmVersion version)
{
	return version == eVrmVersion::Vrm0 ? glm::vec3(-v.x, v.y, -v.z) : v;
}

// The 1.0 bone spelling for a 0.x name (only the thumb differs)
inline std::string vrm1BoneName(const std::string& vrm0Name)
{
	if (vrm0Name == "leftThumbProximal") return "leftThumbMetacarpal";
	if (vrm0Name == "leftThumbIntermediate") return "leftThumbProximal";
	if (vrm0Name == "rightThumbProximal") return "rightThumbMetacarpal";
	if (vrm0Name == "rightThumbIntermediate") return "rightThumbProximal";
	return vrm0Name;
}

// Resolves the rig into per-node world rotations and the compensated local
// translations a glTF node carries: a child under a rotated node keeps its
// authored WORLD position
struct ResolvedNode
{
	glm::vec3 world{0.f};
	glm::quat worldRotation{1.f, 0.f, 0.f, 0.f};
	glm::vec3 localTranslation{0.f};
};

inline std::vector<ResolvedNode> resolveRig(const std::vector<RigNode>& rig, eVrmVersion version)
{
	std::vector<ResolvedNode> resolved(rig.size());
	for (size_t index= 0; index < rig.size(); ++index)
	{
		const RigNode& node= rig[index];
		ResolvedNode& out= resolved[index];
		out.world= facingAdjusted(node.world, version);
		const glm::quat parentRotation=
			node.parent >= 0 ? resolved[node.parent].worldRotation : glm::quat(1.f, 0.f, 0.f, 0.f);
		const glm::vec3 parentWorld= node.parent >= 0 ? resolved[node.parent].world : glm::vec3(0.f);
		out.worldRotation= parentRotation * node.localRotation;
		out.localTranslation= glm::inverse(parentRotation) * (out.world - parentWorld);
	}
	return resolved;
}

// A loaded-looking AvatarModel straight from the rig: nodes and the humanoid
// map only (no mesh), enough for AvatarSkeleton and the retarget
inline AvatarModel makeModel(eVrmVersion version, bool rotateLeftUpperArmNode)
{
	const std::vector<RigNode> rig= makeRig(rotateLeftUpperArmNode);
	const std::vector<ResolvedNode> resolved= resolveRig(rig, version);

	AvatarModel model;
	model.version= version;
	model.specVersion= version == eVrmVersion::Vrm1 ? "1.0" : "0.0";
	model.meta.name= "Synthetic";
	model.nodes.resize(rig.size());
	for (size_t index= 0; index < rig.size(); ++index)
	{
		AvatarNode& node= model.nodes[index];
		node.name= rig[index].name;
		node.parent= rig[index].parent;
		glm::mat4 local= glm::mat4_cast(rig[index].localRotation);
		local[3]= glm::vec4(resolved[index].localTranslation, 1.f);
		node.localTransform= local;
		if (node.parent >= 0)
			model.nodes[node.parent].children.push_back((int)index);
		else
			model.rootNodes.push_back((int)index);

		// Humanoid map by the 0.x spelling, which is the enum's name in lowerCamel
		for (int boneIndex= 0; boneIndex < HUMANOID_BONE_COUNT; ++boneIndex)
		{
			std::string unityName= humanoidBoneName((eHumanoidBone)boneIndex);
			unityName[0]= (char)std::tolower((unsigned char)unityName[0]);
			if (unityName == rig[index].name)
				model.humanoidNodes[boneIndex]= (int)index;
		}
	}
	model.fileHumanoidNodes= model.humanoidNodes;
	return model;
}
} // namespace SyntheticAvatar
