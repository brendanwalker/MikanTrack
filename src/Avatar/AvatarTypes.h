#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/ext/vector_uint4.hpp"

// The VRM humanoid skeleton and the CPU-side model a VRM file loads into.
// VRM is glTF 2.0 plus a humanoid extension that names which node plays which
// bone, so a loaded avatar is a plain glTF scene (nodes, skins, meshes,
// materials, images) plus the bone-to-node map and the metadata. Both VRM
// generations load into this one representation: version-specific differences
// are resolved in the loader and the skeleton, never downstream.

// The 55 VRM humanoid bones, in Unity HumanBodyBones vocabulary. Unity
// spelling is the one every consumer matches on (VMC receivers, and the
// VRM 0.x extension itself), so the enum names and humanoidBoneName() use it.
// VRM 1.0 renamed the thumb bones (metacarpal/proximal/distal instead of
// proximal/intermediate/distal); the loader maps them back onto this list.
// The app's thumb phalanx [0] is the metacarpal (CMC -> MCP), so app thumb
// proximal = VRM 1.0 metacarpal = VRM 0.x proximal, and the three thumb slots
// line up with the three app phalanges just like the other fingers.
enum class eHumanoidBone : int
{
	Hips= 0,
	Spine,
	Chest,
	UpperChest,
	Neck,
	Head,
	LeftEye,
	RightEye,
	Jaw,

	LeftShoulder,
	LeftUpperArm,
	LeftLowerArm,
	LeftHand,
	RightShoulder,
	RightUpperArm,
	RightLowerArm,
	RightHand,

	LeftUpperLeg,
	LeftLowerLeg,
	LeftFoot,
	LeftToes,
	RightUpperLeg,
	RightLowerLeg,
	RightFoot,
	RightToes,

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
constexpr int HUMANOID_BONE_COUNT= (int)eHumanoidBone::Count;

// Unity HumanBodyBones spelling ("LeftUpperArm"); "" for an out-of-range value
const char* humanoidBoneName(eHumanoidBone bone);

// The parent each bone hangs off in the VRM humanoid hierarchy (Hips -> none).
// This is the SPEC parent, which may be an optional bone the avatar lacks
// (UpperChest, the shoulders, the toes, the eyes, the jaw, and all fingers
// are optional). AvatarSkeleton resolves the nearest PRESENT ancestor.
eHumanoidBone humanoidBoneParent(eHumanoidBone bone);
constexpr eHumanoidBone HUMANOID_BONE_NONE= eHumanoidBone::Count;

// Bones every VRM must name. Load fails without them.
bool isHumanoidBoneRequired(eHumanoidBone bone);

// First finger bone slot of one hand (left = 0, right = 1), so a (finger,
// phalanx) pair indexes as firstFingerBone(side) + finger * 3 + phalanx
eHumanoidBone firstHumanoidFingerBone(int sideIndex);

// The two extension schemas.
enum class eVrmVersion
{
	Vrm0, // extension "VRM", specVersion "0.0", model faces -Z
	Vrm1, // extension "VRMC_vrm", specVersion "1.0", model faces +Z
};

// -- The loaded model ----------------------------------------------------------

struct AvatarNode
{
	std::string name;
	int parent= -1;
	std::vector<int> children;
	// Parent-relative, as authored (TRS composed, or the matrix as given)
	glm::mat4 localTransform{1.f};
	int mesh= -1;
	int skin= -1;
};

struct AvatarSkin
{
	std::string name;
	std::vector<int> joints; // node indices
	std::vector<glm::mat4> inverseBindMatrices; // one per joint, identity when absent
};

struct AvatarPrimitive
{
	std::vector<glm::vec3> positions;
	std::vector<glm::vec3> normals;  // empty when the file has none
	std::vector<glm::vec2> uvs;      // TEXCOORD_0, empty when absent
	std::vector<glm::uvec4> joints;  // JOINTS_0, empty on an unskinned primitive
	std::vector<glm::vec4> weights;  // WEIGHTS_0, renormalized to sum 1
	std::vector<uint32_t> indices;   // triangle list, generated when the file has none
	int material= -1;
};

struct AvatarMesh
{
	std::string name;
	std::vector<AvatarPrimitive> primitives;
};

enum class eAlphaMode
{
	Opaque,
	Mask,
	Blend,
};

// glTF base material plus the MToon subset the renderer implements. The MToon
// fields come from VRMC_materials_mtoon on a 1.0 file and from the VRM 0.x
// materialProperties bag (mapped the way UniVRM and three-vrm migrate them),
// and carry the spec defaults when the material is not MToon at all.
struct AvatarMaterial
{
	std::string name;
	glm::vec4 baseColorFactor{1.f};
	int baseColorTexture= -1; // texture index
	eAlphaMode alphaMode= eAlphaMode::Opaque;
	float alphaCutoff= 0.5f;
	bool doubleSided= false;

	bool isMToon= false;
	glm::vec3 shadeColorFactor{0.f};
	int shadeMultiplyTexture= -1;
	float shadingShiftFactor= 0.f;
	float shadingToonyFactor= 0.9f;
	// Draw order inside the BLEND group: -9..0 without z-write, 0..9 with
	int renderQueueOffset= 0;
	bool transparentWithZWrite= false;
};

struct AvatarTexture
{
	int image= -1;
	bool repeatS= true;
	bool repeatT= true;
	bool nearestFilter= false;
};

// Decoded RGBA8, rows top-down (glTF's UV origin is the top-left texel, which
// is also the first row GL receives, so no flip anywhere)
struct AvatarImage
{
	std::string name;
	int width= 0;
	int height= 0;
	std::vector<uint8_t> rgba;
};

struct AvatarMeta
{
	std::string name;    // 1.0 "name", 0.x "title"
	std::string version;
	std::string author;  // 1.0 authors joined with ", "
	std::string license; // 1.0 licenseUrl, 0.x licenseName
};

struct AvatarModel
{
	eVrmVersion version= eVrmVersion::Vrm0;
	std::string specVersion;
	std::string sourcePath; // UTF-8, empty for a memory load
	AvatarMeta meta;

	std::vector<AvatarNode> nodes;
	std::vector<int> rootNodes; // nodes with no parent, in file order
	std::vector<AvatarSkin> skins;
	std::vector<AvatarMesh> meshes;
	std::vector<AvatarMaterial> materials;
	std::vector<AvatarTexture> textures;
	std::vector<AvatarImage> images;

	// Node index per humanoid bone, -1 when the avatar lacks that bone
	std::array<int, HUMANOID_BONE_COUNT> humanoidNodes{};

	AvatarModel() { humanoidNodes.fill(-1); }

	int boneNode(eHumanoidBone bone) const { return humanoidNodes[(int)bone]; }
	bool hasBone(eHumanoidBone bone) const { return humanoidNodes[(int)bone] >= 0; }

	size_t triangleCount() const;
};
