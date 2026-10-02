#include "TestCommon.h"

#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtx/quaternion.hpp"

#include "AvatarSkeleton.h"
#include "SyntheticAvatar.h"
#include "VrmLoader.h"

// The VRM loader and the rest skeleton, on files built in memory: a minimal
// humanoid in both VRM generations with one skinned triangle and one texture,
// varied across the accessor and node encodings a real exporter may use. No
// file on disk is read, so this runs in the self-test batch; the shipped VRoid
// samples are exercised by --test-vrm-samples instead.

namespace
{
using json= nlohmann::json;

constexpr int kFloat= 5126;
constexpr int kUnsignedByte= 5121;
constexpr int kUnsignedShort= 5123;
constexpr int kUnsignedInt= 5125;

// -- GLB assembly -------------------------------------------------------------

class GlbBuilder
{
public:
	json root;
	std::vector<uint8_t> bin;

	GlbBuilder()
	{
		root["asset"]= {{"version", "2.0"}, {"generator", "MikanTrack test"}};
		root["bufferViews"]= json::array();
		root["accessors"]= json::array();
	}

	void align()
	{
		while (bin.size() % 4 != 0)
			bin.push_back(0);
	}

	int addBufferView(const void* data, size_t byteCount, size_t byteStride= 0)
	{
		align();
		const size_t offset= bin.size();
		bin.insert(bin.end(), (const uint8_t*)data, (const uint8_t*)data + byteCount);
		json view= {{"buffer", 0}, {"byteOffset", offset}, {"byteLength", byteCount}};
		if (byteStride != 0)
			view["byteStride"]= byteStride;
		root["bufferViews"].push_back(view);
		return (int)root["bufferViews"].size() - 1;
	}

	int addAccessor(int bufferView, int componentType, const char* type, size_t count, bool normalized= false,
					size_t byteOffset= 0)
	{
		json accessor= {{"bufferView", bufferView}, {"componentType", componentType}, {"type", type}, {"count", count}};
		if (normalized)
			accessor["normalized"]= true;
		if (byteOffset != 0)
			accessor["byteOffset"]= byteOffset;
		root["accessors"].push_back(accessor);
		return (int)root["accessors"].size() - 1;
	}

	std::vector<uint8_t> finish()
	{
		align();
		root["buffers"]= json::array({{{"byteLength", bin.size()}}});

		std::string jsonText= root.dump();
		while (jsonText.size() % 4 != 0)
			jsonText.push_back(' ');

		std::vector<uint8_t> out;
		auto putU32= [&](uint32_t value) {
			for (int byte= 0; byte < 4; ++byte)
				out.push_back((uint8_t)(value >> (8 * byte)));
		};
		const uint32_t total= (uint32_t)(12 + 8 + jsonText.size() + 8 + bin.size());
		putU32(0x46546C67u); // "glTF"
		putU32(2);
		putU32(total);
		putU32((uint32_t)jsonText.size());
		putU32(0x4E4F534Au); // "JSON"
		out.insert(out.end(), jsonText.begin(), jsonText.end());
		putU32((uint32_t)bin.size());
		putU32(0x004E4942u); // "BIN\0"
		out.insert(out.end(), bin.begin(), bin.end());
		return out;
	}
};

// -- The synthetic rig --------------------------------------------------------

enum class eExtensions
{
	Vrm0,
	Vrm1,
	Both,
	None,
};

struct RigOptions
{
	eExtensions extensions= eExtensions::Vrm1;
	bool rotateUpperArmNode= false; // 30 degrees about Y on LeftUpperArm, subtree compensated
	bool spineAsMatrix= false;      // the spine node authored as "matrix" instead of TRS
	bool ushortJoints= false;       // JOINTS_0 unsigned short instead of byte
	bool floatWeights= false;       // WEIGHTS_0 float instead of normalized ushort
	int indexComponentType= kUnsignedByte;
	bool sparsePosition= false;     // vertex 2 overridden through a sparse accessor
	bool omitHips= false;
	bool addUnknownBone= false;
};

using SyntheticAvatar::RigNode;

eVrmVersion versionOf(eExtensions extensions)
{
	return extensions == eExtensions::Vrm0 ? eVrmVersion::Vrm0 : eVrmVersion::Vrm1;
}

std::vector<RigNode> makeRig(const RigOptions& options)
{
	return SyntheticAvatar::makeRig(options.rotateUpperArmNode);
}

glm::vec3 facingAdjusted(const glm::vec3& v, eExtensions extensions)
{
	return SyntheticAvatar::facingAdjusted(v, versionOf(extensions));
}

std::string vrm1BoneName(const std::string& vrm0Name)
{
	return SyntheticAvatar::vrm1BoneName(vrm0Name);
}

struct BuiltAvatar
{
	std::vector<uint8_t> bytes;
	std::vector<glm::vec3> nodeWorld;     // rest positions in the file's own glTF frame
	std::vector<glm::quat> nodeWorldRot;
	int leftLowerArmNode= -1;
	std::array<glm::vec3, 3> trianglePositions; // in the file's glTF frame
};

BuiltAvatar buildAvatar(const RigOptions& options)
{
	BuiltAvatar result;
	const std::vector<RigNode> rig= makeRig(options);
	GlbBuilder glb;

	// World rotations and the compensated locals: a child under a rotated
	// node keeps its authored WORLD position
	result.nodeWorld.resize(rig.size());
	result.nodeWorldRot.resize(rig.size());
	json nodes= json::array();
	for (size_t index= 0; index < rig.size(); ++index)
	{
		const RigNode& node= rig[index];
		const glm::vec3 world= facingAdjusted(node.world, options.extensions);
		const glm::quat parentRot= node.parent >= 0 ? result.nodeWorldRot[node.parent] : glm::quat(1.f, 0.f, 0.f, 0.f);
		const glm::vec3 parentWorld= node.parent >= 0 ? result.nodeWorld[node.parent] : glm::vec3(0.f);
		result.nodeWorld[index]= world;
		result.nodeWorldRot[index]= parentRot * node.localRotation;
		const glm::vec3 local= glm::inverse(parentRot) * (world - parentWorld);

		json nodeJson= {{"name", node.name}};
		if (options.spineAsMatrix && node.name == "spine")
		{
			glm::mat4 matrix= glm::mat4_cast(node.localRotation);
			matrix[3]= glm::vec4(local, 1.f);
			json values= json::array();
			for (int column= 0; column < 4; ++column)
				for (int row= 0; row < 4; ++row)
					values.push_back(matrix[column][row]);
			nodeJson["matrix"]= values;
		}
		else
		{
			nodeJson["translation"]= {local.x, local.y, local.z};
			const glm::quat& q= node.localRotation;
			if (q.w != 1.f)
				nodeJson["rotation"]= {q.x, q.y, q.z, q.w};
		}
		json children= json::array();
		for (size_t child= 0; child < rig.size(); ++child)
			if (rig[child].parent == (int)index)
				children.push_back(child);
		if (!children.empty())
			nodeJson["children"]= children;
		nodes.push_back(nodeJson);

		if (node.name == "leftLowerArm")
			result.leftLowerArmNode= (int)index;
	}

	// One triangle beside the left forearm, fully weighted to it
	const glm::vec3 triangle[3]= {
		facingAdjusted({0.40f, 1.30f, 0.02f}, options.extensions),
		facingAdjusted({0.50f, 1.30f, 0.02f}, options.extensions),
		facingAdjusted({0.50f, 1.35f, 0.02f}, options.extensions),
	};
	for (int vertex= 0; vertex < 3; ++vertex)
		result.trianglePositions[vertex]= triangle[vertex];

	// POSITION with a 16-byte stride (padded vec3)
	float positions[12]= {};
	for (int vertex= 0; vertex < 3; ++vertex)
	{
		positions[vertex * 4 + 0]= triangle[vertex].x;
		positions[vertex * 4 + 1]= triangle[vertex].y;
		positions[vertex * 4 + 2]= triangle[vertex].z;
	}
	const int positionView= glb.addBufferView(positions, sizeof(positions), 16);
	const int positionAccessor= glb.addAccessor(positionView, kFloat, "VEC3", 3);
	if (options.sparsePosition)
	{
		// Vertex 2 lifted by 5 cm through a sparse override
		const glm::vec3 lifted= triangle[2] + glm::vec3(0.f, 0.05f, 0.f);
		result.trianglePositions[2]= lifted;
		const uint16_t sparseIndex= 2;
		const float sparseValue[3]= {lifted.x, lifted.y, lifted.z};
		const int indicesView= glb.addBufferView(&sparseIndex, sizeof(sparseIndex));
		const int valuesView= glb.addBufferView(sparseValue, sizeof(sparseValue));
		glb.root["accessors"][positionAccessor]["sparse"]= {
			{"count", 1},
			{"indices", {{"bufferView", indicesView}, {"componentType", kUnsignedShort}}},
			{"values", {{"bufferView", valuesView}}},
		};
	}

	const float uvs[6]= {0.f, 0.f, 1.f, 0.f, 1.f, 1.f};
	const int uvAccessor= glb.addAccessor(glb.addBufferView(uvs, sizeof(uvs)), kFloat, "VEC2", 3);

	const int joint= result.leftLowerArmNode;
	int jointsAccessor;
	if (options.ushortJoints)
	{
		const uint16_t joints[12]= {(uint16_t)joint, 0, 0, 0, (uint16_t)joint, 0, 0, 0, (uint16_t)joint, (uint16_t)joint, 0, 0};
		jointsAccessor= glb.addAccessor(glb.addBufferView(joints, sizeof(joints)), kUnsignedShort, "VEC4", 3);
	}
	else
	{
		const uint8_t joints[12]= {(uint8_t)joint, 0, 0, 0, (uint8_t)joint, 0, 0, 0, (uint8_t)joint, (uint8_t)joint, 0, 0};
		jointsAccessor= glb.addAccessor(glb.addBufferView(joints, sizeof(joints)), kUnsignedByte, "VEC4", 3);
	}
	int weightsAccessor;
	if (options.floatWeights)
	{
		// Vertex 2 deliberately sums to 0.4: the loader renormalizes
		const float weights[12]= {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.2f, 0.2f, 0.f, 0.f};
		weightsAccessor= glb.addAccessor(glb.addBufferView(weights, sizeof(weights)), kFloat, "VEC4", 3);
	}
	else
	{
		const uint16_t weights[12]= {65535, 0, 0, 0, 65535, 0, 0, 0, 32767, 32768, 0, 0};
		weightsAccessor=
			glb.addAccessor(glb.addBufferView(weights, sizeof(weights)), kUnsignedShort, "VEC4", 3, true);
	}

	int indicesAccessor;
	if (options.indexComponentType == kUnsignedByte)
	{
		const uint8_t indices[3]= {0, 1, 2};
		indicesAccessor= glb.addAccessor(glb.addBufferView(indices, sizeof(indices)), kUnsignedByte, "SCALAR", 3);
	}
	else if (options.indexComponentType == kUnsignedShort)
	{
		const uint16_t indices[3]= {0, 1, 2};
		indicesAccessor= glb.addAccessor(glb.addBufferView(indices, sizeof(indices)), kUnsignedShort, "SCALAR", 3);
	}
	else
	{
		const uint32_t indices[3]= {0, 1, 2};
		indicesAccessor= glb.addAccessor(glb.addBufferView(indices, sizeof(indices)), kUnsignedInt, "SCALAR", 3);
	}

	// Skin over every node, inverse bind = inverse of the rest global
	std::vector<float> inverseBinds;
	json skinJoints= json::array();
	for (size_t index= 0; index < rig.size(); ++index)
	{
		glm::mat4 global= glm::mat4_cast(result.nodeWorldRot[index]);
		global[3]= glm::vec4(result.nodeWorld[index], 1.f);
		const glm::mat4 inverseBind= glm::inverse(global);
		for (int column= 0; column < 4; ++column)
			for (int row= 0; row < 4; ++row)
				inverseBinds.push_back(inverseBind[column][row]);
		skinJoints.push_back(index);
	}
	const int inverseBindAccessor= glb.addAccessor(
		glb.addBufferView(inverseBinds.data(), inverseBinds.size() * sizeof(float)), kFloat, "MAT4", rig.size());
	glb.root["skins"]= json::array({{{"joints", skinJoints}, {"inverseBindMatrices", inverseBindAccessor}}});

	// A 2x2 RGBA PNG: red, half-transparent green / transparent blue, white
	cv::Mat pixels(2, 2, CV_8UC4);
	pixels.at<cv::Vec4b>(0, 0)= cv::Vec4b(0, 0, 255, 255);
	pixels.at<cv::Vec4b>(0, 1)= cv::Vec4b(0, 255, 0, 128);
	pixels.at<cv::Vec4b>(1, 0)= cv::Vec4b(255, 0, 0, 0);
	pixels.at<cv::Vec4b>(1, 1)= cv::Vec4b(255, 255, 255, 255);
	std::vector<uint8_t> png;
	cv::imencode(".png", pixels, png);
	const int imageView= glb.addBufferView(png.data(), png.size());
	glb.root["images"]= json::array({{{"name", "swatch"}, {"mimeType", "image/png"}, {"bufferView", imageView}}});
	glb.root["samplers"]= json::array({{{"wrapS", 33071}, {"wrapT", 10497}}});
	glb.root["textures"]= json::array({{{"source", 0}, {"sampler", 0}}});

	glb.root["materials"]= json::array({
		{{"name", "skin"},
		 {"pbrMetallicRoughness", {{"baseColorFactor", {0.5f, 0.6f, 0.7f, 1.f}}, {"baseColorTexture", {{"index", 0}}}}},
		 {"alphaMode", "MASK"},
		 {"doubleSided", true}},
		{{"name", "glass"}, {"alphaMode", "BLEND"}},
		{{"name", "glass2"}, {"alphaMode", "BLEND"}},
	});
	if (options.extensions != eExtensions::Vrm0)
	{
		glb.root["materials"][1]["extensions"]["VRMC_materials_mtoon"]= {
			{"specVersion", "1.0"},
			{"shadeColorFactor", {0.2f, 0.3f, 0.4f}},
			{"shadingToonyFactor", 0.7f},
			{"shadingShiftFactor", 0.1f},
			{"renderQueueOffsetNumber", -3},
		};
		glb.root["materials"][2]["extensions"]["VRMC_materials_mtoon"]= {
			{"specVersion", "1.0"}, {"renderQueueOffsetNumber", -1}};
	}

	glb.root["meshes"]= json::array({{
		{"name", "tri"},
		{"primitives", json::array({{
			{"attributes",
			 {{"POSITION", positionAccessor},
			  {"TEXCOORD_0", uvAccessor},
			  {"JOINTS_0", jointsAccessor},
			  {"WEIGHTS_0", weightsAccessor}}},
			{"indices", indicesAccessor},
			{"material", 0},
		}})},
	}});
	// The mesh hangs off a node whose own transform must NOT move skinned
	// vertices
	nodes.push_back({{"name", "meshNode"}, {"mesh", 0}, {"skin", 0}, {"translation", {5.f, 5.f, 5.f}}});
	glb.root["nodes"]= nodes;
	glb.root["scenes"]= json::array({{{"nodes", {0, (int)rig.size()}}}});
	glb.root["scene"]= 0;

	// The humanoid map and meta
	auto writeVrm1= [&]() {
		json humanBones= json::object();
		for (size_t index= 0; index < rig.size(); ++index)
		{
			if (rig[index].bone == nullptr)
				continue;
			if (options.omitHips && rig[index].name == "hips")
				continue;
			humanBones[vrm1BoneName(rig[index].name)]= {{"node", index}};
		}
		if (options.addUnknownBone)
			humanBones["leftTail"]= {{"node", 0}};
		glb.root["extensions"]["VRMC_vrm"]= {
			{"specVersion", "1.0"},
			{"meta", {{"name", "Synthetic"}, {"version", "7"}, {"authors", {"MikanTrack", "tests"}},
					  {"licenseUrl", "https://example.org/license"}}},
			{"humanoid", {{"humanBones", humanBones}}},
		};
	};
	auto writeVrm0= [&]() {
		json humanBones= json::array();
		for (size_t index= 0; index < rig.size(); ++index)
		{
			if (rig[index].bone == nullptr)
				continue;
			if (options.omitHips && rig[index].name == "hips")
				continue;
			humanBones.push_back({{"bone", rig[index].name}, {"node", index}});
		}
		if (options.addUnknownBone)
			humanBones.push_back({{"bone", "leftTail"}, {"node", 0}});
		glb.root["extensions"]["VRM"]= {
			{"specVersion", "0.0"},
			{"exporterVersion", "MikanTrack test"},
			{"meta", {{"title", "Synthetic"}, {"version", "7"}, {"author", "MikanTrack"}, {"licenseName", "CC0"}}},
			{"humanoid", {{"humanBones", humanBones}}},
			{"materialProperties",
			 json::array({
				 {{"name", "skin"},
				  {"shader", "VRM/MToon"},
				  {"renderQueue", 2450},
				  {"floatProperties", {{"_ShadeShift", 0.f}, {"_ShadeToony", 0.9f}, {"_CullMode", 0.f}}},
				  {"vectorProperties", {{"_Color", {1.f, 1.f, 1.f, 1.f}}, {"_ShadeColor", {0.5f, 0.5f, 0.5f, 1.f}}}},
				  {"textureProperties", {{"_MainTex", 0}, {"_ShadeTexture", 0}}}},
				 {{"name", "glass"},
				  {"shader", "VRM/MToon"},
				  {"renderQueue", 3000},
				  {"floatProperties", {{"_ZWrite", 0.f}}}},
				 {{"name", "glass2"},
				  {"shader", "VRM/MToon"},
				  {"renderQueue", 2990},
				  {"floatProperties", {{"_ZWrite", 0.f}}}},
			 })},
		};
	};
	switch (options.extensions)
	{
	case eExtensions::Vrm1: writeVrm1(); break;
	case eExtensions::Vrm0: writeVrm0(); break;
	case eExtensions::Both:
		writeVrm1();
		writeVrm0();
		break;
	case eExtensions::None: break;
	}
	if (options.extensions != eExtensions::None)
	{
		glb.root["extensionsUsed"]= json::array();
		for (auto it= glb.root["extensions"].begin(); it != glb.root["extensions"].end(); ++it)
			glb.root["extensionsUsed"].push_back(it.key());
	}

	result.bytes= glb.finish();
	return result;
}

bool nearlyEqual(const glm::vec3& a, const glm::vec3& b, float tolerance)
{
	return glm::length(a - b) <= tolerance;
}

bool nearlyEqual(const glm::vec2& a, const glm::vec2& b, float tolerance)
{
	return glm::length(a - b) <= tolerance;
}

bool nearlyEqual(float a, float b, float tolerance)
{
	return std::fabs(a - b) <= tolerance;
}

int boneNode(const AvatarModel& model, eHumanoidBone bone)
{
	return model.humanoidNodes[(int)bone];
}
} // namespace

static int runVrmTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-vrm") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-vrm") << "FAIL " << name;
			failures++;
		}
	};
	auto load= [](const BuiltAvatar& avatar) { return VrmLoader::loadMemory(avatar.bytes.data(), avatar.bytes.size()); };

	using B= eHumanoidBone;

	// (a) Failure modes: a plain glTF, a truncated file, and a missing
	// required bone are refused with a message, never a crash
	{
		RigOptions plain;
		plain.extensions= eExtensions::None;
		const VrmLoader::LoadResult result= load(buildAvatar(plain));
		check(result.model == nullptr && result.error.find("Not a VRM") != std::string::npos,
			  "(a) a glTF without a VRM extension is refused");

		const BuiltAvatar full= buildAvatar(RigOptions());
		const VrmLoader::LoadResult truncated= VrmLoader::loadMemory(full.bytes.data(), 100);
		check(truncated.model == nullptr && !truncated.error.empty(), "(a) a truncated file is refused");

		RigOptions noHips;
		noHips.omitHips= true;
		const VrmLoader::LoadResult missing= load(buildAvatar(noHips));
		check(missing.model == nullptr && missing.error.find("Hips") != std::string::npos,
			  "(a) a file without hips is refused naming the bone");
	}

	// (b) Version detection and the bone map in both schemas
	RigOptions options1;
	options1.addUnknownBone= true;
	const BuiltAvatar avatar1= buildAvatar(options1);
	const VrmLoader::LoadResult result1= load(avatar1);
	check(result1.model != nullptr, "(b) the VRM 1.0 file loads");
	if (result1.model == nullptr)
		return 1;
	const AvatarModel& model1= *result1.model;
	check(model1.version == eVrmVersion::Vrm1 && model1.specVersion == "1.0", "(b) VRM 1.0 detected");
	check(model1.meta.name == "Synthetic" && model1.meta.author == "MikanTrack, tests" &&
			  model1.meta.license == "https://example.org/license" && model1.meta.version == "7",
		  "(b) VRM 1.0 meta read");
	check(!result1.warnings.empty() && result1.warnings[0].find("leftTail") != std::string::npos,
		  "(b) an unknown bone name warns and is ignored");
	check(boneNode(model1, B::LeftThumbProximal) >= 0 &&
			  model1.nodes[boneNode(model1, B::LeftThumbProximal)].name == "leftThumbProximal" &&
			  model1.nodes[boneNode(model1, B::LeftThumbIntermediate)].name == "leftThumbIntermediate" &&
			  model1.nodes[boneNode(model1, B::LeftThumbDistal)].name == "leftThumbDistal",
		  "(b) VRM 1.0 thumb metacarpal/proximal/distal map onto proximal/intermediate/distal");
	check(boneNode(model1, B::UpperChest) < 0 && boneNode(model1, B::Jaw) < 0 && boneNode(model1, B::LeftToes) < 0,
		  "(b) absent optional bones read as missing");

	RigOptions options0;
	options0.extensions= eExtensions::Vrm0;
	options0.indexComponentType= kUnsignedInt;
	const BuiltAvatar avatar0= buildAvatar(options0);
	const VrmLoader::LoadResult result0= load(avatar0);
	check(result0.model != nullptr, "(b) the VRM 0.x file loads");
	if (result0.model == nullptr)
		return 1;
	const AvatarModel& model0= *result0.model;
	check(model0.version == eVrmVersion::Vrm0 && model0.specVersion == "0.0", "(b) VRM 0.x detected");
	check(model0.meta.name == "Synthetic" && model0.meta.author == "MikanTrack" && model0.meta.license == "CC0",
		  "(b) VRM 0.x meta read");
	int presentCount0= 0;
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		presentCount0+= model0.humanoidNodes[index] >= 0 ? 1 : 0;
	check(presentCount0 == 49 && model0.nodes[boneNode(model0, B::RightLittleDistal)].name == "rightLittleDistal",
		  "(b) VRM 0.x bone names map one to one");

	RigOptions optionsBoth;
	optionsBoth.extensions= eExtensions::Both;
	const VrmLoader::LoadResult resultBoth= load(buildAvatar(optionsBoth));
	check(resultBoth.model != nullptr && resultBoth.model->version == eVrmVersion::Vrm1,
		  "(b) VRMC_vrm wins when both extensions are present");

	// (c) Accessors: strided positions, byte joints, normalized short
	// weights, byte indices (model 1); short joints, float weights,
	// short indices, a sparse position override (model 2); uint indices
	// (model 0)
	RigOptions options2;
	options2.rotateUpperArmNode= true;
	options2.spineAsMatrix= true;
	options2.ushortJoints= true;
	options2.floatWeights= true;
	options2.indexComponentType= kUnsignedShort;
	options2.sparsePosition= true;
	const BuiltAvatar avatar2= buildAvatar(options2);
	const VrmLoader::LoadResult result2= load(avatar2);
	check(result2.model != nullptr, "(c) the encoded-variant file loads");
	if (result2.model == nullptr)
		return 1;
	const AvatarModel& model2= *result2.model;

	const AvatarPrimitive& primitive1= model1.meshes[0].primitives[0];
	const AvatarPrimitive& primitive2= model2.meshes[0].primitives[0];
	const AvatarPrimitive& primitive0= model0.meshes[0].primitives[0];
	check(primitive1.positions.size() == 3 && nearlyEqual(primitive1.positions[1], avatar1.trianglePositions[1], 1e-6f),
		  "(c) strided float positions read");
	check(primitive2.positions.size() == 3 && nearlyEqual(primitive2.positions[2], avatar2.trianglePositions[2], 1e-6f) &&
			  nearlyEqual(primitive2.positions[1], avatar2.trianglePositions[1], 1e-6f),
		  "(c) a sparse position override is applied");
	check(primitive1.indices == std::vector<uint32_t>({0, 1, 2}) && primitive2.indices == std::vector<uint32_t>({0, 1, 2}) &&
			  primitive0.indices == std::vector<uint32_t>({0, 1, 2}),
		  "(c) byte, short and int indices read");
	check(primitive1.joints.size() == 3 && primitive1.joints[0].x == (unsigned)avatar1.leftLowerArmNode &&
			  primitive2.joints[2].y == (unsigned)avatar2.leftLowerArmNode,
		  "(c) byte and short joint indices read");
	check(nearlyEqual(primitive1.weights[0].x, 1.f, 1e-6f) && nearlyEqual(primitive1.weights[2].x, 0.5f, 1e-4f) &&
			  nearlyEqual(primitive1.weights[2].y, 0.5f, 1e-4f),
		  "(c) normalized short weights read as fractions");
	check(nearlyEqual(primitive2.weights[2].x, 0.5f, 1e-6f) && nearlyEqual(primitive2.weights[2].y, 0.5f, 1e-6f),
		  "(c) float weights summing to 0.4 are renormalized to 1");
	check(primitive1.uvs.size() == 3 && nearlyEqual(primitive1.uvs[2], glm::vec2(1.f, 1.f), 1e-6f) &&
			  primitive1.normals.empty(),
		  "(c) uvs read and absent normals stay empty");
	check(primitive1.material == 0 && model1.nodes.back().mesh == 0 && model1.nodes.back().skin == 0,
		  "(c) the mesh node references its mesh, skin and material");

	// (d) Node transforms and rest globals: a matrix node and a rotated node
	// with compensated children land on the same world positions as the
	// plain TRS rig
	const AvatarSkeleton skeleton1(model1);
	const AvatarSkeleton skeleton2(model2);
	const AvatarSkeleton skeleton0(model0);
	{
		bool bSame= true;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		{
			const AvatarSkeleton::Bone& a= skeleton1.getBone((B)index);
			const AvatarSkeleton::Bone& b= skeleton2.getBone((B)index);
			if (a.present != b.present)
				bSame= false;
			else if (a.present && !nearlyEqual(a.restPositionWorld, b.restPositionWorld, 1e-5f))
				bSame= false;
		}
		check(bSame, "(d) matrix and rotated-rest nodes compose to the same rest world positions");

		const int spine= boneNode(model2, B::Spine);
		const int chest= boneNode(model2, B::Chest);
		const glm::vec3 chestGlobal(skeleton2.getRestGlobalsAvatar()[chest][3]);
		const glm::vec3 chestExpected= avatar2.nodeWorld[chest];
		check(nearlyEqual(chestGlobal, chestExpected, 1e-6f) && model2.nodes[spine].parent == boneNode(model2, B::Hips),
			  "(d) grandchild global = parent global * local");
	}

	// (e) Skin data: global * inverseBind is the identity at rest, so the
	// triangle maps onto itself; rotating the forearm node moves it exactly as
	// a CPU skinning reference says, and the mesh node's own transform is
	// ignored (its 5,5,5 translation never appears)
	{
		const AvatarSkin& skin= model2.skins[0];
		const int joint= avatar2.leftLowerArmNode;
		const std::vector<glm::mat4>& restGlobals= skeleton2.getRestGlobalsAvatar();
		const glm::mat4 restSkin= restGlobals[skin.joints[joint]] * skin.inverseBindMatrices[joint];
		const glm::vec3 atRest(restSkin * glm::vec4(primitive2.positions[0], 1.f));
		check(nearlyEqual(atRest, primitive2.positions[0], 1e-5f), "(e) rest skinning maps a vertex onto itself");

		std::vector<glm::mat4> locals= skeleton2.getRestLocalsAvatar();
		const glm::quat turn= glm::angleAxis(glm::half_pi<float>(), glm::vec3(0.f, 1.f, 0.f));
		locals[joint]= locals[joint] * glm::mat4_cast(turn);
		std::vector<glm::mat4> globals;
		computeNodeGlobals(model2, locals, globals);
		const glm::mat4 posedSkin= globals[skin.joints[joint]] * skin.inverseBindMatrices[joint];
		const glm::vec3 posed(posedSkin * glm::vec4(primitive2.positions[0], 1.f));

		const glm::vec3 forearmWorld= avatar2.nodeWorld[joint];
		const glm::quat forearmRestRot= avatar2.nodeWorldRot[joint];
		const glm::vec3 expected=
			forearmWorld + (forearmRestRot * turn * glm::inverse(forearmRestRot)) * (primitive2.positions[0] - forearmWorld);
		check(nearlyEqual(posed, expected, 1e-5f), "(e) a rotated forearm skins the vertex like the CPU reference");
	}

	// (f) The avatar-to-world conversion, both generations
	{
		const glm::mat4 w1= makeWorldFromAvatar(eVrmVersion::Vrm1);
		const glm::mat4 w0= makeWorldFromAvatar(eVrmVersion::Vrm0);
		check(nearlyEqual(glm::vec3(w1 * glm::vec4(0.f, 0.f, 1.f, 0.f)), glm::vec3(1.f, 0.f, 0.f), 1e-6f) &&
				  nearlyEqual(glm::vec3(w1 * glm::vec4(1.f, 0.f, 0.f, 0.f)), glm::vec3(0.f, 1.f, 0.f), 1e-6f) &&
				  nearlyEqual(glm::vec3(w1 * glm::vec4(0.f, 1.f, 0.f, 0.f)), glm::vec3(0.f, 0.f, 1.f), 1e-6f),
			  "(f) VRM 1.0: glTF facing +Z, left +X, up +Y land on world +X, +Y, +Z");
		check(nearlyEqual(glm::vec3(w0 * glm::vec4(0.f, 0.f, -1.f, 0.f)), glm::vec3(1.f, 0.f, 0.f), 1e-6f) &&
				  nearlyEqual(glm::vec3(w0 * glm::vec4(-1.f, 0.f, 0.f, 0.f)), glm::vec3(0.f, 1.f, 0.f), 1e-6f),
			  "(f) VRM 0.x: glTF facing -Z and left -X land on world +X and +Y");
		check(nearlyEqual(glm::determinant(glm::mat3(w1)), 1.f, 1e-6f) &&
				  nearlyEqual(glm::determinant(glm::mat3(w0)), 1.f, 1e-6f),
			  "(f) both conversions are proper rotations");
	}

	// (g) The rest skeleton in world space
	{
		const glm::vec3 leftArm=
			skeleton1.getBone(B::LeftLowerArm).restPositionWorld - skeleton1.getBone(B::LeftUpperArm).restPositionWorld;
		const glm::vec3 rightArm=
			skeleton1.getBone(B::RightLowerArm).restPositionWorld - skeleton1.getBone(B::RightUpperArm).restPositionWorld;
		check(nearlyEqual(glm::normalize(leftArm), VmcRetarget::restArmDirection(eHandSide::Left), 1e-5f) &&
				  nearlyEqual(glm::normalize(rightArm), VmcRetarget::restArmDirection(eHandSide::Right), 1e-5f),
			  "(g) the arms rest along the T-pose directions VmcRetarget assumes");
		check(nearlyEqual(skeleton1.getUpperArmLength(eHandSide::Left), 0.25f, 1e-5f) &&
				  nearlyEqual(skeleton1.getForearmLength(eHandSide::Right), 0.22f, 1e-5f) &&
				  nearlyEqual(skeleton1.getShoulderWidth(), 0.30f, 1e-5f) &&
				  nearlyEqual(skeleton1.getHandLength(eHandSide::Left), 0.085f, 1e-5f),
			  "(g) the derived lengths equal the authored ones");
		check(skeleton1.getBone(B::Neck).parent == B::Chest && skeleton1.getBone(B::LeftShoulder).parent == B::Chest &&
				  skeleton1.getBone(B::Hips).parent == HUMANOID_BONE_NONE &&
				  skeleton1.getBone(B::LeftIndexProximal).parent == B::LeftHand,
			  "(g) a missing optional bone is skipped when resolving humanoid parents");
		check(nearlyEqual(skeleton2.getBone(B::LeftLowerArm).restOffsetFromParentWorld, glm::vec3(0.f, 0.25f, 0.f), 1e-5f) &&
				  nearlyEqual(skeleton2.getBone(B::LeftHand).restOffsetFromParentWorld, glm::vec3(0.f, 0.22f, 0.f), 1e-5f),
			  "(g) parent offsets are world rest deltas even under a rotated rest node");
		check(nearlyEqual(skeleton1.getHeightAboveHips(), 0.55f, 1e-5f), "(g) height above the hips reaches the head");

		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const eHandSide side= (eHandSide)sideIndex;
			const AvatarSkeleton::HandRest& hand= skeleton1.getHand(side);
			const glm::mat3 restPalm= VmcRetarget::restPalmFrame(side);
			const bool bFrameMatches= hand.valid &&
				nearlyEqual(glm::vec3(hand.palmFrameWorld[0]), restPalm[0], 1e-4f) &&
				nearlyEqual(glm::vec3(hand.palmFrameWorld[1]), restPalm[1], 1e-4f) &&
				nearlyEqual(glm::vec3(hand.palmFrameWorld[2]), restPalm[2], 1e-4f);
			check(bFrameMatches, sideIndex == 0 ? "(g) left palm frame matches the VMC rest palm frame"
												: "(g) right palm frame matches the VMC rest palm frame");
			const float indexY= hand.skeleton.baseInPalm[(int)eFinger::Index].y;
			check(sideIndex == 0 ? indexY > 0.01f : indexY < -0.01f,
				  sideIndex == 0 ? "(g) left index base sits at positive palm Y (thumb side)"
								 : "(g) right index base sits at negative palm Y (thumb side)");
			check(nearlyEqual(hand.skeleton.baseInPalm[(int)eFinger::Middle].x, 0.0425f, 1e-5f),
				  "(g) the middle base sits half a hand length from the palm center");
			check(nearlyEqual(hand.skeleton.phalanxLengths[(int)eFinger::Index][0], 0.03f, 1e-5f) &&
					  nearlyEqual(hand.skeleton.phalanxLengths[(int)eFinger::Index][1], 0.02f, 1e-5f) &&
					  nearlyEqual(hand.skeleton.phalanxLengths[(int)eFinger::Index][2], 0.016f, 1e-5f),
				  "(g) phalanx lengths follow the authored joints, tip extrapolated");
			check(nearlyEqual(hand.skeleton.neutralDirInPalm[(int)eFinger::Index], glm::vec3(1.f, 0.f, 0.f), 1e-4f) &&
					  hand.skeleton.neutralDirInPalm[(int)eFinger::Thumb].z > 0.2f,
				  "(g) neutral directions are the avatar's rest fingers, thumb palmar");
		}
	}

	// (h) Images and materials
	{
		check(model1.images.size() == 1 && model1.images[0].width == 2 && model1.images[0].height == 2 &&
				  model1.images[0].rgba[0] == 255 && model1.images[0].rgba[3] == 255 && model1.images[0].rgba[5] == 255 &&
				  model1.images[0].rgba[7] == 128 && model1.images[0].rgba[11] == 0 && model1.images[0].rgba[8] == 0 &&
				  model1.images[0].rgba[10] == 255,
			  "(h) the PNG decodes to top-down RGBA with its alpha intact");
		check(model1.textures.size() == 1 && model1.textures[0].image == 0 && !model1.textures[0].repeatS &&
				  model1.textures[0].repeatT,
			  "(h) texture sampler wrap modes read");

		const AvatarMaterial& skin1= model1.materials[0];
		check(skin1.alphaMode == eAlphaMode::Mask && nearlyEqual(skin1.alphaCutoff, 0.5f, 1e-6f) && skin1.doubleSided &&
				  skin1.baseColorTexture == 0 && nearlyEqual(glm::vec3(skin1.baseColorFactor), glm::vec3(0.5f, 0.6f, 0.7f), 1e-6f) &&
				  !skin1.isMToon,
			  "(h) glTF material fields and the default alpha cutoff");
		const AvatarMaterial& glass1= model1.materials[1];
		check(glass1.isMToon && nearlyEqual(glass1.shadeColorFactor, glm::vec3(0.2f, 0.3f, 0.4f), 1e-6f) &&
				  nearlyEqual(glass1.shadingToonyFactor, 0.7f, 1e-6f) && nearlyEqual(glass1.shadingShiftFactor, 0.1f, 1e-6f) &&
				  glass1.renderQueueOffset == -3 && !glass1.transparentWithZWrite &&
				  model1.materials[2].renderQueueOffset == -1,
			  "(h) VRMC_materials_mtoon fields read");

		const AvatarMaterial& skin0= model0.materials[0];
		const float linearHalf= std::pow((0.5f + 0.055f) / 1.055f, 2.4f);
		check(skin0.isMToon && nearlyEqual(skin0.shadeColorFactor, glm::vec3(linearHalf), 1e-4f) &&
				  nearlyEqual(skin0.shadingToonyFactor, 0.95f, 1e-5f) && nearlyEqual(skin0.shadingShiftFactor, -0.05f, 1e-5f) &&
				  skin0.doubleSided && skin0.shadeMultiplyTexture == 0 && skin0.baseColorTexture == 0,
			  "(h) VRM 0.x MToon properties migrate to the 1.0 fields");
		check(model0.materials[1].renderQueueOffset == 0 && model0.materials[2].renderQueueOffset == -1 &&
				  !model0.materials[1].transparentWithZWrite,
			  "(h) VRM 0.x transparent render queues become ordered offsets ending at 0");
	}

	// (i) Both generations describe the same avatar in world space
	{
		bool bSame= true;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		{
			const AvatarSkeleton::Bone& a= skeleton1.getBone((B)index);
			const AvatarSkeleton::Bone& b= skeleton0.getBone((B)index);
			if (a.present != b.present || (a.present && !nearlyEqual(a.restPositionWorld, b.restPositionWorld, 1e-5f)))
				bSame= false;
		}
		const AvatarSkeleton::HandRest& left1= skeleton1.getHand(eHandSide::Left);
		const AvatarSkeleton::HandRest& left0= skeleton0.getHand(eHandSide::Left);
		check(bSame && nearlyEqual(glm::vec3(left1.palmFrameWorld[2]), glm::vec3(left0.palmFrameWorld[2]), 1e-4f),
			  "(i) VRM 0.x and 1.0 rigs produce the same world rest skeleton");
	}

	MIKAN_LOG_INFO("test-vrm") << (failures == 0 ? "ALL PASSED" : "FAILURES") << " (" << failures << " failed)";
	return failures == 0 ? 0 : 1;
}
MIKAN_REGISTER_TEST("--test-vrm", "VRM loader and rest skeleton on synthetic in-memory files",
					eTestCategory::SelfTest, runVrmTest);

// The shipped VRoid samples (fetched by InitialSetup_x64.bat): every one must
// load with its full humanoid map and T-pose rest geometry. Hardware category
// because it reads files; skips with a warning when they are absent.
static int runVrmSamplesTest(const TestArgs&)
{
	const char* samples[]= {"models/avatars/fem_vroid.vrm", "models/avatars/masc_vroid.vrm"};
	int failures= 0;
	int loaded= 0;
	for (const char* sample : samples)
	{
		if (!std::filesystem::exists(sample))
		{
			MIKAN_LOG_WARNING("test-vrm-samples") << sample << " missing - run InitialSetup_x64.bat; SKIPPING";
			continue;
		}
		const VrmLoader::LoadResult result= VrmLoader::loadFile(sample);
		if (result.model == nullptr)
		{
			MIKAN_LOG_ERROR("test-vrm-samples") << "FAIL " << sample << ": " << result.error;
			failures++;
			continue;
		}
		++loaded;
		const AvatarModel& model= *result.model;
		const AvatarSkeleton skeleton(model);

		int presentCount= 0;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
			presentCount+= model.humanoidNodes[index] >= 0 ? 1 : 0;
		const bool bHands= skeleton.getHand(eHandSide::Left).valid && skeleton.getHand(eHandSide::Right).valid;
		const glm::vec3 leftArm= glm::normalize(skeleton.getBone(eHumanoidBone::LeftLowerArm).restPositionWorld -
												skeleton.getBone(eHumanoidBone::LeftUpperArm).restPositionWorld);
		const bool bTPose= glm::dot(leftArm, VmcRetarget::restArmDirection(eHandSide::Left)) > 0.95f;
		const bool bPalmDown= skeleton.getHand(eHandSide::Left).palmFrameWorld[2].z < -0.9f &&
			skeleton.getHand(eHandSide::Right).palmFrameWorld[2].z < -0.9f;
		bool bSkinsBounded= true;
		for (const AvatarSkin& skin : model.skins)
			bSkinsBounded&= skin.joints.size() <= 1024;
		const bool bPass= presentCount >= 50 && bHands && bTPose && bPalmDown && bSkinsBounded &&
			model.triangleCount() > 0 && !model.images.empty();
		MIKAN_LOG_INFO("test-vrm-samples") << (bPass ? "PASS " : "FAIL ") << sample << ": " << presentCount
										   << " bones, hands " << (bHands ? "valid" : "invalid") << ", T-pose "
										   << (bTPose ? "yes" : "no") << ", palms down " << (bPalmDown ? "yes" : "no");
		if (!bPass)
			failures++;
	}
	if (loaded == 0)
		MIKAN_LOG_WARNING("test-vrm-samples") << "no sample loaded";
	return failures == 0 ? 0 : 1;
}
MIKAN_REGISTER_TEST("--test-vrm-samples", "Load the shipped VRoid sample avatars (needs models/avatars)",
					eTestCategory::Hardware, runVrmSamplesTest);
