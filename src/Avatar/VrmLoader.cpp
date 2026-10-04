#include "VrmLoader.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"

#include "nlohmann/json.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Logger.h"
#include "PathUtils.h"

using json= nlohmann::json;

namespace
{
constexpr const char* k_logLabel= "VrmLoader";

struct LoadContext
{
	const cgltf_data* data= nullptr;
	AvatarModel* model= nullptr;
	std::vector<std::string>* warnings= nullptr;

	void warn(const std::string& message)
	{
		MIKAN_LOG_WARNING(k_logLabel) << message;
		warnings->push_back(message);
	}
};

// -- Humanoid bone names ------------------------------------------------------

// Lowercases the first letter: "LeftUpperArm" -> "leftUpperArm", the spelling
// both VRM schemas use for their keys
std::string lowerCamel(const char* unityName)
{
	std::string name= unityName;
	if (!name.empty())
		name[0]= (char)std::tolower((unsigned char)name[0]);
	return name;
}

// VRM 1.0 renamed the thumb bones; everything else matches the 0.x spelling
const char* vrm1ThumbAlias(const std::string& key)
{
	static const std::pair<const char*, const char*> k_aliases[]= {
		{"leftThumbMetacarpal", "leftThumbProximal"},
		{"leftThumbProximal", "leftThumbIntermediate"},
		{"rightThumbMetacarpal", "rightThumbProximal"},
		{"rightThumbProximal", "rightThumbIntermediate"},
	};
	for (const auto& alias : k_aliases)
	{
		if (key == alias.first)
			return alias.second;
	}
	return nullptr;
}

bool lookupBone(const std::string& vrm0Key, eHumanoidBone& outBone)
{
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		if (lowerCamel(humanoidBoneName((eHumanoidBone)index)) == vrm0Key)
		{
			outBone= (eHumanoidBone)index;
			return true;
		}
	}
	return false;
}

void assignBone(LoadContext& context, const std::string& key, eVrmVersion version, int node)
{
	std::string vrm0Key= key;
	if (version == eVrmVersion::Vrm1)
	{
		if (const char* alias= vrm1ThumbAlias(key))
			vrm0Key= alias;
	}

	eHumanoidBone bone;
	if (!lookupBone(vrm0Key, bone))
	{
		context.warn("Unknown humanoid bone '" + key + "' ignored");
		return;
	}
	if (node < 0 || node >= (int)context.model->nodes.size())
	{
		context.warn("Humanoid bone '" + key + "' names node " + std::to_string(node) + ", out of range");
		return;
	}
	if (context.model->humanoidNodes[(int)bone] >= 0)
		context.warn("Humanoid bone '" + key + "' assigned twice, keeping the first");
	else
		context.model->humanoidNodes[(int)bone]= node;
}

// -- VRM extension JSON -------------------------------------------------------

const cgltf_extension* findExtension(const cgltf_extension* extensions, cgltf_size count, const char* name)
{
	for (cgltf_size index= 0; index < count; ++index)
	{
		if (extensions[index].name != nullptr && std::strcmp(extensions[index].name, name) == 0)
			return &extensions[index];
	}
	return nullptr;
}

bool parseExtensionJson(LoadContext& context, const cgltf_extension& extension, json& outJson)
{
	if (extension.data == nullptr)
		return false;
	outJson= json::parse(extension.data, nullptr, false);
	if (outJson.is_discarded() || !outJson.is_object())
	{
		context.warn(std::string("Extension '") + extension.name + "' is not a JSON object");
		return false;
	}
	return true;
}

void readHumanoidAndMeta(LoadContext& context, const json& vrm, eVrmVersion version)
{
	AvatarModel& model= *context.model;
	const json& humanoid= vrm.value("humanoid", json::object());
	const json& bones= humanoid.value("humanBones", json());

	if (version == eVrmVersion::Vrm1)
	{
		// {"hips": {"node": 3}, ...}
		if (bones.is_object())
		{
			for (auto it= bones.begin(); it != bones.end(); ++it)
			{
				if (it.value().is_object())
					assignBone(context, it.key(), version, it.value().value("node", -1));
			}
		}

		const json& meta= vrm.value("meta", json::object());
		model.meta.name= meta.value("name", "");
		model.meta.version= meta.value("version", "");
		if (meta.contains("authors") && meta["authors"].is_array())
		{
			for (const json& author : meta["authors"])
			{
				if (!author.is_string())
					continue;
				if (!model.meta.author.empty())
					model.meta.author+= ", ";
				model.meta.author+= author.get<std::string>();
			}
		}
		model.meta.license= meta.value("licenseUrl", "");
	}
	else
	{
		// [{"bone": "hips", "node": 3}, ...]
		if (bones.is_array())
		{
			for (const json& entry : bones)
			{
				if (entry.is_object() && entry.contains("bone") && entry["bone"].is_string())
					assignBone(context, entry["bone"].get<std::string>(), version, entry.value("node", -1));
			}
		}

		const json& meta= vrm.value("meta", json::object());
		model.meta.name= meta.value("title", "");
		model.meta.version= meta.value("version", "");
		model.meta.author= meta.value("author", "");
		model.meta.license= meta.value("licenseName", "");
	}
}

// -- Expressions --------------------------------------------------------------

// VRM 0.x presetName -> the VRM 1.0 preset vocabulary; "" for unknown or an
// unmapped name
const char* vrm0PresetToPreset(const std::string& presetName)
{
	static const std::pair<const char*, const char*> k_presets[]= {
		{"a", "aa"},
		{"i", "ih"},
		{"u", "ou"},
		{"e", "ee"},
		{"o", "oh"},
		{"blink", "blink"},
		{"blink_l", "blinkLeft"},
		{"blink_r", "blinkRight"},
		{"joy", "happy"},
		{"angry", "angry"},
		{"sorrow", "sad"},
		{"fun", "relaxed"},
		{"lookup", "lookUp"},
		{"lookdown", "lookDown"},
		{"lookleft", "lookLeft"},
		{"lookright", "lookRight"},
		{"neutral", "neutral"},
	};
	for (const auto& preset : k_presets)
	{
		if (presetName == preset.first)
			return preset.second;
	}
	return "";
}

// Adds one bind when its mesh and morph index exist, warns and drops it otherwise
void addMorphBind(LoadContext& context, AvatarExpression& expression, int mesh, int morphIndex, float weight)
{
	const AvatarModel& model= *context.model;
	if (mesh < 0 || mesh >= (int)model.meshes.size())
	{
		context.warn("Expression '" + expression.name + "' binds mesh " + std::to_string(mesh) + ", out of range");
		return;
	}
	if (morphIndex < 0 || morphIndex >= (int)model.meshes[mesh].morphTargetNames.size())
	{
		context.warn("Expression '" + expression.name + "' binds morph target " + std::to_string(morphIndex) +
					 " of mesh " + std::to_string(mesh) + ", out of range");
		return;
	}
	expression.morphBinds.push_back({mesh, morphIndex, weight});
}

void readExpressions1(LoadContext& context, const json& vrm)
{
	const AvatarModel& model= *context.model;
	const json& expressions= vrm.value("expressions", json::object());
	if (!expressions.is_object())
		return;

	for (const char* group : {"preset", "custom"})
	{
		const bool bPreset= std::strcmp(group, "preset") == 0;
		const json& entries= expressions.value(group, json::object());
		if (!entries.is_object())
			continue;

		for (auto it= entries.begin(); it != entries.end(); ++it)
		{
			if (!it.value().is_object())
				continue;
			AvatarExpression expression;
			expression.name= it.key();
			expression.preset= bPreset ? it.key() : "";
			expression.isBinary= it.value().value("isBinary", false);

			const json& binds= it.value().value("morphTargetBinds", json::array());
			if (binds.is_array())
			{
				for (const json& bind : binds)
				{
					if (!bind.is_object())
						continue;
					// 1.0 binds name a node, which resolves to the mesh it carries
					const int node= bind.value("node", -1);
					if (node < 0 || node >= (int)model.nodes.size() || model.nodes[node].mesh < 0)
					{
						context.warn("Expression '" + expression.name + "' binds node " + std::to_string(node) +
									 ", which is out of range or has no mesh");
						continue;
					}
					addMorphBind(context, expression, model.nodes[node].mesh, bind.value("index", -1),
								 bind.value("weight", 1.f));
				}
			}
			context.model->expressions.push_back(std::move(expression));
		}
	}
}

void readExpressions0(LoadContext& context, const json& vrm)
{
	const json& master= vrm.value("blendShapeMaster", json::object());
	if (!master.is_object())
		return;
	const json& groups= master.value("blendShapeGroups", json::array());
	if (!groups.is_array())
		return;

	for (const json& group : groups)
	{
		if (!group.is_object())
			continue;
		AvatarExpression expression;
		expression.name= group.value("name", "");
		expression.preset= vrm0PresetToPreset(group.value("presetName", ""));
		expression.isBinary= group.value("isBinary", false);

		const json& binds= group.value("binds", json::array());
		if (binds.is_array())
		{
			for (const json& bind : binds)
			{
				if (!bind.is_object())
					continue;
				// The file weight runs 0..100
				addMorphBind(context, expression, bind.value("mesh", -1), bind.value("index", -1),
							 bind.value("weight", 100.f) / 100.f);
			}
		}
		context.model->expressions.push_back(std::move(expression));
	}
}

void readExpressions(LoadContext& context, const json& vrm, eVrmVersion version)
{
	if (version == eVrmVersion::Vrm1)
		readExpressions1(context, vrm);
	else
		readExpressions0(context, vrm);
}

// -- Materials ----------------------------------------------------------------

// sRGB -> linear, the way three-vrm migrates VRM 0.x colors, which are stored
// in gamma space while the 1.0 factors are linear
float gammaToLinear(float value)
{
	return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

int textureIndexFromJson(const json& textureInfo)
{
	return textureInfo.is_object() ? textureInfo.value("index", -1) : -1;
}

void readMToon1(AvatarMaterial& material, const json& mtoon)
{
	material.isMToon= true;
	if (mtoon.contains("shadeColorFactor") && mtoon["shadeColorFactor"].is_array() &&
		mtoon["shadeColorFactor"].size() >= 3)
	{
		const json& c= mtoon["shadeColorFactor"];
		material.shadeColorFactor= glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
	}
	material.shadeMultiplyTexture= textureIndexFromJson(mtoon.value("shadeMultiplyTexture", json()));
	material.shadingShiftFactor= mtoon.value("shadingShiftFactor", 0.f);
	material.shadingToonyFactor= mtoon.value("shadingToonyFactor", 0.9f);
	material.renderQueueOffset= std::clamp(mtoon.value("renderQueueOffsetNumber", 0), -9, 9);
	material.transparentWithZWrite= mtoon.value("transparentWithZWrite", false);
}

// VRM 0.x stores MToon as a Unity property bag per material. The conversion
// into the 1.0 fields follows UniVRM's and three-vrm's migration.
void readMToon0(LoadContext& context, const json& vrm)
{
	const json& properties= vrm.value("materialProperties", json::array());
	if (!properties.is_array())
		return;

	AvatarModel& model= *context.model;

	// Unity render queues of the transparent MToon materials, so each one's
	// position in the sorted set becomes the 1.0 offset: zwrite-off queues
	// map onto -9..0 ending at 0, zwrite-on queues onto 0..9 starting at 0
	std::set<int> transparentQueues;
	std::set<int> transparentZWriteQueues;

	struct Parsed
	{
		int materialIndex= -1;
		int renderQueue= 0;
	};
	std::vector<Parsed> parsed;

	for (size_t propertyIndex= 0; propertyIndex < properties.size(); ++propertyIndex)
	{
		const json& entry= properties[propertyIndex];
		if (!entry.is_object())
			continue;
		const std::string shader= entry.value("shader", "");
		if (shader != "VRM/MToon")
			continue;

		// Materials are exported in the same order as their properties;
		// fall back to a name match when the counts disagree
		const std::string name= entry.value("name", "");
		int materialIndex= -1;
		if (propertyIndex < model.materials.size() && model.materials[propertyIndex].name == name)
			materialIndex= (int)propertyIndex;
		else
		{
			for (size_t index= 0; index < model.materials.size(); ++index)
			{
				if (model.materials[index].name == name)
				{
					materialIndex= (int)index;
					break;
				}
			}
		}
		if (materialIndex < 0)
		{
			context.warn("MToon properties for unknown material '" + name + "' ignored");
			continue;
		}

		AvatarMaterial& material= model.materials[materialIndex];
		material.isMToon= true;

		const json& floats= entry.value("floatProperties", json::object());
		const json& vectors= entry.value("vectorProperties", json::object());
		const json& textures= entry.value("textureProperties", json::object());

		if (vectors.contains("_Color") && vectors["_Color"].is_array() && vectors["_Color"].size() >= 4)
		{
			const json& c= vectors["_Color"];
			material.baseColorFactor= glm::vec4(gammaToLinear(c[0].get<float>()), gammaToLinear(c[1].get<float>()),
												gammaToLinear(c[2].get<float>()), c[3].get<float>());
		}
		glm::vec3 shadeColor(0.97f, 0.81f, 0.86f);
		if (vectors.contains("_ShadeColor") && vectors["_ShadeColor"].is_array() && vectors["_ShadeColor"].size() >= 3)
		{
			const json& c= vectors["_ShadeColor"];
			shadeColor= glm::vec3(c[0].get<float>(), c[1].get<float>(), c[2].get<float>());
		}
		material.shadeColorFactor=
			glm::vec3(gammaToLinear(shadeColor.x), gammaToLinear(shadeColor.y), gammaToLinear(shadeColor.z));

		if (textures.contains("_MainTex") && textures["_MainTex"].is_number_integer())
			material.baseColorTexture= textures["_MainTex"].get<int>();
		if (textures.contains("_ShadeTexture") && textures["_ShadeTexture"].is_number_integer())
			material.shadeMultiplyTexture= textures["_ShadeTexture"].get<int>();

		// 0.x shades with smoothstep(shift, shift + (1 - toony)) over a
		// half-Lambert term; the 1.0 ramp is linearstep(-1 + toony, 1 - toony)
		// over NdotL + shift. This is the parameter remap the migrations use.
		const float shadeShift= floats.value("_ShadeShift", 0.f);
		float shadeToony= floats.value("_ShadeToony", 0.9f);
		shadeToony= shadeToony + (1.f - shadeToony) * (0.5f + 0.5f * shadeShift);
		material.shadingToonyFactor= shadeToony;
		material.shadingShiftFactor= -shadeShift - (1.f - shadeToony);

		if (floats.contains("_Cutoff") && material.alphaMode == eAlphaMode::Mask)
			material.alphaCutoff= floats["_Cutoff"].get<float>();
		if (floats.contains("_CullMode"))
			material.doubleSided= floats["_CullMode"].get<float>() == 0.f;

		const bool bZWrite= floats.value("_ZWrite", 1.f) == 1.f;
		material.transparentWithZWrite= material.alphaMode == eAlphaMode::Blend && bZWrite;

		const int renderQueue= entry.value("renderQueue", 0);
		if (material.alphaMode == eAlphaMode::Blend)
		{
			if (material.transparentWithZWrite)
				transparentZWriteQueues.insert(renderQueue);
			else
				transparentQueues.insert(renderQueue);
		}
		parsed.push_back({materialIndex, renderQueue});
	}

	for (const Parsed& item : parsed)
	{
		AvatarMaterial& material= model.materials[item.materialIndex];
		if (material.alphaMode != eAlphaMode::Blend)
			continue;

		if (material.transparentWithZWrite)
		{
			const int position= (int)std::distance(transparentZWriteQueues.begin(),
												   transparentZWriteQueues.find(item.renderQueue));
			material.renderQueueOffset= std::min(position, 9);
		}
		else
		{
			const int position=
				(int)std::distance(transparentQueues.begin(), transparentQueues.find(item.renderQueue));
			const int last= (int)transparentQueues.size() - 1;
			material.renderQueueOffset= std::max(position - last, -9);
		}
	}
}

int textureIndex(const LoadContext& context, const cgltf_texture_view& view)
{
	return view.texture != nullptr ? (int)(view.texture - context.data->textures) : -1;
}

void readMaterials(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	for (cgltf_size index= 0; index < data.materials_count; ++index)
	{
		const cgltf_material& source= data.materials[index];
		AvatarMaterial material;
		material.name= source.name != nullptr ? source.name : "";
		if (source.has_pbr_metallic_roughness)
		{
			const cgltf_float* factor= source.pbr_metallic_roughness.base_color_factor;
			material.baseColorFactor= glm::vec4(factor[0], factor[1], factor[2], factor[3]);
			material.baseColorTexture= textureIndex(context, source.pbr_metallic_roughness.base_color_texture);
		}
		switch (source.alpha_mode)
		{
		case cgltf_alpha_mode_mask: material.alphaMode= eAlphaMode::Mask; break;
		case cgltf_alpha_mode_blend: material.alphaMode= eAlphaMode::Blend; break;
		default: material.alphaMode= eAlphaMode::Opaque; break;
		}
		material.alphaCutoff= source.alpha_cutoff;
		material.doubleSided= source.double_sided != 0;

		if (const cgltf_extension* mtoon=
				findExtension(source.extensions, source.extensions_count, "VRMC_materials_mtoon"))
		{
			json mtoonJson;
			if (parseExtensionJson(context, *mtoon, mtoonJson))
				readMToon1(material, mtoonJson);
		}

		model.materials.push_back(material);
	}
}

// -- Textures and images ------------------------------------------------------

void readTextures(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	for (cgltf_size index= 0; index < data.textures_count; ++index)
	{
		const cgltf_texture& source= data.textures[index];
		AvatarTexture texture;
		texture.image= source.image != nullptr ? (int)(source.image - data.images) : -1;
		if (source.sampler != nullptr)
		{
			texture.repeatS= source.sampler->wrap_s != cgltf_wrap_mode_clamp_to_edge;
			texture.repeatT= source.sampler->wrap_t != cgltf_wrap_mode_clamp_to_edge;
			texture.nearestFilter= source.sampler->mag_filter == cgltf_filter_type_nearest;
		}
		model.textures.push_back(texture);
	}
}

AvatarImage makePlaceholderImage(const std::string& name)
{
	AvatarImage image;
	image.name= name;
	image.width= 1;
	image.height= 1;
	image.rgba= {255, 255, 255, 255};
	return image;
}

bool decodeImage(const uint8_t* bytes, size_t byteCount, AvatarImage& outImage)
{
	const cv::Mat encoded(1, (int)byteCount, CV_8UC1, const_cast<uint8_t*>(bytes));
	cv::Mat decoded= cv::imdecode(encoded, cv::IMREAD_UNCHANGED);
	if (decoded.empty())
		return false;

	// To 8 bits per channel, then to RGBA. imdecode delivers BGR order.
	if (decoded.depth() == CV_16U)
		decoded.convertTo(decoded, CV_8U, 1.0 / 257.0);
	else if (decoded.depth() != CV_8U)
		decoded.convertTo(decoded, CV_8U);

	cv::Mat rgba;
	switch (decoded.channels())
	{
	case 1: cv::cvtColor(decoded, rgba, cv::COLOR_GRAY2RGBA); break;
	case 3: cv::cvtColor(decoded, rgba, cv::COLOR_BGR2RGBA); break;
	case 4: cv::cvtColor(decoded, rgba, cv::COLOR_BGRA2RGBA); break;
	default: return false;
	}

	outImage.width= rgba.cols;
	outImage.height= rgba.rows;
	outImage.rgba.resize((size_t)rgba.cols * rgba.rows * 4);
	for (int row= 0; row < rgba.rows; ++row)
		std::memcpy(outImage.rgba.data() + (size_t)row * rgba.cols * 4, rgba.ptr(row), (size_t)rgba.cols * 4);
	return true;
}

void readImages(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	for (cgltf_size index= 0; index < data.images_count; ++index)
	{
		const cgltf_image& source= data.images[index];
		const std::string name= source.name != nullptr ? source.name : ("image" + std::to_string(index));

		const cgltf_buffer_view* view= source.buffer_view;
		if (view == nullptr || view->buffer == nullptr || view->buffer->data == nullptr)
		{
			context.warn("Image '" + name + "' has no embedded data, using a white placeholder");
			model.images.push_back(makePlaceholderImage(name));
			continue;
		}

		const uint8_t* bytes= (const uint8_t*)view->buffer->data + view->offset;
		AvatarImage image;
		image.name= name;
		if (!decodeImage(bytes, view->size, image))
		{
			context.warn("Image '" + name + "' failed to decode, using a white placeholder");
			model.images.push_back(makePlaceholderImage(name));
			continue;
		}
		model.images.push_back(std::move(image));
	}
}

// -- Nodes and skins ----------------------------------------------------------

int nodeIndex(const LoadContext& context, const cgltf_node* node)
{
	return node != nullptr ? (int)(node - context.data->nodes) : -1;
}

void readNodes(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	model.nodes.resize(data.nodes_count);
	for (cgltf_size index= 0; index < data.nodes_count; ++index)
	{
		const cgltf_node& source= data.nodes[index];
		AvatarNode& node= model.nodes[index];
		node.name= source.name != nullptr ? source.name : "";
		node.parent= nodeIndex(context, source.parent);
		for (cgltf_size child= 0; child < source.children_count; ++child)
			node.children.push_back(nodeIndex(context, source.children[child]));

		cgltf_float local[16];
		cgltf_node_transform_local(&source, local);
		std::memcpy(&node.localTransform[0][0], local, sizeof(local));

		node.mesh= source.mesh != nullptr ? (int)(source.mesh - data.meshes) : -1;
		node.skin= source.skin != nullptr ? (int)(source.skin - data.skins) : -1;

		if (node.parent < 0)
			model.rootNodes.push_back((int)index);
	}
}

void readSkins(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	for (cgltf_size index= 0; index < data.skins_count; ++index)
	{
		const cgltf_skin& source= data.skins[index];
		AvatarSkin skin;
		skin.name= source.name != nullptr ? source.name : "";
		for (cgltf_size joint= 0; joint < source.joints_count; ++joint)
			skin.joints.push_back(nodeIndex(context, source.joints[joint]));

		skin.inverseBindMatrices.assign(source.joints_count, glm::mat4(1.f));
		if (source.inverse_bind_matrices != nullptr)
		{
			const cgltf_accessor& accessor= *source.inverse_bind_matrices;
			for (cgltf_size joint= 0; joint < source.joints_count && joint < accessor.count; ++joint)
			{
				cgltf_float matrix[16];
				if (cgltf_accessor_read_float(&accessor, joint, matrix, 16))
					std::memcpy(&skin.inverseBindMatrices[joint][0][0], matrix, sizeof(matrix));
			}
		}
		model.skins.push_back(std::move(skin));
	}
}

// -- Meshes -------------------------------------------------------------------

const cgltf_accessor* findAttribute(const cgltf_primitive& primitive, cgltf_attribute_type type, int setIndex)
{
	for (cgltf_size index= 0; index < primitive.attributes_count; ++index)
	{
		const cgltf_attribute& attribute= primitive.attributes[index];
		if (attribute.type == type && attribute.index == setIndex)
			return attribute.data;
	}
	return nullptr;
}

template <typename T, int N>
void readFloatAttribute(const cgltf_accessor& accessor, std::vector<T>& out)
{
	out.resize(accessor.count);
	for (cgltf_size index= 0; index < accessor.count; ++index)
	{
		cgltf_float values[4]= {0.f, 0.f, 0.f, 0.f};
		cgltf_accessor_read_float(&accessor, index, values, N);
		std::memcpy(&out[index], values, sizeof(float) * N);
	}
}

bool readPrimitive(LoadContext& context, const cgltf_primitive& source, const std::string& meshName,
				   AvatarPrimitive& outPrimitive)
{
	const cgltf_data& data= *context.data;

	if (source.type != cgltf_primitive_type_triangles)
	{
		context.warn("Mesh '" + meshName + "' has a non-triangle primitive, skipped");
		return false;
	}
	const cgltf_accessor* positions= findAttribute(source, cgltf_attribute_type_position, 0);
	if (positions == nullptr || positions->count == 0)
	{
		context.warn("Mesh '" + meshName + "' has a primitive without POSITION, skipped");
		return false;
	}

	readFloatAttribute<glm::vec3, 3>(*positions, outPrimitive.positions);
	const size_t vertexCount= outPrimitive.positions.size();

	if (const cgltf_accessor* normals= findAttribute(source, cgltf_attribute_type_normal, 0))
	{
		readFloatAttribute<glm::vec3, 3>(*normals, outPrimitive.normals);
		if (outPrimitive.normals.size() != vertexCount)
			outPrimitive.normals.clear();
	}
	if (const cgltf_accessor* uvs= findAttribute(source, cgltf_attribute_type_texcoord, 0))
	{
		readFloatAttribute<glm::vec2, 2>(*uvs, outPrimitive.uvs);
		if (outPrimitive.uvs.size() != vertexCount)
			outPrimitive.uvs.clear();
	}

	const cgltf_accessor* joints= findAttribute(source, cgltf_attribute_type_joints, 0);
	const cgltf_accessor* weights= findAttribute(source, cgltf_attribute_type_weights, 0);
	if (joints != nullptr && weights != nullptr && joints->count == vertexCount && weights->count == vertexCount)
	{
		outPrimitive.joints.resize(vertexCount);
		outPrimitive.weights.resize(vertexCount);
		for (size_t index= 0; index < vertexCount; ++index)
		{
			cgltf_uint jointIds[4]= {0, 0, 0, 0};
			cgltf_accessor_read_uint(joints, index, jointIds, 4);
			outPrimitive.joints[index]= glm::uvec4(jointIds[0], jointIds[1], jointIds[2], jointIds[3]);

			cgltf_float w[4]= {0.f, 0.f, 0.f, 0.f};
			cgltf_accessor_read_float(weights, index, w, 4);
			glm::vec4 weight(w[0], w[1], w[2], w[3]);
			const float sum= weight.x + weight.y + weight.z + weight.w;
			// A vertex with no weight at all follows joint 0 fully rather
			// than collapsing to the origin
			outPrimitive.weights[index]= sum > 1e-6f ? weight / sum : glm::vec4(1.f, 0.f, 0.f, 0.f);
		}
	}

	if (source.indices != nullptr)
	{
		outPrimitive.indices.resize(source.indices->count);
		for (cgltf_size index= 0; index < source.indices->count; ++index)
			outPrimitive.indices[index]= (uint32_t)cgltf_accessor_read_index(source.indices, index);
	}
	else
	{
		outPrimitive.indices.resize(vertexCount);
		for (size_t index= 0; index < vertexCount; ++index)
			outPrimitive.indices[index]= (uint32_t)index;
	}
	// Drop a trailing partial triangle rather than reading past the end
	outPrimitive.indices.resize(outPrimitive.indices.size() - outPrimitive.indices.size() % 3);

	for (uint32_t index : outPrimitive.indices)
	{
		if (index >= vertexCount)
		{
			context.warn("Mesh '" + meshName + "' has an index past its vertex count, primitive skipped");
			return false;
		}
	}

	outPrimitive.material= source.material != nullptr ? (int)(source.material - data.materials) : -1;
	return true;
}

void readMeshes(LoadContext& context)
{
	const cgltf_data& data= *context.data;
	AvatarModel& model= *context.model;

	for (cgltf_size index= 0; index < data.meshes_count; ++index)
	{
		const cgltf_mesh& source= data.meshes[index];
		AvatarMesh mesh;
		mesh.name= source.name != nullptr ? source.name : ("mesh" + std::to_string(index));
		for (cgltf_size primitiveIndex= 0; primitiveIndex < source.primitives_count; ++primitiveIndex)
		{
			AvatarPrimitive primitive;
			if (readPrimitive(context, source.primitives[primitiveIndex], mesh.name, primitive))
				mesh.primitives.push_back(std::move(primitive));
		}

		// Names only: the target count is the primitives' (they all share it)
		const size_t targetCount=
			source.target_names_count > 0
				? source.target_names_count
				: (source.primitives_count > 0 ? source.primitives[0].targets_count : 0);
		for (size_t target= 0; target < targetCount; ++target)
		{
			const bool bNamed= target < source.target_names_count && source.target_names[target] != nullptr;
			mesh.morphTargetNames.push_back(bNamed ? source.target_names[target] : std::to_string(target));
		}
		model.meshes.push_back(std::move(mesh));
	}
}

// -- Validation ---------------------------------------------------------------

bool checkRequiredBones(LoadContext& context, std::string& outError)
{
	std::string missing;
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const eHumanoidBone bone= (eHumanoidBone)index;
		if (isHumanoidBoneRequired(bone) && !context.model->hasBone(bone))
		{
			if (!missing.empty())
				missing+= ", ";
			missing+= humanoidBoneName(bone);
		}
	}
	if (!missing.empty())
	{
		outError= "Required humanoid bones missing: " + missing;
		return false;
	}
	return true;
}

const char* cgltfResultName(cgltf_result result)
{
	switch (result)
	{
	case cgltf_result_success: return "success";
	case cgltf_result_data_too_short: return "data too short";
	case cgltf_result_unknown_format: return "unknown format";
	case cgltf_result_invalid_json: return "invalid JSON";
	case cgltf_result_invalid_gltf: return "invalid glTF";
	case cgltf_result_invalid_options: return "invalid options";
	case cgltf_result_file_not_found: return "file not found";
	case cgltf_result_io_error: return "I/O error";
	case cgltf_result_out_of_memory: return "out of memory";
	case cgltf_result_legacy_gltf: return "legacy glTF 1.0";
	default: return "unknown error";
	}
}
} // namespace

namespace VrmLoader
{
const char* versionName(eVrmVersion version)
{
	return version == eVrmVersion::Vrm1 ? "VRM 1.0" : "VRM 0.x";
}

std::string sha256Hex(const uint8_t* bytes, size_t byteCount)
{
	BCRYPT_ALG_HANDLE algorithm= nullptr;
	if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
		return "";

	uint8_t digest[32]= {};
	// BCryptHash takes a 32-bit length, so a (theoretical) 4 GB+ file hashes in chunks
	BCRYPT_HASH_HANDLE hash= nullptr;
	bool bOk= BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) >= 0;
	for (size_t offset= 0; bOk && offset < byteCount;)
	{
		const size_t chunk= std::min<size_t>(byteCount - offset, 0x40000000);
		bOk= BCryptHashData(hash, const_cast<PUCHAR>(bytes + offset), (ULONG)chunk, 0) >= 0;
		offset+= chunk;
	}
	bOk= bOk && BCryptFinishHash(hash, digest, sizeof(digest), 0) >= 0;
	if (hash != nullptr)
		BCryptDestroyHash(hash);
	BCryptCloseAlgorithmProvider(algorithm, 0);
	if (!bOk)
		return "";

	static const char k_hex[]= "0123456789abcdef";
	std::string text;
	for (uint8_t byte : digest)
	{
		text.push_back(k_hex[byte >> 4]);
		text.push_back(k_hex[byte & 15]);
	}
	return text;
}

LoadResult loadMemory(const uint8_t* bytes, size_t byteCount)
{
	LoadResult result;

	cgltf_options options= {};
	cgltf_data* data= nullptr;
	cgltf_result parseResult= cgltf_parse(&options, bytes, byteCount, &data);
	if (parseResult != cgltf_result_success)
	{
		result.error= std::string("glTF parse failed: ") + cgltfResultName(parseResult);
		return result;
	}

	// No base path: everything a VRM references is inside the GLB
	parseResult= cgltf_load_buffers(&options, data, nullptr);
	if (parseResult != cgltf_result_success)
	{
		result.error= std::string("glTF buffer load failed: ") + cgltfResultName(parseResult);
		cgltf_free(data);
		return result;
	}
	parseResult= cgltf_validate(data);
	if (parseResult != cgltf_result_success)
	{
		result.error= std::string("glTF validation failed: ") + cgltfResultName(parseResult);
		cgltf_free(data);
		return result;
	}

	auto model= std::make_shared<AvatarModel>();
	LoadContext context;
	context.data= data;
	context.model= model.get();
	context.warnings= &result.warnings;

	// The 1.0 extension wins when a file carries both
	const cgltf_extension* vrm1= findExtension(data->data_extensions, data->data_extensions_count, "VRMC_vrm");
	const cgltf_extension* vrm0= findExtension(data->data_extensions, data->data_extensions_count, "VRM");
	json vrmJson;
	if (vrm1 != nullptr && parseExtensionJson(context, *vrm1, vrmJson))
	{
		model->version= eVrmVersion::Vrm1;
		model->specVersion= vrmJson.value("specVersion", "1.0");
	}
	else if (vrm0 != nullptr && parseExtensionJson(context, *vrm0, vrmJson))
	{
		model->version= eVrmVersion::Vrm0;
		model->specVersion= vrmJson.value("specVersion", "0.0");
	}
	else
	{
		result.error= "Not a VRM file: no VRMC_vrm or VRM extension";
		cgltf_free(data);
		return result;
	}

	readNodes(context);
	readSkins(context);
	readMaterials(context);
	readTextures(context);
	readImages(context);
	readMeshes(context);
	readHumanoidAndMeta(context, vrmJson, model->version);
	readExpressions(context, vrmJson, model->version);
	model->sha256Hex= sha256Hex(bytes, byteCount);
	if (model->version == eVrmVersion::Vrm0)
		readMToon0(context, vrmJson);

	cgltf_free(data);

	if (!checkRequiredBones(context, result.error))
		return result;

	model->fileHumanoidNodes= model->humanoidNodes;
	result.model= model;
	return result;
}

LoadResult loadFile(const std::filesystem::path& path)
{
	LoadResult result;

	std::ifstream file(path, std::ios::binary);
	if (!file)
	{
		result.error= "Cannot open " + PathUtils::pathToUtf8(path);
		return result;
	}
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
	if (bytes.empty())
	{
		result.error= "Empty file " + PathUtils::pathToUtf8(path);
		return result;
	}

	result= loadMemory(bytes.data(), bytes.size());
	if (result.model != nullptr)
	{
		result.model->sourcePath= PathUtils::pathToUtf8(path);
		MIKAN_LOG_INFO(k_logLabel) << "Loaded " << versionName(result.model->version) << " '"
								   << result.model->meta.name << "' from " << PathUtils::pathToUtf8(path) << " (" << result.model->nodes.size()
								   << " nodes, " << result.model->skins.size() << " skins, "
								   << result.model->triangleCount() << " triangles, " << result.model->images.size()
								   << " images)";
	}
	else
	{
		MIKAN_LOG_ERROR(k_logLabel) << "Failed to load " << PathUtils::pathToUtf8(path) << ": " << result.error;
	}
	return result;
}
} // namespace VrmLoader
