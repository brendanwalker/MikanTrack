#include "AvatarRig.h"

#include <fstream>
#include <sstream>

#include "nlohmann/json.hpp"

#include "Logger.h"
#include "PathUtils.h"

using json= nlohmann::json;

namespace
{
const char* k_logLabel= "AvatarRig";
const char* k_sideKeys[2]= {"left", "right"};

json vec3ToJson(const glm::vec3& v)
{
	return json::array({v.x, v.y, v.z});
}

glm::vec3 vec3FromJson(const json& j, const glm::vec3& fallback)
{
	if (!j.is_array() || j.size() != 3)
		return fallback;
	return glm::vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}
} // namespace

std::filesystem::path getAvatarRigPath(const std::filesystem::path& vrmPath)
{
	std::filesystem::path rigPath= vrmPath;
	rigPath.replace_extension(".mikanrig.json");
	return rigPath;
}

std::string avatarRigToJson(const AvatarRigSettings& settings)
{
	json j;
	j["version"]= AvatarRigSettings::kVersion;

	json boneNodes= json::object();
	json trims= json::object();
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const char* boneName= humanoidBoneName((eHumanoidBone)index);
		if (settings.boneNodeOverrides[index].has_value())
			boneNodes[boneName]= *settings.boneNodeOverrides[index];
		if (settings.rotationTrimDegrees[index] != glm::vec3(0.f))
			trims[boneName]= vec3ToJson(settings.rotationTrimDegrees[index]);
	}
	j["boneNodes"]= boneNodes;
	j["rotationTrimDegrees"]= trims;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const AvatarRigSettings::Side& side= settings.sides[sideIndex];
		json fingers= json::array();
		for (bool bEnabled : side.fingerEnabled)
			fingers.push_back(bEnabled);
		j[k_sideKeys[sideIndex]]= {
			{"elbowHintOffset", vec3ToJson(side.elbowHintOffset)},
			{"elbowHintConfidence", side.elbowHintConfidence},
			{"thumbPronationAuto", side.bThumbPronationAuto},
			{"thumbPronationDegrees", side.thumbPronationDegrees},
			{"curlGain", side.curlGain},
			{"splayGain", side.splayGain},
			{"fingerEnabled", fingers},
		};
	}
	return j.dump(2);
}

bool avatarRigFromJson(const std::string& text, AvatarRigSettings& outSettings, std::string& outError)
{
	AvatarRigSettings settings;
	try
	{
		const json j= json::parse(text);
		if (!j.is_object())
		{
			outError= "not a JSON object";
			return false;
		}

		const json& boneNodes= j.value("boneNodes", json::object());
		for (auto it= boneNodes.begin(); it != boneNodes.end(); ++it)
		{
			const eHumanoidBone bone= humanoidBoneFromName(it.key().c_str());
			if (bone == HUMANOID_BONE_NONE || !it.value().is_string())
			{
				MIKAN_LOG_WARNING(k_logLabel) << "Ignoring bone mapping entry '" << it.key() << "'";
				continue;
			}
			settings.boneNodeOverrides[(int)bone]= it.value().get<std::string>();
		}

		const json& trims= j.value("rotationTrimDegrees", json::object());
		for (auto it= trims.begin(); it != trims.end(); ++it)
		{
			const eHumanoidBone bone= humanoidBoneFromName(it.key().c_str());
			if (bone == HUMANOID_BONE_NONE)
			{
				MIKAN_LOG_WARNING(k_logLabel) << "Ignoring rotation trim entry '" << it.key() << "'";
				continue;
			}
			settings.rotationTrimDegrees[(int)bone]= vec3FromJson(it.value(), glm::vec3(0.f));
		}

		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			if (!j.contains(k_sideKeys[sideIndex]))
				continue;
			const json& sj= j[k_sideKeys[sideIndex]];
			AvatarRigSettings::Side& side= settings.sides[sideIndex];
			side.elbowHintOffset= vec3FromJson(sj.value("elbowHintOffset", json()), side.elbowHintOffset);
			side.elbowHintConfidence= sj.value("elbowHintConfidence", side.elbowHintConfidence);
			side.bThumbPronationAuto= sj.value("thumbPronationAuto", side.bThumbPronationAuto);
			side.thumbPronationDegrees= sj.value("thumbPronationDegrees", side.thumbPronationDegrees);
			side.curlGain= sj.value("curlGain", side.curlGain);
			side.splayGain= sj.value("splayGain", side.splayGain);
			const json& fingers= sj.value("fingerEnabled", json::array());
			for (int finger= 0; finger < FINGER_COUNT && finger < (int)fingers.size(); ++finger)
				side.fingerEnabled[finger]= fingers[finger].get<bool>();
		}
	}
	catch (const std::exception& exception)
	{
		outError= exception.what();
		return false;
	}

	outSettings= settings;
	return true;
}

AvatarRigSettings loadAvatarRig(const std::filesystem::path& vrmPath, std::vector<std::string>* outWarnings)
{
	const std::filesystem::path rigPath= getAvatarRigPath(vrmPath);
	std::ifstream file(rigPath, std::ios::binary);
	if (!file.is_open())
		return AvatarRigSettings();

	std::stringstream buffer;
	buffer << file.rdbuf();

	AvatarRigSettings settings;
	std::string error;
	if (!avatarRigFromJson(buffer.str(), settings, error))
	{
		const std::string warning= "Ignoring unreadable rig file " + PathUtils::pathToUtf8(rigPath) + ": " + error;
		MIKAN_LOG_WARNING(k_logLabel) << warning;
		if (outWarnings != nullptr)
			outWarnings->push_back(warning);
		return AvatarRigSettings();
	}
	MIKAN_LOG_INFO(k_logLabel) << "Loaded rig settings from " << PathUtils::pathToUtf8(rigPath);
	return settings;
}

bool saveAvatarRig(const std::filesystem::path& vrmPath, const AvatarRigSettings& settings)
{
	const std::filesystem::path rigPath= getAvatarRigPath(vrmPath);
	std::ofstream file(rigPath, std::ios::binary);
	if (!file.is_open())
	{
		MIKAN_LOG_ERROR(k_logLabel) << "Failed to open " << PathUtils::pathToUtf8(rigPath) << " for writing";
		return false;
	}
	file << avatarRigToJson(settings);
	return true;
}

void applyRigToModel(const AvatarRigSettings& settings, AvatarModel& model, std::vector<std::string>& outWarnings)
{
	model.humanoidNodes= model.fileHumanoidNodes;

	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const std::optional<std::string>& nodeName= settings.boneNodeOverrides[index];
		if (!nodeName.has_value())
			continue;
		const eHumanoidBone bone= (eHumanoidBone)index;

		if (nodeName->empty())
		{
			// The skeleton and the retarget assume the required bones exist
			if (isHumanoidBoneRequired(bone))
			{
				outWarnings.push_back(std::string(humanoidBoneName(bone)) +
									  " is required and cannot be unmapped; keeping the file's mapping");
				continue;
			}
			model.humanoidNodes[index]= -1;
			continue;
		}

		int found= -1;
		for (int node= 0; node < (int)model.nodes.size(); ++node)
		{
			if (model.nodes[node].name == *nodeName)
			{
				found= node;
				break;
			}
		}
		if (found < 0)
		{
			outWarnings.push_back(std::string(humanoidBoneName(bone)) + ": no node named '" + *nodeName +
								  "' in this model; keeping the file's mapping");
			continue;
		}
		model.humanoidNodes[index]= found;
	}

	// One node driven as two bones poses as whichever is composed last;
	// this is the normal state halfway through swapping two fingers
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const int node= model.humanoidNodes[index];
		if (node < 0)
			continue;
		for (int other= index + 1; other < HUMANOID_BONE_COUNT; ++other)
		{
			if (model.humanoidNodes[other] == node)
			{
				outWarnings.push_back(std::string(humanoidBoneName((eHumanoidBone)index)) + " and " +
									  humanoidBoneName((eHumanoidBone)other) + " are both mapped to node '" +
									  model.nodes[node].name + "'");
			}
		}
	}
}
