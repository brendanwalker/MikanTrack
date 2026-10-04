#include "AvatarFaceMap.h"

#include "AvatarTypes.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace
{
// ARKit's own spelling only, compared without case: avatars spell perfect
// sync names both "eyeBlinkLeft" and "EyeBlinkLeft"
int arkitIndexIgnoringCase(const std::string& name)
{
	for (int index= 0; index < ARKIT_BLENDSHAPE_COUNT; ++index)
	{
		const char* arkitName= arkitBlendshapeName(index);
		if (name.size() != strlen(arkitName))
			continue;

		bool bMatch= true;
		for (size_t charIndex= 0; charIndex < name.size() && bMatch; ++charIndex)
		{
			bMatch= std::tolower((unsigned char)name[charIndex]) == std::tolower((unsigned char)arkitName[charIndex]);
		}
		if (bMatch)
			return index;
	}
	return -1;
}

AvatarFaceMap::Term term(const char* arkitName, float weight)
{
	return AvatarFaceMap::Term{arkitBlendshapeFromName(arkitName), weight};
}

// A pair of columns averaged, the shape of every two-sided preset
std::vector<AvatarFaceMap::Term> average(const char* first, const char* second)
{
	return {term(first, 0.5f), term(second, 0.5f)};
}

struct PresetRule
{
	const char* preset;
	std::vector<AvatarFaceMap::Term> terms;
};

std::vector<PresetRule> makePresetRules()
{
	return {
		{"blinkLeft", {term("eyeBlinkLeft", 1.f)}},
		{"blinkRight", {term("eyeBlinkRight", 1.f)}},
		{"aa", {term("jawOpen", 1.f)}},
		{"ih", average("mouthSmileLeft", "mouthSmileRight")},
		{"ee", average("mouthStretchLeft", "mouthStretchRight")},
		{"ou", {term("mouthPucker", 1.f)}},
		{"oh", {term("mouthFunnel", 1.f)}},
		{"lookUp", average("eyeLookUpLeft", "eyeLookUpRight")},
		{"lookDown", average("eyeLookDownLeft", "eyeLookDownRight")},
		// The person's left: the left eye looks out, the right eye in
		{"lookLeft", average("eyeLookOutLeft", "eyeLookInRight")},
		{"lookRight", average("eyeLookInLeft", "eyeLookOutRight")},
	};
}
} // namespace

std::shared_ptr<const AvatarFaceMap> AvatarFaceMap::buildRaw()
{
	auto map= std::make_shared<AvatarFaceMap>();
	for (int index= 0; index < ARKIT_BLENDSHAPE_COUNT; ++index)
		map->m_outputs.push_back(Output{arkitBlendshapeName(index), {Term{index, 1.f}}});
	return map;
}

std::shared_ptr<const AvatarFaceMap> AvatarFaceMap::build(const AvatarModel& model)
{
	auto map= std::make_shared<AvatarFaceMap>();

	for (const AvatarExpression& expression : model.expressions)
	{
		const int arkitIndex= arkitIndexIgnoringCase(expression.name);
		if (arkitIndex >= 0)
			map->m_outputs.push_back(Output{expression.name, {Term{arkitIndex, 1.f}}});
	}
	if (!map->m_outputs.empty())
	{
		map->m_bPerfectSync= true;
		return map;
	}

	for (const PresetRule& rule : makePresetRules())
	{
		if (const AvatarExpression* expression= model.findExpressionByPreset(rule.preset))
			map->m_outputs.push_back(Output{expression->name, rule.terms});
	}
	// The two-eyed blink only when the avatar cannot blink each eye on its
	// own: driving both would close the eyes twice
	const AvatarExpression* blink= model.findExpressionByPreset("blink");
	const bool bHasSidedBlinks=
		model.findExpressionByPreset("blinkLeft") != nullptr && model.findExpressionByPreset("blinkRight") != nullptr;
	if (blink != nullptr && !bHasSidedBlinks)
		map->m_outputs.push_back(Output{blink->name, average("eyeBlinkLeft", "eyeBlinkRight")});

	for (const AvatarMesh& mesh : model.meshes)
	{
		for (const std::string& morphName : mesh.morphTargetNames)
		{
			const int arkitIndex= arkitIndexIgnoringCase(morphName);
			if (arkitIndex < 0)
				continue;

			const bool bAlreadyMapped=
				std::any_of(map->m_outputs.begin(), map->m_outputs.end(),
							[&morphName](const Output& output) { return output.name == morphName; });
			if (!bAlreadyMapped)
				map->m_outputs.push_back(Output{morphName, {Term{arkitIndex, 1.f}}});
		}
	}

	return map;
}

void AvatarFaceMap::evaluate(const std::array<float, ARKIT_BLENDSHAPE_COUNT>& arkit,
							 std::vector<float>& outValues) const
{
	outValues.resize(m_outputs.size());
	for (size_t outputIndex= 0; outputIndex < m_outputs.size(); ++outputIndex)
	{
		float value= 0.f;
		for (const Term& outputTerm : m_outputs[outputIndex].terms)
		{
			if (outputTerm.arkitIndex >= 0)
				value+= outputTerm.weight * arkit[outputTerm.arkitIndex];
		}
		outValues[outputIndex]= std::clamp(value, 0.f, 1.f);
	}
}
