#include "TestCommon.h"

#include "AvatarSkeleton.h"
#include "PathUtils.h"
#include "VrmLoader.h"

// Headless inspection of a VRM file: what the loader read from it and what
// the skeleton derived. The quickest way to see whether an avatar someone
// hands over will load before opening the app.
static int runVrmInfoTool(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("vrm-info") << "Usage: --vrm-info <file.vrm>";
		return 1;
	}

	const std::filesystem::path path= PathUtils::utf8ToPath(args[0]);
	const VrmLoader::LoadResult result= VrmLoader::loadFile(path);
	if (result.model == nullptr)
	{
		MIKAN_LOG_ERROR("vrm-info") << "Load failed: " << result.error;
		return 1;
	}
	const AvatarModel& model= *result.model;

	MIKAN_LOG_INFO("vrm-info") << VrmLoader::versionName(model.version) << " (specVersion '" << model.specVersion
							   << "')";
	MIKAN_LOG_INFO("vrm-info") << "name '" << model.meta.name << "' version '" << model.meta.version << "' author '"
							   << model.meta.author << "' license '" << model.meta.license << "'";
	MIKAN_LOG_INFO("vrm-info") << model.nodes.size() << " nodes, " << model.skins.size() << " skins, "
							   << model.meshes.size() << " meshes, " << model.triangleCount() << " triangles, "
							   << model.materials.size() << " materials, " << model.textures.size() << " textures, "
							   << model.images.size() << " images";
	for (size_t index= 0; index < model.skins.size(); ++index)
	{
		MIKAN_LOG_INFO("vrm-info") << "  skin " << index << " '" << model.skins[index].name << "': "
								   << model.skins[index].joints.size() << " joints";
	}
	for (size_t index= 0; index < model.materials.size(); ++index)
	{
		const AvatarMaterial& material= model.materials[index];
		const char* alphaMode= material.alphaMode == eAlphaMode::Blend  ? "BLEND"
							   : material.alphaMode == eAlphaMode::Mask ? "MASK"
																		: "OPAQUE";
		MIKAN_LOG_INFO("vrm-info") << "  material " << index << " '" << material.name << "': " << alphaMode
								   << (material.doubleSided ? " double-sided" : "")
								   << (material.isMToon ? " mtoon" : "") << " baseTex " << material.baseColorTexture
								   << " shadeTex " << material.shadeMultiplyTexture << " shift "
								   << material.shadingShiftFactor << " toony " << material.shadingToonyFactor
								   << " queue " << material.renderQueueOffset
								   << (material.transparentWithZWrite ? " zwrite" : "");
	}
	for (size_t index= 0; index < model.images.size(); ++index)
	{
		MIKAN_LOG_INFO("vrm-info") << "  image " << index << " '" << model.images[index].name << "': "
								   << model.images[index].width << "x" << model.images[index].height;
	}

	int presentCount= 0;
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		const int node= model.humanoidNodes[index];
		if (node >= 0)
		{
			++presentCount;
			MIKAN_LOG_INFO("vrm-info") << "  bone " << humanoidBoneName((eHumanoidBone)index) << " -> node " << node
									   << " '" << model.nodes[node].name << "'";
		}
		else
		{
			MIKAN_LOG_INFO("vrm-info") << "  bone " << humanoidBoneName((eHumanoidBone)index) << " -> missing";
		}
	}
	MIKAN_LOG_INFO("vrm-info") << presentCount << " of " << HUMANOID_BONE_COUNT << " humanoid bones present";

	const AvatarSkeleton skeleton(model);
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const eHandSide side= (eHandSide)sideIndex;
		const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
		MIKAN_LOG_INFO("vrm-info") << (sideIndex == 0 ? "left" : "right") << " arm: upper "
								   << skeleton.getUpperArmLength(side) << " m, forearm "
								   << skeleton.getForearmLength(side) << " m, hand " << skeleton.getHandLength(side)
								   << " m, finger bones " << hand.fingerBoneCount[0] << "/" << hand.fingerBoneCount[1]
								   << "/" << hand.fingerBoneCount[2] << "/" << hand.fingerBoneCount[3] << "/"
								   << hand.fingerBoneCount[4];
		if (hand.valid)
		{
			const glm::vec3 x(hand.palmFrameWorld[0]);
			const glm::vec3 z(hand.palmFrameWorld[2]);
			MIKAN_LOG_INFO("vrm-info") << "  palm X (" << x.x << ", " << x.y << ", " << x.z << ") palm Z (" << z.x
									   << ", " << z.y << ", " << z.z << ") index base y "
									   << hand.skeleton.baseInPalm[(int)eFinger::Index].y;
		}
	}
	using B= eHumanoidBone;
	const glm::vec3 leftArm=
		skeleton.getBone(B::LeftLowerArm).restPositionWorld - skeleton.getBone(B::LeftUpperArm).restPositionWorld;
	MIKAN_LOG_INFO("vrm-info") << "shoulder width " << skeleton.getShoulderWidth() << " m, height above hips "
							   << skeleton.getHeightAboveHips() << " m, left upper arm direction (" << leftArm.x
							   << ", " << leftArm.y << ", " << leftArm.z << ")";

	MIKAN_LOG_INFO("vrm-info") << "sha256 " << model.sha256Hex;
	for (const AvatarExpression& expression : model.expressions)
	{
		MIKAN_LOG_INFO("vrm-info") << "  expression '" << expression.name << "' preset '" << expression.preset
								   << "' " << expression.morphBinds.size() << " binds";
	}

	for (const std::string& warning : result.warnings)
		MIKAN_LOG_INFO("vrm-info") << "warning: " << warning;
	for (const std::string& warning : skeleton.getWarnings())
		MIKAN_LOG_INFO("vrm-info") << "skeleton warning: " << warning;
	return 0;
}
MIKAN_REGISTER_TEST("--vrm-info", "Print what the loader reads from a VRM file and the derived skeleton",
					eTestCategory::Tool, runVrmInfoTool);
