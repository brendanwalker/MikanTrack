#include "AvatarPanel.h"

#include "imgui.h"

#include "LocText.h"

#include "App.h"
#include "AppConfig.h"
#include "AvatarSkeleton.h"
#include "BodyPoseSolver.h"
#include "PathUtils.h"
#include "VrmLoader.h"

#include "tinyfiledialogs.h"

AvatarPanel::AvatarPanel(App* app, AppConfig* config)
	: m_app(app)
	, m_config(config)
{
}

void AvatarPanel::draw(bool* pOpen)
{
	if (!ImGui::Begin(locWindowTitle("windows.avatar"), pOpen))
	{
		ImGui::End();
		return;
	}

	drawLoadControls();

	if (m_app->getAvatarModel() != nullptr)
	{
		ImGui::Separator();
		drawModelSummary();
		ImGui::Separator();
		drawPlacement();
		ImGui::Separator();
		drawBoneTable();
	}

	ImGui::End();
}

void AvatarPanel::drawLoadControls()
{
	AvatarConfig& avatar= m_config->avatar;

	if (ImGui::Button(locLabel("avatarPanel.browseButton")))
	{
		// Seed the dialog at the shipped samples the first time, at the
		// current model afterwards
		std::string defaultPath= (PathUtils::getModulePath() / "models" / "avatars" / "").string();
		if (!avatar.modelPath.empty())
			defaultPath= m_app->resolveAvatarPath(avatar.modelPath).string();

		const char* filterPatterns[]= {"*.vrm"};
		const char* selectedPath= tinyfd_openFileDialog(
			locText("avatarPanel.browseDialogTitle"), defaultPath.c_str(), 1, filterPatterns,
			locText("avatarPanel.browseDialogFilterDescription"), 0);
		if (selectedPath != nullptr)
		{
			const std::filesystem::path path= PathUtils::utf8ToPath(selectedPath);
			if (m_app->loadAvatar(path))
			{
				avatar.modelPath= PathUtils::pathToUtf8(path);
				m_config->markDirty();
			}
		}
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("avatarPanel.unloadButton")))
	{
		m_app->clearAvatar();
		avatar.modelPath.clear();
		m_config->markDirty();
	}

	if (avatar.modelPath.empty())
	{
		ImGui::TextDisabled("%s", locText("avatarPanel.noModelText"));
	}
	else
	{
		const std::filesystem::path shown= PathUtils::utf8ToPath(avatar.modelPath);
		ImGui::TextWrapped("%s", PathUtils::createTrimmedPathString(shown, 60).c_str());
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", avatar.modelPath.c_str());
	}

	const std::string& error= m_app->getAvatarLoadError();
	if (!error.empty())
		ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", error.c_str());
}

void AvatarPanel::drawModelSummary()
{
	const AvatarModel& model= *m_app->getAvatarModel();
	const AvatarSkeleton& skeleton= *m_app->getAvatarSkeleton();

	ImGui::Text("%s", locFormat("avatarPanel.modelNameFmt", model.meta.name.c_str(),
								VrmLoader::versionName(model.version))
						  .c_str());
	if (!model.meta.author.empty())
		ImGui::Text("%s", locFormat("avatarPanel.authorFmt", model.meta.author.c_str()).c_str());
	if (!model.meta.license.empty())
		ImGui::TextWrapped("%s", locFormat("avatarPanel.licenseFmt", model.meta.license.c_str()).c_str());
	ImGui::Text("%s", locFormat("avatarPanel.contentsFmt", (int)model.nodes.size(), (int)model.skins.size(),
								(int)model.triangleCount(), (int)model.images.size())
						  .c_str());

	int presentCount= 0;
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		presentCount+= model.humanoidNodes[index] >= 0 ? 1 : 0;
	ImGui::Text("%s", locFormat("avatarPanel.bonesPresentFmt", presentCount, HUMANOID_BONE_COUNT).c_str());

	// The avatar's arm against the user's measured one: the ratio the hand
	// retarget will scale reach by
	const BodyDimensions user= makeBodyDimensions(*m_config);
	ImGui::Text("%s", locFormat("avatarPanel.armLengthsFmt",
								skeleton.getUpperArmLength(eHandSide::Left) * 100.f,
								skeleton.getForearmLength(eHandSide::Left) * 100.f,
								user.upperArmLengthMeters * 100.f, user.forearmLengthMeters * 100.f)
						  .c_str());
	ImGui::Text("%s", locFormat("avatarPanel.shoulderWidthFmt", skeleton.getShoulderWidth() * 100.f,
								user.shoulderWidthMeters * 100.f)
						  .c_str());
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!skeleton.getHand((eHandSide)sideIndex).valid)
		{
			ImGui::TextColored(ImVec4(1.f, 0.8f, 0.4f, 1.f), "%s",
							   locText(sideIndex == 0 ? "avatarPanel.leftHandIncompleteText"
													  : "avatarPanel.rightHandIncompleteText"));
		}
	}
}

void AvatarPanel::drawPlacement()
{
	AvatarConfig& avatar= m_config->avatar;
	bool bChanged= false;

	bChanged|= ImGui::Checkbox(locLabel("avatarPanel.showInSceneCheckbox"), &avatar.showInScene);
	bChanged|= ImGui::Checkbox(locLabel("avatarPanel.followShouldersCheckbox"), &avatar.followShoulders);
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("%s", locText("avatarPanel.followShouldersTooltip"));

	ImGui::TextDisabled("%s", locText("avatarPanel.fixedRootHeader"));
	bChanged|= ImGui::DragFloat3(locLabel("avatarPanel.rootPositionDrag"), &avatar.rootPositionWorld.x, 0.005f,
								 -5.f, 5.f, "%.3f m");
	bChanged|= ImGui::DragFloat(locLabel("avatarPanel.rootYawDrag"), &avatar.rootYawDegrees, 0.5f, -180.f, 180.f,
								"%.1f deg");
	if (ImGui::Button(locLabel("avatarPanel.resetRootButton")))
	{
		avatar.rootPositionWorld= glm::vec3(0.f);
		avatar.rootYawDegrees= 0.f;
		bChanged= true;
	}

	if (bChanged)
		m_config->markDirty();
}

void AvatarPanel::drawBoneTable()
{
	if (!ImGui::CollapsingHeader(locLabel("avatarPanel.boneMapHeader")))
		return;

	const AvatarModel& model= *m_app->getAvatarModel();
	const ImGuiTableFlags flags=
		ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
	if (!ImGui::BeginTable("avatarBoneTable", 2, flags))
		return;

	ImGui::TableSetupColumn(locText("avatarPanel.boneColumn"));
	ImGui::TableSetupColumn(locText("avatarPanel.nodeColumn"));
	ImGui::TableHeadersRow();

	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		ImGui::TableNextRow();
		ImGui::TableSetColumnIndex(0);
		ImGui::TextUnformatted(humanoidBoneName((eHumanoidBone)index));
		ImGui::TableSetColumnIndex(1);
		const int node= model.humanoidNodes[index];
		if (node >= 0)
			ImGui::TextUnformatted(model.nodes[node].name.c_str());
		else
			ImGui::TextDisabled("%s", locText("avatarPanel.boneMissingText"));
	}
	ImGui::EndTable();
}
