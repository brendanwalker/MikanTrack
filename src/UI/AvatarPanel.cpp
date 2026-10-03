#include "AvatarPanel.h"

#include <algorithm>
#include <cctype>

#include "imgui.h"

#include "LocText.h"

#include "App.h"
#include "AppConfig.h"
#include "AvatarRig.h"
#include "AvatarSkeleton.h"
#include "BodyPoseSolver.h"
#include "HandPoseModel.h"
#include "PathUtils.h"
#include "VrmLoader.h"

#include "tinyfiledialogs.h"

namespace
{
using B= eHumanoidBone;

const ImVec4 k_warningColor(1.f, 0.8f, 0.4f, 1.f);

eHumanoidBone fingerBone(int sideIndex, int finger, int phalanx)
{
	return (eHumanoidBone)((int)firstHumanoidFingerBone(sideIndex) + finger * 3 + phalanx);
}

// The bones whose rest frame the retarget measures a full frame against, and
// so the only ones a trim means anything on
bool hasFrameTrim(eHumanoidBone bone)
{
	return bone == B::Head || bone == B::LeftHand || bone == B::RightHand;
}

bool hasRollTrim(eHumanoidBone bone)
{
	return bone == B::LeftLowerArm || bone == B::RightLowerArm;
}

bool containsCaseInsensitive(const std::string& text, const char* filter)
{
	if (filter[0] == '\0')
		return true;
	const std::string needle(filter);
	auto it= std::search(text.begin(), text.end(), needle.begin(), needle.end(), [](char a, char b) {
		return std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
	});
	return it != text.end();
}

const char* nodeName(const AvatarModel& model, int node)
{
	return node >= 0 && node < (int)model.nodes.size() ? model.nodes[node].name.c_str() : "";
}
} // namespace

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
		if (ImGui::BeginTabBar("avatarTabs"))
		{
			if (ImGui::BeginTabItem(locLabel("avatarPanel.tabModel")))
			{
				drawModelTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem(locLabel("avatarPanel.tabMapping")))
			{
				drawMappingTab();
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem(locLabel("avatarPanel.tabRetarget")))
			{
				drawRetargetTab();
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
		}
	}
	else
	{
		m_previewPose= ePreviewPose::Live;
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
				m_selectedBone= HUMANOID_BONE_NONE;
			}
		}
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("avatarPanel.unloadButton")))
	{
		m_app->clearAvatar();
		avatar.modelPath.clear();
		m_config->markDirty();
		m_selectedBone= HUMANOID_BONE_NONE;
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

void AvatarPanel::drawModelTab()
{
	drawModelSummary();
	ImGui::Separator();
	drawPlacement();
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
		if (!skeleton.getHand((eHandSide)sideIndex).hasAnyFinger())
		{
			ImGui::TextColored(k_warningColor, "%s",
							   locText(sideIndex == 0 ? "avatarPanel.leftHandIncompleteText"
													  : "avatarPanel.rightHandIncompleteText"));
		}
	}
	for (const std::string& warning : m_app->getAvatarRigWarnings())
		ImGui::TextWrapped("%s", warning.c_str());

	if (!model.sourcePath.empty())
	{
		const std::filesystem::path rigPath= getAvatarRigPath(PathUtils::utf8ToPath(model.sourcePath));
		ImGui::TextDisabled("%s", locFormat("avatarPanel.rigFileFmt",
											PathUtils::pathToUtf8(rigPath.filename()).c_str())
									  .c_str());
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", PathUtils::pathToUtf8(rigPath).c_str());
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

std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT> AvatarPanel::computeDotStates() const
{
	const AvatarModel& model= *m_app->getAvatarModel();
	const AvatarSkeleton& skeleton= *m_app->getAvatarSkeleton();

	std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT> states;
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		if (model.humanoidNodes[index] >= 0)
			states[index]= eAvatarMapDotState::Mapped;
		else if (isHumanoidBoneRequired((eHumanoidBone)index))
			states[index]= eAvatarMapDotState::RequiredUnmapped;
		else
			states[index]= eAvatarMapDotState::OptionalUnmapped;
	}

	// One node on two bones
	for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
	{
		for (int other= index + 1; other < HUMANOID_BONE_COUNT; ++other)
		{
			if (model.humanoidNodes[index] >= 0 && model.humanoidNodes[index] == model.humanoidNodes[other])
			{
				states[index]= eAvatarMapDotState::Warning;
				states[other]= eAvatarMapDotState::Warning;
			}
		}
	}

	// The index sitting on the far side of the middle finger from the thumb
	// (the skeleton's swapped-finger warning) tags both fingers
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const AvatarSkeleton::HandRest& hand= skeleton.getHand((eHandSide)sideIndex);
		if (!hand.hasFinger((int)eFinger::Thumb) || !hand.hasFinger((int)eFinger::Index))
			continue;
		if (hand.skeleton.baseInPalm[(int)eFinger::Thumb].y * hand.skeleton.baseInPalm[(int)eFinger::Index].y >= 0.f)
			continue;
		for (int finger : {(int)eFinger::Index, (int)eFinger::Middle})
		{
			for (int phalanx= 0; phalanx < 3; ++phalanx)
			{
				const int bone= (int)fingerBone(sideIndex, finger, phalanx);
				if (states[bone] == eAvatarMapDotState::Mapped)
					states[bone]= eAvatarMapDotState::Warning;
			}
		}
	}
	return states;
}

void AvatarPanel::drawMappingTab()
{
	const std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT> states= computeDotStates();

	ImGui::TextWrapped("%s", locText("avatarPanel.mappingHelpText"));
	if (m_figure.draw(states, m_selectedBone))
		m_bScrollToSelected= true;

	if (ImGui::Button(locLabel("avatarPanel.resetAllMappingButton")))
	{
		AvatarRigSettings rig= m_app->getAvatarRig();
		rig.boneNodeOverrides.fill(std::nullopt);
		m_app->setAvatarRig(rig);
	}
	ImGui::SameLine();
	ImGui::SetNextItemWidth(-FLT_MIN);
	ImGui::InputTextWithHint("##nodeFilter", locText("avatarPanel.nodeFilterHint"), m_nodeFilter,
							 sizeof(m_nodeFilter));

	const ImGuiTableFlags flags=
		ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
	if (!ImGui::BeginTable("avatarBoneTable", 3, flags))
		return;
	ImGui::TableSetupColumn(locText("avatarPanel.boneColumn"), ImGuiTableColumnFlags_WidthStretch, 1.f);
	ImGui::TableSetupColumn(locText("avatarPanel.nodeColumn"), ImGuiTableColumnFlags_WidthStretch, 1.4f);
	ImGui::TableSetupColumn(locText("avatarPanel.trimColumn"), ImGuiTableColumnFlags_WidthStretch, 1.2f);
	ImGui::TableHeadersRow();
	for (eHumanoidBone bone : AvatarMapFigure::getSectionBones(m_figure.getSection()))
		drawBoneRow(bone, states);
	ImGui::EndTable();
	m_bScrollToSelected= false;
}

void AvatarPanel::drawBoneRow(eHumanoidBone bone, const std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT>& states)
{
	const AvatarModel& model= *m_app->getAvatarModel();
	const int index= (int)bone;
	const std::optional<std::string>& nodeOverride= m_app->getAvatarRig().boneNodeOverrides[index];

	ImGui::PushID(index);
	ImGui::TableNextRow();

	// Bone: selectable, in the dot's warning color when it carries one
	ImGui::TableSetColumnIndex(0);
	const bool bSelected= m_selectedBone == bone;
	if (bSelected && m_bScrollToSelected)
		ImGui::SetScrollHereY(0.5f);
	const eAvatarMapDotState state= states[index];
	if (state == eAvatarMapDotState::Warning || state == eAvatarMapDotState::RequiredUnmapped)
		ImGui::PushStyleColor(ImGuiCol_Text, state == eAvatarMapDotState::Warning ? k_warningColor
																				  : ImVec4(1.f, 0.4f, 0.4f, 1.f));
	else
		ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_Text));
	if (ImGui::Selectable(humanoidBoneName(bone), bSelected))
		m_selectedBone= bone;
	ImGui::PopStyleColor();

	// Node: a picker over the model's node names, plus unmapped, plus a
	// reset to the file's own mapping when overridden
	ImGui::TableSetColumnIndex(1);
	const int node= model.humanoidNodes[index];
	const std::string preview= node >= 0 ? nodeName(model, node) : locText("avatarPanel.nodeUnmappedItem");
	const float resetWidth= nodeOverride.has_value() ? ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x : 0.f;
	ImGui::SetNextItemWidth(-FLT_MIN - resetWidth);
	if (ImGui::BeginCombo("##node", preview.c_str(), ImGuiComboFlags_HeightLarge))
	{
		std::optional<std::string> chosen;
		bool bChose= false;
		const bool bRequired= isHumanoidBoneRequired(bone);
		ImGui::BeginDisabled(bRequired);
		if (ImGui::Selectable(locText("avatarPanel.nodeUnmappedItem"), node < 0))
		{
			chosen= std::string();
			bChose= true;
		}
		ImGui::EndDisabled();
		if (bRequired && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
			ImGui::SetTooltip("%s", locText("avatarPanel.requiredUnmapTooltip"));
		for (int candidate= 0; candidate < (int)model.nodes.size(); ++candidate)
		{
			const std::string& name= model.nodes[candidate].name;
			if (!containsCaseInsensitive(name, m_nodeFilter))
				continue;
			ImGui::PushID(candidate);
			if (ImGui::Selectable(name.c_str(), candidate == node))
			{
				chosen= name;
				bChose= true;
			}
			if (candidate == node)
				ImGui::SetItemDefaultFocus();
			ImGui::PopID();
		}
		ImGui::EndCombo();

		if (bChose)
		{
			// Choosing what the file already says clears the override
			AvatarRigSettings rig= m_app->getAvatarRig();
			const int fileNode= model.fileHumanoidNodes[index];
			const bool bSameAsFile= chosen->empty() ? fileNode < 0 : (fileNode >= 0 && *chosen == nodeName(model, fileNode));
			rig.boneNodeOverrides[index]= bSameAsFile ? std::nullopt : chosen;
			m_app->setAvatarRig(rig);
			m_selectedBone= bone;
		}
	}
	if (nodeOverride.has_value())
	{
		if (ImGui::IsItemHovered())
		{
			const int fileNode= model.fileHumanoidNodes[index];
			ImGui::SetTooltip("%s", locFormat("avatarPanel.overriddenTooltipFmt",
											  fileNode >= 0 ? nodeName(model, fileNode)
															: locText("avatarPanel.nodeUnmappedItem"))
										.c_str());
		}
		ImGui::SameLine();
		if (ImGui::Button("X##reset", ImVec2(ImGui::GetFrameHeight(), 0.f)))
		{
			AvatarRigSettings rig= m_app->getAvatarRig();
			rig.boneNodeOverrides[index]= std::nullopt;
			m_app->setAvatarRig(rig);
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", locText("avatarPanel.resetBoneTooltip"));
	}

	// Trim: three axes on the frame-bearing bones, a roll on the forearms
	ImGui::TableSetColumnIndex(2);
	if (hasFrameTrim(bone) || hasRollTrim(bone))
	{
		glm::vec3 trim= m_app->getAvatarRig().rotationTrimDegrees[index];
		ImGui::SetNextItemWidth(-FLT_MIN);
		const bool bChanged= hasFrameTrim(bone) ? ImGui::DragFloat3("##trim", &trim.x, 0.25f, -180.f, 180.f, "%.1f")
												: ImGui::DragFloat("##roll", &trim.x, 0.25f, -180.f, 180.f, "%.1f");
		if (ImGui::IsItemHovered())
		{
			ImGui::SetTooltip("%s", locText(hasFrameTrim(bone) ? "avatarPanel.frameTrimTooltip"
															   : "avatarPanel.rollTrimTooltip"));
		}
		if (bChanged)
		{
			AvatarRigSettings rig= m_app->getAvatarRig();
			rig.rotationTrimDegrees[index]= trim;
			m_app->setAvatarRig(rig);
		}
	}

	ImGui::PopID();
}

void AvatarPanel::drawRetargetTab()
{
	ImGui::TextUnformatted(locText("avatarPanel.previewPoseLabel"));
	ImGui::SameLine();
	int preview= (int)m_previewPose;
	ImGui::RadioButton(locLabel("avatarPanel.previewLiveRadio"), &preview, (int)ePreviewPose::Live);
	ImGui::SameLine();
	ImGui::RadioButton(locLabel("avatarPanel.previewRestRadio"), &preview, (int)ePreviewPose::Rest);
	ImGui::SameLine();
	ImGui::RadioButton(locLabel("avatarPanel.previewDemoRadio"), &preview, (int)ePreviewPose::Demo);
	m_previewPose= (ePreviewPose)preview;
	ImGui::SameLine();
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
		ImGui::SetTooltip("%s", locText("avatarPanel.previewPoseTooltip"));

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		drawArmControls(sideIndex);
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		drawHandControls(sideIndex);
}

void AvatarPanel::drawArmControls(int sideIndex)
{
	ImGui::PushID(sideIndex);
	if (ImGui::CollapsingHeader(locLabel(sideIndex == 0 ? "avatarPanel.leftArmHeader" : "avatarPanel.rightArmHeader"),
								ImGuiTreeNodeFlags_DefaultOpen))
	{
		AvatarRigSettings rig= m_app->getAvatarRig();
		AvatarRigSettings::Side& side= rig.sides[sideIndex];
		bool bChanged= false;

		bChanged|= ImGui::DragFloat3(locLabel("avatarPanel.elbowHintDrag"), &side.elbowHintOffset.x, 0.005f, -1.f, 1.f,
									 "%.3f m");
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", locText("avatarPanel.elbowHintTooltip"));
		bChanged|= ImGui::SliderFloat(locLabel("avatarPanel.elbowHintConfidenceSlider"), &side.elbowHintConfidence,
									  0.f, 1.f, "%.2f");
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", locText("avatarPanel.elbowHintConfidenceTooltip"));
		ImGui::Checkbox(locLabel("avatarPanel.showGizmoCheckbox"), &m_bShowElbowGizmo[sideIndex]);
		ImGui::SameLine();
		if (ImGui::Button(locLabel("avatarPanel.resetArmButton")))
		{
			const AvatarRigSettings::Side defaults;
			side.elbowHintOffset= defaults.elbowHintOffset;
			side.elbowHintConfidence= defaults.elbowHintConfidence;
			bChanged= true;
		}

		if (bChanged)
			m_app->setAvatarRig(rig);
	}
	ImGui::PopID();
}

void AvatarPanel::drawHandControls(int sideIndex)
{
	ImGui::PushID(2 + sideIndex);
	if (ImGui::CollapsingHeader(
			locLabel(sideIndex == 0 ? "avatarPanel.leftHandHeader" : "avatarPanel.rightHandHeader"),
			ImGuiTreeNodeFlags_DefaultOpen))
	{
		AvatarRigSettings rig= m_app->getAvatarRig();
		AvatarRigSettings::Side& side= rig.sides[sideIndex];
		bool bChanged= false;

		if (ImGui::Checkbox(locLabel("avatarPanel.thumbPronationAutoCheckbox"), &side.bThumbPronationAuto))
		{
			// Leaving auto starts from what auto was doing
			if (!side.bThumbPronationAuto)
			{
				const AvatarSkeleton& skeleton= *m_app->getAvatarSkeleton();
				side.thumbPronationDegrees=
					glm::degrees(HandPoseModel::getThumbPronationRad(skeleton.getHand((eHandSide)sideIndex).skeleton));
			}
			bChanged= true;
		}
		if (ImGui::IsItemHovered())
			ImGui::SetTooltip("%s", locText("avatarPanel.thumbPronationTooltip"));
		if (!side.bThumbPronationAuto)
		{
			bChanged|= ImGui::DragFloat(locLabel("avatarPanel.thumbPronationDrag"), &side.thumbPronationDegrees, 0.5f,
										-180.f, 180.f, "%.1f deg");
		}
		bChanged|= ImGui::SliderFloat(locLabel("avatarPanel.curlGainSlider"), &side.curlGain, 0.f, 2.f, "%.2f");
		bChanged|= ImGui::SliderFloat(locLabel("avatarPanel.splayGainSlider"), &side.splayGain, 0.f, 2.f, "%.2f");

		ImGui::TextUnformatted(locText("avatarPanel.fingersDrivenLabel"));
		const char* k_fingerKeys[FINGER_COUNT]= {
			"avatarPanel.fingerThumbCheckbox", "avatarPanel.fingerIndexCheckbox", "avatarPanel.fingerMiddleCheckbox",
			"avatarPanel.fingerRingCheckbox", "avatarPanel.fingerLittleCheckbox",
		};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			if (finger > 0)
				ImGui::SameLine();
			bool bEnabled= side.fingerEnabled[finger];
			if (ImGui::Checkbox(locLabel(k_fingerKeys[finger]), &bEnabled))
			{
				side.fingerEnabled[finger]= bEnabled;
				bChanged= true;
			}
		}

		if (ImGui::Button(locLabel("avatarPanel.resetHandButton")))
		{
			const AvatarRigSettings::Side defaults;
			side.bThumbPronationAuto= defaults.bThumbPronationAuto;
			side.thumbPronationDegrees= defaults.thumbPronationDegrees;
			side.curlGain= defaults.curlGain;
			side.splayGain= defaults.splayGain;
			side.fingerEnabled= defaults.fingerEnabled;
			bChanged= true;
		}

		if (bChanged)
			m_app->setAvatarRig(rig);
	}
	ImGui::PopID();
}
