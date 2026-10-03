#pragma once

#include <array>
#include <string>

#include "AvatarMapFigure.h"

class App;
class AppConfig;

// The avatar panel: which VRM is loaded and what the loader read from it
// (Model), the humanoid map with its per-bone overrides and rotation trims
// (Mapping), and the elbow hints and hand tweaks plus a preview pose to tune
// them against (Retarget). Loading goes through App (the model is
// main-thread, GL-adjacent state); placement edits go to the persisted
// AvatarConfig and rig edits to App::setAvatarRig.
class AvatarPanel
{
public:
	AvatarPanel(App* app, AppConfig* config);

	void draw(bool* pOpen);

	// What the 3D scene's avatar is posed from. Rest and Demo are canned
	// frames for tuning the rig without cameras; the VMC output always
	// follows live tracking.
	enum class ePreviewPose
	{
		Live,
		Rest,
		Demo,
	};
	ePreviewPose getPreviewPose() const { return m_previewPose; }
	// Whether an arm's elbow hint is shown as a draggable gizmo in the scene
	bool getShowElbowGizmo(int sideIndex) const { return m_bShowElbowGizmo[sideIndex]; }

private:
	void drawLoadControls();
	void drawModelTab();
	void drawModelSummary();
	void drawPlacement();
	void drawMappingTab();
	void drawBoneRow(eHumanoidBone bone, const std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT>& states);
	void drawRetargetTab();
	void drawArmControls(int sideIndex);
	void drawHandControls(int sideIndex);
	std::array<eAvatarMapDotState, HUMANOID_BONE_COUNT> computeDotStates() const;

	App* m_app;
	AppConfig* m_config;

	AvatarMapFigure m_figure;
	eHumanoidBone m_selectedBone= eHumanoidBone::Count;
	bool m_bScrollToSelected= false;
	char m_nodeFilter[64]= {};

	ePreviewPose m_previewPose= ePreviewPose::Live;
	bool m_bShowElbowGizmo[2]= {false, false};
};
