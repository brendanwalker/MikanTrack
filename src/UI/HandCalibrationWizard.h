#pragma once

#include "VisionThread.h" // BoneCalibrationCapture / RestPoseCapture
#include "WizardResult.h"

class AppConfig;
struct TrackingFrameResult;

// One guided flow for both hand calibrations, in the only order that works:
// bones first, rest pose second. The order is the point of the wizard -
// saving measured bones moves the THUMB's angle zero (its neutral direction
// follows the metacarpal, unlike the four fingers, which are pinned to palm
// forward), so a rest pose captured before a bone save is captured against a
// zero that no longer exists. Two separate buttons made that mistake easy;
// one flow makes it impossible.
//
// Both stages need stereo: bones are medians over triangulated landmarks (a
// monocular pose carries the landmark model's shape, the thing being
// replaced), and the rest capture reads the raw multi-view angles of the
// fuse that services it.
//
// Leaves tracking running, like the mounting and body wizards: the tracked
// hands ARE the measurement.
class HandCalibrationWizard
{
public:
	enum class eState
	{
		BonesIntro,     // instructions + live stereo status
		BonesCountdown,
		BonesSampling,  // vision thread samples; live counts shown
		BonesReview,    // measured numbers; Accept writes skeleton + scale
		RestIntro,      // instructions (flat hands)
		RestCountdown,
		RestResult,     // per-side captured/missed; Retry or Done
	};

	HandCalibrationWizard(AppConfig* config, VisionThread* visionThread);

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }
	eWizardResult getResult() const { return m_result; }

	// Returns false once the wizard wants to close
	bool update(float deltaSeconds, const TrackingFrameResult& fusedResult);

private:
	void drawBonesIntro(const TrackingFrameResult& fusedResult);
	void drawBonesCountdown(float deltaSeconds);
	void drawBonesSampling();
	void drawBonesReview();
	void drawRestIntro(const TrackingFrameResult& fusedResult);
	void drawRestCountdown(float deltaSeconds);
	void drawRestResult();

	void saveBones();
	void saveRest();

	AppConfig* m_config= nullptr;
	VisionThread* m_visionThread= nullptr;

	bool m_bActive= false;
	bool m_bWantsClose= false;
	eWizardResult m_result= eWizardResult::None;
	eState m_state= eState::BonesIntro;
	float m_countdown= 0.f;

	VisionThread::BoneCalibrationCapture m_boneCapture;
	VisionThread::RestPoseCapture m_restCapture;
};
