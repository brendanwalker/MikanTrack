#pragma once

#include <vector>

#include "BodyDimensionCalibrator.h"
#include "VisionThread.h" // VisionPreviewFrame
#include "WizardResult.h"

class AppConfig;
struct TrackingFrameResult;

// Measures the body lengths the pose solve assumes.
//
// Those lengths are NOT anatomical: they are separations between the model's
// own landmarks, which sit inside the real joints by an amount that varies
// with the person and the model. Assuming an anatomical shoulder width put a
// live shoulder 0.8 m too far away, so they have to be measured on the actual
// body in front of the actual model.
//
// Two stages, each shaped by what it can see:
//
//  1. ARMS OUT, forearms level. That puts the arms, shoulders and head in one
//     plane facing the camera, sharing the fused wrists' distance from it -
//     and the wrists are the only metric, world-anchored points available,
//     stereo-triangulated by a pipeline that knows nothing about any of this.
//     The pose is verified rather than trusted: an arm angled toward the
//     camera reads short against the known forearm length, and those samples
//     are thrown away.
//  2. HEAD TURNED. Facing the camera the nose sits on the ear midpoint and
//     says nothing about how far it juts forward; turning the head swings that
//     offset into the image plane where it can be measured.
class BodyCalibrationWizard
{
public:
	enum class eState
	{
		VerifyReady,  // a body-pose camera, extrinsics, and both hands tracked
		FrontalPose,  // arms out, forearms level
		HeadTurn,     // look left and right
		Review,
	};

	BodyCalibrationWizard(AppConfig* config, VisionThread* visionThread);

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }
	int getCameraIndex() const { return m_cameraIndex; }
	eWizardResult getResult() const { return m_wizardResult; }

	// Returns false once the wizard wants to close
	bool update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
				const TrackingFrameResult& fusedResult);

private:
	// The camera running body pose, or -1
	int findBodyPoseCamera() const;
	// Assembles the camera's calibration (undistorted matrix + extrinsics)
	// around its latest observation, the same way the vision thread does when
	// it builds a fusion candidate
	bool makeCameraFrame(const std::vector<VisionPreviewFrame>& previews, CameraFrameResult& outCamera) const;

	bool drawVerifyStage(const std::vector<VisionPreviewFrame>& previews, const TrackingFrameResult& fusedResult);
	bool drawFrontalStage(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
						  const TrackingFrameResult& fusedResult);
	bool drawHeadTurnStage(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews);
	bool drawReviewStage();

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifyReady;
	bool m_bActive= false;
	// m_result is the calibrator's measurement result, hence the longer name
	eWizardResult m_wizardResult= eWizardResult::None;
	int m_cameraIndex= -1;

	BodyDimensionCalibrator m_calibrator;
	BodyDimensionCalibrator::Sample m_lastSample;
	bool m_bLastSampleAccepted= false;
	float m_countdownSeconds= 0.f;
	bool m_bCollecting= false;

	BodyDimensionCalibrator::Result m_result;
};
