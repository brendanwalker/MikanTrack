#include "BodyCalibrationWizard.h"

#include <algorithm>

#include "imgui.h"

#include "LocText.h"

#include "AppConfig.h"
#include "TrackingTypes.h"

namespace
{
constexpr float k_countdownSeconds= 3.f;
constexpr float k_collectSeconds= 4.f;

const ImVec4 k_colorGood(0.4f, 1.f, 0.4f, 1.f);
const ImVec4 k_colorWait(1.f, 0.6f, 0.3f, 1.f);
const ImVec4 k_colorBad(1.f, 0.4f, 0.4f, 1.f);

void drawStatusLine(bool bOk, const char* text)
{
	ImGui::TextColored(bOk ? k_colorGood : k_colorWait, "%s %s", bOk ? "[ok]" : "[..]", text);
}

// A measured length beside the one in use, so the review reads as a decision
// rather than a number dump
void drawComparison(const char* label, float measured, float current)
{
	const float changePercent= current > 1e-4f ? 100.f * (measured - current) / current : 0.f;
	ImGui::Text(locText("bodyWizard.comparisonFmt"), label, measured * 100.f, current * 100.f,
				changePercent);
}
} // namespace

BodyCalibrationWizard::BodyCalibrationWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

void BodyCalibrationWizard::enter()
{
	m_bActive= true;
	m_wizardResult= eWizardResult::None;
	m_state= eState::VerifyReady;
	m_cameraIndex= findBodyPoseCamera();
	m_calibrator.reset();
	m_result= BodyDimensionCalibrator::Result();
	m_bCollecting= false;
	m_countdownSeconds= 0.f;
	m_bLastSampleAccepted= false;
}

void BodyCalibrationWizard::exit()
{
	m_bActive= false;
	m_bCollecting= false;
}

int BodyCalibrationWizard::findBodyPoseCamera() const
{
	for (int cameraIndex= 0; cameraIndex < (int)m_config->cameraCount(); ++cameraIndex)
	{
		if (m_config->camera(cameraIndex).bodyPose.enabled)
			return cameraIndex;
	}
	return -1;
}

bool BodyCalibrationWizard::makeCameraFrame(const std::vector<VisionPreviewFrame>& previews,
											CameraFrameResult& outCamera) const
{
	if (m_cameraIndex < 0 || m_cameraIndex >= (int)previews.size() || !previews[m_cameraIndex].valid)
		return false;

	const CameraProfile& profile= m_config->camera(m_cameraIndex);
	if (!profile.intrinsics.present || !profile.extrinsics.present)
		return false;

	outCamera= CameraFrameResult();
	outCamera.cameraIndex= m_cameraIndex;
	outCamera.valid= true;
	outCamera.hasExtrinsics= true;
	outCamera.markerFromCamera= profile.extrinsics.markerFromCamera;
	outCamera.hasIntrinsics= true;

	const MikanMatrix3d& cameraMatrix= profile.intrinsics.intrinsics.undistorted_camera_matrix;
	outCamera.fx= (float)cameraMatrix.x0;
	outCamera.fy= (float)cameraMatrix.y1;
	outCamera.cx= (float)cameraMatrix.z0;
	outCamera.cy= (float)cameraMatrix.z1;
	outCamera.result= previews[m_cameraIndex].result;
	return true;
}

bool BodyCalibrationWizard::update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
								   const TrackingFrameResult& fusedResult)
{
	if (!m_bActive)
		return false;

	bool bKeepOpen= true;
	ImGui::SetNextWindowSize(ImVec2(560, 0), ImGuiCond_Appearing);
	if (ImGui::Begin(locWindowTitle("windows.bodyWizard"), nullptr, ImGuiWindowFlags_NoCollapse))
	{
		switch (m_state)
		{
		case eState::VerifyReady:
			bKeepOpen= drawVerifyStage(previews, fusedResult);
			break;
		case eState::FrontalPose:
			bKeepOpen= drawFrontalStage(deltaSeconds, previews, fusedResult);
			break;
		case eState::HeadTurn:
			bKeepOpen= drawHeadTurnStage(deltaSeconds, previews);
			break;
		case eState::Review:
			bKeepOpen= drawReviewStage();
			break;
		}
	}
	ImGui::End();

	if (!bKeepOpen)
		exit();
	return bKeepOpen;
}

bool BodyCalibrationWizard::drawVerifyStage(const std::vector<VisionPreviewFrame>& previews,
											const TrackingFrameResult& fusedResult)
{
	ImGui::TextWrapped("%s", locText("bodyWizard.verifyIntro"));
	ImGui::Separator();

	const bool bHaveCamera= m_cameraIndex >= 0;
	drawStatusLine(bHaveCamera, bHaveCamera ? locText("bodyWizard.cameraEnabledStatus")
											: locText("bodyWizard.cameraNotEnabledStatus"));

	CameraFrameResult camera;
	const bool bHaveCalibration= makeCameraFrame(previews, camera);
	drawStatusLine(bHaveCalibration, bHaveCalibration ? locText("bodyWizard.calibrationPresentStatus")
													  : locText("bodyWizard.calibrationMissingStatus"));

	const bool bHaveBody= bHaveCalibration && camera.result.body.valid;
	drawStatusLine(bHaveBody, bHaveBody ? locText("bodyWizard.bodyTrackingStatus")
										: locText("bodyWizard.bodyNotTrackingStatus"));

	const bool bBothHands= fusedResult.poses[0].tracked && fusedResult.poses[0].hasWorldPose &&
		fusedResult.poses[1].tracked && fusedResult.poses[1].hasWorldPose;
	drawStatusLine(bBothHands, bBothHands ? locText("bodyWizard.bothHandsTrackedStatus")
										  : locText("bodyWizard.bothHandsMissingStatus"));
	ImGui::TextDisabled("%s", locText("bodyWizard.fusedWristsNoteLine1"));
	ImGui::TextDisabled("%s", locText("bodyWizard.fusedWristsNoteLine2"));

	ImGui::Separator();
	const bool bReady= bHaveCamera && bHaveCalibration && bHaveBody && bBothHands;
	ImGui::BeginDisabled(!bReady);
	if (ImGui::Button(locLabel("bodyWizard.beginButton"), ImVec2(180, 0)))
	{
		m_calibrator.reset();
		m_state= eState::FrontalPose;
		m_countdownSeconds= k_countdownSeconds;
		m_bCollecting= false;
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		m_wizardResult= eWizardResult::Cancelled;
		return false;
	}
	return true;
}

bool BodyCalibrationWizard::drawFrontalStage(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
											 const TrackingFrameResult& fusedResult)
{
	ImGui::TextWrapped("%s", locText("bodyWizard.frontalRaiseHandInstructions"));
	ImGui::TextWrapped("%s", locText("bodyWizard.frontalOneHandEnoughInstructions"));
	ImGui::TextDisabled("%s", locText("bodyWizard.frontalExplainLine1"));
	ImGui::TextDisabled("%s", locText("bodyWizard.frontalExplainLine2"));
	ImGui::TextDisabled("%s", locText("bodyWizard.frontalExplainLine3"));
	ImGui::TextDisabled("%s", locText("bodyWizard.frontalExplainLine4"));
	ImGui::TextDisabled("%s", locText("bodyWizard.frontalExplainLine5"));
	ImGui::Separator();

	CameraFrameResult camera;
	const bool bHaveCamera= makeCameraFrame(previews, camera);

	if (m_countdownSeconds > 0.f)
	{
		m_countdownSeconds-= deltaSeconds;
		ImGui::TextColored(k_colorWait, locText("bodyWizard.frontalCountdownFmt"), std::ceil(m_countdownSeconds));
		if (m_countdownSeconds <= 0.f)
			m_bCollecting= true;
	}
	else if (m_bCollecting)
	{
		if (bHaveCamera)
		{
			BodyDimensionCalibrator::Sample sample;
			m_bLastSampleAccepted= m_calibrator.addFrontalSample(
				camera, fusedResult, makeBodyDimensions(*m_config), sample);
			if (m_bLastSampleAccepted)
				m_lastSample= sample;
		}

		ImGui::TextColored(k_colorGood, locText("bodyWizard.holdItSamplesFmt"), m_calibrator.getFrontalSampleCount());
		// The pose check, shown live: a forearm angled toward the camera
		// reads short, and this is the number that says so
		// Per hand, because either can carry a frame on its own
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const bool bAccepted=
				sideIndex == 0 ? m_lastSample.bAcceptedLeft : m_lastSample.bAcceptedRight;
			const float reach=
				sideIndex == 0 ? m_lastSample.handOffsetLeft : m_lastSample.handOffsetRight;

			const char* statusKey;
			if (bAccepted)
				statusKey= sideIndex == 0 ? "bodyWizard.leftCounting" : "bodyWizard.rightCounting";
			else if (reach <= 0.f)
				statusKey= sideIndex == 0 ? "bodyWizard.leftNotTracked" : "bodyWizard.rightNotTracked";
			else
				statusKey= sideIndex == 0 ? "bodyWizard.leftTooFar" : "bodyWizard.rightTooFar";
			ImGui::TextColored(bAccepted ? k_colorGood : k_colorWait, "%s", locText(statusKey));
			if (reach > 0.f)
				ImGui::TextDisabled(locText("bodyWizard.handOffsetFmt"), reach * 100.f,
									BodyDimensionCalibrator::k_maxRaisedHandOffset * 100.f);
		}

		const int needed= BodyDimensionCalibrator::k_minSamples;
		ImGui::ProgressBar(std::min(1.f, (float)m_calibrator.getFrontalSampleCount() / (float)(needed * 2)),
						   ImVec2(-1, 0));

		if (m_calibrator.getFrontalSampleCount() >= needed * 2)
		{
			m_bCollecting= false;
			m_state= eState::HeadTurn;
			m_countdownSeconds= k_countdownSeconds;
		}
	}

	ImGui::Separator();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		m_wizardResult= eWizardResult::Cancelled;
		return false;
	}
	return true;
}

bool BodyCalibrationWizard::drawHeadTurnStage(float deltaSeconds,
											  const std::vector<VisionPreviewFrame>& previews)
{
	ImGui::TextWrapped("%s", locText("bodyWizard.headTurnInstructions"));
	ImGui::TextDisabled("%s", locText("bodyWizard.headTurnExplainLine1"));
	ImGui::TextDisabled("%s", locText("bodyWizard.headTurnExplainLine2"));
	ImGui::TextDisabled("%s", locText("bodyWizard.headTurnExplainLine3"));
	ImGui::Separator();

	CameraFrameResult camera;
	const bool bHaveCamera= makeCameraFrame(previews, camera);

	if (m_countdownSeconds > 0.f)
	{
		m_countdownSeconds-= deltaSeconds;
		ImGui::TextColored(k_colorWait, locText("bodyWizard.headTurnCountdownFmt"), std::ceil(m_countdownSeconds));
		if (m_countdownSeconds <= 0.f)
			m_bCollecting= true;
	}
	else if (m_bCollecting)
	{
		float noseForward= 0.f;
		const bool bAccepted= bHaveCamera && m_calibrator.addHeadTurnSample(camera, noseForward);

		ImGui::TextColored(bAccepted ? k_colorGood : k_colorWait,
						   bAccepted ? locText("bodyWizard.headTurnSamplesFmt")
									 : locText("bodyWizard.headTurnSamplesNotAcceptedFmt"),
						   m_calibrator.getHeadTurnSampleCount());

		const int needed= BodyDimensionCalibrator::k_minSamples;
		ImGui::ProgressBar(std::min(1.f, (float)m_calibrator.getHeadTurnSampleCount() / (float)needed),
						   ImVec2(-1, 0));

		if (m_calibrator.getHeadTurnSampleCount() >= needed)
		{
			m_bCollecting= false;
			m_result= m_calibrator.solve(makeBodyDimensions(*m_config));
			m_state= eState::Review;
		}
	}

	ImGui::Separator();
	// The nose measurement is the least critical of the four: it only sets
	// head yaw and pitch, so skipping it must not cost the other three
	if (ImGui::Button(locLabel("bodyWizard.skipThisStepButton"), ImVec2(160, 0)))
	{
		m_bCollecting= false;
		m_result= m_calibrator.solve(makeBodyDimensions(*m_config));
		m_state= eState::Review;
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		m_wizardResult= eWizardResult::Cancelled;
		return false;
	}
	return true;
}

bool BodyCalibrationWizard::drawReviewStage()
{
	const BodyConfig& body= m_config->body;

	if (!m_result.bValid)
	{
		ImGui::TextColored(k_colorBad, locText("bodyWizard.notEnoughSamplesFmt"), m_result.sampleCount);
		ImGui::TextWrapped("%s", locText("bodyWizard.notEnoughSamplesExplain"));
		if (ImGui::Button(locLabel("common.close"), ImVec2(120, 0)))
		{
			m_wizardResult= eWizardResult::Cancelled;
			return false;
		}
		return true;
	}

	ImGui::Text(locText("bodyWizard.measuredFromSamplesFmt"), m_result.sampleCount);
	ImGui::Separator();
	drawComparison(locText("bodyWizard.shoulderWidthLabel"), m_result.shoulderWidth, body.shoulderWidthMeters);
	drawComparison(locText("bodyWizard.headWidthLabel"), m_result.headWidth, body.headWidthMeters);
	drawComparison(locText("bodyWizard.upperArmLabel"), m_result.upperArmLength, body.upperArmLengthMeters);
	if (m_result.bHaveNoseForward)
		drawComparison(locText("bodyWizard.noseForwardLabel"), m_result.noseForward, body.noseForwardMeters);
	else
		ImGui::TextDisabled(locText("bodyWizard.notMeasuredFmt"), locText("bodyWizard.noseForwardLabel"));

	ImGui::Separator();
	ImGui::TextDisabled(locText("bodyWizard.upperArmNoteLine1Fmt"),
						m_config->body.upperArmPerShoulderWidth);
	ImGui::TextDisabled("%s", locText("bodyWizard.upperArmNoteLine2"));
	ImGui::TextDisabled("%s", locText("bodyWizard.upperArmNoteLine3"));
	ImGui::TextDisabled(locText("bodyWizard.upperArmNoteLine4Fmt"));

	const float worstSpread=
		std::max(m_result.shoulderWidthSpread, m_result.headWidthSpread);
	if (worstSpread > 0.25f)
	{
		ImGui::TextColored(k_colorBad, locText("bodyWizard.sampleSpreadFmt"),
						   worstSpread * 100.f);
	}

	ImGui::Separator();
	if (ImGui::Button(locLabel("bodyWizard.acceptAndSaveButton"), ImVec2(180, 0)))
	{
		m_config->body.shoulderWidthMeters= m_result.shoulderWidth;
		m_config->body.headWidthMeters= m_result.headWidth;
		m_config->body.upperArmLengthMeters= m_result.upperArmLength; // kept in step for the manual path
		if (m_result.bHaveNoseForward)
			m_config->body.noseForwardMeters= m_result.noseForward;
		m_config->markDirty();
		m_visionThread->requestConfigRefresh();
		m_wizardResult= eWizardResult::Completed;
		return false;
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("bodyWizard.recaptureButton"), ImVec2(140, 0)))
	{
		m_calibrator.reset();
		m_state= eState::FrontalPose;
		m_countdownSeconds= k_countdownSeconds;
		m_bCollecting= false;
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("bodyWizard.discardButton"), ImVec2(120, 0)))
	{
		m_wizardResult= eWizardResult::Cancelled;
		return false;
	}
	return true;
}
