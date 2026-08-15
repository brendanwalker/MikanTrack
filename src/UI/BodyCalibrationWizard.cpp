#include "BodyCalibrationWizard.h"

#include <algorithm>

#include "imgui.h"

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
	ImGui::Text("%-14s %5.1f cm   (was %5.1f cm, %+.0f%%)", label, measured * 100.f, current * 100.f,
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
	if (ImGui::Begin("Body Measurements", nullptr, ImGuiWindowFlags_NoCollapse))
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
	ImGui::TextWrapped(
		"Measures the body lengths the elbow, shoulder and head estimates rest on. "
		"These are not anatomical numbers: they are distances between the pose "
		"model's own landmarks, which sit inside your real joints by an amount "
		"that differs per person and per model, so they have to be measured "
		"rather than assumed.");
	ImGui::Separator();

	const bool bHaveCamera= m_cameraIndex >= 0;
	drawStatusLine(bHaveCamera, bHaveCamera ? "Body pose enabled on a camera" : "No camera has body pose enabled");

	CameraFrameResult camera;
	const bool bHaveCalibration= makeCameraFrame(previews, camera);
	drawStatusLine(bHaveCalibration, bHaveCalibration ? "Camera intrinsics + extrinsics calibrated"
													  : "That camera needs intrinsics AND extrinsics first");

	const bool bHaveBody= bHaveCalibration && camera.result.body.valid;
	drawStatusLine(bHaveBody, bHaveBody ? "Body landmarks tracking" : "Body landmarks not tracking");

	const bool bBothHands= fusedResult.poses[0].tracked && fusedResult.poses[0].hasWorldPose &&
		fusedResult.poses[1].tracked && fusedResult.poses[1].hasWorldPose;
	drawStatusLine(bBothHands, bBothHands ? "Both hands tracked" : "Both hands must be tracked");
	ImGui::TextDisabled("  The fused wrists are the ruler: they are the only metric,");
	ImGui::TextDisabled("  world-anchored points, measured by the other cameras.");

	ImGui::Separator();
	const bool bReady= bHaveCamera && bHaveCalibration && bHaveBody && bBothHands;
	ImGui::BeginDisabled(!bReady);
	if (ImGui::Button("Begin", ImVec2(180, 0)))
	{
		m_calibrator.reset();
		m_state= eState::FrontalPose;
		m_countdownSeconds= k_countdownSeconds;
		m_bCollecting= false;
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button("Cancel", ImVec2(120, 0)))
		return false;
	return true;
}

bool BodyCalibrationWizard::drawFrontalStage(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
											 const TrackingFrameResult& fusedResult)
{
	ImGui::TextWrapped(
		"Face the camera and simply RAISE ONE HAND up beside your shoulder, as "
		"though waving. Keep it near your shoulder rather than reaching out.");
	ImGui::TextWrapped(
		"ONE HAND IS ENOUGH, and either will do - hold one up, let it count, and "
		"you are done. The other can stay on the desk.");
	ImGui::TextDisabled("A raised hand puts a measured wrist at about the distance your");
	ImGui::TextDisabled("torso is from the camera, and that is all the shoulders and ears");
	ImGui::TextDisabled("need in order to be measured. Nothing has to be straight: the");
	ImGui::TextDisabled("upper arm is worked out from your shoulder width instead, because");
	ImGui::TextDisabled("measuring it directly needed a pose that was too hard to hold.");
	ImGui::Separator();

	CameraFrameResult camera;
	const bool bHaveCamera= makeCameraFrame(previews, camera);

	if (m_countdownSeconds > 0.f)
	{
		m_countdownSeconds-= deltaSeconds;
		ImGui::TextColored(k_colorWait, "Get into position... %.0f", std::ceil(m_countdownSeconds));
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

		ImGui::TextColored(k_colorGood, "Hold it - %d samples", m_calibrator.getFrontalSampleCount());
		// The pose check, shown live: a forearm angled toward the camera
		// reads short, and this is the number that says so
		// Per hand, because either can carry a frame on its own
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const bool bAccepted=
				sideIndex == 0 ? m_lastSample.bAcceptedLeft : m_lastSample.bAcceptedRight;
			const float reach=
				sideIndex == 0 ? m_lastSample.handOffsetLeft : m_lastSample.handOffsetRight;

			const char* reason= "";
			if (!bAccepted)
			{
				reason= reach <= 0.f ? " - not tracked" : " - too far from your shoulder, raise it";
			}
			ImGui::TextColored(bAccepted ? k_colorGood : k_colorWait, "%s hand: %s%s",
							   sideIndex == 0 ? "Left " : "Right",
							   bAccepted ? "counting" : "not counting", reason);
			if (reach > 0.f)
				ImGui::TextDisabled("   %.0f%% of a shoulder width away (needs under %.0f%%)", reach * 100.f,
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
	if (ImGui::Button("Cancel", ImVec2(120, 0)))
		return false;
	return true;
}

bool BodyCalibrationWizard::drawHeadTurnStage(float deltaSeconds,
											  const std::vector<VisionPreviewFrame>& previews)
{
	ImGui::TextWrapped(
		"Now keep your body still and slowly turn your head to look left, then "
		"right, as though checking both ends of your desk.");
	ImGui::TextDisabled("Facing the camera, your nose sits on top of the midpoint between");
	ImGui::TextDisabled("your ears and says nothing about how far it juts forward. Turning");
	ImGui::TextDisabled("swings that offset across the image, where it can be measured.");
	ImGui::Separator();

	CameraFrameResult camera;
	const bool bHaveCamera= makeCameraFrame(previews, camera);

	if (m_countdownSeconds > 0.f)
	{
		m_countdownSeconds-= deltaSeconds;
		ImGui::TextColored(k_colorWait, "Starting... %.0f", std::ceil(m_countdownSeconds));
		if (m_countdownSeconds <= 0.f)
			m_bCollecting= true;
	}
	else if (m_bCollecting)
	{
		float noseForward= 0.f;
		const bool bAccepted= bHaveCamera && m_calibrator.addHeadTurnSample(camera, noseForward);

		ImGui::TextColored(bAccepted ? k_colorGood : k_colorWait, "%d samples%s",
						   m_calibrator.getHeadTurnSampleCount(),
						   bAccepted ? "" : " - turn further, the head still reads face-on");

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
	if (ImGui::Button("Skip this step", ImVec2(160, 0)))
	{
		m_bCollecting= false;
		m_result= m_calibrator.solve(makeBodyDimensions(*m_config));
		m_state= eState::Review;
	}
	ImGui::SameLine();
	if (ImGui::Button("Cancel", ImVec2(120, 0)))
		return false;
	return true;
}

bool BodyCalibrationWizard::drawReviewStage()
{
	const BodyConfig& body= m_config->body;

	if (!m_result.bValid)
	{
		ImGui::TextColored(k_colorBad, "Not enough usable samples (%d).", m_result.sampleCount);
		ImGui::TextWrapped(
			"Every sample needs both hands tracked, the shoulders, elbows and ears "
			"visible, and the arms held across the camera rather than toward it.");
		if (ImGui::Button("Close", ImVec2(120, 0)))
			return false;
		return true;
	}

	ImGui::Text("Measured from %d samples:", m_result.sampleCount);
	ImGui::Separator();
	drawComparison("Shoulders", m_result.shoulderWidth, body.shoulderWidthMeters);
	drawComparison("Head width", m_result.headWidth, body.headWidthMeters);
	drawComparison("Upper arm", m_result.upperArmLength, body.upperArmLengthMeters);
	if (m_result.bHaveNoseForward)
		drawComparison("Nose forward", m_result.noseForward, body.noseForwardMeters);
	else
		ImGui::TextDisabled("%-14s not measured (step skipped)", "Nose forward");

	ImGui::Separator();
	ImGui::TextDisabled("The upper arm is not measured: it is taken as %.2f x the shoulder",
						m_config->body.upperArmPerShoulderWidth);
	ImGui::TextDisabled("width, which only needs a width that is easy to measure. Measuring");
	ImGui::TextDisabled("it directly needed the arm straight and square to the camera, and");
	ImGui::TextDisabled("a missed pose read it 20%% short - enough to bend the elbow wrongly.");

	const float worstSpread=
		std::max(m_result.shoulderWidthSpread, m_result.headWidthSpread);
	if (worstSpread > 0.25f)
	{
		ImGui::TextColored(k_colorBad, "Sample spread %.0f%% - hold stiller for a cleaner result.",
						   worstSpread * 100.f);
	}

	ImGui::Separator();
	if (ImGui::Button("Accept & Save", ImVec2(180, 0)))
	{
		m_config->body.shoulderWidthMeters= m_result.shoulderWidth;
		m_config->body.headWidthMeters= m_result.headWidth;
		m_config->body.upperArmLengthMeters= m_result.upperArmLength; // kept in step for the manual path
		if (m_result.bHaveNoseForward)
			m_config->body.noseForwardMeters= m_result.noseForward;
		m_config->markDirty();
		m_visionThread->requestConfigRefresh();
		return false;
	}
	ImGui::SameLine();
	if (ImGui::Button("Recapture", ImVec2(140, 0)))
	{
		m_calibrator.reset();
		m_state= eState::FrontalPose;
		m_countdownSeconds= k_countdownSeconds;
		m_bCollecting= false;
	}
	ImGui::SameLine();
	if (ImGui::Button("Discard", ImVec2(120, 0)))
		return false;
	return true;
}
