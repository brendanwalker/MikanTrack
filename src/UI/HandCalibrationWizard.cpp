#include "HandCalibrationWizard.h"

#include "imgui.h"

#include "AppConfig.h"

namespace
{
constexpr float k_countdownSeconds= 3.f;
constexpr float k_boneSampleSeconds= 10.f;
// Per-bone spread above this suggests the capture window was unlucky (bones
// aligned with view rays) and a retry is worth it
constexpr float k_boneSpreadWarnMm= 4.f;

void drawCenteredCountdown(float remainingSeconds, float totalSeconds)
{
	char countdownText[16];
	snprintf(countdownText, sizeof(countdownText), "%d", (int)ceilf(remainingSeconds));
	ImGui::SetWindowFontScale(3.f);
	const float textWidth= ImGui::CalcTextSize(countdownText).x;
	ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
	ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "%s", countdownText);
	ImGui::SetWindowFontScale(1.f);
	ImGui::ProgressBar(1.f - remainingSeconds / totalSeconds, ImVec2(-1, 4), "");
}

// Both sides stereo-quality right now (what both stages need to succeed)
void stereoStatus(const TrackingFrameResult& fused, bool outStereo[2])
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= fused.poses[sideIndex];
		outStereo[sideIndex]= pose.tracked && pose.hasWorldPose && pose.stereoTriangulated;
	}
}

void drawStereoStatusLine(const TrackingFrameResult& fused)
{
	bool bStereo[2];
	stereoStatus(fused, bStereo);
	ImGui::Text("Stereo tracking  L:");
	ImGui::SameLine();
	ImGui::TextColored(bStereo[0] ? ImVec4(0.4f, 1.f, 0.4f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f),
					   bStereo[0] ? "yes" : "no");
	ImGui::SameLine();
	ImGui::Text(" R:");
	ImGui::SameLine();
	ImGui::TextColored(bStereo[1] ? ImVec4(0.4f, 1.f, 0.4f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f),
					   bStereo[1] ? "yes" : "no");
}
} // namespace

HandCalibrationWizard::HandCalibrationWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

void HandCalibrationWizard::enter()
{
	m_bActive= true;
	m_bWantsClose= false;
	m_result= eWizardResult::None;
	m_state= eState::BonesIntro;
	m_countdown= 0.f;
	m_boneCapture= VisionThread::BoneCalibrationCapture();
	m_restCapture= VisionThread::RestPoseCapture();
}

void HandCalibrationWizard::exit()
{
	m_visionThread->cancelBoneCalibration();
	m_bActive= false;
}

bool HandCalibrationWizard::update(float deltaSeconds, const TrackingFrameResult& fusedResult)
{
	if (!m_bActive)
		return false;

	ImGui::SetNextWindowSize(ImVec2(560, 0), ImGuiCond_Appearing);
	if (ImGui::Begin("Hand Calibration", nullptr, ImGuiWindowFlags_NoCollapse))
	{
		switch (m_state)
		{
		case eState::BonesIntro: drawBonesIntro(fusedResult); break;
		case eState::BonesCountdown: drawBonesCountdown(deltaSeconds); break;
		case eState::BonesSampling: drawBonesSampling(); break;
		case eState::BonesReview: drawBonesReview(); break;
		case eState::RestIntro: drawRestIntro(fusedResult); break;
		case eState::RestCountdown: drawRestCountdown(deltaSeconds); break;
		case eState::RestResult: drawRestResult(); break;
		}

		ImGui::Separator();
		if (ImGui::Button("Cancel Calibration"))
		{
			m_result= eWizardResult::Cancelled;
			m_bWantsClose= true;
		}
	}
	ImGui::End();

	return !m_bWantsClose;
}

void HandCalibrationWizard::drawBonesIntro(const TrackingFrameResult& fusedResult)
{
	ImGui::TextWrapped(
		"Step 1 of 2: measure your hand bones.\n\n"
		"The landmark model's metric hand is not your hand - its proximal "
		"phalanges run about half length - which biases the single-camera "
		"depth solve and ships the wrong skeleton to clients. This samples "
		"every stereo-triangulated frame for %d seconds and takes the median "
		"of each bone.\n\n"
		"During the window: keep BOTH hands where two cameras see them, and "
		"move them through varied poses and rotations, so no bone stays "
		"aligned with a camera's view ray.",
		(int)k_boneSampleSeconds);
	ImGui::Spacing();
	drawStereoStatusLine(fusedResult);
	ImGui::Spacing();

	if (ImGui::Button("Start Bone Measurement", ImVec2(-1, 0)))
	{
		m_countdown= k_countdownSeconds;
		m_state= eState::BonesCountdown;
	}
}

void HandCalibrationWizard::drawBonesCountdown(float deltaSeconds)
{
	m_countdown-= deltaSeconds;
	if (m_countdown <= 0.f)
	{
		m_visionThread->requestBoneCalibration(k_boneSampleSeconds);
		m_state= eState::BonesSampling;
		return;
	}
	ImGui::TextWrapped("Get both hands into view of two cameras...");
	drawCenteredCountdown(m_countdown, k_countdownSeconds);
}

void HandCalibrationWizard::drawBonesSampling()
{
	bool bSamplingActive= false;
	int liveSamples[2]= {0, 0};
	m_visionThread->getBoneCalibrationProgress(bSamplingActive, liveSamples[0], liveSamples[1]);

	ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Sampling - move both hands");
	ImGui::Text("Samples  L: %d  R: %d (need %d)", liveSamples[0], liveSamples[1],
				HandBoneCalibrator::k_minSamples);

	if (m_visionThread->fetchBoneCalibration(m_boneCapture))
	{
		m_state= eState::BonesReview;
		return;
	}

	if (ImGui::Button("Cancel Sampling", ImVec2(-1, 0)))
	{
		m_visionThread->cancelBoneCalibration();
		m_state= eState::BonesIntro;
	}
}

void HandCalibrationWizard::drawBonesReview()
{
	// Review before anything is written: this replaces the geometry every
	// client rebuilds the hand from, so it gets looked at first
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const char* sideName= sideIndex == 0 ? "Left" : "Right";
		if (!m_boneCapture.bCaptured[sideIndex])
		{
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s: only %d stereo samples, need %d",
							   sideName, m_boneCapture.quality[sideIndex].sampleCount,
							   HandBoneCalibrator::k_minSamples);
			continue;
		}

		const HandSkeleton& skeleton= m_boneCapture.skeleton[sideIndex];
		const float referenceBone= skeleton.baseInPalm[(int)eFinger::Middle].x * 2.f;
		const float spreadMm= m_boneCapture.quality[sideIndex].worstPhalanxSpread * 1000.f;
		const ImVec4 spreadColor= spreadMm > k_boneSpreadWarnMm ? ImVec4(1.f, 0.85f, 0.3f, 1.f)
																: ImVec4(0.4f, 1.f, 0.4f, 1.f);
		ImGui::TextColored(spreadColor, "%s: wrist->knuckle %.1f mm, %d samples, worst spread %.1f mm",
						   sideName, referenceBone * 1000.f,
						   m_boneCapture.quality[sideIndex].sampleCount, spreadMm);

		// Proximal phalanxes only: they are the longest bones and where the
		// model is most wrong, so they are what a glance should check
		ImGui::Text("   proximal  index %.1f  middle %.1f  ring %.1f  pinky %.1f mm",
					skeleton.phalanxLengths[(int)eFinger::Index][0] * 1000.f,
					skeleton.phalanxLengths[(int)eFinger::Middle][0] * 1000.f,
					skeleton.phalanxLengths[(int)eFinger::Ring][0] * 1000.f,
					skeleton.phalanxLengths[(int)eFinger::Pinky][0] * 1000.f);
	}

	const bool bAnyCaptured= m_boneCapture.bCaptured[0] || m_boneCapture.bCaptured[1];
	ImGui::Spacing();
	ImGui::BeginDisabled(!bAnyCaptured);
	if (ImGui::Button("Accept and Continue", ImVec2(-1, 0)))
	{
		saveBones();
		m_state= eState::RestIntro;
	}
	ImGui::EndDisabled();
	if (ImGui::Button("Retry Measurement", ImVec2(-1, 0)))
		m_state= eState::BonesIntro;
}

void HandCalibrationWizard::drawRestIntro(const TrackingFrameResult& fusedResult)
{
	ImGui::TextWrapped(
		"Step 2 of 2: capture your rest pose.\n\n"
		"This pose becomes the all-zero-angles reference. Without it, zero "
		"means an idealized flat hand, and a hand hovering over a keyboard "
		"genuinely holds tens of degrees of knuckle flexion.\n\n"
		"It is captured NOW, after the bone save, on purpose: measured bones "
		"move the thumb's angle zero, so a rest pose captured before them "
		"would be measured against a zero that no longer exists.\n\n"
		"Hold both hands in your rest pose - flat, fingers together and "
		"straight - where two cameras see them.");
	ImGui::Spacing();
	drawStereoStatusLine(fusedResult);
	ImGui::Spacing();

	if (ImGui::Button("Capture Rest Pose", ImVec2(-1, 0)))
	{
		m_countdown= k_countdownSeconds;
		m_state= eState::RestCountdown;
	}
}

void HandCalibrationWizard::drawRestCountdown(float deltaSeconds)
{
	m_countdown-= deltaSeconds;
	if (m_countdown <= 0.f)
	{
		m_visionThread->requestRestPoseCapture();
		m_state= eState::RestResult;
		return;
	}
	ImGui::TextWrapped("Hold your rest pose...");
	drawCenteredCountdown(m_countdown, k_countdownSeconds);
}

void HandCalibrationWizard::drawRestResult()
{
	// The capture is serviced by the vision thread within a fuse; poll for it
	VisionThread::RestPoseCapture capture;
	if (m_visionThread->fetchRestPoseCapture(capture))
	{
		m_restCapture= capture;
		saveRest();
	}

	const bool bLeft= m_restCapture.bCaptured[0];
	const bool bRight= m_restCapture.bCaptured[1];
	if (bLeft && bRight)
		ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "Captured both hands");
	else if (bLeft || bRight)
		ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
						   "Captured %s hand only - the %s hand was not stereo-tracked",
						   bLeft ? "left" : "right", bLeft ? "right" : "left");
	else
		ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
						   "Nothing captured - both hands need two cameras on them");

	ImGui::Spacing();
	if (ImGui::Button("Retry Rest Pose", ImVec2(-1, 0)))
		m_state= eState::RestIntro;
	ImGui::BeginDisabled(!bLeft && !bRight);
	if (ImGui::Button("Done", ImVec2(-1, 0)))
	{
		m_result= eWizardResult::Completed;
		m_bWantsClose= true;
	}
	ImGui::EndDisabled();
}

void HandCalibrationWizard::saveBones()
{
	double referenceSum= 0.0;
	int referenceCount= 0;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!m_boneCapture.bCaptured[sideIndex])
			continue;
		m_config->handSkeleton.skeleton[sideIndex]= m_boneCapture.skeleton[sideIndex];
		m_config->handSkeleton.present[sideIndex]= true;
		referenceSum+= m_boneCapture.skeleton[sideIndex].baseInPalm[(int)eFinger::Middle].x * 2.0;
		++referenceCount;
	}

	// One scale story: the reference bone the rest of the app reads now
	// comes from the measurement, not the EMA
	if (referenceCount > 0)
	{
		m_config->handScale.refLengthMeters= referenceSum / (double)referenceCount;
		m_config->handScale.present= true;
	}

	m_config->save();
	m_visionThread->requestConfigRefresh();
}

void HandCalibrationWizard::saveRest()
{
	bool bAnySide= false;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!m_restCapture.bCaptured[sideIndex])
			continue;
		m_config->fusedRestAngles.angles[sideIndex]= m_restCapture.angles[sideIndex];
		m_config->fusedRestAngles.present[sideIndex]= true;
		bAnySide= true;
	}
	if (!bAnySide)
		return;

	m_config->save();
	m_visionThread->requestConfigRefresh();
}
