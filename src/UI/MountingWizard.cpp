#include "MountingWizard.h"

#include <algorithm>
#include <cmath>

#include "imgui.h"

#include "AppConfig.h"
#include "ImuService.h"
#include "TrackingTypes.h"
#include "VisionThread.h"

namespace
{
constexpr float k_holdCountdownSeconds= 3.f;
// A margin over ImuService's own capture gate, so a capture is never taken
// right on the boundary
constexpr float k_twistReadyDominance= 0.8f;
constexpr float k_twistReadyReversal= 0.6f;

const char* sideName(int sideIndex)
{
	return sideIndex == 0 ? "Left" : "Right";
}

void drawStatusLine(bool bOk, const char* text)
{
	const ImVec4 color= bOk ? ImVec4(0.4f, 1.f, 0.4f, 1.f) : ImVec4(1.f, 0.6f, 0.3f, 1.f);
	ImGui::TextColored(color, "%s %s", bOk ? "[ok]" : "[..]", text);
}

float quaternionAngleDegrees(const glm::quat& q)
{
	const float axisLength= std::min(1.f, glm::length(glm::vec3(q.x, q.y, q.z)));
	return glm::degrees(2.f * asinf(axisLength));
}
} // namespace

MountingWizard::MountingWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

void MountingWizard::enter()
{
	m_bActive= true;
	m_bWantsClose= false;
	m_state= eState::VerifyDevices;
	m_bCaptureRequested= false;
	m_holdCountdown= 0.f;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_bParticipating[sideIndex]= false;
		m_bTwistReady[sideIndex]= false;
		m_bAccepted[sideIndex]= false;
		m_captured[sideIndex]= MountingCaptureResult();
	}
}

void MountingWizard::exit()
{
	// Leaving mid-measurement must not strand the service collecting forever
	m_visionThread->cancelImuBiasCalibration();
	m_bActive= false;
}

bool MountingWizard::isSideParticipating(int sideIndex) const
{
	return m_bParticipating[sideIndex];
}

void MountingWizard::beginTwistStage()
{
	// Start from a clean slate: whatever the arms happened to be doing before
	// the wizard opened is not part of this measurement
	m_visionThread->requestImuMotionReset();
	m_epochAtReset= m_visionThread->getImuSideStatus(eHandSide::Left).motionEpoch;
	m_bWaitingForMotionReset= true;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		m_bTwistReady[sideIndex]= false;
	m_state= eState::TwistForearms;
}

bool MountingWizard::update(float deltaSeconds, const TrackingFrameResult& fusedResult)
{
	if (!m_bActive)
		return false;

	const ImuSideStatus status[2]= {
		m_visionThread->getImuSideStatus(eHandSide::Left),
		m_visionThread->getImuSideStatus(eHandSide::Right),
	};

	ImGui::SetNextWindowSize(ImVec2(460, 0), ImGuiCond_Appearing);
	if (!ImGui::Begin("Wrist IMU Mounting Calibration", nullptr, ImGuiWindowFlags_NoCollapse))
	{
		ImGui::End();
		return true;
	}

	switch (m_state)
	{
		case eState::VerifyDevices:
		{
			ImGui::TextWrapped(
				"This measures how each controller is strapped to your wrist, so the app "
				"can turn a sensor orientation into a FOREARM orientation. It takes two "
				"steps: twisting your forearms (which measures the arm axis) and then "
				"holding them straight (which sets the roll about that axis).");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"Strap each controller to the top of the wrist, pointing along the "
				"forearm. The exact orientation does not matter - that is what this "
				"calibration absorbs - but it must not shift afterwards.");
			ImGui::Spacing();

			bool bAnyReady= false;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const ImuSideStatus& sideStatus= status[sideIndex];
				const bool bReady= sideStatus.streaming && sideStatus.filterConverged;
				bAnyReady|= bReady;

				ImGui::SeparatorText(sideName(sideIndex));
				if (!sideStatus.deviceConnected)
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "  No controller assigned");
					continue;
				}
				ImGui::Text("  %s", sideStatus.deviceName.c_str());
				drawStatusLine(sideStatus.streaming, "streaming");
				drawStatusLine(sideStatus.filterConverged,
							   sideStatus.filterConverged ? "orientation settled"
														  : "orientation settling - hold it still");
			}

			ImGui::Spacing();
			if (!bAnyReady)
			{
				ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
								   "Waiting for at least one controller to stream and settle");
			}

			ImGui::BeginDisabled(!bAnyReady);
			// Lock in who takes part NOW - a side that is not usable at the
			// start must not silently join half way and get calibrated against
			// motion it was not present for
			auto lockParticipants= [this, &status]() {
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					m_bParticipating[sideIndex]= status[sideIndex].streaming && status[sideIndex].filterConverged;
			};
			if (ImGui::Button("Start", ImVec2(180, 0)))
			{
				lockParticipants();
				m_visionThread->requestImuBiasCalibration();
				m_state= eState::CalibrateBias;
			}
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;
			break;
		}

		case eState::CalibrateBias:
		{
			ImGui::TextWrapped(
				"Step 1 of 3: REST. Take the controllers off and lay them flat on the "
				"desk. Do not touch them until this finishes.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"A gyro reads a small nonzero rate even when perfectly still, and that "
				"offset integrates into drift. Sitting still is the one situation where "
				"the true rate is known to be zero, so the reading IS the error - "
				"including about the vertical axis, which nothing else in the system can "
				"measure.");
			ImGui::Spacing();

			bool bAllDone= true;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;

				const float progress= status[sideIndex].biasCalibrationProgress;
				const bool bDone= progress < 0.f;
				bAllDone&= bDone;

				ImGui::Text("%-6s", sideName(sideIndex));
				ImGui::SameLine();
				if (bDone)
				{
					ImGui::ProgressBar(1.f, ImVec2(-90, 0), "measured");
					ImGui::SameLine();
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "done");
				}
				else
				{
					ImGui::ProgressBar(progress, ImVec2(-90, 0),
									   status[sideIndex].biasCalibrationDisturbed ? "restarted" : "measuring");
					ImGui::SameLine();
					ImGui::TextDisabled("%.0f%%", progress * 100.f);
				}
			}

			if (bAllDone)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "Put the controllers back on your wrists.");
				ImGui::Spacing();
				if (ImGui::Button("Continue", ImVec2(180, 0)))
					beginTwistStage();
			}
			else
			{
				ImGui::TextDisabled("Any movement restarts the measurement.");
				ImGui::Spacing();
				if (ImGui::Button("Skip", ImVec2(180, 0)))
				{
					// The filter estimates the bias online anyway - just not
					// about the vertical axis, which is the one that drifts
					m_visionThread->cancelImuBiasCalibration();
					beginTwistStage();
				}
				ImGui::SetItemTooltip(
					"The mounting calibration still works; yaw will just drift\n"
					"faster and lean harder on the camera to correct it.");
			}
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;
			break;
		}

		case eState::TwistForearms:
		{
			ImGui::TextWrapped(
				"Step 2 of 3: TWIST. Rotate each forearm as if slowly turning a doorknob "
				"- palm up, palm down, and back again, several times. Keep your elbows "
				"still and your wrists relaxed; it is the twist that is being measured.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"Turning one way only does not count, and neither does waving your arms "
				"around - the bar needs back-and-forth rotation about the forearm itself. "
				"Your hands do not need to be visible to the cameras during this step.");
			ImGui::Spacing();

			// The reset is serviced on the vision thread; until it lands, the
			// status still describes the previous session
			if (m_bWaitingForMotionReset &&
				status[0].motionEpoch != m_epochAtReset && status[1].motionEpoch != m_epochAtReset)
			{
				m_bWaitingForMotionReset= false;
			}

			bool bAllReady= true;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;

				const ImuSideStatus& sideStatus= status[sideIndex];
				const float dominance= std::max(0.f, sideStatus.armAxisDominance);
				if (!m_bWaitingForMotionReset && sideStatus.twistProgress >= 1.f &&
					sideStatus.twistReversal >= k_twistReadyReversal && dominance >= k_twistReadyDominance)
				{
					m_bTwistReady[sideIndex]= true;
				}
				bAllReady&= m_bTwistReady[sideIndex];

				ImGui::Text("%-6s", sideName(sideIndex));
				ImGui::SameLine();
				const float shownProgress= m_bWaitingForMotionReset ? 0.f : sideStatus.twistProgress;
				ImGui::ProgressBar(shownProgress, ImVec2(-90, 0),
								   m_bTwistReady[sideIndex] ? "ready" : "keep twisting");
				ImGui::SameLine();
				if (m_bTwistReady[sideIndex])
				{
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "done");
				}
				else
				{
					ImGui::TextDisabled("%.0f%%", shownProgress * 100.f);
					// Once there IS enough motion but it still does not qualify,
					// say which way it is wrong - the two need opposite fixes
					if (!m_bWaitingForMotionReset && sideStatus.twistProgress >= 1.f)
					{
						if (sideStatus.twistReversal < k_twistReadyReversal)
							ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
											   "  %s: rotate back the other way too", sideName(sideIndex));
						else if (dominance < k_twistReadyDominance)
							ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
											   "  %s: keep the elbow still - twist only",
											   sideName(sideIndex));
					}
				}

				if (sideStatus.biasSaturated)
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  %s: this controller's orientation filter has diverged.\n"
									   "  Set it down, redo the rest step, and try again.",
									   sideName(sideIndex));
				}
			}

			if (bAllReady)
				m_state= eState::HoldStraight;

			ImGui::Spacing();
			ImGui::BeginDisabled(!m_bTwistReady[0] && !m_bTwistReady[1]);
			if (ImGui::Button("Continue", ImVec2(180, 0)))
				m_state= eState::HoldStraight;
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;
			break;
		}

		case eState::HoldStraight:
		{
			ImGui::TextWrapped(
				"Step 3 of 3: HOLD STRAIGHT. Hold each hand flat and in line with its "
				"forearm - no bend at the wrist, as if your hand and forearm were one "
				"board - where the cameras can see it. Then capture.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"This pose is what defines a straight wrist as zero, so any bend you hold "
				"here is baked in as the new neutral.");
			ImGui::Spacing();

			bool bAllTracked= true;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;
				const HandPose& pose= fusedResult.poses[sideIndex];
				const bool bTracked= pose.tracked && pose.hasWorldPose;
				bAllTracked&= bTracked;

				char label[64];
				snprintf(label, sizeof(label), "%s hand visible to the cameras", sideName(sideIndex));
				drawStatusLine(bTracked, label);
			}

			ImGui::Spacing();
			if (m_holdCountdown > 0.f)
			{
				m_holdCountdown-= deltaSeconds;
				if (m_holdCountdown <= 0.f)
				{
					m_holdCountdown= 0.f;
					m_visionThread->requestImuMountingCapture();
					m_bCaptureRequested= true;
				}
				else
				{
					char countdownText[16];
					snprintf(countdownText, sizeof(countdownText), "%d", (int)ceilf(m_holdCountdown));
					ImGui::SetWindowFontScale(3.f);
					const float textWidth= ImGui::CalcTextSize(countdownText).x;
					ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "%s", countdownText);
					ImGui::SetWindowFontScale(1.f);
					if (ImGui::Button("Cancel countdown", ImVec2(180, 0)))
						m_holdCountdown= 0.f;
				}
			}
			else if (!m_bCaptureRequested)
			{
				ImGui::BeginDisabled(!bAllTracked);
				if (ImGui::Button("Capture", ImVec2(180, 0)))
					m_holdCountdown= k_holdCountdownSeconds;
				ImGui::EndDisabled();
				if (!bAllTracked)
					ImGui::TextDisabled("Both participating hands must be tracked");
				ImGui::SameLine();
				if (ImGui::Button("Back", ImVec2(120, 0)))
					beginTwistStage();
			}

			// The capture is serviced on the vision thread; collect the result
			if (m_bCaptureRequested)
			{
				VisionThread::ImuMountingCapture capture;
				if (m_visionThread->fetchImuMountingCapture(capture))
				{
					m_bCaptureRequested= false;
					bool bAnyChange= false;
					for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					{
						m_captured[sideIndex]= capture.sides[sideIndex];
						m_bAccepted[sideIndex]=
							capture.sides[sideIndex].bCaptured && capture.sides[sideIndex].bMotionUsable;
						if (!m_bAccepted[sideIndex])
							continue;

						m_config->imu.forearmToSensor[sideIndex]= capture.sides[sideIndex].forearmToSensor;
						m_config->imu.mountingPresent[sideIndex]= true;
						bAnyChange= true;
					}
					if (bAnyChange)
					{
						m_config->markDirty();
						m_visionThread->requestConfigRefresh();
					}
					m_state= eState::Review;
				}
			}
			break;
		}

		case eState::Review:
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;

				const MountingCaptureResult& result= m_captured[sideIndex];
				ImGui::SeparatorText(sideName(sideIndex));
				if (m_bAccepted[sideIndex])
				{
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "  Calibrated (arm axis %.2f)",
									   result.axisDominance);
				}
				else if (!result.bCaptured)
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Not captured - the hand was not tracked, or the\n"
									   "  controller stopped streaming");
				}
				else
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Rejected - the twist was not good enough to locate\n"
									   "  the forearm axis (amount %.0f%%, back-and-forth %.2f,\n"
									   "  single-axis %.2f). Nothing was saved for this side.",
									   result.twistProgress * 100.f, result.twistReversal,
									   result.axisDominance);
				}
			}

			ImGui::Spacing();
			ImGui::SeparatorText("Check it");
			ImGui::TextWrapped(
				"Keep your hands straight in line with your forearms. The wrist bend below "
				"should read near zero, and should stay near zero as you twist your "
				"forearms. If it climbs while you twist, the mounting is still wrong.");
			ImGui::Spacing();

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;
				const HandPose& pose= fusedResult.poses[sideIndex];
				if (!pose.tracked || !pose.hasWorldPose || !pose.hasForearmPose)
				{
					ImGui::TextDisabled("  %s: no reading (hand not tracked)", sideName(sideIndex));
					continue;
				}
				const float bendDegrees= quaternionAngleDegrees(pose.getWristRotation());
				const ImVec4 color= bendDegrees < 15.f ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
													   : ImVec4(1.f, 0.85f, 0.3f, 1.f);
				ImGui::TextColored(color, "  %s wrist bend: %.0f deg", sideName(sideIndex), bendDegrees);
			}

			ImGui::Spacing();
			if (ImGui::Button("Finish", ImVec2(180, 0)))
				m_bWantsClose= true;
			ImGui::SameLine();
			if (ImGui::Button("Redo", ImVec2(120, 0)))
			{
				m_bCaptureRequested= false;
				beginTwistStage();
			}
			break;
		}
	}

	ImGui::End();
	return !m_bWantsClose;
}
