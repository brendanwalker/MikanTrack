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
// Below this the measured arm axis is not trustworthy and the capture is
// refused. Matches the gate the capture itself documents.
constexpr float k_minAxisDominance= 0.7f;
// Twisting is declared "enough" a bit above the gate, so a capture never
// lands right on the boundary
constexpr float k_twistReadyDominance= 0.85f;
constexpr float k_holdCountdownSeconds= 3.f;

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
		m_bestDominance[sideIndex]= 0.f;
		m_bAccepted[sideIndex]= false;
		m_bAttempted[sideIndex]= false;
		m_capturedDominance[sideIndex]= -1.f;
	}
}

void MountingWizard::exit()
{
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
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_bTwistReady[sideIndex]= false;
		m_bestDominance[sideIndex]= 0.f;
	}
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
			if (ImGui::Button("Start", ImVec2(180, 0)))
			{
				// Lock in who takes part NOW - a side that is not usable at the
				// start must not silently join half way and get calibrated
				// against motion it was not present for
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					m_bParticipating[sideIndex]= status[sideIndex].streaming && status[sideIndex].filterConverged;
				beginTwistStage();
			}
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;
			break;
		}

		case eState::TwistForearms:
		{
			ImGui::TextWrapped(
				"Step 1 of 2: TWIST. Rotate each forearm as if slowly turning a doorknob "
				"- palm up, palm down, back and forth. Keep your elbows still and your "
				"wrists relaxed; it is the twist that is being measured.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"Waving your arms around does not help - it makes the motion less "
				"single-axis and the bars below go DOWN. Your hands do not need to be "
				"visible to the cameras during this step.");
			ImGui::Spacing();

			bool bAllReady= true;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;

				const float dominance= std::max(0.f, status[sideIndex].armAxisDominance);
				m_bestDominance[sideIndex]= std::max(m_bestDominance[sideIndex], dominance);
				if (dominance >= k_twistReadyDominance)
					m_bTwistReady[sideIndex]= true;
				bAllReady&= m_bTwistReady[sideIndex];

				// Scale the bar so the ready threshold sits at full - the number
				// itself (a ratio of eigenvalues) means nothing to the user
				const float progress= std::min(1.f, dominance / k_twistReadyDominance);
				ImGui::Text("%-6s", sideName(sideIndex));
				ImGui::SameLine();
				ImGui::ProgressBar(progress, ImVec2(-90, 0), m_bTwistReady[sideIndex] ? "ready" : "keep twisting");
				ImGui::SameLine();
				if (m_bTwistReady[sideIndex])
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "done");
				else
					ImGui::TextDisabled("%.2f", dominance);
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
				"Step 2 of 2: HOLD STRAIGHT. Hold each hand flat and in line with its "
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
						m_bAttempted[sideIndex]= capture.bCaptured[sideIndex];
						m_capturedDominance[sideIndex]=
							capture.bCaptured[sideIndex] ? capture.axisDominance[sideIndex] : -1.f;
						m_bAccepted[sideIndex]= capture.bCaptured[sideIndex] &&
												capture.axisDominance[sideIndex] >= k_minAxisDominance;
						if (!m_bAccepted[sideIndex])
							continue;

						m_config->imu.forearmToSensor[sideIndex]= capture.forearmToSensor[sideIndex];
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

				ImGui::SeparatorText(sideName(sideIndex));
				if (m_bAccepted[sideIndex])
				{
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "  Calibrated (arm axis %.2f)",
									   m_capturedDominance[sideIndex]);
				}
				else if (!m_bAttempted[sideIndex])
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Not captured - the hand was not tracked, or the\n"
									   "  controller stopped streaming");
				}
				else
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Rejected - arm axis only %.2f, needs %.2f.\n"
									   "  Redo and twist more purely about the forearm.",
									   m_capturedDominance[sideIndex], k_minAxisDominance);
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
