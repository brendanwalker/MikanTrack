#include "MountingWizard.h"

#include <algorithm>
#include <cmath>

#include "imgui.h"

#include "LocText.h"

#include "AppConfig.h"
#include "ImuService.h"
#include "TrackingTypes.h"
#include "VisionThread.h"

namespace
{
// A margin over ImuService's own capture gate, so a capture is never taken
// right on the boundary
constexpr float k_twistReadyDominance= 0.8f;
constexpr float k_twistReadyReversal= 0.6f;

const char* sideName(int sideIndex)
{
	return sideIndex == 0 ? locText("mountingWizard.sideLeft") : locText("mountingWizard.sideRight");
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
	m_result= eWizardResult::None;
	m_state= eState::VerifyDevices;
	m_bCaptureRequested= false;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_bParticipating[sideIndex]= false;
		m_bTwistReady[sideIndex]= false;
		m_bCurlReady[sideIndex]= false;
		m_bAccepted[sideIndex]= false;
		m_captured[sideIndex]= MountingCaptureResult();
	}
}

void MountingWizard::exit()
{
	// Leaving mid-measurement must not strand the service collecting forever
	m_visionThread->cancelImuBiasCalibration();
	m_visionThread->requestImuMotionRecording(eMountingMotion::None);
	m_bActive= false;
}

bool MountingWizard::isSideParticipating(int sideIndex) const
{
	return m_bParticipating[sideIndex];
}

void MountingWizard::beginMotionStage(eState state, eMountingMotion motion)
{
	// Start from a clean slate: whatever the arms happened to be doing before
	// this stage is not part of this measurement
	m_visionThread->requestImuMotionRecording(motion);
	m_epochAtReset= m_visionThread->getImuSideStatus(eHandSide::Left).motionEpoch;
	m_bWaitingForMotionReset= true;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (motion == eMountingMotion::Twist)
			m_bTwistReady[sideIndex]= false;
		else
			m_bCurlReady[sideIndex]= false;
	}
	m_state= state;
}

bool MountingWizard::drawMotionStage(const ImuSideStatus status[2], bool bReady[2], const char* elbowHint)
{
	// The recording is serviced on the vision thread; until it lands, the
	// status still describes the previous stage
	if (m_bWaitingForMotionReset && status[0].motionEpoch != m_epochAtReset &&
		status[1].motionEpoch != m_epochAtReset)
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
			bReady[sideIndex]= true;
		}
		bAllReady&= bReady[sideIndex];

		ImGui::Text("%-6s", sideName(sideIndex));
		ImGui::SameLine();
		const float shownProgress= m_bWaitingForMotionReset ? 0.f : sideStatus.twistProgress;
		ImGui::ProgressBar(shownProgress, ImVec2(-90, 0),
						   bReady[sideIndex] ? locText("mountingWizard.readyOverlay")
											 : locText("mountingWizard.keepGoingOverlay"));
		ImGui::SameLine();
		if (bReady[sideIndex])
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "%s", locText("mountingWizard.doneStatus"));
		}
		else
		{
			ImGui::TextDisabled("%.0f%%", shownProgress * 100.f);
			// Once there IS enough motion but it still does not qualify, say
			// which way it is wrong - the two need opposite fixes
			if (!m_bWaitingForMotionReset && sideStatus.twistProgress >= 1.f)
			{
				if (sideStatus.twistReversal < k_twistReadyReversal)
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
									   locText("mountingWizard.moveBackOtherWayFmt"), sideName(sideIndex));
				else if (dominance < k_twistReadyDominance)
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "  %s: %s", sideName(sideIndex),
									   elbowHint);
			}
		}

		if (sideStatus.biasSaturated)
		{
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
							   locText("mountingWizard.filterDivergedFmt"),
							   sideName(sideIndex));
		}
	}
	return bAllReady;
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
	if (!ImGui::Begin(locWindowTitle("windows.mountingWizard"), nullptr, ImGuiWindowFlags_NoCollapse))
	{
		ImGui::End();
		return true;
	}

	switch (m_state)
	{
		case eState::VerifyDevices:
		{
			ImGui::TextWrapped("%s", locText("mountingWizard.verifyIntro"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.verifyStrapInstructions"));
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
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s",
									   locText("mountingWizard.noControllerAssigned"));
					continue;
				}
				ImGui::Text("  %s", sideStatus.deviceName.c_str());
				drawStatusLine(sideStatus.streaming, locText("mountingWizard.streaming"));
				drawStatusLine(sideStatus.filterConverged,
							   sideStatus.filterConverged ? locText("mountingWizard.orientationSettled")
														  : locText("mountingWizard.orientationSettling"));
			}

			ImGui::Spacing();
			if (!bAnyReady)
			{
				ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
								   "%s", locText("mountingWizard.waitingForController"));
			}

			ImGui::BeginDisabled(!bAnyReady);
			// Lock in who takes part NOW - a side that is not usable at the
			// start must not silently join half way and get calibrated against
			// motion it was not present for
			auto lockParticipants= [this, &status]() {
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					m_bParticipating[sideIndex]= status[sideIndex].streaming && status[sideIndex].filterConverged;
			};
			if (ImGui::Button(locLabel("common.start"), ImVec2(180, 0)))
			{
				lockParticipants();
				m_visionThread->requestImuBiasCalibration();
				m_state= eState::CalibrateBias;
			}
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
			{
				m_result= eWizardResult::Cancelled;
				m_bWantsClose= true;
			}
			break;
		}

		case eState::CalibrateBias:
		{
			ImGui::TextWrapped("%s", locText("mountingWizard.biasStep1Intro"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.biasStep1Explain"));
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
					ImGui::ProgressBar(1.f, ImVec2(-90, 0), locText("mountingWizard.measuredOverlay"));
					ImGui::SameLine();
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "%s", locText("mountingWizard.doneStatus"));
				}
				else
				{
					ImGui::ProgressBar(progress, ImVec2(-90, 0),
									   status[sideIndex].biasCalibrationDisturbed
										   ? locText("mountingWizard.restartedOverlay")
										   : locText("mountingWizard.measuringOverlay"));
					ImGui::SameLine();
					ImGui::TextDisabled("%.0f%%", progress * 100.f);
				}
			}

			if (bAllDone)
			{
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "%s",
								   locText("mountingWizard.putControllersBack"));
				ImGui::Spacing();
				if (ImGui::Button(locLabel("common.continueLabel"), ImVec2(180, 0)))
					beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
			}
			else
			{
				ImGui::TextDisabled("%s", locText("mountingWizard.anyMovementRestarts"));
				ImGui::Spacing();
				if (ImGui::Button(locLabel("common.skip"), ImVec2(180, 0)))
				{
					// The filter estimates the bias online anyway - just not
					// about the vertical axis, which is the one that drifts
					m_visionThread->cancelImuBiasCalibration();
					beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
				}
				ImGui::SetItemTooltip("%s", locText("mountingWizard.skipBiasTooltip"));
			}
			ImGui::SameLine();
			if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
			{
				m_result= eWizardResult::Cancelled;
				m_bWantsClose= true;
			}
			break;
		}

		case eState::TwistForearms:
		{
			ImGui::TextWrapped("%s", locText("mountingWizard.twistStep2Intro"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.twistNote"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.twistPurpose"));
			ImGui::Spacing();

			drawMotionStage(status, m_bTwistReady, locText("mountingWizard.twistElbowHint"));

			// Deliberately NOT auto-advanced: the next stage is a different
			// motion with its own instructions, and dropping the user into it
			// mid-twist means they perform the first half of it wrong
			ImGui::Spacing();
			ImGui::BeginDisabled(!m_bTwistReady[0] && !m_bTwistReady[1]);
			if (ImGui::Button(locLabel("common.continueLabel"), ImVec2(180, 0)))
				beginMotionStage(eState::CurlElbows, eMountingMotion::Curl);
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
			{
				m_result= eWizardResult::Cancelled;
				m_bWantsClose= true;
			}
			break;
		}

		case eState::CurlElbows:
		{
			ImGui::TextWrapped("%s", locText("mountingWizard.curlStep3Intro"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.curlPurpose"));
			ImGui::Spacing();
			ImGui::TextWrapped("%s", locText("mountingWizard.curlVisibility"));
			ImGui::Spacing();

			const bool bAllReady=
				drawMotionStage(status, m_bCurlReady, locText("mountingWizard.curlElbowHint"));

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;
				const HandPose& pose= fusedResult.poses[sideIndex];
				if (!pose.tracked || !pose.hasWorldPose)
				{
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
									   locText("mountingWizard.handNotVisibleFmt"),
									   sideName(sideIndex));
				}
			}

			if (bAllReady && !m_bCaptureRequested)
			{
				m_visionThread->requestImuMountingCapture();
				m_bCaptureRequested= true;
			}

			ImGui::Spacing();
			ImGui::BeginDisabled(m_bCaptureRequested || (!m_bCurlReady[0] && !m_bCurlReady[1]));
			if (ImGui::Button(locLabel("mountingWizard.finishNowButton"), ImVec2(180, 0)))
			{
				m_visionThread->requestImuMountingCapture();
				m_bCaptureRequested= true;
			}
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
			{
				m_result= eWizardResult::Cancelled;
				m_bWantsClose= true;
			}

			// The capture is serviced on the vision thread; collect the result
			if (m_bCaptureRequested)
			{
				VisionThread::ImuMountingCapture capture;
				if (m_visionThread->fetchImuMountingCapture(capture))
				{
					m_bCaptureRequested= false;
					m_visionThread->requestImuMotionRecording(eMountingMotion::None);
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

					// The curl measured the elbow-to-controller distance as a
					// by-product. It is the radius the sensor swept, so it runs
					// slightly short of a true elbow-to-wrist length, but it
					// beats the fixed default it replaces - and the setting
					// stays editable either way.
					float lengthSum= 0.f;
					int lengthCount= 0;
					for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					{
						if (m_bAccepted[sideIndex] && capture.sides[sideIndex].bLengthMeasured)
						{
							lengthSum+= capture.sides[sideIndex].forearmLengthMeters;
							lengthCount++;
						}
					}
					if (lengthCount > 0)
					{
						m_config->body.forearmLengthMeters= lengthSum / (float)lengthCount;
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
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f),
									   locText("mountingWizard.calibratedFmt"),
									   result.axisDominance, result.interAxisAngleDegrees);
					if (result.bLengthMeasured)
					{
						ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f),
										   locText("mountingWizard.elbowToControllerFmt"),
										   result.forearmLengthMeters * 100.f);
					}
				}
				else if (!result.bCaptured)
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "%s", locText("mountingWizard.notCaptured"));
				}
				else
				{
					// Name the gate that actually failed: they need different
					// corrections, and "rejected" alone told the user nothing
					const char* reasonKey= "mountingWizard.reasonGeneric";
					if (!imuIsTwistUsable(result.axisDominance, result.twistProgress, result.twistReversal))
						reasonKey= "mountingWizard.reasonTwistWeak";
					else if (result.curlProgress < 1.f || result.curlReversal < 0.5f)
						reasonKey= "mountingWizard.reasonCurlSmall";
					else if (result.interAxisAngleDegrees < 60.f)
						reasonKey= "mountingWizard.reasonSameAxis";
					else if (result.curlStrokes < 3 || result.hingeSpreadDegrees > 15.f)
						reasonKey= "mountingWizard.reasonStrokesDisagreed";
					else if (result.lengthFitCorrelation < 0.5f)
						reasonKey= "mountingWizard.reasonHandDirectionUnclear";
					else if (result.palmarSource == ePalmarSource::None)
						reasonKey= "mountingWizard.reasonPalmUnknown";

					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", locText(reasonKey));
				}
			}

			ImGui::Spacing();
			ImGui::SeparatorText(locText("mountingWizard.checkItHeader"));
			ImGui::TextWrapped("%s", locText("mountingWizard.reviewCheckIntro"));
			ImGui::Spacing();

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;
				const HandPose& pose= fusedResult.poses[sideIndex];
				if (!pose.tracked || !pose.hasWorldPose || !pose.hasForearmPose)
				{
					ImGui::TextDisabled(locText("mountingWizard.noReadingFmt"), sideName(sideIndex));
					continue;
				}
				const float bendDegrees= quaternionAngleDegrees(pose.getWristRotation());
				const ImVec4 color= bendDegrees < 15.f ? ImVec4(0.4f, 1.f, 0.4f, 1.f)
													   : ImVec4(1.f, 0.85f, 0.3f, 1.f);
				ImGui::TextColored(color, locText("mountingWizard.wristBendFmt"), sideName(sideIndex), bendDegrees);
			}

			ImGui::Spacing();
			if (ImGui::Button(locLabel("common.finish"), ImVec2(180, 0)))
			{
				m_result= eWizardResult::Completed;
				m_bWantsClose= true;
			}
			ImGui::SameLine();
			if (ImGui::Button(locLabel("mountingWizard.redoButton"), ImVec2(120, 0)))
			{
				m_bCaptureRequested= false;
				beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
			}
			break;
		}
	}

	ImGui::End();
	return !m_bWantsClose;
}
