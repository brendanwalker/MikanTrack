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
		ImGui::ProgressBar(shownProgress, ImVec2(-90, 0), bReady[sideIndex] ? "ready" : "keep going");
		ImGui::SameLine();
		if (bReady[sideIndex])
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "done");
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
									   "  %s: move back the other way too", sideName(sideIndex));
				else if (dominance < k_twistReadyDominance)
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "  %s: %s", sideName(sideIndex),
									   elbowHint);
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
				"can turn a sensor orientation into a FOREARM orientation. Two motions "
				"supply it: a twist of the forearms, then a curl at the elbows.");
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
					beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
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
					beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
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
				"Step 2 of 3: TWIST. Hold your forearms roughly HORIZONTAL and rotate each "
				"one as if slowly turning a doorknob - palm up, palm down, and back again, "
				"several times. Keep your elbows still; it is the twist that is measured.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"Turning one way only does not count, and neither does waving your arms "
				"around - the bar needs back-and-forth rotation about the forearm itself.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"This finds the forearm's long axis, which is what places the elbow.");
			ImGui::Spacing();

			drawMotionStage(status, m_bTwistReady, "keep the elbow still - twist only");

			// Deliberately NOT auto-advanced: the next stage is a different
			// motion with its own instructions, and dropping the user into it
			// mid-twist means they perform the first half of it wrong
			ImGui::Spacing();
			ImGui::BeginDisabled(!m_bTwistReady[0] && !m_bTwistReady[1]);
			if (ImGui::Button("Continue", ImVec2(180, 0)))
				beginMotionStage(eState::CurlElbows, eMountingMotion::Curl);
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;
			break;
		}

		case eState::CurlElbows:
		{
			ImGui::TextWrapped(
				"Step 3 of 3: CURL. Keep your PALMS FACING DOWN and your upper arms still, "
				"then raise and lower each hand at the elbow, several times - a slow bicep "
				"curl. Do not twist your forearms during this.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"The elbow hinge points across the forearm, so this second direction of "
				"rotation is what fixes the roll that a twist alone cannot see. It also "
				"measures the distance from your elbow to the controller.");
			ImGui::Spacing();
			ImGui::TextWrapped(
				"KEEP BOTH HANDS VISIBLE to the cameras. The cameras decide only which way "
				"round your palm faces, so a rough view is enough.");
			ImGui::Spacing();

			const bool bAllReady=
				drawMotionStage(status, m_bCurlReady, "keep the shoulder still - bend at the elbow only");

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!isSideParticipating(sideIndex))
					continue;
				const HandPose& pose= fusedResult.poses[sideIndex];
				if (!pose.tracked || !pose.hasWorldPose)
				{
					ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f),
									   "  %s: hand not visible - the cameras must see it",
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
			if (ImGui::Button("Finish now", ImVec2(180, 0)))
			{
				m_visionThread->requestImuMountingCapture();
				m_bCaptureRequested= true;
			}
			ImGui::EndDisabled();
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(120, 0)))
				m_bWantsClose= true;

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
						m_config->imu.forearmLengthMeters= lengthSum / (float)lengthCount;
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
									   "  Calibrated (arm axis %.2f, hinge %.0f deg off it)",
									   result.axisDominance, result.interAxisAngleDegrees);
					if (result.bLengthMeasured)
					{
						ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f),
										   "  Elbow to controller: %.0f cm (measured)",
										   result.forearmLengthMeters * 100.f);
					}
				}
				else if (!result.bCaptured)
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Not captured - the controller stopped streaming");
				}
				else
				{
					// Name the gate that actually failed: they need different
					// corrections, and "rejected" alone told the user nothing
					const char* reason= "the motions were not good enough";
					if (!imuIsTwistUsable(result.axisDominance, result.twistProgress, result.twistReversal))
						reason= "the TWIST did not pin the forearm axis - rotate\n  further, and back the other way too";
					else if (result.curlProgress < 1.f || result.curlReversal < 0.5f)
						reason= "the CURL was too small - raise and lower your hand\n  further, several times";
					else if (result.interAxisAngleDegrees < 60.f)
						reason= "the curl turned about the same axis as the twist -\n  keep the shoulder still and bend only at the elbow";
					else if (result.curlStrokes < 3 || result.hingeSpreadDegrees > 15.f)
						reason= "the curl strokes disagreed - keep the palms facing\n  down and do not twist while curling";
					else if (result.lengthFitCorrelation < 0.5f)
						reason= "the curl did not show which way the hand points -\n  bend at the elbow rather than moving the whole arm";
					else if (result.palmarSource == ePalmarSource::None)
						reason= "the cameras never saw the hand, so which way the\n  palm faces is unknown";

					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
									   "  Rejected - %s.\n  Nothing was saved for this side.", reason);
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
				beginMotionStage(eState::TwistForearms, eMountingMotion::Twist);
			}
			break;
		}
	}

	ImGui::End();
	return !m_bWantsClose;
}
