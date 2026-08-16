#include "TimelinePanel.h"

#include <algorithm>
#include <filesystem>

#include "imgui.h"

#include "AppConfig.h"
#include "VisionThread.h"

static const ImVec4 k_colorGood(0.4f, 1.f, 0.5f, 1.f);
static const ImVec4 k_colorWarn(1.f, 0.85f, 0.3f, 1.f);
static const ImVec4 k_colorBad(1.f, 0.4f, 0.4f, 1.f);

void TimelinePanel::draw(AppConfig* config, VisionThread* visionThread)
{
	if (!ImGui::Begin("Timeline"))
	{
		ImGui::End();
		return;
	}

	drawRecordingSection(config, visionThread);
	ImGui::SeparatorText("Replay");
	drawLoadSection(visionThread);
	if (m_bLoaded && m_replay.didRun())
	{
		drawTransportSection();
		drawFrameInfoSection();
		ImGui::SeparatorText("What-if");
		drawWhatIfSection();
	}

	rebuildDisplayFrame();
	ImGui::End();
}

void TimelinePanel::drawRecordingSection(AppConfig* config, VisionThread* visionThread)
{
	ImGui::SeparatorText("Recording");

	if (visionThread->isRecording())
	{
		ImGui::TextColored(k_colorBad, "REC");
		ImGui::SameLine();
		ImGui::Text("%lld frames, %.1f MB", (long long)visionThread->getRecordingFrameCount(),
					(double)visionThread->getRecordingBytes() / (1024.0 * 1024.0));
		if (visionThread->isRecordingRawFrames())
		{
			ImGui::TextColored(k_colorBad, "+ RAW FRAMES");
			ImGui::SameLine();
			ImGui::Text("%lld images, %.0f MB", (long long)visionThread->getRecordedFrameCount(),
						(double)visionThread->getRecordedFrameBytes() / (1024.0 * 1024.0));
			const int64_t dropped= visionThread->getDroppedFrameCount();
			if (dropped > 0)
				ImGui::TextColored(k_colorBad, "%lld frames dropped (encoder behind)", (long long)dropped);
		}
		if (ImGui::Button("Stop Recording (F10)", ImVec2(-1, 0)))
			visionThread->requestRecordingStop();
	}
	else
	{
		if (ImGui::Button("Start Recording (F10)", ImVec2(-1, 0)))
			visionThread->requestRecordingStart(AppConfig::makeRecordingFilePath());
		ImGui::SetItemTooltip(
			"Captures every input to the tracking stages (per-camera\n"
			"landmarks, depth samples, timestamps) so the whole pipeline can\n"
			"be re-run offline bit-exactly. Starting RESETS the transient\n"
			"tracking state (brief blip), and editing any tracking/fusion\n"
			"setting while recording finalizes the file. Roughly 0.5-1 MB\n"
			"per second.");

		// Deliberately below the start button and unchecked by default: this
		// is the difference between recording abstract landmarks and
		// recording video of the room
		bool bRecordFrames= config->recording.recordRawFrames;
		if (ImGui::Checkbox("Also record raw camera frames", &bRecordFrames))
		{
			config->recording.recordRawFrames= bRecordFrames;
			config->markDirty();
		}
		ImGui::SetItemTooltip(
			"PRIVACY: this writes the actual camera images to disk as JPEGs,\n"
			"next to the recording. Off by default, and worth leaving off\n"
			"unless you are chasing a problem.\n\n"
			"What it buys: a landmark recording cannot answer whether a\n"
			"different pose model would have done better, because the model's\n"
			"input is gone. With frames, that becomes an offline measurement\n"
			"(--replay-bodypose) instead of a live impression.\n\n"
			"Costs roughly 3-6 MB per second per camera.");
		if (bRecordFrames)
			ImGui::TextColored(k_colorBad, "Raw frames WILL be written next recording");

		const std::string lastPath= visionThread->getLastRecordingPath();
		if (!lastPath.empty())
		{
			if (visionThread->didLastRecordingAbort())
				ImGui::TextColored(k_colorBad, "Last recording ABORTED (writer overflow)");
			ImGui::TextWrapped("Last: %s", lastPath.c_str());
		}
	}
}

void TimelinePanel::refreshRecordingList()
{
	m_recordingFiles.clear();
	std::error_code ec;
	for (const auto& entry :
		 std::filesystem::directory_iterator(AppConfig::getRecordingsDirectoryPath(), ec))
	{
		if (entry.is_regular_file() && entry.path().extension() == ".jsonl")
			m_recordingFiles.push_back(entry.path().string());
	}
	std::sort(m_recordingFiles.begin(), m_recordingFiles.end());
	if (m_selectedRecording >= (int)m_recordingFiles.size())
		m_selectedRecording= (int)m_recordingFiles.size() - 1;
	if (m_selectedRecording < 0 && !m_recordingFiles.empty())
		m_selectedRecording= (int)m_recordingFiles.size() - 1; // newest by timestamp name
}

void TimelinePanel::drawLoadSection(VisionThread* visionThread)
{
	if (m_recordingFiles.empty())
		refreshRecordingList();

	const char* selectedName= m_selectedRecording >= 0
		? m_recordingFiles[m_selectedRecording].c_str()
		: "<no recordings>";
	// Show just the filename in the combo (paths are long)
	auto fileLabel= [](const std::string& path) {
		return std::filesystem::path(path).filename().string();
	};

	if (ImGui::BeginCombo("Recording", m_selectedRecording >= 0
											? fileLabel(m_recordingFiles[m_selectedRecording]).c_str()
											: selectedName))
	{
		refreshRecordingList();
		for (int fileIndex= 0; fileIndex < (int)m_recordingFiles.size(); ++fileIndex)
		{
			if (ImGui::Selectable(fileLabel(m_recordingFiles[fileIndex]).c_str(),
								  fileIndex == m_selectedRecording))
				m_selectedRecording= fileIndex;
		}
		ImGui::EndCombo();
	}

	const bool bCanLoad= m_selectedRecording >= 0 && !visionThread->isRecording();
	ImGui::BeginDisabled(!bCanLoad);
	if (ImGui::Button("Load + Verify", ImVec2(-1, 0)))
	{
		std::string error;
		if (m_replay.load(m_recordingFiles[m_selectedRecording], error))
		{
			m_replay.runAll();
			m_bLoaded= true;
			m_bWhatIfSeeded= false;
			m_whatIfStatus.clear();
			m_builtFrame= -1;
			setCurrentFrame(0);
			m_bPlaying= false;

			char status[256];
			snprintf(status, sizeof(status), "%d frames, %d divergent%s", m_replay.getFrameCount(),
					 (int)m_replay.getDivergentFrames().size(),
					 m_replay.hasFooter()
						 ? (m_replay.getFooter().bAborted ? " (recording aborted)" : "")
						 : " (no footer - truncated)");
			m_loadStatus= status;
			m_bReplayViewActive= true;
		}
		else
		{
			m_bLoaded= false;
			m_loadStatus= "Load failed: " + error;
		}
	}
	ImGui::EndDisabled();
	ImGui::SetItemTooltip(
		"Loads the recording, re-runs the full tracking pipeline on its\n"
		"inputs and verifies every frame's checksum against what live\n"
		"produced. Divergent frames indicate a build mismatch (recordings\n"
		"verify bit-exactly only against the binary that made them).");

	if (!m_loadStatus.empty())
	{
		const bool bDivergent= m_bLoaded && !m_replay.getDivergentFrames().empty();
		ImGui::TextColored(m_bLoaded ? (bDivergent ? k_colorWarn : k_colorGood) : k_colorBad, "%s",
						   m_loadStatus.c_str());
	}

	if (m_bLoaded && m_replay.didRun())
	{
		ImGui::Checkbox("Show replay in 3D scene", &m_bReplayViewActive);
		ImGui::SetItemTooltip(
			"Feeds the 3D scene from the replay at the scrub position\n"
			"instead of live tracking. Live tracking and OSC keep running\n"
			"either way.");
	}
}

void TimelinePanel::setCurrentFrame(int frameIndex)
{
	m_currentFrame= std::clamp(frameIndex, 0, std::max(0, m_replay.getFrameCount() - 1));
	if (m_replay.getFrameCount() > 0)
		m_playheadMs= m_replay.getFrame(m_currentFrame).recorded->nowTimestampMs;
}

void TimelinePanel::drawTransportSection()
{
	const int frameCount= m_replay.getFrameCount();
	const double firstMs= m_replay.getFrame(0).recorded->nowTimestampMs;

	// Playback advances the playhead against recorded time
	if (m_bPlaying)
	{
		m_playheadMs+= (double)(ImGui::GetIO().DeltaTime * 1000.f) * m_playSpeed;
		while (m_currentFrame + 1 < frameCount &&
			   m_replay.getFrame(m_currentFrame + 1).recorded->nowTimestampMs <= m_playheadMs)
			++m_currentFrame;
		if (m_currentFrame >= frameCount - 1)
			m_bPlaying= false;
	}

	// Frame slider with divergence tick marks overlaid on the track
	int sliderFrame= m_currentFrame;
	ImGui::SetNextItemWidth(-1.f);
	if (ImGui::SliderInt("##frame", &sliderFrame, 0, frameCount - 1, "frame %d"))
	{
		setCurrentFrame(sliderFrame);
		m_bPlaying= false;
	}
	{
		const ImVec2 rectMin= ImGui::GetItemRectMin();
		const ImVec2 rectMax= ImGui::GetItemRectMax();
		ImDrawList* drawList= ImGui::GetWindowDrawList();
		for (int divergentFrame : m_replay.getDivergentFrames())
		{
			const float x= rectMin.x +
				(rectMax.x - rectMin.x) * ((float)divergentFrame / (float)std::max(1, frameCount - 1));
			drawList->AddLine(ImVec2(x, rectMin.y), ImVec2(x, rectMax.y), IM_COL32(255, 80, 80, 220), 2.f);
		}
	}

	const double currentMs= m_replay.getFrame(m_currentFrame).recorded->nowTimestampMs;
	ImGui::Text("t= %.3f s", (currentMs - firstMs) / 1000.0);
	ImGui::SameLine();

	if (ImGui::Button("|<"))
	{
		setCurrentFrame(0);
		m_bPlaying= false;
	}
	ImGui::SameLine();
	if (ImGui::Button("<"))
	{
		setCurrentFrame(m_currentFrame - 1);
		m_bPlaying= false;
	}
	ImGui::SameLine();
	if (ImGui::Button(m_bPlaying ? "Pause" : "Play"))
	{
		m_bPlaying= !m_bPlaying;
		if (m_bPlaying)
		{
			if (m_currentFrame >= frameCount - 1)
				setCurrentFrame(0);
			m_playheadMs= m_replay.getFrame(m_currentFrame).recorded->nowTimestampMs;
		}
	}
	ImGui::SameLine();
	if (ImGui::Button(">"))
	{
		setCurrentFrame(m_currentFrame + 1);
		m_bPlaying= false;
	}
	ImGui::SameLine();
	if (ImGui::Button(">|"))
	{
		setCurrentFrame(frameCount - 1);
		m_bPlaying= false;
	}
	ImGui::SameLine();
	ImGui::SetNextItemWidth(70.f);
	const char* kSpeedNames[]= {"0.1x", "0.25x", "0.5x", "1x", "2x"};
	const float kSpeedValues[]= {0.1f, 0.25f, 0.5f, 1.f, 2.f};
	static int s_speedIndex= 3;
	if (ImGui::Combo("##speed", &s_speedIndex, kSpeedNames, IM_ARRAYSIZE(kSpeedNames)))
		m_playSpeed= kSpeedValues[s_speedIndex];

	// Left/right arrows step while the panel is focused
	if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows))
	{
		if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
		{
			setCurrentFrame(m_currentFrame - 1);
			m_bPlaying= false;
		}
		if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
		{
			setCurrentFrame(m_currentFrame + 1);
			m_bPlaying= false;
		}
	}

	// View selection + divergence jump list
	ImGui::RadioButton("Recorded", &m_viewMode, 0);
	ImGui::SameLine();
	ImGui::RadioButton("Replayed", &m_viewMode, 1);
	ImGui::SameLine();
	ImGui::BeginDisabled(!m_replay.hasWhatIf());
	ImGui::RadioButton("What-if", &m_viewMode, 2);
	ImGui::EndDisabled();
	ImGui::SetItemTooltip("Which output the 3D scene shows at the scrub position");

	const std::vector<int>& divergent= m_replay.getDivergentFrames();
	if (!divergent.empty())
	{
		ImGui::TextColored(k_colorWarn, "%d divergent frames:", (int)divergent.size());
		ImGui::SameLine();
		for (size_t index= 0; index < divergent.size() && index < 8; ++index)
		{
			char label[32];
			snprintf(label, sizeof(label), "%d##div%d", divergent[index], (int)index);
			if (ImGui::SmallButton(label))
			{
				setCurrentFrame(divergent[index]);
				m_bPlaying= false;
			}
			ImGui::SameLine();
		}
		ImGui::NewLine();
	}
}

void TimelinePanel::drawFrameInfoSection()
{
	const TrackingReplay::ReplayFrame& frame= m_replay.getFrame(m_currentFrame);
	const RecordedFrame& recorded= *frame.recorded;

	char freshList[64]= "";
	for (const RecordedCameraInput& fresh : recorded.freshCameras)
	{
		const size_t len= strlen(freshList);
		snprintf(freshList + len, sizeof(freshList) - len, "%s%d%s", len > 0 ? " " : "",
				 fresh.cameraIndex + 1, fresh.valid ? "" : "(invalid)");
	}
	ImGui::Text("%s | fresh cams: %s", recorded.bFused ? "fused" : "passthrough", freshList);

	if (!frame.bChecksumMatch)
		ImGui::TextColored(k_colorBad, "CHECKSUM MISMATCH (replay != recorded on this build)");

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const RecordedPoseOutput& pose= recorded.outPoses[sideIndex];
		if (!pose.tracked)
		{
			ImGui::TextDisabled("%s: untracked", sideIndex == 0 ? "L" : "R");
			continue;
		}
		ImGui::Text("%s: conf %.2f %s", sideIndex == 0 ? "L" : "R", pose.confidence,
					pose.stereoTriangulated ? "stereo" : "mono");
	}
}

void TimelinePanel::drawWhatIfSection()
{
	if (!m_bWhatIfSeeded)
	{
		m_whatIfParams= m_replay.makeDefaultWhatIfParams();
		m_bWhatIfSeeded= true;
	}

	HandFusionConfig& fusion= m_whatIfParams.fusionConfig;
	ImGui::Checkbox("Triangulation", &fusion.triangulationEnabled);
	ImGui::SameLine();
	ImGui::Checkbox("Use depth", &m_whatIfParams.bUseRecordedDepth);
	ImGui::SameLine();
	ImGui::Checkbox("Rest angles", &m_whatIfParams.bApplyRestAngles);
	ImGui::SameLine();
	ImGui::Checkbox("Hand bones", &m_whatIfParams.bApplyCalibratedSkeleton);
	ImGui::SetItemTooltip(
		"Off = fall back to the landmark model's own proportions, which is\n"
		"the A/B for a bone calibration. No effect on a recording made\n"
		"before one was measured.");

	float jitterReferenceMm= fusion.jitterReferenceM * 1000.f;
	if (ImGui::SliderFloat("Jitter reference", &jitterReferenceMm, 3.f, 60.f, "%.0f mm"))
		fusion.jitterReferenceM= jitterReferenceMm * 0.001f;
	ImGui::SliderFloat("Max tri residual", &fusion.triangulationMaxResidualPx, 5.f, 80.f, "%.0f px");
	ImGui::SliderFloat("Residual reference", &fusion.residualReferencePx, 2.f, 30.f, "%.0f px");
	ImGui::SliderFloat("Min camera confidence", &fusion.minCameraConfidence, 0.f, 1.f, "%.2f");
	float stalenessMs= (float)fusion.stalenessWindowMs;
	if (ImGui::SliderFloat("Staleness window", &stalenessMs, 20.f, 200.f, "%.0f ms"))
		fusion.stalenessWindowMs= stalenessMs;
	ImGui::SliderFloat("Hand scale x", &m_whatIfParams.refLengthScale, 0.8f, 1.2f, "%.3f");

	if (ImGui::Button("Re-simulate", ImVec2(-1, 0)))
	{
		m_replay.runWhatIf(m_whatIfParams);
		m_viewMode= 2;
		m_builtFrame= -1;

		// Summarize the divergence from the recorded output
		int differentFrames= 0;
		double maxPalmDeltaMm= 0.0;
		for (int frameIndex= 0; frameIndex < m_replay.getFrameCount(); ++frameIndex)
		{
			const TrackingReplay::ReplayFrame& frame= m_replay.getFrame(frameIndex);
			if (!frame.bHasWhatIf)
				continue;
			bool bDifferent= false;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const HandPose& whatIfPose= frame.whatIfFused.poses[sideIndex];
				const RecordedPoseOutput& recordedPose= frame.recorded->outPoses[sideIndex];
				if (whatIfPose.tracked != recordedPose.tracked)
				{
					bDifferent= true;
					continue;
				}
				if (!whatIfPose.tracked || !whatIfPose.hasWorldPose || !recordedPose.hasWorldPose)
					continue;
				const double deltaMm=
					glm::length(whatIfPose.palmPositionWorld - recordedPose.palmPositionWorld) * 1000.0;
				maxPalmDeltaMm= std::max(maxPalmDeltaMm, deltaMm);
				if (deltaMm > 0.01)
					bDifferent= true;
			}
			if (bDifferent)
				++differentFrames;
		}
		char status[128];
		snprintf(status, sizeof(status), "%d/%d frames differ, max palm delta %.1f mm",
				 differentFrames, m_replay.getFrameCount(), maxPalmDeltaMm);
		m_whatIfStatus= status;
	}
	ImGui::SetItemTooltip(
		"Re-runs the whole recording with these parameters. The original\n"
		"recorded output is untouched - flip the view radio (or compare\n"
		"the deltas below) to judge a candidate fix against the incident.");

	if (!m_whatIfStatus.empty())
		ImGui::Text("%s", m_whatIfStatus.c_str());

	// Current-frame delta readout
	if (m_replay.hasWhatIf())
	{
		const TrackingReplay::ReplayFrame& frame= m_replay.getFrame(m_currentFrame);
		if (frame.bHasWhatIf)
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const HandPose& whatIfPose= frame.whatIfFused.poses[sideIndex];
				const RecordedPoseOutput& recordedPose= frame.recorded->outPoses[sideIndex];
				if (!whatIfPose.tracked || !recordedPose.tracked || !whatIfPose.hasWorldPose ||
					!recordedPose.hasWorldPose)
					continue;
				const float deltaMm=
					glm::length(whatIfPose.palmPositionWorld - recordedPose.palmPositionWorld) * 1000.f;
				float maxAngleDeltaDeg= 0.f;
				for (int fingerIndex= 0; fingerIndex < FINGER_COUNT; ++fingerIndex)
				{
					const FingerAngles& whatIfAngles= whatIfPose.fingers[fingerIndex];
					const FingerAngles& recordedAngles= recordedPose.fingers[fingerIndex];
					maxAngleDeltaDeg= std::max(
						{maxAngleDeltaDeg, fabsf(whatIfAngles.lateral - recordedAngles.lateral),
						 fabsf(whatIfAngles.proximal - recordedAngles.proximal),
						 fabsf(whatIfAngles.intermediate - recordedAngles.intermediate),
						 fabsf(whatIfAngles.distal - recordedAngles.distal)});
				}
				maxAngleDeltaDeg*= 57.2958f;
				ImGui::Text("%s delta vs recorded: palm %.1f mm, angles %.1f deg",
							sideIndex == 0 ? "L" : "R", deltaMm, maxAngleDeltaDeg);
			}
		}
	}
}

void TimelinePanel::buildRecordedResult(const RecordedFrame& recorded,
										TrackingFrameResult& outResult) const
{
	outResult= TrackingFrameResult();
	outResult.timestampMs= recorded.nowTimestampMs;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const RecordedPoseOutput& recordedPose= recorded.outPoses[sideIndex];
		HandPose& pose= outResult.poses[sideIndex];
		pose.tracked= recordedPose.tracked;
		if (!recordedPose.tracked)
			continue;
		pose.side= (eHandSide)recordedPose.side;
		pose.presence= recordedPose.presence;
		pose.visibility= recordedPose.visibility;
		pose.confidence= recordedPose.confidence;
		pose.fkReprojectionPx= recordedPose.fkReprojectionPx;
		pose.stereoTriangulated= recordedPose.stereoTriangulated;
		pose.hasCameraPose= recordedPose.hasCameraPose;
		pose.palmPositionCamera= recordedPose.palmPositionCamera;
		pose.palmOrientationCamera= recordedPose.palmOrientationCamera;
		pose.hasWorldPose= recordedPose.hasWorldPose;
		pose.palmPositionWorld= recordedPose.palmPositionWorld;
		pose.palmOrientationWorld= recordedPose.palmOrientationWorld;
		pose.fingers= recordedPose.fingers;
		pose.skeleton= recordedPose.skeleton;

		const RecordedImuOutput& imu= recorded.imu[sideIndex];
		pose.hasForearmPose= imu.hasForearmPose;
		if (imu.hasForearmPose)
		{
			pose.forearmOrientationWorld= imu.forearmOrientationWorld;
			pose.forearmConfidence= imu.forearmConfidence;
		}
	}
}

void TimelinePanel::rebuildDisplayFrame()
{
	if (!isReplayViewActive())
		return;
	if (m_builtFrame == m_currentFrame && m_builtViewMode == m_viewMode)
		return;

	const TrackingReplay::ReplayFrame& frame= m_replay.getFrame(m_currentFrame);
	switch (m_viewMode)
	{
	case 0:
		buildRecordedResult(*frame.recorded, m_displayFused);
		break;
	case 2:
		if (frame.bHasWhatIf)
		{
			m_displayFused= frame.whatIfFused;
			break;
		}
		[[fallthrough]];
	default:
		m_displayFused= frame.replayedFused;
		break;
	}

	m_perCameraCopies= frame.perCamera;
	m_perCameraCopyValid.assign(frame.perCameraValid.begin(), frame.perCameraValid.end());
	m_builtFrame= m_currentFrame;
	m_builtViewMode= m_viewMode;
}

std::vector<SceneCameraView> TimelinePanel::getSceneCameras() const
{
	std::vector<SceneCameraView> cameras;
	const AppConfig& config= m_replay.getRecordedConfig();
	for (size_t cameraIndex= 0; cameraIndex < config.cameraCount(); ++cameraIndex)
	{
		const CameraProfile& profile= config.camera(cameraIndex);
		SceneCameraView view;
		view.cameraToWorld= glm::mat4(profile.extrinsics.markerFromCamera);
		view.bHasExtrinsics= profile.extrinsics.present;
		view.intrinsics= profile.intrinsics.present ? &profile.intrinsics.intrinsics : nullptr;
		cameras.push_back(view);
	}
	return cameras;
}

std::vector<const TrackingFrameResult*> TimelinePanel::getPerCameraResults() const
{
	std::vector<const TrackingFrameResult*> results;
	for (size_t cameraIndex= 0; cameraIndex < m_perCameraCopies.size(); ++cameraIndex)
	{
		results.push_back(cameraIndex < m_perCameraCopyValid.size() && m_perCameraCopyValid[cameraIndex]
							  ? &m_perCameraCopies[cameraIndex]
							  : nullptr);
	}
	return results;
}
