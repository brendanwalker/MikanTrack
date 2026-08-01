#include "SettingsPanels.h"

#include "imgui.h"

#include "AppConfig.h"
#include "Scene3dPanel.h"
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

void SettingsPanels::drawTrackingPanel(AppConfig* config, VisionThread* visionThread,
									   VideoPreviewPanel* previewPanel, Scene3dPanel* scene3dPanel)
{
	if (!ImGui::Begin("Tracking"))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	TrackingConfig& tracking= config->tracking;

	bChanged|= ImGui::Checkbox("Flip handedness", &tracking.flipHandedness);
	ImGui::SetItemTooltip("MediaPipe assumes a mirrored selfie view.\nEnable for a normal (non-mirrored) camera.");

	bChanged|= ImGui::SliderInt("Detector interval", &tracking.detectorIntervalFrames, 5, 120);
	ImGui::SetItemTooltip("Palm detector re-runs at least every N frames");

	ImGui::SeparatorText("Depth Estimation");
	bChanged|= ImGui::Checkbox("solvePnP depth", &tracking.usePnpDepth);
	ImGui::SetItemTooltip(
		"Solves the hand's rigid 6-DoF pose from all 21 landmark\n"
		"correspondences instead of estimating depth from the single\n"
		"wrist->knuckle bone - substantially less z noise. Off = legacy\n"
		"two-point estimator (for A/B comparison).");
	ImGui::BeginDisabled(!tracking.usePnpDepth);
	bChanged|= ImGui::Checkbox("PnP: palm points only", &tracking.pnpPalmOnly);
	ImGui::SetItemTooltip(
		"Restrict the solve to the 6 quasi-rigid palm points (wrist,\n"
		"thumb CMC, finger MCPs). Try if occluded fingertips appear to\n"
		"drag the full solve.");
	ImGui::EndDisabled();

	ImGui::SeparatorText("Smoothing");
	bChanged|= ImGui::Checkbox("Enabled", &tracking.smoothingEnabled);
	bChanged|= ImGui::SliderFloat("Min cutoff", &tracking.smoothingMinCutoff, 0.1f, 5.f, "%.2f Hz");
	bChanged|= ImGui::SliderFloat("Beta", &tracking.smoothingBeta, 0.f, 0.5f, "%.3f");

	if (config->cameraCount() > 1)
	{
		ImGui::SeparatorText("Fusion");
		FusionConfig& fusion= config->fusion;

		float stalenessMs= (float)fusion.stalenessWindowMs;
		if (ImGui::SliderFloat("Staleness window", &stalenessMs, 20.f, 200.f, "%.0f ms"))
		{
			fusion.stalenessWindowMs= stalenessMs;
			bChanged= true;
		}
		ImGui::SetItemTooltip("A camera's last result older than this is\nexcluded from fusion");

		const char* kSidePriorOptions[]= {"Off", "+X", "-X", "+Y", "-Y"};
		bChanged|= ImGui::Combo("Right hand toward", &fusion.spatialSidePriorAxis,
								kSidePriorOptions, IM_ARRAYSIZE(kSidePriorOptions));
		ImGui::SetItemTooltip(
			"If you never cross your hands: the world axis (marker frame,\n"
			"origin at the marker) pointing toward where your RIGHT hand\n"
			"lives. Adds a spatial prior to left/right assignment, which\n"
			"stops a camera that sees only one hand from hijacking the\n"
			"wrong side. Find the axis with the palm position readout or\n"
			"the 3D view's world axes.");

		bChanged|= ImGui::Checkbox("Cross-camera search seeding", &tracking.crossCameraSeeding);
		ImGui::SetItemTooltip(
			"When one camera tracks a hand another camera lost, project it\n"
			"into the lost camera's image and try the landmark model there\n"
			"directly - much faster reacquisition after claps/occlusion.");

		ImGui::SeparatorText("Observation Confidence");
		bChanged|= ImGui::SliderFloat("Jitter reference", &fusion.jitterReferenceMm, 3.f, 60.f, "%.0f mm");
		ImGui::SetItemTooltip(
			"Palm jitter at which a camera's view counts as half as\n"
			"trustworthy. Confidence = presence x stability, and the blend\n"
			"weight is confidence x how face-on the palm is - so a camera\n"
			"seeing a hand edge-on stops polluting the fused pose.\n"
			"LOWER = stricter. Raise it if fast hand motion is being\n"
			"treated as noise.");

		bChanged|= ImGui::SliderFloat("Min camera confidence", &fusion.minCameraConfidence, 0.f, 1.f, "%.2f");
		ImGui::SetItemTooltip(
			"Drop a camera's observation entirely below this confidence.\n"
			"0 = never drop, rely on the soft weighting alone (usually\n"
			"enough). Watch the per-camera readout below to pick a value.");

		// Which camera won each hand in the last fusion
		const int leftCam= visionThread->getDominantCamera(eHandSide::Left);
		const int rightCam= visionThread->getDominantCamera(eHandSide::Right);
		ImGui::Text("Dominant camera  L: %s  R: %s",
					leftCam >= 0 ? std::to_string(leftCam + 1).c_str() : "-",
					rightCam >= 0 ? std::to_string(rightCam + 1).c_str() : "-");

		// Live per-camera confidence: the number the thresholds act on
		if (ImGui::BeginTable("confidence", 3, ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("");
			ImGui::TableSetupColumn("Left");
			ImGui::TableSetupColumn("Right");
			ImGui::TableHeadersRow();
			for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("Camera %d", cameraIndex + 1);
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					ImGui::TableNextColumn();
					const float confidence= visionThread->getObservationConfidence(cameraIndex, (eHandSide)sideIndex);
					if (confidence < 0.f)
						ImGui::TextDisabled("-");
					else
						ImGui::Text("%.2f", confidence);
				}
			}
			ImGui::EndTable();
		}

		ImGui::SeparatorText("Stereo Hand Scale");
		bChanged|= ImGui::Checkbox("Auto hand scale", &tracking.autoHandScaleFromStereo);
		ImGui::SetItemTooltip(
			"With two calibrated cameras seeing the same hand, its wrist can\n"
			"be triangulated - which implies the true hand scale. Refines the\n"
			"wizard-measured scale continuously while enabled.");

		const float scaleFactor= visionThread->getAutoHandScaleFactor();
		const double autoScaleMeters= config->handScale.refLengthMeters * (double)scaleFactor;
		ImGui::Text("Configured: %.2f cm   Stereo: %.2f cm (x%.3f)",
					config->handScale.refLengthMeters * 100.0, autoScaleMeters * 100.0, scaleFactor);
		if (tracking.autoHandScaleFromStereo && fabsf(scaleFactor - 1.f) > 0.01f)
		{
			if (ImGui::Button("Save stereo scale as calibrated"))
			{
				config->handScale.refLengthMeters= autoScaleMeters;
				config->handScale.present= true;
				config->markDirty();
				// The refresh resets the correction EMA to 1 over the new baseline
				visionThread->requestConfigRefresh();
			}
		}
	}

	ImGui::SeparatorText("Rest Pose");
	{
		HandRestPoseConfig& restPose= config->handRestPose;
		ImGui::TextWrapped(
			"Defines which pose reports all-zero angles. Hold both hands in "
			"your rest pose (flat, fingers together and straight) and capture.");
		ImGui::Text("Calibrated  L: %s  R: %s", restPose.present[0] ? "yes" : "no",
					restPose.present[1] ? "yes" : "no");

		if (ImGui::Button("Capture Rest Pose"))
			visionThread->requestRestPoseCapture();
		ImGui::SetItemTooltip(
			"Captures the currently tracked hands as the zero reference.\n"
			"Without it, zero means the flat-hand default (fingers parallel\n"
			"to the palm's forward axis), which ignores how your own hand\n"
			"rests - a hand hovering over a keyboard genuinely holds tens of\n"
			"degrees of knuckle flexion.");

		if (restPose.present[0] || restPose.present[1])
		{
			ImGui::SameLine();
			if (ImGui::Button("Clear"))
			{
				restPose.present[0]= false;
				restPose.present[1]= false;
				bChanged= true;
			}
		}

		// Poll for a completed capture (the vision thread does the work)
		std::array<HandPoseModel::NeutralDirections, 2> capturedDirs;
		bool bCaptured[2]= {false, false};
		if (visionThread->fetchRestPoseCapture(capturedDirs, bCaptured))
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!bCaptured[sideIndex])
					continue;
				restPose.neutralDirInPalm[sideIndex]= capturedDirs[sideIndex];
				restPose.present[sideIndex]= true;
			}
			if (bCaptured[0] || bCaptured[1])
				bChanged= true;
		}
	}

	ImGui::SeparatorText("Overlay");
	bool bShowOverlay= previewPanel->getShowOverlay();
	if (ImGui::Checkbox("Show skeleton overlay", &bShowOverlay))
		previewPanel->setShowOverlay(bShowOverlay);
	bool bShowBoxes= previewPanel->getShowDetectionBoxes();
	if (ImGui::Checkbox("Show detection boxes", &bShowBoxes))
		previewPanel->setShowDetectionBoxes(bShowBoxes);
	if (config->cameraCount() > 1)
	{
		bool bShowPerCamera= scene3dPanel->getShowPerCameraSkeletons();
		if (ImGui::Checkbox("3D: per-camera skeletons", &bShowPerCamera))
			scene3dPanel->setShowPerCameraSkeletons(bShowPerCamera);
		ImGui::SetItemTooltip(
			"Draws each camera's unfused skeleton dimmed in that camera's\n"
			"color - use to verify the cameras agree in world space\n"
			"(they should overlap within a few cm)");
	}

	ImGui::SeparatorText("Inference");
	for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		ImGui::Text("Camera %d EP: %s", cameraIndex + 1, visionThread->getActiveExecutionProvider(cameraIndex));
	ImGui::Text("Inference (all cameras): %.1f ms", visionThread->getLastInferenceMs());

	ImGui::SeparatorText("Diagnostics");
	if (ImGui::Button("Dump tracking state (F9)"))
		visionThread->requestDiagnosticDump(AppConfig::makeDumpDirectoryPath());
	ImGui::SetItemTooltip(
		"Writes the last few seconds of tracking/fusion history\n"
		"(including cluster + side-assignment scores), the live config\n"
		"and each camera's current frame (raw + annotated PNG) to a\n"
		"timestamped folder - hit it the moment tracking misbehaves.");
	{
		const std::string lastDump= visionThread->getLastDumpPath();
		if (!lastDump.empty())
			ImGui::TextWrapped("Last dump: %s", lastDump.c_str());
	}

	if (bChanged)
	{
		config->markDirty();
		visionThread->requestConfigRefresh();
	}

	ImGui::End();
}

void SettingsPanels::drawOscPanel(AppConfig* config, VisionThread* visionThread,
								  const TrackingFrameResult& fusedResult)
{
	if (!ImGui::Begin("OSC Output"))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	OscConfig& osc= config->osc;

	bChanged|= ImGui::Checkbox("Enabled", &osc.enabled);

	char ipBuffer[64];
	snprintf(ipBuffer, sizeof(ipBuffer), "%s", osc.targetIp.c_str());
	if (ImGui::InputText("Target IP", ipBuffer, sizeof(ipBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
	{
		osc.targetIp= ipBuffer;
		bChanged= true;
	}

	int port= osc.targetPort;
	if (ImGui::InputInt("Port", &port, 0) && port > 0 && port <= 65535)
	{
		osc.targetPort= port;
		bChanged= true;
	}

	bChanged|= ImGui::SliderInt("Max rate", &osc.maxRateHz, 10, 120, "%d Hz");

	bChanged|= ImGui::SliderFloat("Min confidence", &osc.minConfidence, 0.f, 1.f, "%.2f");
	ImGui::SetItemTooltip(
		"Below this fused confidence a hand is streamed as tracked=0 with\n"
		"NO palm/finger messages, so the client can hold its last good\n"
		"pose or blend to a rest pose instead of following jitter.\n"
		"0 = always send. Confidence is on /tracked as the 3rd value;\n"
		"watch the live values below to pick a threshold.");

	ImGui::Separator();
	ImGui::TextDisabled("Space: marker-anchored, meters,\nright-handed, +Z up from table");
	ImGui::TextDisabled("Palm frame: +X fingers, +Z out of palm\nAngles: radians on the wire, degrees below");

	// Live readout of exactly what's being streamed: palm transform + the 20
	// finger angles per hand (shown in degrees)
	static const char* s_fingerNames[FINGER_COUNT]= {"Thumb", "Index", "Middle", "Ring", "Pinky"};
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= fusedResult.poses[sideIndex];
		const char* sideName= sideIndex == (int)eHandSide::Left ? "Left Hand" : "Right Hand";

		if (!ImGui::CollapsingHeader(sideName, ImGuiTreeNodeFlags_DefaultOpen))
			continue;

		// Confidence vs the gate: red while the pose is being withheld
		const bool bGated= pose.tracked && pose.confidence < osc.minConfidence;
		if (bGated)
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "Confidence: %.2f (WITHHELD - below %.2f)",
							   pose.confidence, osc.minConfidence);
		else
			ImGui::Text("Confidence: %.2f", pose.confidence);

		if (!pose.tracked)
		{
			ImGui::TextDisabled("not tracked");
			continue;
		}

		const bool bWorld= pose.hasWorldPose;
		const glm::vec3& palmPos= bWorld ? pose.palmPositionWorld : pose.palmPositionCamera;
		ImGui::Text("Palm: (%.3f, %.3f, %.3f) m %s", palmPos.x, palmPos.y, palmPos.z,
					bWorld ? "" : "(camera space)");

		ImGui::PushID(sideIndex);
		if (ImGui::BeginTable("angles", 5,
							  ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn("Finger");
			ImGui::TableSetupColumn("Lat");
			ImGui::TableSetupColumn("Prox");
			ImGui::TableSetupColumn("Inter");
			ImGui::TableSetupColumn("Dist");
			ImGui::TableHeadersRow();

			constexpr float kRadToDeg= 57.29578f;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				const FingerAngles& angles= pose.fingers[finger];
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::TextUnformatted(s_fingerNames[finger]);
				ImGui::TableNextColumn();
				ImGui::Text("%+.0f", angles.lateral * kRadToDeg);
				ImGui::TableNextColumn();
				ImGui::Text("%+.0f", angles.proximal * kRadToDeg);
				ImGui::TableNextColumn();
				ImGui::Text("%+.0f", angles.intermediate * kRadToDeg);
				ImGui::TableNextColumn();
				ImGui::Text("%+.0f", angles.distal * kRadToDeg);
			}
			ImGui::EndTable();
		}
		ImGui::PopID();
	}

	if (bChanged)
	{
		config->markDirty();
		visionThread->requestConfigRefresh();
	}

	ImGui::End();
}
