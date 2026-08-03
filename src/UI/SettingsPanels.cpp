#include "SettingsPanels.h"

#include "imgui.h"

#include "AppConfig.h"
#include "Scene3dPanel.h"
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

// Seconds between pressing Capture Rest Pose and the sample being taken -
// long enough to get the mouse hand back into the rest pose
static constexpr float k_restPoseCountdownSeconds= 3.f;
// How long the "captured" banner stays up afterwards
static constexpr float k_restPoseResultSeconds= 4.f;

void SettingsPanels::drawTrackingPanel(AppConfig* config, VisionThread* visionThread,
									   VideoPreviewPanel* previewPanel, Scene3dPanel* scene3dPanel,
									   TrackingPanelState& panelState)
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

	bChanged|= ImGui::SliderFloat("Reacquire threshold", &tracking.palmScoreThresholdRelaxed, 0.05f, 0.5f, "%.2f");
	ImGui::SetItemTooltip(
		"Relaxed palm-detection cutoff used for ~1.5s after a tracked\n"
		"hand is lost (normal cutoff is 0.5). Lower = faster reacquisition\n"
		"of a hand that dipped below the strict threshold, at the cost of\n"
		"more detector false positives DURING reacquisition only - the\n"
		"fusion gates (clustering, residual veto, priors) filter those.");

	ImGui::SeparatorText("Depth Estimation");
	bChanged|= ImGui::Checkbox("RealSense hardware depth", &tracking.useRealSenseDepth);
	ImGui::SetItemTooltip(
		"For RealSense cameras (rs:// devices): sample the depth stream at\n"
		"each 2D landmark and build the palm transform from MEASURED metric\n"
		"depth instead of the monocular PnP estimate. Fingertip depth holes\n"
		"fall back to parent-joint depth. No effect on plain webcams.");

	ImGui::SeparatorText("Smoothing");
	bChanged|= ImGui::Checkbox("Enabled", &tracking.smoothingEnabled);
	ImGui::TextDisabled("Palm transform (latency is visible - keep responsive)");
	bChanged|= ImGui::SliderFloat("Palm min cutoff", &tracking.palmMinCutoff, 0.1f, 10.f, "%.2f Hz");
	bChanged|= ImGui::SliderFloat("Palm beta", &tracking.palmBeta, 0.f, 0.5f, "%.3f");
	ImGui::TextDisabled("Finger angles (latency is invisible - keep steady)");
	bChanged|= ImGui::SliderFloat("Angle min cutoff", &tracking.angleMinCutoff, 0.1f, 5.f, "%.2f Hz");
	bChanged|= ImGui::SliderFloat("Angle beta", &tracking.angleBeta, 0.f, 0.5f, "%.3f");

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

		bChanged|= ImGui::Checkbox("Stereo landmark triangulation", &fusion.triangulationEnabled);
		ImGui::SetItemTooltip(
			"When two cameras see the same hand, triangulate all 21\n"
			"landmarks from the 2D image points and extract the pose from\n"
			"real stereo geometry. The network's monocular depth estimate\n"
			"(the noisy, view-dependent part) stays out of the loop\n"
			"entirely. Off = blend the per-camera monocular poses (the\n"
			"previous behavior).");

		ImGui::BeginDisabled(!fusion.triangulationEnabled);
		bChanged|= ImGui::SliderFloat("Max tri residual", &fusion.triangulationMaxResidualPx, 5.f, 80.f, "%.0f px");
		ImGui::SetItemTooltip(
			"RMS reprojection residual above which a triangulated pairing\n"
			"is rejected: two DIFFERENT physical hands wrongly merged\n"
			"triangulate to garbage that projects nowhere near what either\n"
			"camera saw, so this doubles as a correspondence check.");
		ImGui::EndDisabled();

		ImGui::SeparatorText("Observation Confidence");
		bChanged|= ImGui::SliderFloat("Jitter reference", &fusion.jitterReferenceMm, 3.f, 60.f, "%.0f mm");
		ImGui::SetItemTooltip(
			"Palm jitter at which a camera's view counts as half as\n"
			"trustworthy. Confidence = presence x stability, and the blend\n"
			"weight is confidence x how face-on the palm is - so a camera\n"
			"seeing a hand edge-on stops polluting the fused pose.\n"
			"LOWER = stricter. Raise it if fast hand motion is being\n"
			"treated as noise.");

		ImGui::BeginDisabled(!fusion.triangulationEnabled);
		bChanged|= ImGui::SliderFloat("Residual reference", &fusion.residualReferencePx, 2.f, 30.f, "%.0f px");
		ImGui::SetItemTooltip(
			"Triangulation reprojection residual at which a stereo pose\n"
			"counts as half as trustworthy. Unlike presence, the residual\n"
			"directly measures how well the pose explains what BOTH\n"
			"cameras saw this frame. LOWER = stricter.");
		ImGui::EndDisabled();

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

		ImGui::SeparatorText("Hand Scale");
		// Measured live and applied live - there is nothing to press. The
		// wrist->knuckle bone is re-measured from stereo/depth every session,
		// so persisting it would only save the few seconds the EMA takes to
		// converge, at the cost of a button and a stale value to explain.
		const float scaleFactor= visionThread->getAutoHandScaleFactor();
		const double autoScaleMeters= config->handScale.refLengthMeters * (double)scaleFactor;
		ImGui::Text("Measured: %.2f cm (x%.3f)", autoScaleMeters * 100.0, scaleFactor);
		ImGui::SetItemTooltip(
			"Your wrist->knuckle bone length, measured continuously from\n"
			"stereo triangulation / depth. Nothing to configure.");
	}

	ImGui::SeparatorText("Wrist IMU");
	{
		ImuConfig& imu= config->imu;
		bChanged|= ImGui::Checkbox("Enable wrist IMU", &imu.enabled);
		ImGui::SetItemTooltip(
			"Wrist-strapped inertial trackers supply FOREARM orientation at\n"
			"~200 Hz, immune to occlusion. A wrist strap sits proximal to the\n"
			"wrist joint, so it measures the forearm - not the palm - which is\n"
			"what makes the wrist joint angle measurable (streamed on\n"
			"/mikan/hand/{s}/wrist).");

		ImGui::BeginDisabled(!imu.enabled);

		// No scan button: the service rescans on its own while a wrist has no
		// controller, so pairing one in Windows Bluetooth settings is enough
		if (ImGui::BeginTable("imu", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
		{
			ImGui::TableSetupColumn("Wrist");
			ImGui::TableSetupColumn("Device");
			ImGui::TableSetupColumn("Rate");
			ImGui::TableSetupColumn("Yaw drift");
			ImGui::TableSetupColumn("Mounting");
			ImGui::TableHeadersRow();

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const ImuSideStatus status= visionThread->getImuSideStatus((eHandSide)sideIndex);
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("%s", sideIndex == 0 ? "Left" : "Right");

				ImGui::TableNextColumn();
				if (!status.deviceConnected)
					ImGui::TextDisabled("none");
				else if (!status.calibrated)
					ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "%s (uncalibrated)",
									   status.deviceName.c_str());
				else if (status.orientationValid)
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s", status.deviceName.c_str());
				else
					ImGui::TextDisabled("%s (converging)", status.deviceName.c_str());

				ImGui::TableNextColumn();
				if (status.streaming)
				{
					ImGui::Text("%.0f Hz  %d%%", status.sampleRateHz, (int)(status.batteryLevel * 100.f));
				}
				else if (status.deviceConnected && status.millisecondsSinceLastSample > 0.0)
				{
					// Silent but open: asleep or the link dropped. The service
					// reopens it automatically; this just makes it visible.
					ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "silent %.0fs",
									   status.millisecondsSinceLastSample / 1000.0);
				}
				else
				{
					ImGui::TextDisabled("-");
				}

				ImGui::TableNextColumn();
				// The yaw-axis gyro bias is the number that matters: it is the
				// drift vision has to keep correcting
				if (status.streaming)
					ImGui::Text("%.2f deg/s", status.gyroBiasDegreesPerSecond.z);
				else
					ImGui::TextDisabled("-");

				// Mounting quality, scored against real forearm twist
				ImGui::TableNextColumn();
				if (status.forearmAxisConsistency < 0.f)
				{
					ImGui::TextDisabled("twist to test");
				}
				else
				{
					const ImVec4 color= status.forearmAxisConsistency > 0.8f
						? ImVec4(0.4f, 1.f, 0.5f, 1.f)
						: (status.forearmAxisConsistency > 0.5f ? ImVec4(1.f, 0.85f, 0.3f, 1.f)
																: ImVec4(1.f, 0.4f, 0.4f, 1.f));
					ImGui::TextColored(color, "%.2f", status.forearmAxisConsistency);
				}
				ImGui::SetItemTooltip(
					"Mounting quality, measured from your own motion. Twisting a\n"
					"forearm about its long axis must leave the forearm frame's\n"
					"forward axis fixed - so this scores how well it does.\n"
					">0.8 = good. Low means the calibration pose was not a\n"
					"straight wrist, and the elbow will sweep a cone as you\n"
					"twist. Rotate your forearms for a few seconds to fill it in.");
			}
			ImGui::EndTable();
		}

		bChanged|= ImGui::Checkbox("Swap wrists", &imu.swapSides);
		ImGui::SetItemTooltip("If the Joy-Con L is strapped to your RIGHT wrist");

		bChanged|= ImGui::SliderFloat("Forearm length", &imu.forearmLengthMeters, 0.15f, 0.40f, "%.2f m");
		ImGui::SetItemTooltip(
			"Wrist-to-elbow distance, used to place the elbow back along the\n"
			"MEASURED forearm direction. Only the length is assumed - the\n"
			"direction comes from the IMU - so an error here slides the elbow\n"
			"along the forearm axis without rotating it. Tune it by watching\n"
			"the elbow marker against your actual elbow in the camera view.");

		// No "vision yaw anchor" slider: the value is settled, and it is still
		// in the config file for anyone who needs to retune it

		if (ImGui::Button("Calibrate Mounting...", ImVec2(-1, 0)))
			panelState.bLaunchMountingWizard= true;
		ImGui::SetItemTooltip(
			"Opens a guided calibration: twist your forearms (which measures\n"
			"each arm's axis from the motion itself), then hold them straight\n"
			"(which sets the roll about that axis).");

		ImGui::EndDisabled();
	}

	ImGui::SeparatorText("Rest Pose");
	{
		ImGui::TextWrapped(
			"Defines which pose reports all-zero angles. Hold both hands in "
			"your rest pose (flat, fingers together and straight) and capture. "
			"Recorded per camera - each sees the articulation differently.");

		for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		{
			const RestAnglesConfig& cameraRest= config->camera(cameraIndex).restAngles;
			ImGui::Text("Camera %d  L: %s  R: %s", cameraIndex + 1,
						cameraRest.present[0] ? "yes" : "no", cameraRest.present[1] ? "yes" : "no");
		}
		if (config->cameraCount() > 1)
		{
			ImGui::Text("Stereo    L: %s  R: %s",
						config->fusedRestAngles.present[0] ? "yes" : "no",
						config->fusedRestAngles.present[1] ? "yes" : "no");
			ImGui::SetItemTooltip(
				"Zero reference for the triangulated (two-camera) pose path.\n"
				"Captured when both cameras saw the hand during the capture.");
		}

		const float deltaSeconds= ImGui::GetIO().DeltaTime;

		if (panelState.restPoseCountdown > 0.f)
		{
			// Counting down: both hands need to be free, so the sample is
			// taken well after the mouse click that started this
			panelState.restPoseCountdown-= deltaSeconds;
			if (panelState.restPoseCountdown <= 0.f)
			{
				panelState.restPoseCountdown= 0.f;
				visionThread->requestRestPoseCapture();
			}

			char countdownText[16];
			snprintf(countdownText, sizeof(countdownText), "%d",
					 (int)ceilf(panelState.restPoseCountdown));
			ImGui::SetWindowFontScale(3.f);
			const float textWidth= ImGui::CalcTextSize(countdownText).x;
			ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "%s", countdownText);
			ImGui::SetWindowFontScale(1.f);
			ImGui::ProgressBar(1.f - panelState.restPoseCountdown / k_restPoseCountdownSeconds,
							   ImVec2(-1, 4), "");

			if (ImGui::Button("Cancel", ImVec2(-1, 0)))
				panelState.restPoseCountdown= 0.f;
		}
		else
		{
			if (ImGui::Button("Capture Rest Pose"))
			{
				panelState.restPoseCountdown= k_restPoseCountdownSeconds;
				panelState.restPoseResultTimer= 0.f;
			}
			ImGui::SetItemTooltip(
				"Counts down, then captures the tracked hands as the zero\n"
				"reference. Without it, zero means the flat-hand default\n"
				"(fingers parallel to the palm's forward axis), which ignores\n"
				"how your own hand rests - a hand hovering over a keyboard\n"
				"genuinely holds tens of degrees of knuckle flexion.");

			bool bAnyCalibrated= false;
			for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
			{
				const RestAnglesConfig& cameraRest= config->camera(cameraIndex).restAngles;
				bAnyCalibrated|= cameraRest.present[0] || cameraRest.present[1];
			}
			if (bAnyCalibrated)
			{
				ImGui::SameLine();
				if (ImGui::Button("Clear"))
				{
					for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
					{
						RestAnglesConfig& cameraRest= config->camera(cameraIndex).restAngles;
						cameraRest.present[0]= false;
						cameraRest.present[1]= false;
					}
					config->fusedRestAngles.present[0]= false;
					config->fusedRestAngles.present[1]= false;
					bChanged= true;
				}
			}
		}

		// Poll for a completed capture (the vision thread does the work)
		std::vector<VisionThread::RestPoseCapture> captures;
		VisionThread::RestPoseCapture fusedCapture;
		if (visionThread->fetchRestPoseCapture(captures, fusedCapture))
		{
			// A side counts as captured only if EVERY camera got it - a
			// partially calibrated side would make the cameras disagree
			bool bAllCameras[2]= {!captures.empty(), !captures.empty()};
			for (size_t cameraIndex= 0; cameraIndex < captures.size(); ++cameraIndex)
			{
				if (cameraIndex >= config->cameraCount())
					break;

				RestAnglesConfig& cameraRest= config->camera(cameraIndex).restAngles;
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					if (!captures[cameraIndex].bCaptured[sideIndex])
					{
						bAllCameras[sideIndex]= false;
						continue;
					}
					cameraRest.angles[sideIndex]= captures[cameraIndex].angles[sideIndex];
					cameraRest.present[sideIndex]= true;
					bChanged= true;
				}
			}

			// Stereo zero reference (only fills when the fuse that serviced the
			// capture actually triangulated that side)
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (!fusedCapture.bCaptured[sideIndex])
					continue;
				config->fusedRestAngles.angles[sideIndex]= fusedCapture.angles[sideIndex];
				config->fusedRestAngles.present[sideIndex]= true;
				bChanged= true;
			}

			panelState.bRestPoseResultCaptured[0]= bAllCameras[0];
			panelState.bRestPoseResultCaptured[1]= bAllCameras[1];
			panelState.restPoseResultTimer= k_restPoseResultSeconds;
		}

		// Result banner: a hand that was not tracked at the moment of capture
		// is silently skipped otherwise
		if (panelState.restPoseResultTimer > 0.f)
		{
			panelState.restPoseResultTimer-= deltaSeconds;

			const bool bLeft= panelState.bRestPoseResultCaptured[0];
			const bool bRight= panelState.bRestPoseResultCaptured[1];
			if (bLeft && bRight)
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.4f, 1.f), "Captured both hands");
			else if (bLeft || bRight)
				ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Captured %s hand only - the %s hand was not tracked",
								   bLeft ? "left" : "right", bLeft ? "right" : "left");
			else
				ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
								   "Nothing captured - every camera must see both hands");
		}
	}

	ImGui::SeparatorText("Overlay");
	bool bShowOverlay= previewPanel->getShowOverlay();
	if (ImGui::Checkbox("Show skeleton overlay", &bShowOverlay))
		previewPanel->setShowOverlay(bShowOverlay);

	// Depth preview toggle per RealSense camera (colorized depth instead of
	// color in that camera's preview pane; tracking still runs on color)
	for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
	{
		const std::string& devicePath= config->camera(cameraIndex).video.devicePath;
		if (devicePath.rfind("rs://", 0) != 0)
			continue;

		char label[48];
		snprintf(label, sizeof(label), "Depth view (camera %d)", cameraIndex + 1);
		bool bDepthPreview= visionThread->isDepthPreviewEnabled(cameraIndex);
		if (ImGui::Checkbox(label, &bDepthPreview))
			visionThread->setDepthPreviewEnabled(cameraIndex, bDepthPreview);
		ImGui::SetItemTooltip(
			"Show the colorized depth stream in this camera's preview pane.\n"
			"Near = warm, far = cool, BLACK = depth holes - watch which\n"
			"fingers go black to see exactly what the depth sensor loses.\n"
			"Tracking is unaffected (it always consumes the color image).");
	}
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

	bChanged|= ImGui::SliderFloat("Dropout hold", &osc.holdOnDropoutMs, 0.f, 1000.f, "%.0f ms");
	ImGui::SetItemTooltip(
		"After a hand goes untracked (or below min confidence), keep\n"
		"streaming its last good pose with the confidence decaying to\n"
		"zero for this long before reporting tracked=0. Bridges brief\n"
		"2-10 frame losses so the client doesn't slam to its rest-pose\n"
		"blend and back. 0 = report dropouts immediately.");

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

		// Wrist/forearm (what /mikan/hand/{s}/wrist carries). The wrist angle
		// is shown in DEGREES because it is the number that tells you whether
		// the mounting calibration is good: the wrist rotation is measured
		// relative to the pose you captured, so a straight wrist should read
		// near zero. A large angle with your wrist actually straight means
		// recalibrate (or yaw has drifted).
		if (pose.hasForearmPose && bWorld)
		{
			const glm::quat wristRotation= pose.getWristRotation();
			const float wristDegrees= glm::degrees(2.f * asinf(std::min(
				glm::length(glm::vec3(wristRotation.x, wristRotation.y, wristRotation.z)), 1.f)));

			const ImVec4 wristColor= wristDegrees < 25.f  ? ImVec4(0.4f, 1.f, 0.5f, 1.f)
									 : wristDegrees < 60.f ? ImVec4(1.f, 0.85f, 0.3f, 1.f)
														   : ImVec4(1.f, 0.5f, 0.4f, 1.f);
			ImGui::TextColored(wristColor, "Wrist bend: %.0f deg", wristDegrees);
			ImGui::SetItemTooltip(
				"Rotation of the palm relative to the forearm, measured from\n"
				"your mounting-calibration pose. Hold your wrist STRAIGHT: if\n"
				"this doesn't drop near zero, recalibrate the mounting (or the\n"
				"IMU yaw has drifted since you did).");

			const glm::quat& forearm= pose.forearmOrientationWorld;
			ImGui::Text("  forearm quat: (%.3f, %.3f, %.3f, %.3f)", forearm.x, forearm.y, forearm.z,
						forearm.w);
			ImGui::Text("  wrist quat:   (%.3f, %.3f, %.3f, %.3f)", wristRotation.x, wristRotation.y,
						wristRotation.z, wristRotation.w);

			const glm::vec3 elbow= pose.getElbowPositionWorld(config->imu.forearmLengthMeters);
			ImGui::Text("  elbow: (%.3f, %.3f, %.3f) m", elbow.x, elbow.y, elbow.z);
		}
		else
		{
			ImGui::TextDisabled("Wrist: no IMU (streams valid=0)");
		}

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
