#include "SettingsPanels.h"

#include <algorithm>
#include <cstring>

#include "imgui.h"

#include "glm/gtc/quaternion.hpp"

#include "AppConfig.h"
#include "HandPoseModel.h"
#include "HandRoiQuality.h"
#include "Scene3dPanel.h"
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

// Seconds between pressing Capture Rest Pose and the sample being taken -
// long enough to get the mouse hand back into the rest pose
static constexpr float k_restPoseCountdownSeconds= 3.f;
// How long the "captured" banner stays up afterwards
static constexpr float k_restPoseResultSeconds= 4.f;

// Hand bone calibration: the countdown gets the mouse hand back into view,
// then a long window - the estimate is a median over many poses, and one pose
// only measures how well that pose happened to triangulate
static constexpr float k_boneCountdownSeconds= 3.f;
static constexpr float k_boneSampleSeconds= 10.f;
// Per-bone spread (median absolute deviation) above which the capture is
// flagged rather than trusted. Synthetic noise of 4 mm per landmark produces
// about 2.6 mm of spread (--test-bonecalib), so past this the hand was moving
// through poses the cameras could not agree on.
static constexpr float k_boneSpreadWarnMm= 4.f;

// Hold-still jitter test: countdown mirrors the rest-pose capture (the mouse
// hand needs to get back into position), then the sampling window
static constexpr float k_holdStillCountdownSeconds= 3.f;
static constexpr float k_holdStillSampleSeconds= 3.f;
// A side needs at least this many fused samples for its deviation to mean
// anything (one second at 30fps)
static constexpr int k_holdStillMinSamples= 30;

// -- Image quality readout ---------------------------------------------------

// Band classification: good inside [goodLo, goodHi], warn inside
// [warnLo, warnHi] (which contains the good range), bad outside both
struct MetricBand
{
	float goodLo, goodHi;
	float warnLo, warnHi;
};

static int classifyBand(float value, const MetricBand& band)
{
	if (value >= band.goodLo && value <= band.goodHi)
		return 0;
	if (value >= band.warnLo && value <= band.warnHi)
		return 1;
	return 2;
}

static ImVec4 bandColor(int bandClass)
{
	switch (bandClass)
	{
	case 0: return ImVec4(0.4f, 1.f, 0.5f, 1.f);
	case 1: return ImVec4(1.f, 0.85f, 0.3f, 1.f);
	default: return ImVec4(1.f, 0.4f, 0.4f, 1.f);
	}
}

// How far outside the good range a value sits (0 inside), used to pick the
// worst hand when both classify the same
static float bandSeverity(float value, const MetricBand& band)
{
	return std::max(0.f, std::max(band.goodLo - value, value - band.goodHi));
}

struct QualityRowDesc
{
	const char* label;
	const char* format; // printf format for the value in display units
	float displayScale; // raw metric -> display units (100 for ratio -> %)
	MetricBand band;    // in display units
	float (*extract)(const HandImageQuality&);
	const char* tooltip;
};

// Bands live at the analyzer's fixed ~160px working scale, tuned against real
// captures: a well-lit low-jitter camera reads noise ~2.5 and sharpness in the
// low thousands, so sharpness only means something when it drops (blur), and
// high values are just texture/scale
static const QualityRowDesc k_qualityRows[]= {
	{"Luminance", "%.0f", 1.f, {90.f, 180.f, 60.f, 220.f},
	 [](const HandImageQuality& q) { return q.meanLuma; },
	 "Mean brightness of the hand region, 0-255.\n"
	 "Low = underexposed: raise exposure or add light.\n"
	 "High = overexposed: lower exposure before highlights clip."},
	{"Highlight clip", "%.1f%%", 100.f, {0.f, 1.f, 0.f, 5.f},
	 [](const HandImageQuality& q) { return q.highlightClipRatio; },
	 "Blown-out pixels in the hand region. Clipping erases the skin\n"
	 "texture and joint boundaries the landmark model reads - reduce\n"
	 "exposure, or diffuse/angle the light to kill specular hot spots."},
	{"Shadow clip", "%.1f%%", 100.f, {0.f, 5.f, 0.f, 15.f},
	 [](const HandImageQuality& q) { return q.shadowClipRatio; },
	 "Pixels stuck at black in the hand region. Some background is\n"
	 "normal; a high value together with low luminance means real\n"
	 "underexposure."},
	{"Contrast", "%.0f", 1.f, {30.f, 1e9f, 15.f, 1e9f},
	 [](const HandImageQuality& q) { return q.contrast; },
	 "Gray-level spread inside the hand region - the texture the\n"
	 "landmark model actually reads. Raise with more (soft) light;\n"
	 "flat frontal glare and near-clipping both flatten it."},
	{"Separation", "%.0f", 1.f, {20.f, 1e9f, 8.f, 1e9f},
	 [](const HandImageQuality& q) { return q.backgroundSeparation; },
	 "Hand brightness vs the surrounding background. Low = the hand\n"
	 "blends in, which starves palm DETECTION (internal contrast can\n"
	 "still be fine). Change the surface under your hands, or light\n"
	 "the hands rather than the table."},
	{"Sharpness", "%.0f", 1.f, {1000.f, 1e9f, 300.f, 1e9f},
	 [](const HandImageQuality& q) { return q.sharpness; },
	 "Edge strength with sensor noise filtered out first. Low = motion\n"
	 "blur or defocus - shorten the exposure time (adding light to\n"
	 "compensate) or refocus the camera."},
	{"Noise", "%.1f", 1.f, {0.f, 4.f, 0.f, 7.f},
	 [](const HandImageQuality& q) { return q.noise; },
	 "Sensor noise (median-filter residual). High = auto-exposure has\n"
	 "cranked the gain in dim light, which directly destabilizes the\n"
	 "landmarks - add light so the gain comes back down."},
};
// The flicker row is appended after these (per camera, not per hand)
static constexpr int k_flickerRowIndex= (int)IM_ARRAYSIZE(k_qualityRows);

static void drawImageQualitySection(AppConfig* config, TrackingPanelState& panelState,
									const std::vector<VisionPreviewFrame>& latestPreviews)
{
	const int cameraCount= std::min((int)config->cameraCount(),
									(int)TrackingPanelState::kQualityMaxCameras);
	if (cameraCount <= 0)
		return;

	ImGui::TextDisabled("Worst tracked hand per camera, ~1s average");
	ImGui::SetItemTooltip(
		"Diagnoses WHY tracking jitters, in terms of the knobs that fix\n"
		"it: exposure, gain, lighting, background. Statistics come from\n"
		"each hand's region in the exact image the model consumed,\n"
		"at a fixed working scale so values are comparable across\n"
		"resolutions. Hover a metric name for what to adjust. The\n"
		"diagnostic dump (F9) records the raw per-frame series.");

	if (!ImGui::BeginTable("imageQuality", 1 + cameraCount,
						   ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
		return;

	ImGui::TableSetupColumn("Metric");
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		char header[32];
		snprintf(header, sizeof(header), "Camera %d", cameraIndex + 1);
		ImGui::TableSetupColumn(header);
	}
	ImGui::TableHeadersRow();

	// ~1s EMA so the readout is legible at camera rate
	const float emaAlpha= std::min(1.f, ImGui::GetIO().DeltaTime);

	for (int rowIndex= 0; rowIndex <= k_flickerRowIndex; ++rowIndex)
	{
		const bool bFlickerRow= rowIndex == k_flickerRowIndex;
		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::Text("%s", bFlickerRow ? "Flicker" : k_qualityRows[rowIndex].label);
		ImGui::SetItemTooltip("%s", bFlickerRow
			? "Frame-to-frame brightness oscillation of the whole image.\n"
			  "A steady frequency = light PWM/mains flicker beating against\n"
			  "the shutter (change lights, or match the shutter to the mains\n"
			  "rate). Oscillation with no clear frequency = auto-exposure\n"
			  "hunting - lock exposure and gain."
			: k_qualityRows[rowIndex].tooltip);

		for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
		{
			ImGui::TableNextColumn();

			const bool bHavePreview= cameraIndex < (int)latestPreviews.size() &&
									 latestPreviews[cameraIndex].valid;
			float& ema= panelState.qualityEma[cameraIndex][rowIndex];
			bool& bEmaValid= panelState.bQualityEmaValid[cameraIndex][rowIndex];

			float rawValue= 0.f;
			bool bHaveValue= false;
			float flickerHz= 0.f;
			MetricBand band;
			const char* format;
			if (bFlickerRow)
			{
				band= {0.f, 2.f, 0.f, 5.f};
				format= "%.1f%%";
				if (bHavePreview)
				{
					rawValue= latestPreviews[cameraIndex].result.lumaInstability * 100.f;
					flickerHz= latestPreviews[cameraIndex].result.lumaFlickerHz;
					bHaveValue= true;
				}
			}
			else
			{
				const QualityRowDesc& row= k_qualityRows[rowIndex];
				band= row.band;
				format= row.format;
				if (bHavePreview)
				{
					// Worst tracked hand: higher band class first, then the
					// distance outside the good range
					int worstClass= -1;
					float worstSeverity= 0.f;
					for (const TrackedHand& hand : latestPreviews[cameraIndex].result.hands)
					{
						if (!hand.tracked || !hand.imageQuality.valid)
							continue;
						const float value= row.extract(hand.imageQuality) * row.displayScale;
						const int bandClass= classifyBand(value, band);
						const float severity= bandSeverity(value, band);
						if (bandClass > worstClass ||
							(bandClass == worstClass && severity > worstSeverity))
						{
							worstClass= bandClass;
							worstSeverity= severity;
							rawValue= value;
							bHaveValue= true;
						}
					}
				}
			}

			if (!bHaveValue)
			{
				bEmaValid= false;
				ImGui::TextDisabled("-");
				continue;
			}

			ema= bEmaValid ? ema + (rawValue - ema) * emaAlpha : rawValue;
			bEmaValid= true;

			char text[48];
			snprintf(text, sizeof(text), format, ema);
			if (bFlickerRow && flickerHz > 0.f)
			{
				const size_t len= strlen(text);
				snprintf(text + len, sizeof(text) - len, " @%.1fHz", flickerHz);
			}
			ImGui::TextColored(bandColor(classifyBand(ema, band)), "%s", text);
		}
	}
	ImGui::EndTable();
}

void SettingsPanels::drawTrackingPanel(AppConfig* config, VisionThread* visionThread,
									   VideoPreviewPanel* previewPanel, Scene3dPanel* scene3dPanel,
									   TrackingPanelState& panelState,
									   const std::vector<VisionPreviewFrame>& latestPreviews,
									   const TrackingFrameResult& fusedResult)
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

	ImGui::SeparatorText("Image Quality");
	drawImageQualitySection(config, panelState, latestPreviews);

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
			ImGui::TableSetupColumn("Roll error");
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

				// Mounting error, measured against anatomy. Preferred over the
				// twist-axis score because it is SIGNED and means something
				// during any motion, not only a deliberate twist.
				ImGui::TableNextColumn();
				if (status.wristAxialTwistDegrees < -900.f)
				{
					ImGui::TextDisabled("measuring");
				}
				else
				{
					const float residual= fabsf(status.wristAxialTwistDegrees);
					const ImVec4 color= residual < 5.f
						? ImVec4(0.4f, 1.f, 0.5f, 1.f)
						: (residual < 15.f ? ImVec4(1.f, 0.85f, 0.3f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f));
					ImGui::TextColored(color, "%+.0f deg", status.wristAxialTwistDegrees);
				}
				ImGui::SetItemTooltip(
					"Mounting roll error, read straight off anatomy: the wrist\n"
					"cannot rotate about the forearm's long axis, so any twist\n"
					"measured in the wrist joint is calibration error.\n"
					"Near 0 = good. A large value rolls the forearm frame, which\n"
					"makes the elbow bend along a rotated arc and shows up as\n"
					"phantom wrist bend when you pronate.\n"
					"Recalibrate the mounting if this stays large.");
			}
			ImGui::EndTable();
		}

		bChanged|= ImGui::Checkbox("Swap wrists", &imu.swapSides);
		ImGui::SetItemTooltip("If the Joy-Con L is strapped to your RIGHT wrist");

		bChanged|= ImGui::SliderFloat("Forearm length", &config->body.forearmLengthMeters, 0.10f, 0.40f, "%.2f m");
		ImGui::SetItemTooltip(
			"Wrist-to-elbow distance, used to place the elbow back along the\n"
			"MEASURED forearm direction. An error here slides the elbow along\n"
			"the forearm axis without rotating it.\n\n"
			"The mounting wizard's curl stage measures this from the arc the\n"
			"controller sweeps, and overwrites this value. That reads slightly\n"
			"SHORT of a true elbow-to-wrist length, because it is the radius to\n"
			"the controller rather than to the wrist - nudge it up if the elbow\n"
			"marker sits inside your actual elbow in the camera view.");

		// No "vision yaw anchor" slider: the value is settled, and it is still
		// in the config file for anyone who needs to retune it

		if (ImGui::Button("Calibrate Mounting...", ImVec2(-1, 0)))
			panelState.bLaunchMountingWizard= true;
		ImGui::SetItemTooltip(
			"Opens a guided calibration: twist your forearms (which measures\n"
			"each arm's long axis), then curl at the elbows (which measures\n"
			"the roll about that axis, and your forearm length).");

		ImGui::EndDisabled();
	}

	ImGui::SeparatorText("Body Pose");
	{
		ImGui::TextWrapped(
			"Body tracking for measured elbows, shoulders, and head pose. "
			"Opt-in per camera: the person detector only fires on a camera "
			"that sees you upright, so leave overhead cameras off.");

		for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		{
			BodyPoseCameraConfig& bodyPose= config->camera(cameraIndex).bodyPose;
			ImGui::PushID(cameraIndex);

			char label[32];
			snprintf(label, sizeof(label), "Camera %d", cameraIndex + 1);
			bChanged|= ImGui::Checkbox(label, &bodyPose.enabled);

			if (bodyPose.enabled)
			{
				ImGui::SetNextItemWidth(110.f);
				bChanged|= ImGui::SliderInt("divider", &bodyPose.poseFrameDivider, 1, 4);
				ImGui::SetItemTooltip("Pose models run every Nth frame on this camera");

				ImGui::SameLine();
				ImGui::SetNextItemWidth(110.f);
				bChanged|= ImGui::SliderInt("re-detect", &bodyPose.detectorIntervalFrames, 1, 60);
				ImGui::SetItemTooltip(
					"Rebuild the search region from the image every Nth model\n"
					"frame, whatever the model claims about its confidence.\n"
					"Without it the region is only ever grown from the previous\n"
					"landmarks, so a drifting crop feeds itself.");
			}

			ImGui::PopID();
		}

		ImGui::TextWrapped(
			"Body proportions. Only lengths are assumed - every direction is "
			"measured - but they set the scale of the estimates, so correcting "
			"them for your own body is worth a minute.");

		if (ImGui::Button("Measure My Body...", ImVec2(-1, 0)))
			panelState.bLaunchBodyCalibrationWizard= true;
		ImGui::SetItemTooltip(
			"Measures these four lengths against your own body, using the fused\n"
			"wrists as the ruler.\n\n"
			"They are NOT anatomical numbers: they are distances between the\n"
			"pose model's landmarks, which sit inside your real joints by an\n"
			"amount that differs per person and per model. A guessed shoulder\n"
			"width put a measured shoulder 0.8 m too far away.");
		bChanged|=
			ImGui::Checkbox("Upper arm from shoulder width", &config->body.bDeriveUpperArmFromShoulderWidth);
		ImGui::SetItemTooltip(
			"Take the upper arm as a multiple of the shoulder width instead of\n"
			"measuring it. Measuring it needs the arm straight and square to the\n"
			"camera, which is hard to hold at a desk and reads 20%% short when\n"
			"missed - and a short upper arm makes the elbow bend the wrong way.\n"
			"Proportions are stable enough that a multiple of a width that IS\n"
			"easy to measure wins.");
		if (config->body.bDeriveUpperArmFromShoulderWidth)
		{
			bChanged|= ImGui::SliderFloat("Upper arm ratio", &config->body.upperArmPerShoulderWidth, 0.80f,
										  1.40f, "%.2f x shoulders");
			ImGui::SetItemTooltip(
				"NOT the anatomical ratio: the model's shoulder points sit inside\n"
				"your real joints (measured at ~0.74 of a biacromial breadth), so\n"
				"the familiar 'arm is about 1.5 shoulder widths' becomes ~2.0 of\n"
				"THIS width, and the upper arm alone lands near 1.05.");
			ImGui::TextDisabled("   = %.1f cm upper arm", config->body.shoulderWidthMeters *
															 config->body.upperArmPerShoulderWidth * 100.f);
		}
		else
		{
			bChanged|= ImGui::SliderFloat("Upper arm", &config->body.upperArmLengthMeters, 0.20f, 0.45f,
										  "%.2f m");
			ImGui::SetItemTooltip("Shoulder to elbow. Decides which of the two elbow solutions is real.");
		}
		bChanged|= ImGui::SliderFloat("Shoulder width", &config->body.shoulderWidthMeters, 0.25f, 0.60f, "%.2f m");
		ImGui::SetItemTooltip("Between the shoulder joints. Sets the shoulders' distance from the camera.");
		bChanged|= ImGui::SliderFloat("Head width", &config->body.headWidthMeters, 0.10f, 0.22f, "%.2f m");
		ImGui::SetItemTooltip("Ear to ear. Sets the head's distance from the camera.");
		bChanged|= ImGui::SliderFloat("Nose forward", &config->body.noseForwardMeters, 0.05f, 0.18f, "%.2f m");
		ImGui::SetItemTooltip("Ear midpoint to nose tip. Sets head yaw and pitch.");
	}

	ImGui::SeparatorText("Hand Bones");
	{
		ImGui::TextWrapped(
			"Measures your own bone lengths from stereo triangulation. The "
			"landmark model's metric hand is not your hand - its proximal "
			"phalanges run about half length - which both biases the "
			"single-camera depth solve and ships the wrong skeleton to "
			"clients. Move and rotate both hands through varied poses during "
			"the window so no bone stays aligned with a camera's view ray.");

		const bool bCalibrated= config->handSkeleton.present[0] || config->handSkeleton.present[1];
		ImGui::Text("Calibrated  L: %s  R: %s", config->handSkeleton.present[0] ? "yes" : "no",
					config->handSkeleton.present[1] ? "yes" : "no");

		const float deltaSeconds= ImGui::GetIO().DeltaTime;

		bool bSamplingActive= false;
		int liveSamples[2]= {0, 0};
		visionThread->getBoneCalibrationProgress(bSamplingActive, liveSamples[0], liveSamples[1]);

		if (panelState.boneCountdown > 0.f)
		{
			panelState.boneCountdown-= deltaSeconds;
			if (panelState.boneCountdown <= 0.f)
			{
				panelState.boneCountdown= 0.f;
				visionThread->requestBoneCalibration(k_boneSampleSeconds);
			}

			char countdownText[16];
			snprintf(countdownText, sizeof(countdownText), "%d", (int)ceilf(panelState.boneCountdown));
			ImGui::SetWindowFontScale(3.f);
			const float textWidth= ImGui::CalcTextSize(countdownText).x;
			ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "%s", countdownText);
			ImGui::SetWindowFontScale(1.f);

			if (ImGui::Button("Cancel", ImVec2(-1, 0)))
				panelState.boneCountdown= 0.f;
		}
		else if (bSamplingActive)
		{
			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Sampling - move both hands");
			ImGui::Text("Samples  L: %d  R: %d (need %d)", liveSamples[0], liveSamples[1],
						HandBoneCalibrator::k_minSamples);
			if (ImGui::Button("Cancel", ImVec2(-1, 0)))
				visionThread->cancelBoneCalibration();
		}
		else if (!panelState.bBoneReviewPending)
		{
			if (ImGui::Button("Calibrate Hand Bones"))
			{
				panelState.boneCountdown= k_boneCountdownSeconds;
				panelState.bBoneReviewPending= false;
			}
			ImGui::SetItemTooltip(
				"Counts down, then samples every stereo-triangulated frame for\n"
				"a few seconds and takes the median of each bone. Needs BOTH\n"
				"cameras seeing the hand - a monocular pose carries the model's\n"
				"shape, which is the thing being replaced.");

			if (bCalibrated)
			{
				ImGui::SameLine();
				if (ImGui::Button("BoneCalibrationClear##Clear"))
				{
					config->handSkeleton.present[0]= false;
					config->handSkeleton.present[1]= false;
					bChanged= true;
				}
				ImGui::SetItemTooltip(
					"Reverts to the landmark model's proportions and re-enables\n"
					"the stereo auto hand-scale.");
			}
		}

		// Poll for a finished window (the vision thread does the sampling)
		VisionThread::BoneCalibrationCapture capture;
		if (visionThread->fetchBoneCalibration(capture))
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				panelState.bBoneCaptured[sideIndex]= capture.bCaptured[sideIndex];
				panelState.boneSkeleton[sideIndex]= capture.skeleton[sideIndex];
				panelState.boneWorstSpreadMm[sideIndex]=
					capture.quality[sideIndex].worstPhalanxSpread * 1000.f;
				panelState.boneSampleCount[sideIndex]= capture.quality[sideIndex].sampleCount;
			}
			panelState.bBoneReviewPending= true;
		}

		// Review before anything is written: this replaces the geometry every
		// client rebuilds the hand from, so it gets looked at first
		if (panelState.bBoneReviewPending)
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const char* sideName= sideIndex == 0 ? "Left" : "Right";
				if (!panelState.bBoneCaptured[sideIndex])
				{
					ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s: only %d stereo samples, need %d",
									   sideName, panelState.boneSampleCount[sideIndex],
									   HandBoneCalibrator::k_minSamples);
					continue;
				}

				const HandSkeleton& skeleton= panelState.boneSkeleton[sideIndex];
				const float referenceBone= skeleton.baseInPalm[(int)eFinger::Middle].x * 2.f;
				const float spreadMm= panelState.boneWorstSpreadMm[sideIndex];
				const ImVec4 spreadColor= spreadMm > k_boneSpreadWarnMm ? ImVec4(1.f, 0.85f, 0.3f, 1.f)
																		: ImVec4(0.4f, 1.f, 0.4f, 1.f);
				ImGui::TextColored(spreadColor, "%s: wrist->knuckle %.1f mm, %d samples, worst spread %.1f mm",
								   sideName, referenceBone * 1000.f,
								   panelState.boneSampleCount[sideIndex], spreadMm);

				// Proximal phalanxes only: they are the longest bones and where
				// the model is most wrong, so they are what a glance should check
				ImGui::Text("   proximal  index %.1f  middle %.1f  ring %.1f  pinky %.1f mm",
							skeleton.phalanxLengths[(int)eFinger::Index][0] * 1000.f,
							skeleton.phalanxLengths[(int)eFinger::Middle][0] * 1000.f,
							skeleton.phalanxLengths[(int)eFinger::Ring][0] * 1000.f,
							skeleton.phalanxLengths[(int)eFinger::Pinky][0] * 1000.f);
			}

			const bool bAnyCaptured= panelState.bBoneCaptured[0] || panelState.bBoneCaptured[1];
			ImGui::BeginDisabled(!bAnyCaptured);
			if (ImGui::Button("Save"))
			{
				double referenceSum= 0.0;
				int referenceCount= 0;
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					if (!panelState.bBoneCaptured[sideIndex])
						continue;
					config->handSkeleton.skeleton[sideIndex]= panelState.boneSkeleton[sideIndex];
					config->handSkeleton.present[sideIndex]= true;
					referenceSum+=
						panelState.boneSkeleton[sideIndex].baseInPalm[(int)eFinger::Middle].x * 2.0;
					++referenceCount;
				}

				// One scale story: the reference bone the rest of the app
				// reads now comes from the measurement, not the EMA
				if (referenceCount > 0)
				{
					config->handScale.refLengthMeters= referenceSum / (double)referenceCount;
					config->handScale.present= true;
				}

				panelState.bBoneReviewPending= false;
				bChanged= true;
			}
			ImGui::EndDisabled();

			ImGui::SameLine();
			if (ImGui::Button("Discard"))
				panelState.bBoneReviewPending= false;

			ImGui::TextWrapped(
				"Saving moves the thumb's angle zero (its neutral direction "
				"follows the metacarpal, unlike the four fingers, which are "
				"always palm-forward), so a rest-pose recapture is worth doing "
				"afterwards.");
		}
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
				if (ImGui::Button("RestAnglesClear##Clear"))
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

	bool bShowBodyPose= previewPanel->getShowBodyPose();
	if (ImGui::Checkbox("Show body landmarks", &bShowBodyPose))
		previewPanel->setShowBodyPose(bShowBodyPose);
	ImGui::SetItemTooltip(
		"Draws the raw body skeleton on cameras running body pose.\n"
		"Landmarks below the solver's visibility gate are dimmed, and the\n"
		"joints it consumes (shoulders, elbows, wrists) carry their\n"
		"visibility - so a bad elbow can be traced to its source landmark.");
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

	// Frame-loop hitches: one thread serves every camera, so a long phase
	// starves all of them at once and reads downstream as a camera fault
	{
		const int hitchCount= visionThread->getHitchCount();
		if (hitchCount == 0)
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Frame loop: no hitches");
		}
		else
		{
			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Frame loop hitches: %d (worst %s %.0f ms)",
							   hitchCount,
							   VisionThread::getPhaseName(visionThread->getLastHitchPhase()),
							   visionThread->getLastHitchMs());
		}
		ImGui::SetItemTooltip(
			"Loop iterations over 50 ms. Every camera shares this thread, so a\n"
			"hitch drops frames on ALL of them at once and shows up as a\n"
			"synchronized tracking gap. The named phase is where the time\n"
			"went; the log line carries the full breakdown.");
	}

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

	// Hold-still jitter test: THE repeatable A/B number for camera-settings
	// and lighting changes. Rebuilds the 21 world-space joints from each fused
	// pose via forward kinematics (exactly what a client rebuilds) and reports
	// the mean per-landmark deviation over a still window - with the hand
	// genuinely still, deviation IS tracking noise.
	{
		const float deltaSeconds= ImGui::GetIO().DeltaTime;

		if (panelState.holdStillCountdown > 0.f)
		{
			panelState.holdStillCountdown-= deltaSeconds;
			if (panelState.holdStillCountdown <= 0.f)
			{
				panelState.holdStillCountdown= 0.f;
				panelState.holdStillSecondsLeft= k_holdStillSampleSeconds;
				panelState.holdStillLastTimestampMs= 0.0;
				panelState.holdStillAccum[0]= TrackingPanelState::HoldStillAccum();
				panelState.holdStillAccum[1]= TrackingPanelState::HoldStillAccum();
			}

			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "Get ready - hold both hands still... %d",
							   (int)ceilf(panelState.holdStillCountdown));
			ImGui::ProgressBar(1.f - panelState.holdStillCountdown / k_holdStillCountdownSeconds,
							   ImVec2(-1, 4), "");
		}
		else if (panelState.holdStillSecondsLeft > 0.f)
		{
			// Accumulate each NEW fused result (they arrive slower than UI
			// frames; re-adding a held result would fake stability)
			if (fusedResult.timestampMs > 0.0 &&
				fusedResult.timestampMs != panelState.holdStillLastTimestampMs)
			{
				panelState.holdStillLastTimestampMs= fusedResult.timestampMs;
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					const HandPose& pose= fusedResult.poses[sideIndex];
					if (!pose.tracked || !pose.hasWorldPose)
						continue;

					glm::mat4 palmTransform= glm::mat4_cast(pose.palmOrientationWorld);
					palmTransform[3]= glm::vec4(pose.palmPositionWorld, 1.f);
					std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
					HandPoseModel::buildFingerJoints(palmTransform, pose.skeleton, pose.fingers, joints);

					TrackingPanelState::HoldStillAccum& accum= panelState.holdStillAccum[sideIndex];
					int pointIndex= 0;
					auto addPoint= [&accum, &pointIndex](const glm::vec3& point) {
						const glm::dvec3 p(point);
						accum.sum[pointIndex]+= p;
						accum.sumSq[pointIndex]+= p * p;
						++pointIndex;
					};
					addPoint(pose.palmPositionWorld);
					for (int fingerIndex= 0; fingerIndex < FINGER_COUNT; ++fingerIndex)
						for (int jointIndex= 0; jointIndex < 4; ++jointIndex)
							addPoint(joints[fingerIndex][jointIndex]);
					++accum.sampleCount;
				}
			}

			panelState.holdStillSecondsLeft-= deltaSeconds;
			if (panelState.holdStillSecondsLeft <= 0.f)
			{
				panelState.holdStillSecondsLeft= 0.f;
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					const TrackingPanelState::HoldStillAccum& accum= panelState.holdStillAccum[sideIndex];
					if (accum.sampleCount < k_holdStillMinSamples)
					{
						panelState.holdStillDeviationMm[sideIndex]= -1.f;
						continue;
					}

					double deviationSum= 0.0;
					for (int pointIndex= 0; pointIndex < HAND_LANDMARK_COUNT; ++pointIndex)
					{
						const glm::dvec3 mean= accum.sum[pointIndex] / (double)accum.sampleCount;
						const glm::dvec3 variance=
							accum.sumSq[pointIndex] / (double)accum.sampleCount - mean * mean;
						deviationSum+= std::sqrt(std::max(0.0, variance.x + variance.y + variance.z));
					}
					panelState.holdStillDeviationMm[sideIndex]=
						(float)(deviationSum / HAND_LANDMARK_COUNT * 1000.0);
				}
				panelState.bHoldStillHasResult= true;
			}

			ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.f, 1.f), "Sampling - keep holding still...");
			ImGui::ProgressBar(1.f - panelState.holdStillSecondsLeft / k_holdStillSampleSeconds,
							   ImVec2(-1, 4), "");
		}
		else if (ImGui::Button("Hold-Still Jitter Test"))
		{
			panelState.holdStillCountdown= k_holdStillCountdownSeconds;
		}
		if (panelState.holdStillCountdown <= 0.f && panelState.holdStillSecondsLeft <= 0.f)
		{
			ImGui::SetItemTooltip(
				"Hold both hands still for a few seconds and get ONE number\n"
				"per hand: the mean per-landmark deviation over the window.\n"
				"With the hands genuinely still, that deviation IS tracking\n"
				"noise - so run it once per camera-settings or lighting\n"
				"change and compare. Uses the fused world-space output\n"
				"(what clients actually receive).");
		}

		if (panelState.bHoldStillHasResult)
		{
			ImGui::Text("Jitter:");
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				ImGui::SameLine();
				const float deviationMm= panelState.holdStillDeviationMm[sideIndex];
				if (deviationMm < 0.f)
				{
					ImGui::TextDisabled("%s: too few samples", sideIndex == 0 ? "L" : "R");
					continue;
				}
				const ImVec4 color= deviationMm < 2.f
					? ImVec4(0.4f, 1.f, 0.5f, 1.f)
					: (deviationMm < 5.f ? ImVec4(1.f, 0.85f, 0.3f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f));
				ImGui::TextColored(color, "%s: %.1f mm", sideIndex == 0 ? "L" : "R", deviationMm);
			}
		}
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

	int outputMode= (int)osc.outputMode;
	if (ImGui::Combo("Format", &outputMode, "Mikan\0VMC (VRM)\0"))
	{
		osc.outputMode= (eOscOutputMode)outputMode;
		bChanged= true;
	}
	ImGui::SetItemTooltip(
		"Mikan: the native /mikan/* schema - world-space poses, finger\n"
		"angles and per-joint confidences.\n"
		"VMC: the VMC protocol (protocol.vmc.info) - head, clavicle, arm,\n"
		"hand and finger bones as parent-relative Unity transforms, for\n"
		"receivers such as VMC4UE.\n"
		"One at a time: they describe the same pose in incompatible terms.");

	const bool bVmc= osc.outputMode == eOscOutputMode::Vmc;

	char ipBuffer[64];
	snprintf(ipBuffer, sizeof(ipBuffer), "%s", osc.targetIp.c_str());
	if (ImGui::InputText("Target IP", ipBuffer, sizeof(ipBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
	{
		osc.targetIp= ipBuffer;
		bChanged= true;
	}

	// Each format keeps its own port so switching modes cannot silently aim
	// the stream at a listener that speaks the other one
	int& activePort= bVmc ? osc.vmcPort : osc.targetPort;
	int port= activePort;
	if (ImGui::InputInt(bVmc ? "Port (VMC)" : "Port", &port, 0) && port > 0 && port <= 65535)
	{
		activePort= port;
		bChanged= true;
	}
	if (bVmc)
		ImGui::SetItemTooltip("39539 is VMC's conventional Performer -> Marionette port.");

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

	bChanged|= ImGui::Checkbox("Log palm frames", &osc.logPalmFrames);
	ImGui::SetItemTooltip(
		"Writes every palm transform to the log as it goes on the wire,\n"
		"tagged with the frame id, so it can be diffed against a client's\n"
		"own receive log to prove what did or did not arrive.\n"
		"One line per hand per frame - leave it off for normal use.");

	if (bVmc)
	{
		ImGui::Separator();
		ImGui::TextDisabled("VMC");

		bChanged|= ImGui::SliderFloat("Head offset", &osc.vmcHeadOffsetMeters, 0.f, 0.25f, "%.3f m");
		ImGui::SetItemTooltip(
			"Neck -> head bone offset. A VMC receiver replaces the position\n"
			"of every bone it is sent, and nothing here measures a neck, so\n"
			"this is the knob: raise it if the head sinks into the shoulders,\n"
			"lower it if it floats.");

		bChanged|= ImGui::Checkbox("Freeze on loss", &osc.vmcFreezeOnLoss);
		ImGui::SetItemTooltip(
			"VMC carries no confidence, so a lost hand can only be expressed\n"
			"as motion. On: that arm's last bones keep streaming and it holds\n"
			"still. Off: the bones stop, and the receiver returns the arm to\n"
			"the avatar's rest T-pose.");

		ImGui::TextDisabled("Bones: head, clavicles, arms,\nhands, fingers (37)");
		ImGui::TextDisabled("The avatar takes the measured\nbone lengths (Body panel)");
	}

	ImGui::Separator();
	ImGui::TextDisabled("Space: marker-anchored, meters,\nright-handed, +Z up from table");
	ImGui::TextDisabled("Palm frame: +X fingers, +Z out of palm\nAngles: degrees on the wire");

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

			const glm::vec3 elbow= pose.getElbowPositionWorld(config->body.forearmLengthMeters);
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
