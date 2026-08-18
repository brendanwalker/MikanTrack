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

// Hold-still jitter test: countdown mirrors the calibration captures (the mouse
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

	// The tracking/fusion tuning values (staleness window, jitter and
	// residual references, detector cadence) live in config.json and the
	// Timeline what-if panel, where they can be swept against a recording.
	// The live panel holds rig facts, calibration, and readouts only.

	if (config->cameraCount() > 1)
	{
		ImGui::SeparatorText("Fusion");

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
			"what makes the wrist joint angle measurable and places the\n"
			"elbow. The forearm frame is streamed on /mikan/hand/{s}/forearm.");

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

	// -- Hand calibration: one wizard, in the only order that works -----
	ImGui::SeparatorText("Hand Calibration");
	{
		ImGui::Text("Bones  L: %s  R: %s   Rest pose  L: %s  R: %s",
					config->handSkeleton.present[0] ? "yes" : "no",
					config->handSkeleton.present[1] ? "yes" : "no",
					config->fusedRestAngles.present[0] ? "yes" : "no",
					config->fusedRestAngles.present[1] ? "yes" : "no");

		if (ImGui::Button("Calibrate Hands...", ImVec2(-1, 0)))
			panelState.bLaunchHandCalibrationWizard= true;
		ImGui::SetItemTooltip(
			"Guided flow: measure your bone lengths from stereo triangulation,\n"
			"then capture your rest pose - in that order, enforced, because\n"
			"saving bones moves the thumb's angle zero and a rest pose captured\n"
			"first would be measured against a zero that no longer exists.");

		const bool bAnyHandCalibration= config->handSkeleton.present[0] ||
			config->handSkeleton.present[1] || config->fusedRestAngles.present[0] ||
			config->fusedRestAngles.present[1];
		if (bAnyHandCalibration)
		{
			if (ImGui::Button("Clear Hand Calibration"))
			{
				// One Clear for both: they are a matched set (the rest pose is
				// measured against the skeleton's thumb zero)
				config->handSkeleton.present[0]= false;
				config->handSkeleton.present[1]= false;
				config->fusedRestAngles.present[0]= false;
				config->fusedRestAngles.present[1]= false;
				bChanged= true;
			}
			ImGui::SetItemTooltip(
				"Reverts to the landmark model's proportions and the flat-hand\n"
				"zero, and re-enables the stereo auto hand-scale.");
		}
	}

	ImGui::SeparatorText("Overlay");
	bool bShowOverlay= previewPanel->getShowOverlay();
	if (ImGui::Checkbox("Show skeleton overlay", &bShowOverlay))
		previewPanel->setShowOverlay(bShowOverlay);

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
		visionThread->requestDiagnosticDump(config->makeDumpDirectoryPath());
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

		// Wrist bend. NOT a wire value - /mikan/hand/{s}/forearm carries the
		// forearm frame and leaves the joint angle to the consumer, which gets
		// it from the palm the same way this does. Shown in DEGREES because it
		// is the number that tells you whether the mounting calibration is
		// good: the wrist rotation is measured relative to the pose you
		// captured, so a straight wrist should read near zero. A large angle
		// with your wrist actually straight means recalibrate (or yaw has
		// drifted).
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
