#include "SettingsPanels.h"

#include <algorithm>
#include <cstring>

#include "imgui.h"

#include "glm/gtc/quaternion.hpp"

#include "AppConfig.h"
#include "HandPoseModel.h"
#include "HandRoiQuality.h"
#include "LocalizationManager.h"
#include "LocText.h"
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
	const char* labelKey;
	const char* format; // printf format for the value in display units
	float displayScale; // raw metric -> display units (100 for ratio -> %)
	MetricBand band;    // in display units
	float (*extract)(const HandImageQuality&);
	const char* tooltipKey;
};

// Bands live at the analyzer's fixed ~160px working scale, tuned against real
// captures: a well-lit low-jitter camera reads noise ~2.5 and sharpness in the
// low thousands, so sharpness only means something when it drops (blur), and
// high values are just texture/scale
static const QualityRowDesc k_qualityRows[]= {
	{"trackingPanel.luminanceLabel", "%.0f", 1.f, {90.f, 180.f, 60.f, 220.f},
	 [](const HandImageQuality& q) { return q.meanLuma; },
	 "trackingPanel.luminanceTooltip"},
	{"trackingPanel.highlightClipLabel", "%.1f%%", 100.f, {0.f, 1.f, 0.f, 5.f},
	 [](const HandImageQuality& q) { return q.highlightClipRatio; },
	 "trackingPanel.highlightClipTooltip"},
	{"trackingPanel.shadowClipLabel", "%.1f%%", 100.f, {0.f, 5.f, 0.f, 15.f},
	 [](const HandImageQuality& q) { return q.shadowClipRatio; },
	 "trackingPanel.shadowClipTooltip"},
	{"trackingPanel.contrastLabel", "%.0f", 1.f, {30.f, 1e9f, 15.f, 1e9f},
	 [](const HandImageQuality& q) { return q.contrast; },
	 "trackingPanel.contrastTooltip"},
	{"trackingPanel.separationLabel", "%.0f", 1.f, {20.f, 1e9f, 8.f, 1e9f},
	 [](const HandImageQuality& q) { return q.backgroundSeparation; },
	 "trackingPanel.separationTooltip"},
	{"trackingPanel.sharpnessLabel", "%.0f", 1.f, {1000.f, 1e9f, 300.f, 1e9f},
	 [](const HandImageQuality& q) { return q.sharpness; },
	 "trackingPanel.sharpnessTooltip"},
	{"trackingPanel.noiseLabel", "%.1f", 1.f, {0.f, 4.f, 0.f, 7.f},
	 [](const HandImageQuality& q) { return q.noise; },
	 "trackingPanel.noiseTooltip"},
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

	ImGui::TextDisabled("%s", locText("trackingPanel.imageQualityHeader"));
	ImGui::SetItemTooltip("%s", locText("trackingPanel.imageQualityTooltip"));

	if (!ImGui::BeginTable("imageQuality", 1 + cameraCount,
						   ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
		return;

	ImGui::TableSetupColumn(locText("trackingPanel.metricColumn"));
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		const std::string header= locFormat("trackingPanel.cameraFmt", cameraIndex + 1);
		ImGui::TableSetupColumn(header.c_str());
	}
	ImGui::TableHeadersRow();

	// ~1s EMA so the readout is legible at camera rate
	const float emaAlpha= std::min(1.f, ImGui::GetIO().DeltaTime);

	for (int rowIndex= 0; rowIndex <= k_flickerRowIndex; ++rowIndex)
	{
		const bool bFlickerRow= rowIndex == k_flickerRowIndex;
		ImGui::TableNextRow();
		ImGui::TableNextColumn();
		ImGui::Text("%s", bFlickerRow ? locText("trackingPanel.flickerLabel") : locText(k_qualityRows[rowIndex].labelKey));
		ImGui::SetItemTooltip("%s", bFlickerRow
			? locText("trackingPanel.flickerTooltip")
			: locText(k_qualityRows[rowIndex].tooltipKey));

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
	if (!ImGui::Begin(locWindowTitle("windows.tracking")))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	TrackingConfig& tracking= config->tracking;

	bChanged|= ImGui::Checkbox(locLabel("trackingPanel.flipHandedness"), &tracking.flipHandedness);
	ImGui::SetItemTooltip("%s", locText("trackingPanel.flipHandednessTooltip"));

	// The tracking/fusion tuning values (staleness window, jitter and
	// residual references, detector cadence) live in config.json and the
	// Timeline what-if panel, where they can be swept against a recording.
	// The live panel holds rig facts, calibration, and readouts only.

	if (config->cameraCount() > 1)
	{
		ImGui::SeparatorText(locText("trackingPanel.fusionSection"));

		// Which camera won each hand in the last fusion
		const int leftCam= visionThread->getDominantCamera(eHandSide::Left);
		const int rightCam= visionThread->getDominantCamera(eHandSide::Right);
		ImGui::Text(locText("trackingPanel.dominantCameraFmt"),
					leftCam >= 0 ? std::to_string(leftCam + 1).c_str() : "-",
					rightCam >= 0 ? std::to_string(rightCam + 1).c_str() : "-");

		// Live per-camera confidence: the number the thresholds act on
		if (ImGui::BeginTable("confidence", 3, ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("");
			ImGui::TableSetupColumn(locText("trackingPanel.left"));
			ImGui::TableSetupColumn(locText("trackingPanel.right"));
			ImGui::TableHeadersRow();
			for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
			{
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text(locText("trackingPanel.cameraFmt"), cameraIndex + 1);
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

		ImGui::SeparatorText(locText("trackingPanel.handScaleSection"));
		// Measured live and applied live - there is nothing to press. The
		// wrist->knuckle bone is re-measured from stereo/depth every session,
		// so persisting it would only save the few seconds the EMA takes to
		// converge, at the cost of a button and a stale value to explain.
		const float scaleFactor= visionThread->getAutoHandScaleFactor();
		const double autoScaleMeters= config->handScale.refLengthMeters * (double)scaleFactor;
		ImGui::Text(locText("trackingPanel.measuredScaleFmt"), autoScaleMeters * 100.0, scaleFactor);
		ImGui::SetItemTooltip("%s", locText("trackingPanel.measuredScaleTooltip"));
	}

	ImGui::SeparatorText(locText("trackingPanel.imageQualitySection"));
	drawImageQualitySection(config, panelState, latestPreviews);

	ImGui::SeparatorText(locText("trackingPanel.wristImuSection"));
	{
		ImuConfig& imu= config->imu;
		bChanged|= ImGui::Checkbox(locLabel("trackingPanel.enableWristImu"), &imu.enabled);
		ImGui::SetItemTooltip("%s", locText("trackingPanel.enableWristImuTooltip"));

		ImGui::BeginDisabled(!imu.enabled);

		// No scan button: the service rescans on its own while a wrist has no
		// controller, so pairing one in Windows Bluetooth settings is enough
		if (ImGui::BeginTable("imu", 5, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
		{
			ImGui::TableSetupColumn(locText("trackingPanel.wristTableWrist"));
			ImGui::TableSetupColumn(locText("trackingPanel.wristTableDevice"));
			ImGui::TableSetupColumn(locText("trackingPanel.wristTableRate"));
			ImGui::TableSetupColumn(locText("trackingPanel.wristTableYawDrift"));
			ImGui::TableSetupColumn(locText("trackingPanel.wristTableRollError"));
			ImGui::TableHeadersRow();

			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const ImuSideStatus status= visionThread->getImuSideStatus((eHandSide)sideIndex);
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::Text("%s", sideIndex == 0 ? locText("trackingPanel.left") : locText("trackingPanel.right"));

				ImGui::TableNextColumn();
				if (!status.deviceConnected)
					ImGui::TextDisabled("%s", locText("trackingPanel.deviceNone"));
				else if (!status.calibrated)
					ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), locText("trackingPanel.deviceUncalibratedFmt"),
									   status.deviceName.c_str());
				else if (status.orientationValid)
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s", status.deviceName.c_str());
				else
					ImGui::TextDisabled(locText("trackingPanel.deviceConvergingFmt"), status.deviceName.c_str());

				ImGui::TableNextColumn();
				if (status.streaming)
				{
					ImGui::Text("%.0f Hz  %d%%", status.sampleRateHz, (int)(status.batteryLevel * 100.f));
				}
				else if (status.deviceConnected && status.millisecondsSinceLastSample > 0.0)
				{
					// Silent but open: asleep or the link dropped. The service
					// reopens it automatically; this just makes it visible.
					ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), locText("trackingPanel.deviceSilentFmt"),
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
					ImGui::TextDisabled("%s", locText("trackingPanel.measuring"));
				}
				else
				{
					const float residual= fabsf(status.wristAxialTwistDegrees);
					const ImVec4 color= residual < 5.f
						? ImVec4(0.4f, 1.f, 0.5f, 1.f)
						: (residual < 15.f ? ImVec4(1.f, 0.85f, 0.3f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f));
					ImGui::TextColored(color, "%+.0f deg", status.wristAxialTwistDegrees);
				}
				ImGui::SetItemTooltip("%s", locText("trackingPanel.wristRollErrorTooltip"));
			}
			ImGui::EndTable();
		}

		bChanged|= ImGui::Checkbox(locLabel("trackingPanel.swapWrists"), &imu.swapSides);
		ImGui::SetItemTooltip("%s", locText("trackingPanel.swapWristsTooltip"));

		bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.forearmLength"), &config->body.forearmLengthMeters, 0.10f, 0.40f, "%.2f m");
		ImGui::SetItemTooltip("%s", locText("trackingPanel.forearmLengthTooltip"));

		// No "vision yaw anchor" slider: the value is settled, and it is still
		// in the config file for anyone who needs to retune it

		if (ImGui::Button(locLabel("trackingPanel.calibrateMounting"), ImVec2(-1, 0)))
			panelState.bLaunchMountingWizard= true;
		ImGui::SetItemTooltip("%s", locText("trackingPanel.calibrateMountingTooltip"));

		ImGui::EndDisabled();
	}

	ImGui::SeparatorText(locText("trackingPanel.bodyPoseSection"));
	{
		ImGui::TextWrapped("%s", locText("trackingPanel.bodyPoseIntro"));

		for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		{
			BodyPoseCameraConfig& bodyPose= config->camera(cameraIndex).bodyPose;
			ImGui::PushID(cameraIndex);

			const std::string label= locFormat("trackingPanel.cameraFmt", cameraIndex + 1);
			bChanged|= ImGui::Checkbox(label.c_str(), &bodyPose.enabled);

			ImGui::PopID();
		}

		ImGui::TextWrapped("%s", locText("trackingPanel.bodyPoseProportionsIntro"));

		if (ImGui::Button(locLabel("trackingPanel.measureMyBody"), ImVec2(-1, 0)))
			panelState.bLaunchBodyCalibrationWizard= true;
		ImGui::SetItemTooltip("%s", locText("trackingPanel.measureMyBodyTooltip"));
		bChanged|=
			ImGui::Checkbox(locLabel("trackingPanel.deriveUpperArmFromShoulderWidth"), &config->body.bDeriveUpperArmFromShoulderWidth);
		ImGui::SetItemTooltip(locText("trackingPanel.deriveUpperArmTooltip"));
		if (config->body.bDeriveUpperArmFromShoulderWidth)
		{
			bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.upperArmRatio"), &config->body.upperArmPerShoulderWidth, 0.80f,
										  1.40f, locText("trackingPanel.upperArmRatioFmt"));
			ImGui::SetItemTooltip("%s", locText("trackingPanel.upperArmRatioTooltip"));
			ImGui::TextDisabled(locText("trackingPanel.upperArmDerivedFmt"), config->body.shoulderWidthMeters *
															 config->body.upperArmPerShoulderWidth * 100.f);
		}
		else
		{
			bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.upperArm"), &config->body.upperArmLengthMeters, 0.20f, 0.45f,
										  "%.2f m");
			ImGui::SetItemTooltip("%s", locText("trackingPanel.upperArmTooltip"));
		}
		bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.shoulderWidth"), &config->body.shoulderWidthMeters, 0.25f, 0.60f, "%.2f m");
		ImGui::SetItemTooltip("%s", locText("trackingPanel.shoulderWidthTooltip"));
		bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.headWidth"), &config->body.headWidthMeters, 0.10f, 0.22f, "%.2f m");
		ImGui::SetItemTooltip("%s", locText("trackingPanel.headWidthTooltip"));
		bChanged|= ImGui::SliderFloat(locLabel("trackingPanel.noseForward"), &config->body.noseForwardMeters, 0.05f, 0.18f, "%.2f m");
		ImGui::SetItemTooltip("%s", locText("trackingPanel.noseForwardTooltip"));
	}

	// -- Hand calibration: one wizard, in the only order that works -----
	ImGui::SeparatorText(locText("trackingPanel.handCalibrationSection"));
	{
		ImGui::Text(locText("trackingPanel.handCalibrationStatusFmt"),
					config->handSkeleton.present[0] ? locText("common.yes") : locText("common.no"),
					config->handSkeleton.present[1] ? locText("common.yes") : locText("common.no"),
					config->fusedRestAngles.present[0] ? locText("common.yes") : locText("common.no"),
					config->fusedRestAngles.present[1] ? locText("common.yes") : locText("common.no"));

		if (ImGui::Button(locLabel("trackingPanel.calibrateHands"), ImVec2(-1, 0)))
			panelState.bLaunchHandCalibrationWizard= true;
		ImGui::SetItemTooltip("%s", locText("trackingPanel.calibrateHandsTooltip"));

		const bool bAnyHandCalibration= config->handSkeleton.present[0] ||
			config->handSkeleton.present[1] || config->fusedRestAngles.present[0] ||
			config->fusedRestAngles.present[1];
		if (bAnyHandCalibration)
		{
			if (ImGui::Button(locLabel("trackingPanel.clearHandCalibration")))
			{
				// One Clear for both: they are a matched set (the rest pose is
				// measured against the skeleton's thumb zero)
				config->handSkeleton.present[0]= false;
				config->handSkeleton.present[1]= false;
				config->fusedRestAngles.present[0]= false;
				config->fusedRestAngles.present[1]= false;
				bChanged= true;
			}
			ImGui::SetItemTooltip("%s", locText("trackingPanel.clearHandCalibrationTooltip"));
		}
	}

	ImGui::SeparatorText(locText("trackingPanel.overlaySection"));
	bool bShowOverlay= previewPanel->getShowOverlay();
	if (ImGui::Checkbox(locLabel("trackingPanel.showSkeletonOverlay"), &bShowOverlay))
		previewPanel->setShowOverlay(bShowOverlay);

	bool bShowBoxes= previewPanel->getShowDetectionBoxes();
	if (ImGui::Checkbox(locLabel("trackingPanel.showDetectionBoxes"), &bShowBoxes))
		previewPanel->setShowDetectionBoxes(bShowBoxes);

	bool bShowBodyPose= previewPanel->getShowBodyPose();
	if (ImGui::Checkbox(locLabel("trackingPanel.showBodyLandmarks"), &bShowBodyPose))
		previewPanel->setShowBodyPose(bShowBodyPose);
	ImGui::SetItemTooltip("%s", locText("trackingPanel.showBodyLandmarksTooltip"));
	if (config->cameraCount() > 1)
	{
		bool bShowPerCamera= scene3dPanel->getShowPerCameraSkeletons();
		if (ImGui::Checkbox(locLabel("trackingPanel.showPerCameraSkeletons"), &bShowPerCamera))
			scene3dPanel->setShowPerCameraSkeletons(bShowPerCamera);
		ImGui::SetItemTooltip("%s", locText("trackingPanel.showPerCameraSkeletonsTooltip"));
	}

	ImGui::SeparatorText(locText("trackingPanel.inferenceSection"));
	for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		ImGui::Text(locText("trackingPanel.cameraEpFmt"), cameraIndex + 1, visionThread->getActiveExecutionProvider(cameraIndex));
	ImGui::Text(locText("trackingPanel.inferenceMsFmt"), visionThread->getLastInferenceMs());

	// Frame-loop hitches: one thread serves every camera, so a long phase
	// starves all of them at once and reads downstream as a camera fault
	{
		const int hitchCount= visionThread->getHitchCount();
		if (hitchCount == 0)
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s", locText("trackingPanel.frameLoopNoHitches"));
		}
		else
		{
			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), locText("trackingPanel.frameLoopHitchesFmt"),
							   hitchCount,
							   VisionThread::getPhaseName(visionThread->getLastHitchPhase()),
							   visionThread->getLastHitchMs());
		}
		ImGui::SetItemTooltip("%s", locText("trackingPanel.frameLoopHitchesTooltip"));
	}

	ImGui::SeparatorText(locText("trackingPanel.diagnosticsSection"));
	if (ImGui::Button(locLabel("trackingPanel.dumpTrackingState")))
		visionThread->requestDiagnosticDump(config->makeDumpDirectoryPath());
	ImGui::SetItemTooltip("%s", locText("trackingPanel.dumpTrackingStateTooltip"));
	{
		const std::string lastDump= visionThread->getLastDumpPath();
		if (!lastDump.empty())
			ImGui::TextWrapped(locText("trackingPanel.lastDumpFmt"), lastDump.c_str());
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

			ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), locText("trackingPanel.holdStillCountdownFmt"),
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

			ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.f, 1.f), "%s", locText("trackingPanel.holdStillSampling"));
			ImGui::ProgressBar(1.f - panelState.holdStillSecondsLeft / k_holdStillSampleSeconds,
							   ImVec2(-1, 4), "");
		}
		else if (ImGui::Button(locLabel("trackingPanel.holdStillJitterTest")))
		{
			panelState.holdStillCountdown= k_holdStillCountdownSeconds;
		}
		if (panelState.holdStillCountdown <= 0.f && panelState.holdStillSecondsLeft <= 0.f)
		{
			ImGui::SetItemTooltip("%s", locText("trackingPanel.holdStillJitterTestTooltip"));
		}

		if (panelState.bHoldStillHasResult)
		{
			ImGui::Text("%s", locText("trackingPanel.jitterLabel"));
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				ImGui::SameLine();
				const float deviationMm= panelState.holdStillDeviationMm[sideIndex];
				if (deviationMm < 0.f)
				{
					ImGui::TextDisabled(locText("trackingPanel.jitterTooFewSamplesFmt"),
										sideIndex == 0 ? locText("trackingPanel.leftAbbr") : locText("trackingPanel.rightAbbr"));
					continue;
				}
				const ImVec4 color= deviationMm < 2.f
					? ImVec4(0.4f, 1.f, 0.5f, 1.f)
					: (deviationMm < 5.f ? ImVec4(1.f, 0.85f, 0.3f, 1.f) : ImVec4(1.f, 0.4f, 0.4f, 1.f));
				ImGui::TextColored(color, "%s: %.1f mm", sideIndex == 0 ? locText("trackingPanel.leftAbbr") : locText("trackingPanel.rightAbbr"), deviationMm);
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
	if (!ImGui::Begin(locWindowTitle("windows.oscOutput")))
	{
		ImGui::End();
		return;
	}

	bool bChanged= false;
	OscConfig& osc= config->osc;

	bChanged|= ImGui::Checkbox(locLabel("oscPanel.enabled"), &osc.enabled);

	static const char* const k_oscFormatKeys[]= {"oscPanel.formatMikan", "oscPanel.formatVmc"};
	int outputMode= (int)osc.outputMode;
	if (ImGui::BeginCombo(locLabel("oscPanel.format"), locText(k_oscFormatKeys[outputMode])))
	{
		for (int optionIndex= 0; optionIndex < IM_ARRAYSIZE(k_oscFormatKeys); ++optionIndex)
		{
			const bool bSelected= outputMode == optionIndex;
			if (ImGui::Selectable(locLabel(k_oscFormatKeys[optionIndex]), bSelected))
			{
				osc.outputMode= (eOscOutputMode)optionIndex;
				bChanged= true;
			}
			if (bSelected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	ImGui::SetItemTooltip("%s", locText("oscPanel.formatTooltip"));

	const bool bVmc= osc.outputMode == eOscOutputMode::Vmc;

	char ipBuffer[64];
	snprintf(ipBuffer, sizeof(ipBuffer), "%s", osc.targetIp.c_str());
	if (ImGui::InputText(locLabel("oscPanel.targetIp"), ipBuffer, sizeof(ipBuffer), ImGuiInputTextFlags_EnterReturnsTrue))
	{
		osc.targetIp= ipBuffer;
		bChanged= true;
	}

	// Each format keeps its own port so switching modes cannot silently aim
	// the stream at a listener that speaks the other one
	int& activePort= bVmc ? osc.vmcPort : osc.targetPort;
	int port= activePort;
	if (ImGui::InputInt(bVmc ? locLabel("oscPanel.portVmc") : locLabel("oscPanel.port"), &port, 0) && port > 0 && port <= 65535)
	{
		activePort= port;
		bChanged= true;
	}
	if (bVmc)
		ImGui::SetItemTooltip("%s", locText("oscPanel.portVmcTooltip"));

	bChanged|= ImGui::SliderInt(locLabel("oscPanel.maxRate"), &osc.maxRateHz, 10, 120, "%d Hz");

	bChanged|= ImGui::SliderFloat(locLabel("oscPanel.minConfidence"), &osc.minConfidence, 0.f, 1.f, "%.2f");
	ImGui::SetItemTooltip("%s", locText("oscPanel.minConfidenceTooltip"));

	bChanged|= ImGui::SliderFloat(locLabel("oscPanel.dropoutHold"), &osc.holdOnDropoutMs, 0.f, 1000.f, "%.0f ms");
	ImGui::SetItemTooltip("%s", locText("oscPanel.dropoutHoldTooltip"));

	bChanged|= ImGui::Checkbox(locLabel("oscPanel.logPalmFrames"), &osc.logPalmFrames);
	ImGui::SetItemTooltip("%s", locText("oscPanel.logPalmFramesTooltip"));

	if (bVmc)
	{
		ImGui::Separator();
		ImGui::TextDisabled("%s", locText("oscPanel.vmcSection"));

		bChanged|= ImGui::SliderFloat(locLabel("oscPanel.headOffset"), &osc.vmcHeadOffsetMeters, 0.f, 0.25f, "%.3f m");
		ImGui::SetItemTooltip("%s", locText("oscPanel.headOffsetTooltip"));

		bChanged|= ImGui::Checkbox(locLabel("oscPanel.freezeOnLoss"), &osc.vmcFreezeOnLoss);
		ImGui::SetItemTooltip("%s", locText("oscPanel.freezeOnLossTooltip"));

		ImGui::TextDisabled("%s", locText("oscPanel.vmcBonesInfo"));
		ImGui::TextDisabled("%s", locText("oscPanel.vmcAvatarBoneLengths"));
	}

	ImGui::Separator();
	ImGui::TextDisabled("%s", locText("oscPanel.spaceInfo"));
	ImGui::TextDisabled("%s", locText("oscPanel.palmFrameInfo"));

	// Live readout of exactly what's being streamed: palm transform + the 20
	// finger angles per hand (shown in degrees)
	static const char* s_fingerNames[FINGER_COUNT]= {
		"oscPanel.fingerThumb", "oscPanel.fingerIndex", "oscPanel.fingerMiddle",
		"oscPanel.fingerRing", "oscPanel.fingerPinky"};
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= fusedResult.poses[sideIndex];
		const char* sideKey= sideIndex == (int)eHandSide::Left ? "oscPanel.leftHand" : "oscPanel.rightHand";

		if (!ImGui::CollapsingHeader(locLabel(sideKey), ImGuiTreeNodeFlags_DefaultOpen))
			continue;

		// Confidence vs the gate: red while the pose is being withheld
		const bool bGated= pose.tracked && pose.confidence < osc.minConfidence;
		if (bGated)
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), locText("oscPanel.confidenceWithheldFmt"),
							   pose.confidence, osc.minConfidence);
		else
			ImGui::Text(locText("oscPanel.confidenceFmt"), pose.confidence);

		if (!pose.tracked)
		{
			ImGui::TextDisabled("%s", locText("oscPanel.notTracked"));
			continue;
		}

		const bool bWorld= pose.hasWorldPose;
		const glm::vec3& palmPos= bWorld ? pose.palmPositionWorld : pose.palmPositionCamera;
		ImGui::Text(locText("oscPanel.palmFmt"), palmPos.x, palmPos.y, palmPos.z,
					bWorld ? "" : locText("oscPanel.cameraSpaceSuffix"));

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
			ImGui::TextColored(wristColor, locText("oscPanel.wristBendFmt"), wristDegrees);
			ImGui::SetItemTooltip("%s", locText("oscPanel.wristBendTooltip"));

			const glm::quat& forearm= pose.forearmOrientationWorld;
			ImGui::Text(locText("oscPanel.forearmQuatFmt"), forearm.x, forearm.y, forearm.z,
						forearm.w);
			ImGui::Text(locText("oscPanel.wristQuatFmt"), wristRotation.x, wristRotation.y,
						wristRotation.z, wristRotation.w);

			const glm::vec3 elbow= pose.getElbowPositionWorld(config->body.forearmLengthMeters);
			ImGui::Text(locText("oscPanel.elbowFmt"), elbow.x, elbow.y, elbow.z);
		}
		else
		{
			ImGui::TextDisabled("%s", locText("oscPanel.wristNoImu"));
		}

		ImGui::PushID(sideIndex);
		if (ImGui::BeginTable("angles", 5,
							  ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders))
		{
			ImGui::TableSetupColumn(locText("oscPanel.fingerColumn"));
			ImGui::TableSetupColumn(locText("oscPanel.latColumn"));
			ImGui::TableSetupColumn(locText("oscPanel.proxColumn"));
			ImGui::TableSetupColumn(locText("oscPanel.interColumn"));
			ImGui::TableSetupColumn(locText("oscPanel.distColumn"));
			ImGui::TableHeadersRow();

			constexpr float kRadToDeg= 57.29578f;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				const FingerAngles& angles= pose.fingers[finger];
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				ImGui::TextUnformatted(locText(s_fingerNames[finger]));
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

void SettingsPanels::drawLanguageCombo()
{
	LocalizationManager* localization= LocalizationManager::getInstance();
	if (localization == nullptr)
		return;

	const std::vector<LocalizationManager::LanguageInfo> languages=
		localization->getSupportedLanguages();

	const char* currentNativeName= localization->getLanguage().c_str();
	for (const LocalizationManager::LanguageInfo& info : languages)
	{
		if (info.code == localization->getLanguage())
			currentNativeName= info.nativeName.c_str();
	}

	if (ImGui::BeginCombo(locLabel("settingsPanel.language"), currentNativeName))
	{
		for (const LocalizationManager::LanguageInfo& info : languages)
		{
			ImGui::PushID(info.code.c_str());
			if (ImGui::Selectable(info.nativeName.c_str(), info.code == localization->getLanguage()))
				localization->setLanguage(info.code);
			ImGui::PopID();
		}
		ImGui::EndCombo();
	}
}

void SettingsPanels::drawAppSettingsPanel()
{
	if (!ImGui::Begin(locWindowTitle("windows.settings")))
	{
		ImGui::End();
		return;
	}

	drawLanguageCombo();

	ImGui::End();
}
