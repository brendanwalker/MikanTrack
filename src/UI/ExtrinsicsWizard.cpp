#include "ExtrinsicsWizard.h"

#include <algorithm>

#include "opencv2/imgproc.hpp"

#include "glm/gtc/matrix_inverse.hpp"

#include "AppConfig.h"
#include "ArucoMarkerPoseSampler.h"
#include "CalibrationPatternFinder_Aruco.h"
#include "Logger.h"
#include "PatternExport.h"
#include "PathUtils.h"
#include "VisionThread.h"

ExtrinsicsWizard::ExtrinsicsWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

ExtrinsicsWizard::~ExtrinsicsWizard()= default;

void ExtrinsicsWizard::enter(int cameraIndex)
{
	m_cameraIndex= cameraIndex;
	m_bActive= true;
	m_bWantsClose= false;
	m_state= eState::VerifySetup;

	m_markerId= m_config->camera(m_cameraIndex).extrinsics.markerId;
	m_markerLengthMM= (float)m_config->camera(m_cameraIndex).extrinsics.markerLengthMM;
	m_bHasCameraPose= false;
	m_handScaleSamples.clear();
	m_measuredHandScale= 0.0;

	// Marker detection wants undistorted frames; tracking only needed for
	// hand scale. Only this camera is affected - the others keep tracking.
	m_visionThread->setTrackingEnabled(m_cameraIndex, false);
	m_visionThread->setUndistortEnabled(m_cameraIndex, true);
}

void ExtrinsicsWizard::exit()
{
	m_poseSampler= nullptr;
	m_bActive= false;

	m_visionThread->setTrackingEnabled(m_cameraIndex, true);
	m_visionThread->requestConfigRefresh();
}

void ExtrinsicsWizard::beginPoseCapture(int frameWidth, int frameHeight)
{
	// Guard OpenCV parameter asserts (invalid marker size/id) - surface the
	// error instead of crashing the app
	try
	{
		m_poseSampler= std::make_unique<ArucoMarkerPoseSampler>(
			m_config->camera(m_cameraIndex).intrinsics.intrinsics,
			frameWidth, frameHeight,
			m_markerLengthMM,
			m_markerId,
			eCharucoDictionaryType::DICT_6X6,
			12);
		m_state= eState::CaptureCameraPose;
	}
	catch (const cv::Exception& e)
	{
		MIKAN_LOG_ERROR("ExtrinsicsWizard::beginPoseCapture") << "OpenCV error creating pose sampler: " << e.what();
		m_poseSampler= nullptr;
	}
}

bool ExtrinsicsWizard::areMarkerParamsValid(std::string& outError) const
{
	// DICT_6X6_250 has marker ids 0..249
	if (m_markerId < 0 || m_markerId > 249)
	{
		outError= "Marker ID must be 0-249 (DICT_6X6_250)";
		return false;
	}
	if (m_markerLengthMM <= 0.f)
	{
		outError= "Marker size must be positive";
		return false;
	}
	return true;
}

glm::dmat4 ExtrinsicsWizard::computeWorldFromCameraPose(const glm::dmat4& cameraFromMarker)
{
	// The solvePnP object points lie in the marker's local XZ plane, so the
	// marker's plane normal is its local +/-Y axis. Build the world frame so
	// +Z is the normal pointing toward the camera (up out of the table):
	//   world X = marker X, world Z = upSign * marker Y, world Y = -upSign * marker Z
	const glm::dvec3 cameraInMarker= glm::dvec3(glm::inverse(cameraFromMarker)[3]);
	const double upSign= cameraInMarker.y >= 0.0 ? 1.0 : -1.0;

	glm::dmat4 worldFromMarker(1.0);
	worldFromMarker[0]= glm::dvec4(1, 0, 0, 0);        // marker X -> world X
	worldFromMarker[1]= glm::dvec4(0, 0, upSign, 0);   // marker Y -> world Z (up)
	worldFromMarker[2]= glm::dvec4(0, -upSign, 0, 0);  // marker Z -> world Y (right-handed)

	// GL camera space -> marker space -> Z-up world space
	return worldFromMarker * glm::inverse(cameraFromMarker);
}

bool ExtrinsicsWizard::raycastPixelOntoPlane(const MikanMonoIntrinsics& intrinsics,
											 const glm::dmat4& cameraFromMarker,
											 const glm::vec2& pixel,
											 glm::dvec3& outPoint)
{
	// Undistorted pinhole ray in OpenCV camera space, flipped to GL convention
	const double fx= intrinsics.undistorted_camera_matrix.x0;
	const double fy= intrinsics.undistorted_camera_matrix.y1;
	const double cx= intrinsics.undistorted_camera_matrix.z0;
	const double cy= intrinsics.undistorted_camera_matrix.z1;
	if (fx <= 0.0 || fy <= 0.0)
		return false;

	const double cvX= ((double)pixel.x - cx) / fx;
	const double cvY= ((double)pixel.y - cy) / fy;
	const glm::dvec3 rayDir= glm::normalize(glm::dvec3(cvX, -cvY, -1.0)); // GL camera space

	// Marker/table plane in GL camera space. The plane normal is the marker's
	// local Y axis (the solvePnP object points span the marker's XZ plane) -
	// the sign doesn't matter for the intersection
	const glm::dvec3 planeOrigin= glm::dvec3(cameraFromMarker[3]);
	const glm::dvec3 planeNormal= glm::normalize(glm::dvec3(glm::dmat3(cameraFromMarker) * glm::dvec3(0, 1, 0)));

	const double denominator= glm::dot(rayDir, planeNormal);
	if (fabs(denominator) < 1e-6)
		return false;

	const double t= glm::dot(planeOrigin, planeNormal) / denominator;
	if (t <= 0.0)
		return false;

	outPoint= rayDir * t;
	return true;
}

glm::dmat4 ExtrinsicsWizard::computeWorldFromCamera() const
{
	return computeWorldFromCameraPose(m_cameraFromMarker);
}

bool ExtrinsicsWizard::raycastPixelOntoMarkerPlane(const glm::vec2& pixel, glm::dvec3& outPoint) const
{
	return raycastPixelOntoPlane(m_config->camera(m_cameraIndex).intrinsics.intrinsics, m_cameraFromMarker, pixel, outPoint);
}

void ExtrinsicsWizard::updateHandScaleCapture(const TrackingFrameResult& trackingResult)
{
	// Use whichever hand is tracked with good presence
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedHand& hand= trackingResult.hands[sideIndex];
		if (!hand.tracked || hand.presence < 0.7f)
			continue;

		const glm::vec3& wristPx= hand.imagePoints[(int)eHandLandmark::WRIST];
		const glm::vec3& mcpPx= hand.imagePoints[(int)eHandLandmark::MIDDLE_MCP];

		glm::dvec3 wristOnPlane, mcpOnPlane;
		if (raycastPixelOntoMarkerPlane(glm::vec2(wristPx.x, wristPx.y), wristOnPlane) &&
			raycastPixelOntoMarkerPlane(glm::vec2(mcpPx.x, mcpPx.y), mcpOnPlane))
		{
			const double length= glm::length(mcpOnPlane - wristOnPlane);

			// Sanity range: 5-12cm covers human hands
			if (length >= 0.05 && length <= 0.12)
			{
				m_handScaleSamples.push_back(length);

				if ((int)m_handScaleSamples.size() >= k_handScaleSampleCount)
				{
					std::vector<double> sorted= m_handScaleSamples;
					std::sort(sorted.begin(), sorted.end());
					m_measuredHandScale= sorted[sorted.size() / 2]; // median
					m_state= eState::Review;
				}
			}
		}

		break; // only sample one hand per frame
	}
}

bool ExtrinsicsWizard::update(float deltaSeconds, const cv::Mat& bgrPreview,
							  const TrackingFrameResult& trackingResult, ImDrawList* overlayDrawList,
							  const ImageToScreenMapping& mapping)
{
	if (!m_bActive)
		return false;

	if (m_state == eState::CaptureHandScale && m_handScaleCountdown > 0.f)
		m_handScaleCountdown-= deltaSeconds;

	// Marker pose sampling
	if ((m_state == eState::CaptureCameraPose || m_state == eState::VerifySetup) &&
		m_poseSampler != nullptr && !bgrPreview.empty())
	{
		cv::cvtColor(bgrPreview, m_grayFrame, cv::COLOR_BGR2GRAY);
		m_poseSampler->setGrayscaleFrame(&m_grayFrame);

		if (m_state == eState::CaptureCameraPose && !m_poseSampler->hasFinishedSampling())
		{
			if (m_poseSampler->computeApertureRelativeMarkerXform())
				m_poseSampler->sampleLastApertureRelativeMarkerXform();

			if (m_poseSampler->hasFinishedSampling() &&
				m_poseSampler->computeCalibratedMarkerPose(m_cameraFromMarker))
			{
				m_bHasCameraPose= true;
				m_state= eState::PoseTest;
			}
		}
	}

	if ((m_state == eState::CaptureCameraPose) && overlayDrawList != nullptr)
		drawMarkerOverlay(overlayDrawList, mapping);

	drawWizardWindow(bgrPreview, trackingResult);

	if (m_state == eState::CaptureHandScale && m_handScaleCountdown <= 0.f)
		updateHandScaleCapture(trackingResult);

	return !m_bWantsClose;
}

void ExtrinsicsWizard::drawMarkerOverlay(ImDrawList* drawList, const ImageToScreenMapping& mapping)
{
	CalibrationPatternFinder_Aruco* finder= m_poseSampler != nullptr ? m_poseSampler->getPatternFinder() : nullptr;
	if (finder == nullptr)
		return;

	// Side-effect-free read (fetchLastFoundCalibrationPattern commits capture state)
	t_opencv_point2d_list imagePoints;
	cv::Point2f boundingQuad[4];
	if (!finder->getCurrentCalibrationPattern(imagePoints, boundingQuad))
		return;

	const ImU32 color= IM_COL32(80, 255, 120, 255);
	for (int i= 0; i < 4; ++i)
	{
		const cv::Point2f& c0= boundingQuad[i];
		const cv::Point2f& c1= boundingQuad[(i + 1) % 4];
		drawList->AddLine(mapping.toScreen(c0.x, c0.y), mapping.toScreen(c1.x, c1.y), color, 2.f);
	}
}

void ExtrinsicsWizard::drawWizardWindow(const cv::Mat& bgrPreview, const TrackingFrameResult& trackingResult)
{
	ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_Appearing);
	if (!ImGui::Begin("Extrinsics + Hand Scale Calibration", nullptr, ImGuiWindowFlags_NoCollapse))
	{
		ImGui::End();
		return;
	}

	if (!m_config->camera(m_cameraIndex).intrinsics.present)
	{
		ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "Camera intrinsics must be calibrated first");
		if (ImGui::Button("Close"))
			m_bWantsClose= true;
		ImGui::End();
		return;
	}

	switch (m_state)
	{
		case eState::VerifySetup:
		{
			ImGui::TextWrapped(
				"Place the printed origin aruco marker flat on the table where the world "
				"origin should be. The marker stays there during hand tracking - it defines "
				"the tracking space.");

			ImGui::InputInt("Marker ID", &m_markerId);
			ImGui::InputFloat("Marker size (mm)", &m_markerLengthMM, 0.f, 0.f, "%.0f");

			std::string paramError;
			const bool bParamsValid= areMarkerParamsValid(paramError);
			if (!bParamsValid)
				ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", paramError.c_str());

			ImGui::BeginDisabled(!bParamsValid);
			if (ImGui::Button("Export marker PNG..."))
			{
				const std::filesystem::path exportPath=
					PathUtils::getResourceDirectory() / "calibration" / "aruco_marker.png";
				try
				{
					if (generateArucoMarkerPng(exportPath, m_markerId, eCharucoDictionaryType::DICT_6X6, 1000))
					{
						MIKAN_LOG_INFO("ExtrinsicsWizard") << "Exported aruco marker to " << exportPath;
					}
				}
				catch (const cv::Exception& e)
				{
					MIKAN_LOG_ERROR("ExtrinsicsWizard") << "OpenCV error exporting marker: " << e.what();
				}
			}
			ImGui::EndDisabled();

			ImGui::Separator();
			const bool bHasVideo= !bgrPreview.empty();
			if (!bHasVideo)
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Start a video stream first");
			ImGui::BeginDisabled(!bHasVideo || !bParamsValid);
			if (ImGui::Button("Capture Camera Pose", ImVec2(-1, 0)))
				beginPoseCapture(bgrPreview.cols, bgrPreview.rows);
			ImGui::EndDisabled();
			break;
		}

		case eState::CaptureCameraPose:
		{
			ImGui::Text("Sampling marker pose...");
			ImGui::ProgressBar(m_poseSampler->getCalibrationProgress(), ImVec2(-1, 0));
			if (!m_poseSampler->hasValidApertureRelativeMarkerXform())
				ImGui::TextDisabled("Marker not visible - make sure it's in view");
			if (ImGui::Button("Restart"))
				m_poseSampler->resetCalibrationState();
			break;
		}

		case eState::PoseTest:
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Camera pose captured");

			const glm::dvec3 markerPos= glm::dvec3(m_cameraFromMarker[3]);
			ImGui::Text("Marker distance: %.2f m", glm::length(markerPos));

			// Camera height above the marker/table plane (world +Z is the
			// marker plane normal)
			const glm::dmat4 worldFromCamera= computeWorldFromCamera();
			ImGui::Text("Camera height over table: %.2f m", worldFromCamera[3].z);
			ImGui::TextWrapped("Sanity check these numbers against your physical setup.");

			if (ImGui::Button("Looks Right - Measure Hand Scale", ImVec2(-1, 0)))
			{
				m_handScaleSamples.clear();
				m_handScaleCountdown= k_handScaleCountdownSeconds;
				m_visionThread->setTrackingEnabled(m_cameraIndex, true); // hand landmarks needed now
				m_state= eState::CaptureHandScale;
			}
			if (ImGui::Button("Recapture Pose"))
				m_state= eState::VerifySetup;
			break;
		}

		case eState::CaptureHandScale:
		{
			ImGui::TextWrapped(
				"Lay your hand FLAT on the table next to the marker, fingers relaxed. "
				"Hold still while samples are collected.");

			if (m_handScaleCountdown > 0.f)
			{
				// Big countdown so the hand can settle before sampling begins
				char countdownText[16];
				snprintf(countdownText, sizeof(countdownText), "%d", (int)ceilf(m_handScaleCountdown));
				ImGui::SetWindowFontScale(3.f);
				const float textWidth= ImGui::CalcTextSize(countdownText).x;
				ImGui::SetCursorPosX((ImGui::GetWindowWidth() - textWidth) * 0.5f);
				ImGui::TextColored(ImVec4(1.f, 0.85f, 0.3f, 1.f), "%s", countdownText);
				ImGui::SetWindowFontScale(1.f);
				ImGui::ProgressBar(
					1.f - m_handScaleCountdown / k_handScaleCountdownSeconds, ImVec2(-1, 4), "");
			}
			else
			{
				ImGui::ProgressBar(
					(float)m_handScaleSamples.size() / (float)k_handScaleSampleCount, ImVec2(-1, 0));
			}

			bool bHandVisible= false;
			for (const TrackedHand& hand : trackingResult.hands)
				bHandVisible|= hand.tracked;
			if (!bHandVisible)
				ImGui::TextDisabled("No hand detected yet");

			if (ImGui::Button("Skip (keep default 8 cm)"))
			{
				m_measuredHandScale= 0.0;
				m_state= eState::Review;
			}
			break;
		}

		case eState::Review:
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibration ready");

			const glm::dmat4 worldFromCamera= computeWorldFromCamera();
			ImGui::Text("Camera height over table: %.2f m", worldFromCamera[3].z);
			if (m_measuredHandScale > 0.0)
				ImGui::Text("Hand scale (wrist->knuckle): %.1f cm", m_measuredHandScale * 100.0);
			else
				ImGui::Text("Hand scale: default %.1f cm", m_config->handScale.refLengthMeters * 100.0);

			if (ImGui::Button("Accept & Save", ImVec2(-1, 0)))
			{
				// worldFromCamera maps GL camera space to the Z-up world, but
				// SpaceTransforms/LandmarkTo3D work in OpenCV camera convention
				// (+Y down, +Z forward). Right-multiply the GL->CV axis flip so
				// the stored transform maps CV-convention camera points to world.
				const glm::dmat4 cvFromGlFlip(
					glm::dvec4(1, 0, 0, 0),
					glm::dvec4(0, -1, 0, 0),
					glm::dvec4(0, 0, -1, 0),
					glm::dvec4(0, 0, 0, 1));

				m_config->camera(m_cameraIndex).extrinsics.present= true;
				m_config->camera(m_cameraIndex).extrinsics.markerFromCamera= worldFromCamera * cvFromGlFlip;
				m_config->camera(m_cameraIndex).extrinsics.markerId= m_markerId;
				m_config->camera(m_cameraIndex).extrinsics.markerLengthMM= m_markerLengthMM;
				if (m_measuredHandScale > 0.0)
				{
					m_config->handScale.present= true;
					m_config->handScale.refLengthMeters= m_measuredHandScale;
				}
				m_config->markDirty();
				m_config->save();
				m_bWantsClose= true;
			}
			if (ImGui::Button("Redo Hand Scale"))
			{
				m_handScaleSamples.clear();
				m_handScaleCountdown= k_handScaleCountdownSeconds;
				m_state= eState::CaptureHandScale;
			}
			break;
		}
	}

	ImGui::Separator();
	if (ImGui::Button("Cancel / Close"))
		m_bWantsClose= true;

	ImGui::End();
}
