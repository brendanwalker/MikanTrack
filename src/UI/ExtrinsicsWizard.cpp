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
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

ExtrinsicsWizard::ExtrinsicsWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

ExtrinsicsWizard::~ExtrinsicsWizard()= default;

void ExtrinsicsWizard::enter()
{
	m_bActive= true;
	m_bWantsClose= false;
	m_state= eState::VerifySetup;

	m_markerId= m_config->camera(0).extrinsics.markerId;
	m_markerLengthMM= (float)m_config->camera(0).extrinsics.markerLengthMM;

	m_captures.clear();
	m_captures.resize(m_config->cameraCount());

	// Marker detection wants undistorted frames; tracking isn't needed while
	// calibrating - every camera participates in the same session
	for (int cameraIndex= 0; cameraIndex < (int)m_config->cameraCount(); ++cameraIndex)
	{
		m_visionThread->setTrackingEnabled(cameraIndex, false);
		m_visionThread->setUndistortEnabled(cameraIndex, true);
	}
}

void ExtrinsicsWizard::exit()
{
	m_captures.clear();
	m_bActive= false;

	for (int cameraIndex= 0; cameraIndex < (int)m_config->cameraCount(); ++cameraIndex)
		m_visionThread->setTrackingEnabled(cameraIndex, true);
	m_visionThread->requestConfigRefresh();
}

bool ExtrinsicsWizard::allCamerasHaveIntrinsics() const
{
	for (size_t i= 0; i < m_config->cameraCount(); ++i)
	{
		if (!m_config->camera(i).intrinsics.present)
			return false;
	}
	return true;
}

bool ExtrinsicsWizard::beginPoseCapture(const std::vector<VisionPreviewFrame>& previews)
{
	// Guard OpenCV parameter asserts (invalid marker size/id) - surface the
	// error instead of crashing the app
	try
	{
		for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
		{
			const cv::Mat& bgr= previews[cameraIndex].bgr;
			m_captures[cameraIndex].bHasPose= false;
			m_captures[cameraIndex].sampler= std::make_unique<ArucoMarkerPoseSampler>(
				m_config->camera(cameraIndex).intrinsics.intrinsics,
				bgr.cols, bgr.rows,
				m_markerLengthMM,
				m_markerId,
				eCharucoDictionaryType::DICT_6X6,
				12);
		}
		m_state= eState::CaptureCameraPose;
		return true;
	}
	catch (const cv::Exception& e)
	{
		MIKAN_LOG_ERROR("ExtrinsicsWizard::beginPoseCapture") << "OpenCV error creating pose sampler: " << e.what();
		for (CameraCapture& capture : m_captures)
			capture.sampler= nullptr;
		return false;
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

bool ExtrinsicsWizard::update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
							  VideoPreviewPanel* previewPanel)
{
	if (!m_bActive)
		return false;

	// Marker pose sampling: every camera samples the SAME marker placement
	// simultaneously; the state advances only when ALL cameras have a pose
	if (m_state == eState::CaptureCameraPose)
	{
		bool bAllHavePose= true;
		for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
		{
			CameraCapture& capture= m_captures[cameraIndex];
			if (capture.sampler == nullptr)
			{
				bAllHavePose= false;
				continue;
			}
			if (capture.bHasPose)
				continue;

			bAllHavePose= false;
			if (cameraIndex >= previews.size() || previews[cameraIndex].bgr.empty())
				continue;

			cv::cvtColor(previews[cameraIndex].bgr, capture.grayFrame, cv::COLOR_BGR2GRAY);
			capture.sampler->setGrayscaleFrame(&capture.grayFrame);

			if (!capture.sampler->hasFinishedSampling())
			{
				if (capture.sampler->computeApertureRelativeMarkerXform())
					capture.sampler->sampleLastApertureRelativeMarkerXform();
			}
			if (capture.sampler->hasFinishedSampling() &&
				capture.sampler->computeCalibratedMarkerPose(capture.cameraFromMarker))
			{
				capture.bHasPose= true;
			}
		}

		if (bAllHavePose && !m_captures.empty())
			m_state= eState::Review;

		if (previewPanel != nullptr)
			drawMarkerOverlays(previewPanel);
	}

	drawWizardWindow(previews);

	return !m_bWantsClose;
}

void ExtrinsicsWizard::drawMarkerOverlays(VideoPreviewPanel* previewPanel)
{
	ImDrawList* drawList= previewPanel->getLastDrawList();
	if (drawList == nullptr)
		return;

	for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
	{
		CalibrationPatternFinder_Aruco* finder=
			m_captures[cameraIndex].sampler != nullptr ? m_captures[cameraIndex].sampler->getPatternFinder() : nullptr;
		if (finder == nullptr)
			continue;

		// Side-effect-free read (fetchLastFoundCalibrationPattern commits capture state)
		t_opencv_point2d_list imagePoints;
		cv::Point2f boundingQuad[4];
		if (!finder->getCurrentCalibrationPattern(imagePoints, boundingQuad))
			continue;

		const ImageToScreenMapping& mapping= previewPanel->getImageToScreenMapping((int)cameraIndex);
		const ImU32 color= IM_COL32(80, 255, 120, 255);
		for (int i= 0; i < 4; ++i)
		{
			const cv::Point2f& c0= boundingQuad[i];
			const cv::Point2f& c1= boundingQuad[(i + 1) % 4];
			drawList->AddLine(mapping.toScreen(c0.x, c0.y), mapping.toScreen(c1.x, c1.y), color, 2.f);
		}
	}
}

void ExtrinsicsWizard::drawWizardWindow(const std::vector<VisionPreviewFrame>& previews)
{
	ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_Appearing);
	if (!ImGui::Begin("Extrinsics Calibration (all cameras)", nullptr, ImGuiWindowFlags_NoCollapse))
	{
		ImGui::End();
		return;
	}

	if (!allCamerasHaveIntrinsics())
	{
		ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f),
						   "Every camera needs calibrated intrinsics first");
		ImGui::TextWrapped(
			"Extrinsics are always calibrated for ALL cameras together, against the "
			"same marker placement - otherwise the cameras end up in disagreeing "
			"world frames.");
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
				"the tracking space. ALL cameras are calibrated in this one session so they "
				"agree on the same marker placement; make sure every camera can see it.");

			ImGui::Spacing();
			// The marker defines the world AXES, not just the origin, and a
			// sheet laid down turned gives tracking that is self-consistent
			// and points the wrong way - which reads as a tracking fault
			ImGui::TextWrapped(
				"ORIENTATION MATTERS: lay the sheet down the way it reads, with the printed "
				"FORWARD arrow pointing away from you. That makes the tracking world +X "
				"forward, +Y left, +Z up.");

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
					if (generateArucoMarkerPng(exportPath, m_markerId, eCharucoDictionaryType::DICT_6X6, 1000,
											   m_markerLengthMM))
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
			bool bAllHaveVideo= previews.size() >= m_captures.size() && !m_captures.empty();
			for (size_t i= 0; i < m_captures.size() && bAllHaveVideo; ++i)
				bAllHaveVideo= i < previews.size() && !previews[i].bgr.empty();
			if (!bAllHaveVideo)
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Start video streams on all cameras first");

			ImGui::BeginDisabled(!bAllHaveVideo || !bParamsValid);
			if (ImGui::Button("Capture Camera Poses", ImVec2(-1, 0)))
				beginPoseCapture(previews);
			ImGui::EndDisabled();
			break;
		}

		case eState::CaptureCameraPose:
		{
			ImGui::Text("Sampling marker pose on every camera...");
			for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
			{
				const CameraCapture& capture= m_captures[cameraIndex];
				char label[32];
				snprintf(label, sizeof(label), "Camera %d", (int)cameraIndex + 1);
				if (capture.bHasPose)
				{
					ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "%s: captured", label);
				}
				else if (capture.sampler != nullptr)
				{
					ImGui::Text("%s:", label);
					ImGui::SameLine();
					ImGui::ProgressBar(capture.sampler->getCalibrationProgress(), ImVec2(-1, 0));
					if (!capture.sampler->hasValidApertureRelativeMarkerXform())
						ImGui::TextDisabled("  marker not visible to this camera");
				}
			}
			if (ImGui::Button("Restart"))
			{
				for (CameraCapture& capture : m_captures)
				{
					capture.bHasPose= false;
					if (capture.sampler != nullptr)
						capture.sampler->resetCalibrationState();
				}
			}
			break;
		}

		case eState::Review:
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "All camera poses captured");

			for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
			{
				const CameraCapture& capture= m_captures[cameraIndex];
				const glm::dmat4 worldFromCamera= computeWorldFromCameraPose(capture.cameraFromMarker);
				ImGui::Text("Camera %d: marker distance %.2f m, height over table %.2f m",
							(int)cameraIndex + 1,
							glm::length(glm::dvec3(capture.cameraFromMarker[3])),
							worldFromCamera[3].z);
			}
			ImGui::TextWrapped(
				"Sanity check these numbers against your physical setup. Hand scale is "
				"measured automatically from stereo/depth while tracking runs "
				"(current: %.1f cm).",
				m_config->handScale.refLengthMeters * 100.0);

			if (ImGui::Button("Accept & Save All", ImVec2(-1, 0)))
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

				for (size_t cameraIndex= 0; cameraIndex < m_captures.size(); ++cameraIndex)
				{
					const glm::dmat4 worldFromCamera=
						computeWorldFromCameraPose(m_captures[cameraIndex].cameraFromMarker);
					ExtrinsicsConfig& extrinsics= m_config->camera(cameraIndex).extrinsics;
					extrinsics.present= true;
					extrinsics.markerFromCamera= worldFromCamera * cvFromGlFlip;
					extrinsics.markerId= m_markerId;
					extrinsics.markerLengthMM= m_markerLengthMM;
				}
				m_config->markDirty();
				m_config->save();
				m_bWantsClose= true;
			}
			if (ImGui::Button("Recapture Poses"))
				m_state= eState::VerifySetup;
			break;
		}
	}

	ImGui::Separator();
	if (ImGui::Button("Cancel / Close"))
		m_bWantsClose= true;

	ImGui::End();
}
