#include "IntrinsicsWizard.h"

#include "opencv2/imgproc.hpp"

#include "AppConfig.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "Logger.h"
#include "MonoLensDistortionCalibrator.h"
#include "PatternExport.h"
#include "PathUtils.h"
#include "VisionThread.h"

IntrinsicsWizard::IntrinsicsWizard(AppConfig* config, VisionThread* visionThread)
	: m_config(config)
	, m_visionThread(visionThread)
{
}

IntrinsicsWizard::~IntrinsicsWizard()= default;

void IntrinsicsWizard::enter()
{
	m_bActive= true;
	m_bWantsClose= false;
	m_state= eState::SelectBoardParams;

	m_boardCols= m_config->charucoBoard.cols;
	m_boardRows= m_config->charucoBoard.rows;
	m_squareLengthMM= (float)m_config->charucoBoard.squareLengthMM;
	m_markerLengthMM= (float)m_config->charucoBoard.markerLengthMM;

	// Raw distorted frames are required for intrinsics samples
	m_visionThread->setTrackingEnabled(false);
	m_visionThread->setUndistortEnabled(false);
}

void IntrinsicsWizard::exit()
{
	m_calibrator= nullptr;
	m_bActive= false;

	m_visionThread->setTrackingEnabled(true);
	m_visionThread->setUndistortEnabled(true);
	m_visionThread->requestConfigRefresh();
}

void IntrinsicsWizard::beginCapture(int frameWidth, int frameHeight)
{
	// cv::aruco::CharucoBoard throws cv::Exception on invalid board geometry
	// (cols/rows < 2, marker <= 0, square <= marker) - never let that
	// propagate out of the UI as a crash
	try
	{
		m_calibrator= std::make_unique<MonoLensDistortionCalibrator>(
			frameWidth, frameHeight,
			m_boardCols, m_boardRows,
			m_squareLengthMM, m_markerLengthMM,
			eCharucoDictionaryType::DICT_6X6, // matches the exported board
			12);
		m_state= eState::Capture;
	}
	catch (const cv::Exception& e)
	{
		m_lastErrorMessage= e.what();
		MIKAN_LOG_ERROR("IntrinsicsWizard::beginCapture") << "OpenCV error creating calibrator: " << e.what();
		m_calibrator= nullptr;
		m_state= eState::Failed;
	}
}

bool IntrinsicsWizard::areBoardParamsValid(std::string& outError) const
{
	if (m_boardCols < 3 || m_boardRows < 3)
	{
		outError= "Board must be at least 3x3 squares";
		return false;
	}
	if (m_markerLengthMM <= 0.f)
	{
		outError= "Marker size must be positive";
		return false;
	}
	if (m_squareLengthMM <= m_markerLengthMM)
	{
		outError= "Square size must be larger than marker size";
		return false;
	}
	return true;
}

void IntrinsicsWizard::applyResultToConfig()
{
	MikanMonoIntrinsics monoIntrinsics;
	if (m_calibrator->getCameraCalibration(&monoIntrinsics))
	{
		m_config->camera(0).intrinsics.present= true;
		m_config->camera(0).intrinsics.intrinsics= monoIntrinsics;
		m_config->camera(0).intrinsics.reprojectionError= m_calibrator->getReprojectionError();
		m_config->charucoBoard.cols= m_boardCols;
		m_config->charucoBoard.rows= m_boardRows;
		m_config->charucoBoard.squareLengthMM= m_squareLengthMM;
		m_config->charucoBoard.markerLengthMM= m_markerLengthMM;
		m_config->markDirty();
		m_config->save();
	}
}

bool IntrinsicsWizard::update(float deltaSeconds, const cv::Mat& bgrPreview, ImDrawList* overlayDrawList,
							  const ImageToScreenMapping& mapping)
{
	if (!m_bActive)
		return false;

	// Feed detection during capture
	if (m_state == eState::Capture && m_calibrator != nullptr && !bgrPreview.empty())
	{
		cv::cvtColor(bgrPreview, m_grayFrame, cv::COLOR_BGR2GRAY);
		m_calibrator->update(deltaSeconds, &m_grayFrame);

		if (m_calibrator->hasSampledAllCalibrationPatterns())
		{
			m_calibrator->computeCameraCalibration();
			m_state= eState::Solving;
		}
	}

	if (m_state == eState::Solving && m_calibrator->getIsCameraCalibrationComplete())
	{
		MikanMonoIntrinsics monoIntrinsics;
		if (m_calibrator->getCameraCalibration(&monoIntrinsics) && monoIntrinsics.pixel_width > 0)
		{
			applyResultToConfig();
			// Show the live undistorted view for verification
			m_visionThread->setUndistortEnabled(true);
			m_visionThread->requestConfigRefresh();
			m_state= eState::TestUndistort;
		}
		else
		{
			m_lastErrorMessage= "Camera calibration solve did not converge - "
								"try recapturing with better board coverage";
			m_state= eState::Failed;
		}
	}

	if (m_state == eState::Capture && overlayDrawList != nullptr)
		drawPatternOverlay(overlayDrawList, mapping);

	drawWizardWindow(deltaSeconds, bgrPreview);

	return !m_bWantsClose;
}

void IntrinsicsWizard::drawPatternOverlay(ImDrawList* drawList, const ImageToScreenMapping& mapping)
{
	CalibrationPatternFinder_Charuco* finder= m_calibrator->getPatternFinder();
	if (finder == nullptr)
		return;

	// Side-effect-free read: fetchLastFoundCalibrationPattern would commit the
	// points as the last capture and stall the min-separation check
	t_opencv_point2d_list imagePoints;
	cv::Point2f boundingQuad[4];
	if (!finder->getCurrentCalibrationPattern(imagePoints, boundingQuad))
		return;

	// Yellow -> green as the hold-steady window fills; capture fires when full
	const float stability= m_calibrator->getStabilityFraction();
	const ImU32 pointColor= IM_COL32(
		(int)(255.f + (80.f - 255.f) * stability),
		(int)(220.f + (255.f - 220.f) * stability),
		(int)(60.f + (120.f - 60.f) * stability),
		255);

	for (const cv::Point2f& point : imagePoints)
		drawList->AddCircleFilled(mapping.toScreen(point.x, point.y), 3.f, pointColor);

	for (int i= 0; i < 4; ++i)
	{
		const cv::Point2f& c0= boundingQuad[i];
		const cv::Point2f& c1= boundingQuad[(i + 1) % 4];
		drawList->AddLine(mapping.toScreen(c0.x, c0.y), mapping.toScreen(c1.x, c1.y), pointColor, 2.f);
	}
}

void IntrinsicsWizard::drawWizardWindow(float deltaSeconds, const cv::Mat& bgrPreview)
{
	ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_Appearing);
	if (!ImGui::Begin("Intrinsics Calibration", nullptr, ImGuiWindowFlags_NoCollapse))
	{
		ImGui::End();
		return;
	}

	switch (m_state)
	{
		case eState::SelectBoardParams:
		{
			ImGui::TextWrapped(
				"Calibrate the camera lens using a printed charuco board. "
				"Print the board at 100%% scale, measure a square to confirm its size, "
				"then capture it from 12 different angles/positions.");

			ImGui::InputInt("Columns", &m_boardCols);
			ImGui::InputInt("Rows", &m_boardRows);
			ImGui::InputFloat("Square size (mm)", &m_squareLengthMM, 0.f, 0.f, "%.1f");
			ImGui::InputFloat("Marker size (mm)", &m_markerLengthMM, 0.f, 0.f, "%.1f");

			std::string paramError;
			const bool bParamsValid= areBoardParamsValid(paramError);
			if (!bParamsValid)
				ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "%s", paramError.c_str());

			ImGui::BeginDisabled(!bParamsValid);
			if (ImGui::Button("Export board PNG..."))
			{
				const std::filesystem::path exportPath=
					PathUtils::getResourceDirectory() / "calibration" / "charuco_board.png";
				try
				{
					if (generateCharucoBoardPng(exportPath, m_boardCols, m_boardRows, m_squareLengthMM,
												m_markerLengthMM, eCharucoDictionaryType::DICT_6X6, 10.f))
					{
						MIKAN_LOG_INFO("IntrinsicsWizard") << "Exported charuco board to " << exportPath;
					}
				}
				catch (const cv::Exception& e)
				{
					MIKAN_LOG_ERROR("IntrinsicsWizard") << "OpenCV error exporting board: " << e.what();
				}
			}
			ImGui::EndDisabled();

			ImGui::Separator();
			const bool bHasVideo= !bgrPreview.empty();
			if (!bHasVideo)
				ImGui::TextColored(ImVec4(1.f, 0.7f, 0.3f, 1.f), "Start a video stream first");
			ImGui::BeginDisabled(!bHasVideo || !bParamsValid);
			if (ImGui::Button("Begin Capture", ImVec2(-1, 0)))
				beginCapture(bgrPreview.cols, bgrPreview.rows);
			ImGui::EndDisabled();
			break;
		}

		case eState::Capture:
		{
			const float progress= m_calibrator->computeCalibrationProgress();
			const int captured= (int)(progress * (float)m_calibrator->getDesiredPatternCount() + 0.5f);
			ImGui::Text("Captured %d / %d boards", captured, m_calibrator->getDesiredPatternCount());
			ImGui::ProgressBar(progress, ImVec2(-1, 0));

			if (m_calibrator->areCurrentImagePointsValid())
			{
				const float stability= m_calibrator->getStabilityFraction();
				ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Hold steady... %d%%", (int)(stability * 100.f));
				ImGui::ProgressBar(stability, ImVec2(-1, 4), "");
			}
			else
			{
				ImGui::TextDisabled("Show the board / move it to a new position");
			}

			if (ImGui::Button("Restart"))
				m_calibrator->resetCalibrationState();
			break;
		}

		case eState::Solving:
			ImGui::Text("Solving camera calibration...");
			break;

		case eState::TestUndistort:
		{
			ImGui::TextColored(ImVec4(0.4f, 1.f, 0.5f, 1.f), "Calibration complete");
			ImGui::Text("Reprojection error: %.3f px", m_config->camera(0).intrinsics.reprojectionError);
			ImGui::TextWrapped(
				"The preview now shows the undistorted image. "
				"Straight lines in the scene should look straight.");

			if (ImGui::Button("Accept", ImVec2(-1, 0)))
				m_bWantsClose= true;
			if (ImGui::Button("Redo Capture"))
			{
				m_visionThread->setUndistortEnabled(false);
				m_state= eState::SelectBoardParams;
			}
			break;
		}

		case eState::Failed:
			ImGui::TextColored(ImVec4(1.f, 0.4f, 0.4f, 1.f), "Calibration failed");
			if (!m_lastErrorMessage.empty())
				ImGui::TextWrapped("%s", m_lastErrorMessage.c_str());
			if (ImGui::Button("Try Again"))
			{
				m_lastErrorMessage.clear();
				m_state= eState::SelectBoardParams;
			}
			break;
	}

	ImGui::Separator();
	if (ImGui::Button("Cancel / Close"))
		m_bWantsClose= true;

	ImGui::End();
}
