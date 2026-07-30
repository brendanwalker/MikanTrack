// Main implementation file that includes all calibration pattern finder implementations
#include "CalibrationPatternFinder.h"
#include "CalibrationRenderHelpers.h"
#include "CameraComponent.h"
#include "Colors.h"
#include "CameraMath.h"
#include "MikanTextRenderer.h"
#include "Logger.h"
#include "MarkerComponent.h"
#include "MathOpenCV.h"
#include "MathTypeConversion.h"
#include "MathUtility.h"
#include "StageComponent.h"
#include "TextStyle.h"
#include "TrackingVolumeComponent.h"
#include "VideoFrameDistortionView.h"
#include "VideoSourceComponent.h"

#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

//-- CalibrationPatternFinder -----
CalibrationPatternFinder::CalibrationPatternFinder(VideoFrameDistortionView* distortionView)
	: m_distortionView(distortionView)
	, m_frameWidth(distortionView->getFrameWidth())
	, m_frameHeight(distortionView->getFrameHeight())
{
}

CalibrationPatternFinder::CalibrationPatternFinder(int frameWidth, int frameHeight)
	: m_distortionView(nullptr)
	, m_frameWidth((float)frameWidth)
	, m_frameHeight((float)frameHeight)
{
}

CalibrationPatternFinder::~CalibrationPatternFinder() {}

cv::Mat* CalibrationPatternFinder::getGrayscaleVideoFrameInput() const
{
	// By default use the undistorted grayscale image unless explicitly disabled
	// (which should only be the case during distortion calibration)
	return m_distortionView->isGrayscaleUndistortDisabled() ? m_distortionView->getGrayscaleSourceBuffer()
															: m_distortionView->getGrayscaleUndistortBuffer();
}

bool CalibrationPatternFinder::estimateNewCalibrationPatternPose(glm::dmat4& outCameraToPatternXform)
{
	// Make sure mono camera intrinsics are available
	MikanVideoSourceIntrinsics cameraIntrinsics;
	m_distortionView->getVideoSourceComponent()->getCameraIntrinsics(cameraIntrinsics);
	if (cameraIntrinsics.intrinsics_type != MikanIntrinsicsType::MONO_CAMERA_INTRINSICS)
	{
		return false;
	}

	// Look for the calibration pattern in the latest video frame
	if (!findNewCalibrationPattern())
	{
		return false;
	}

	// Get the image points of the calibration pattern
	cv::Point2f boundingQuad[4];
	t_opencv_point2d_list imagePoints;
	t_opencv_pointID_list imagePointIDs;
	if (!fetchLastFoundCalibrationPattern(imagePoints, imagePointIDs, boundingQuad))
	{
		return false;
	}

	// Make a local copy of the mono camera intrinsics
	MikanMonoIntrinsics monoIntrinsics= cameraIntrinsics.getMonoIntrinsics();

	// Given an object model and the image points samples we could be able to compute
	// a position and orientation of the calibration pattern relative to the camera
	cv::Quatd cv_cameraToPatternRot;
	cv::Vec3d cv_cameraToPatternVecMM; // Millimeters
	double meanReprojectionError= 0.0;
	if (!computeOpenCVCameraRelativePatternTransform(monoIntrinsics, imagePoints, m_opencvSolvePnPGeometry.points,
													 cv_cameraToPatternRot, cv_cameraToPatternVecMM,
													 &meanReprojectionError))
	{
		return false;
	}

	// Convert OpenCV pose (in mm) to OpenGL pose (in meters)
	convertOpenCVCameraRelativePoseToGLMMat(cv_cameraToPatternRot, cv_cameraToPatternVecMM, outCameraToPatternXform);

	return true;
}

bool CalibrationPatternFinder::areCurrentImagePointsValid() const { return m_currentImagePoints.size() > 0; }

void CalibrationPatternFinder::renderCalibrationPattern2D() const
{
	if (areCurrentImagePointsValid())
	{
		IMkGraphicsContext* graphicsContext= m_distortionView->getGraphicsContext();

		drawOpenCVChessBoard2D(graphicsContext, m_frameWidth, m_frameHeight,
							   (float*)m_currentImagePoints.data(), // cv::point2f is just two floats
							   (int)m_currentImagePoints.size(), true);
	}
}

void CalibrationPatternFinder::renderSolvePnPPattern3D(const glm::mat4& xform) const
{
	if (areCurrentImagePointsValid())
	{
		IMkGraphicsContext* graphicsContext= m_distortionView->getGraphicsContext();

		drawOpenCVChessBoard3D(graphicsContext, xform,
							   m_openglSolvePnPGeometry.points.data(), // cv::point3f is just three floats
							   (int)m_openglSolvePnPGeometry.points.size(), true);
	}
}

ArucoDictionaryPtr CalibrationPatternFinder::getArucoDictionary(eCharucoDictionaryType dictionaryType)
{
	cv::aruco::PredefinedDictionaryType cvDictionaryType;

	switch (dictionaryType)
	{
	case eCharucoDictionaryType::DICT_4X4:
		cvDictionaryType= cv::aruco::DICT_4X4_250;
		break;
	case eCharucoDictionaryType::DICT_5X5:
		cvDictionaryType= cv::aruco::DICT_5X5_250;
		break;
	case eCharucoDictionaryType::DICT_6X6:
		cvDictionaryType= cv::aruco::DICT_6X6_250;
		break;
	case eCharucoDictionaryType::DICT_7X7:
		cvDictionaryType= cv::aruco::DICT_7X7_250;
		break;
	default:
		cvDictionaryType= cv::aruco::DICT_6X6_250;
		break;
	}

	return std::make_shared<cv::aruco::Dictionary>(cv::aruco::getPredefinedDictionary(cvDictionaryType));
}