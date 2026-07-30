// Main implementation file that includes all calibration pattern finder implementations
#include "CalibrationPatternFinder.h"
#include "CameraMath.h"
#include "Logger.h"
#include "MathOpenCV.h"
#include "MathTypeConversion.h"
#include "MathUtility.h"

#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

//-- CalibrationPatternFinder -----
CalibrationPatternFinder::CalibrationPatternFinder(int frameWidth, int frameHeight)
	: m_grayscaleFrame(nullptr)
	, m_frameWidth((float)frameWidth)
	, m_frameHeight((float)frameHeight)
	, m_hasCameraIntrinsics(false)
{
}

CalibrationPatternFinder::~CalibrationPatternFinder() {}

void CalibrationPatternFinder::setCameraIntrinsics(const MikanMonoIntrinsics& intrinsics)
{
	m_cameraIntrinsics= intrinsics;
	m_hasCameraIntrinsics= true;
}

bool CalibrationPatternFinder::estimateNewCalibrationPatternPose(glm::dmat4& outCameraToPatternXform)
{
	// Make sure mono camera intrinsics are available
	if (!m_hasCameraIntrinsics)
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

	// Build the object point list matching the fetched image points.
	// For partial pattern detections (e.g. a partially visible charuco board) the image
	// point ids index into the solvePnP geometry point list, so look up each detected point.
	const t_opencv_point3d_list& geometryPoints= m_opencvSolvePnPGeometry.points;
	t_opencv_point3d_list objectPoints;
	if (imagePoints.size() == geometryPoints.size())
	{
		objectPoints= geometryPoints;
	}
	else if (imagePoints.size() == imagePointIDs.size())
	{
		objectPoints.reserve(imagePoints.size());
		for (const int pointID : imagePointIDs)
		{
			if (pointID < 0 || pointID >= (int)geometryPoints.size())
				return false;

			objectPoints.push_back(geometryPoints[pointID]);
		}
	}
	else
	{
		return false;
	}

	// Given an object model and the image points samples we could be able to compute
	// a position and orientation of the calibration pattern relative to the camera
	cv::Quatd cv_cameraToPatternRot;
	cv::Vec3d cv_cameraToPatternVecMM; // Millimeters
	double meanReprojectionError= 0.0;
	if (!computeOpenCVCameraRelativePatternTransform(m_cameraIntrinsics, imagePoints, objectPoints,
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
