#include "CalibrationPatternFinder_Charuco.h"
#include "CameraMath.h"
#include "MathOpenCV.h"
#include "MathUtility.h"
#include "MathTypeConversion.h"

#include <algorithm>

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

//-- CharucoBoardData -----
class CharucoBoardData
{
public:
	CharucoBoardData()= default;

	int rows;
	int cols;
	float squareLengthMM;
	float markerLengthMM;

	cv::Ptr<cv::aruco::CharucoDetector> detector;
	t_opencv_point2d_list charucoCorners;
	std::vector<int> charucoIds;
	std::vector<t_opencv_point2d_list> markerCorners;
	std::vector<int> markerVisibleIds;

	// Corner ids from the last valid capture (parallel to m_lastValidImagePoints)
	std::vector<int> lastValidCharucoIds;
};

//-- CalibrationPatternFinder_Charuco -----
CalibrationPatternFinder_Charuco::CalibrationPatternFinder_Charuco(int frameWidth, int frameHeight, int charucoCols,
																   int charucoRows, float charucoSquareLengthMM,
																   float charucoMarkerLengthMM,
																   eCharucoDictionaryType charucoDictionaryType)
	: CalibrationPatternFinder(frameWidth, frameHeight)
	, m_markerData(new CharucoBoardData())
{
	m_opencvLensCalibrationGeometry.points.clear();
	m_opencvSolvePnPGeometry.points.clear();
	m_openglSolvePnPGeometry.points.clear();

	const int cornerRows= charucoRows - 1;
	const int cornerCols= charucoCols - 1;

	for (int row= 0; row < cornerRows; ++row)
	{
		for (int col= 0; col < cornerCols; ++col)
		{
			// Solve PnP points are on the XZ Plane
			cv::Point3f openCVSolvePnPPoint(float(col) * charucoSquareLengthMM, 0.f,
											-float(row) * charucoSquareLengthMM);
			// Lens calibration points are on the XY Plane
			cv::Point3f openCVLensCalibrationPoint(float(col) * charucoSquareLengthMM,
												   float(row) * charucoSquareLengthMM, 0.f);

			// OpenCV -> OpenGL coordinate system transform
			// Rendering world units in meters, not mm
			glm::vec3 openGLPoint(openCVSolvePnPPoint.x * k_millimeters_to_meters,
								  -openCVSolvePnPPoint.y * k_millimeters_to_meters,
								  -openCVSolvePnPPoint.z * k_millimeters_to_meters);

			m_opencvLensCalibrationGeometry.points.push_back(openCVLensCalibrationPoint);
			m_opencvSolvePnPGeometry.points.push_back(openCVSolvePnPPoint);
			m_openglSolvePnPGeometry.points.push_back(openGLPoint);
		}
	}

	ArucoDictionaryPtr dictionary= getArucoDictionary(charucoDictionaryType);
	cv::aruco::CharucoBoard board(cv::Size(charucoCols, charucoRows), charucoSquareLengthMM * k_millimeters_to_meters,
								  charucoMarkerLengthMM * k_millimeters_to_meters, *dictionary.get());
	m_markerData->detector= cv::makePtr<cv::aruco::CharucoDetector>(board);
	m_markerData->rows= charucoRows;
	m_markerData->cols= charucoCols;
	m_markerData->squareLengthMM= charucoSquareLengthMM;
	m_markerData->markerLengthMM= charucoMarkerLengthMM;
}

CalibrationPatternFinder_Charuco::~CalibrationPatternFinder_Charuco() { delete m_markerData; }

bool CalibrationPatternFinder_Charuco::findNewCalibrationPattern(const float minSeperationDist)
{
	const int cornerCount= (m_markerData->cols - 1) * (m_markerData->rows - 1);
	// Accept partial board detections with at least half of the corners visible
	// (computeMonoLensCameraCalibration needs at least 6 point correspondences per sample)
	const int minCornerCount= int_max(cornerCount / 2, 6);

	// Clear out the previous images points
	bool bImagePointsValid= false;
	m_currentImagePoints.clear();

	// Fetch the source image buffer we are searching for the pattern in
	const cv::Mat* gsSourceBuffer= getGrayscaleVideoFrameInput();
	if (gsSourceBuffer == nullptr)
		return false;

	// Find Charuco marker corners in the source image
	m_markerData->markerCorners.clear();
	m_markerData->markerVisibleIds.clear();
	m_markerData->detector->detectBoard(*gsSourceBuffer, m_markerData->charucoCorners, m_markerData->charucoIds,
										m_markerData->markerCorners, m_markerData->markerVisibleIds);
	const bool bFoundMarkers= m_markerData->markerVisibleIds.size() > 0;

	if (bFoundMarkers)
	{
		// Remember the last valid captured points
		m_currentImagePoints= m_markerData->charucoCorners;

		// Append the new chessboard corner pixels into the image_points matrix
		if ((int)m_currentImagePoints.size() >= minCornerCount)
		{
			// If there was a prior image point set,
			// see if this new set is far enough away to be considered unique
			if (m_lastValidImagePoints.size() > 0 && minSeperationDist > 0.f)
			{
				// Compare only the corners that were detected in both captures,
				// matched up by their charuco corner ids
				float error_sum= 0.f;
				int sharedCornerCount= 0;

				for (size_t corner_index= 0; corner_index < m_currentImagePoints.size(); ++corner_index)
				{
					const int cornerId= m_markerData->charucoIds[corner_index];
					const auto it= std::find(m_markerData->lastValidCharucoIds.begin(),
											 m_markerData->lastValidCharucoIds.end(), cornerId);
					if (it != m_markerData->lastValidCharucoIds.end())
					{
						const size_t lastValidIndex=
							(size_t)std::distance(m_markerData->lastValidCharucoIds.begin(), it);
						float squared_error= (float)(cv::norm(m_currentImagePoints[corner_index]
															  - m_lastValidImagePoints[lastValidIndex]));

						error_sum+= squared_error;
						++sharedCornerCount;
					}
				}

				// It's a new location if the shared corners moved far enough on average,
				// or if the two detections share no corners at all
				bImagePointsValid=
					sharedCornerCount == 0 || error_sum >= (float)sharedCornerCount * minSeperationDist;
			}
			else
			{
				// We don't have previous capture.
				bImagePointsValid= true;
			}
		}
	}

	// Re-clear out the image points if we decided the latest captured onces are invalid
	if (!bImagePointsValid)
	{
		m_currentImagePoints.clear();
	}

	return bImagePointsValid;
}

bool CalibrationPatternFinder_Charuco::getCurrentCalibrationPattern(t_opencv_point2d_list& outImagePoints,
																	cv::Point2f outBoundingQuad[4]) const
{
	if (!areCurrentImagePointsValid())
		return false;

	// The number of corners in a row is one less than the number of squares
	const int cornerCols= m_markerData->cols - 1;
	const int cornerCount= (int)m_currentImagePoints.size();

	// Bounding quad of the detected corners
	// (indices are clamped since partial board detections may have fewer corners)
	outBoundingQuad[0]= m_currentImagePoints[0];
	outBoundingQuad[1]= m_currentImagePoints[int_min(cornerCols - 1, cornerCount - 1)];
	outBoundingQuad[2]= m_currentImagePoints[cornerCount - 1];
	outBoundingQuad[3]= m_currentImagePoints[int_max(cornerCount - cornerCols, 0)];

	outImagePoints.clear();
	for (const auto& imagePoint : m_currentImagePoints)
	{
		outImagePoints.push_back(imagePoint);
	}

	return true;
}

bool CalibrationPatternFinder_Charuco::fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
																		t_opencv_pointID_list& outImagePointIDs,
																		cv::Point2f outBoundingQuad[4])
{
	// If it's a valid new location, append it to the board list
	if (getCurrentCalibrationPattern(outImagePoints, outBoundingQuad))
	{
		outImagePointIDs= m_markerData->charucoIds;

		// Remember the last valid captured points; the min-separation check in
		// findNewCalibrationPattern compares against these, so this must only
		// happen when a sample is actually captured
		m_lastValidImagePoints= m_currentImagePoints;
		m_markerData->lastValidCharucoIds= m_markerData->charucoIds;

		return true;
	}

	return false;
}
