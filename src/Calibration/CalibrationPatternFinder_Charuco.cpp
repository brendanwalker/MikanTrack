#include "CalibrationPatternFinder_Charuco.h"
#include "CalibrationRenderHelpers.h"
#include "CameraMath.h"
#include "Colors.h"
#include "MathOpenCV.h"
#include "MathUtility.h"
#include "MathTypeConversion.h"
#include "MikanTextRenderer.h"
#include "TextStyle.h"
#include "VideoFrameDistortionView.h"

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
};

//-- CalibrationPatternFinder_Charuco -----
CalibrationPatternFinder_Charuco::CalibrationPatternFinder_Charuco(VideoFrameDistortionView* distortionView,
																   int charucoRows, int charucoCols,
																   float charucoSquareLengthMM,
																   float charucoMarkerLengthMM,
																   eCharucoDictionaryType charucoDictionaryType)
	: CalibrationPatternFinder(distortionView)
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
	const float newLocationErrorSum= (float)cornerCount * minSeperationDist;

	// Clear out the previous images points
	bool bImagePointsValid= false;
	m_currentImagePoints.clear();

	// Fetch the source image buffer we are searching for the pattern in
	cv::Mat* gsSourceBuffer= getGrayscaleVideoFrameInput();
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
		if (m_currentImagePoints.size() == cornerCount)
		{
			// If there was a prior image point set,
			// see if this new set is far enough away to be considered unique
			if (m_lastValidImagePoints.size() > 0 && minSeperationDist > 0.f)
			{
				float error_sum= 0.f;

				for (int corner_index= 0; corner_index < cornerCount; ++corner_index)
				{
					float squared_error=
						(float)(cv::norm(m_currentImagePoints[corner_index] - m_lastValidImagePoints[corner_index]));

					error_sum+= squared_error;
				}

				bImagePointsValid= error_sum >= newLocationErrorSum;
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

bool CalibrationPatternFinder_Charuco::fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
																		t_opencv_pointID_list& outImagePointIDs,
																		cv::Point2f outBoundingQuad[4])
{
	// If it's a valid new location, append it to the board list
	if (areCurrentImagePointsValid())
	{
		// The number of corners in a row is one less than the number of squares
		const int cornerCols= m_markerData->cols - 1;
		const int cornerCount= (int)m_currentImagePoints.size();

		// Keep track of the corners of all of the chessboards we sample
		outBoundingQuad[0]= m_currentImagePoints[0];
		outBoundingQuad[1]= m_currentImagePoints[cornerCols - 1];
		outBoundingQuad[2]= m_currentImagePoints[cornerCount - 1];
		outBoundingQuad[3]= m_currentImagePoints[cornerCount - cornerCols];

		outImagePoints.clear();
		for (const auto& imagePoint : m_currentImagePoints)
		{
			outImagePoints.push_back(imagePoint);
		}

		outImagePointIDs= m_markerData->charucoIds;

		// Remember the last valid captured points
		m_lastValidImagePoints= m_currentImagePoints;

		return true;
	}

	return false;
}

void CalibrationPatternFinder_Charuco::renderCalibrationPattern2D() const
{
	CalibrationPatternFinder::renderCalibrationPattern2D();

	IMkGraphicsContext* graphicsContext= getDistortionView()->getGraphicsContext();

	// Draw the marker corners, if any
	TextStyle style= getDefaultTextStyle();
	style.horizontalAlignment= eHorizontalTextAlignment::Middle;
	style.verticalAlignment= eVerticalTextAlignment::Middle;
	style.color= Colors::Yellow;

	static int debugDrawIndex= -1;

	for (int quadIndex= 0; quadIndex < m_markerData->markerCorners.size(); quadIndex++)
	{
		if (debugDrawIndex != -1 && debugDrawIndex != quadIndex)
			continue;

		const t_opencv_point2d_list& corners= m_markerData->markerCorners[quadIndex];

		drawQuadList2d(graphicsContext, m_frameWidth, m_frameHeight,
					   (float*)corners.data(), // cv::point2f is just two floats
					   (int)corners.size(), Colors::Yellow);

		if (quadIndex < m_markerData->markerVisibleIds.size())
		{
			int markerId= m_markerData->markerVisibleIds[quadIndex];

			cv::Point2f quadCenter;
			opencv_point2f_compute_average(corners, quadCenter);

			drawTextAtTrackerPosition(graphicsContext, style, m_frameWidth, m_frameHeight,
									  glm::vec2(quadCenter.x, quadCenter.y), L"%d", markerId);
		}
	}
}

void CalibrationPatternFinder_Charuco::renderSolvePnPPattern3D(const glm::mat4& xform) const
{
	CalibrationPatternFinder::renderSolvePnPPattern3D(xform);
}
