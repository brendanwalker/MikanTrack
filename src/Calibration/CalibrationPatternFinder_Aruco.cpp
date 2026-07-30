#include "CalibrationPatternFinder_Aruco.h"
#include "CameraMath.h"
#include "MathOpenCV.h"
#include "MathUtility.h"
#include "MathTypeConversion.h"

#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

//-- ArucoBoardData -----
class ArucoBoardData
{
public:
	ArucoBoardData()= default;

	int desiredArucoId;
	float markerLengthMM;

	cv::Ptr<cv::aruco::ArucoDetector> detector;
	std::vector<t_opencv_point2d_list> markerCorners;
	std::vector<int> markerVisibleIds;
};

//-- CalibrationPatternFinder_Aruco -----
static void initArucoBoardData(ArucoBoardData* markerData, OpenCVCalibrationGeometry& opencvSolvePnPGeometry,
							   OpenCVCalibrationGeometry& opencvLensCalibrationGeometry,
							   OpenGLCalibrationGeometry& openglSolvePnPGeometry, float markerLengthMM,
							   int desiredArucoId, eCharucoDictionaryType arucoDictionaryType)
{
	ArucoDictionaryPtr dictionary= CalibrationPatternFinder::getArucoDictionary(arucoDictionaryType);

	// Use corner refinement to get the best possible corner locations
	cv::aruco::DetectorParameters detectorParams;
	detectorParams.cornerRefinementMethod= cv::aruco::CORNER_REFINE_SUBPIX;

	markerData->desiredArucoId= desiredArucoId;
	markerData->markerLengthMM= markerLengthMM;
	markerData->detector= cv::makePtr<cv::aruco::ArucoDetector>(*dictionary.get(), detectorParams);

	// The Aruco board is a square, so we can hardcode the points in ARUCO_CCW_CENTER style
	// Solve PnP points are on the XZ Plane
	opencvSolvePnPGeometry.points.clear();
	opencvSolvePnPGeometry.points.push_back(cv::Point3f(-markerLengthMM / 2.f, 0.f, markerLengthMM / 2.f));
	opencvSolvePnPGeometry.points.push_back(cv::Point3f(markerLengthMM / 2.f, 0.f, markerLengthMM / 2.f));
	opencvSolvePnPGeometry.points.push_back(cv::Point3f(markerLengthMM / 2.f, 0.f, -markerLengthMM / 2.f));
	opencvSolvePnPGeometry.points.push_back(cv::Point3f(-markerLengthMM / 2.f, 0.f, -markerLengthMM / 2.f));

	// Derive the other geometry from the OpenCV SolvePnP geometry
	opencvLensCalibrationGeometry.points.clear();
	openglSolvePnPGeometry.points.clear();
	for (int index= 0; index < 4; index++)
	{
		// Solve PnP points are on the XZ Plane
		const cv::Point3f& openCVSolvePnPPoint= opencvSolvePnPGeometry.points[index];

		// Lens calibration points are on the XY Plane
		cv::Point3f openCVLensCalibrationPoint(openCVSolvePnPPoint.x, openCVSolvePnPPoint.z, 0.f);
		opencvLensCalibrationGeometry.points.push_back(openCVLensCalibrationPoint);

		// OpenCV -> OpenGL coordinate system transform
		// Rendering world units in meters, not mm
		glm::vec3 openGLPoint(openCVSolvePnPPoint.x * k_millimeters_to_meters,
							  -openCVSolvePnPPoint.y * k_millimeters_to_meters,
							  -openCVSolvePnPPoint.z * k_millimeters_to_meters);
		openglSolvePnPGeometry.points.push_back(openGLPoint);
	}
}

CalibrationPatternFinder_Aruco::CalibrationPatternFinder_Aruco(int frameWidth, int frameHeight, float markerLengthMM,
															   int desiredArucoId,
															   eCharucoDictionaryType arucoDictionaryType)
	: CalibrationPatternFinder(frameWidth, frameHeight)
	, m_markerData(new ArucoBoardData())
{
	initArucoBoardData(m_markerData, m_opencvSolvePnPGeometry, m_opencvLensCalibrationGeometry,
					   m_openglSolvePnPGeometry, markerLengthMM, desiredArucoId, arucoDictionaryType);
}

CalibrationPatternFinder_Aruco::~CalibrationPatternFinder_Aruco() { delete m_markerData; }

bool CalibrationPatternFinder_Aruco::findNewCalibrationPattern(const float minSeperationDist)
{
	// Clear out the previous images points
	bool bImagePointsValid= false;
	m_currentImagePoints.clear();

	// Fetch the source image buffer we are searching for the pattern in
	const cv::Mat* gsSourceBuffer= getGrayscaleVideoFrameInput();
	if (gsSourceBuffer == nullptr)
		return false;

	// Find Arcuo marker corners on the small image
	m_markerData->markerCorners.clear();
	m_markerData->detector->detectMarkers(*gsSourceBuffer, m_markerData->markerCorners, m_markerData->markerVisibleIds);
	const bool bFoundMarkers= m_markerData->markerVisibleIds.size() > 0;

	// Re-clear out the image points if we decided the latest captured onces are invalid
	if (bFoundMarkers)
	{
		for (int index= 0; index < m_markerData->markerVisibleIds.size(); ++index)
		{
			if (m_markerData->markerVisibleIds[index] == m_markerData->desiredArucoId)
			{
				m_currentImagePoints= m_markerData->markerCorners[index];
				break;
			}
		}
	}
	else
	{
		m_currentImagePoints.clear();
	}

	return bFoundMarkers;
}

bool CalibrationPatternFinder_Aruco::fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
																	  t_opencv_pointID_list& outImagePointIDs,
																	  cv::Point2f outBoundingQuad[4])
{
	// If it's a valid new location, append it to the board list
	if (areCurrentImagePointsValid())
	{
		// Keep track of the corners of all of the chessboards we sample
		outBoundingQuad[0]= m_currentImagePoints[0];
		outBoundingQuad[1]= m_currentImagePoints[1];
		outBoundingQuad[2]= m_currentImagePoints[2];
		outBoundingQuad[3]= m_currentImagePoints[3];

		outImagePoints.clear();
		for (const auto& imagePoint : m_currentImagePoints)
		{
			outImagePoints.push_back(imagePoint);
		}

		outImagePointIDs.clear();
		outImagePointIDs.push_back(m_markerData->desiredArucoId);

		// Remember the last valid captured points
		m_lastValidImagePoints= m_currentImagePoints;

		return true;
	}

	return false;
}
