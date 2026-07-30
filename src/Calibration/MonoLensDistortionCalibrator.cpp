#include "CalibrationRenderHelpers.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "Colors.h"
#include "Logger.h"
#include "MathUtility.h"
#include "MarkerObjectSystem.h"
#include "MonoLensDistortionCalibrator.h"
#include "MathTypeConversion.h"
#include "VideoFrameDistortionView.h"
#include "VideoSourceComponent.h"

#include <algorithm>
#include <atomic>
#include <thread>

struct MonoLensDistortionCalibrationState
{
	enum AsyncComputeStatus
	{
		NotStarted,
		Running,
		Succeded,
		Failed
	};

	// Static Input
	OpenCVCalibrationGeometry calibrationGeometry;
	MikanMonoIntrinsics originalCameraIntrinsics;
	int frameWidth, frameHeight;
	int desiredPatternCount;

	// Capture State
	int capturedPatternCount;
	t_opencv_point2d_list quadList;
	std::vector<t_opencv_point2d_list> imagePointsList;
	std::vector<t_opencv_pointID_list> imagePointIDList;

	// Async Calibration Compute
	std::thread* asyncComputeTask;
	std::atomic_int asyncComputeTaskStatus;

	// Result
	MikanMonoIntrinsics outputCameraIntrinsics;
	double reprojectionError;

	void init(CalibrationPatternFinder* patternFinder, VideoSourceComponentPtr videoSourceComponent, int patternCount)
	{
		videoSourceComponent->getVideoPixelDimensions(frameWidth, frameHeight);
		desiredPatternCount= patternCount;

		patternFinder->getOpenCVLensCalibrationGeometry(&calibrationGeometry);

		resetCalibration();
	}

	void resetCalibration()
	{
		// Reset the capture state
		capturedPatternCount= 0;
		quadList.clear();
		imagePointsList.clear();

		// Reset the async task state
		asyncComputeTask= nullptr;
		asyncComputeTaskStatus= NotStarted;

		// Reset the output
		outputCameraIntrinsics= originalCameraIntrinsics;
		reprojectionError= 0.0;
	}
};

//-- MonoDistortionCalibrator ----
MonoLensDistortionCalibrator::MonoLensDistortionCalibrator(MarkerObjectSystemPtr markerObjectSystem,
														   VideoFrameDistortionView* distortionView,
														   int desiredBoardCount)
	: m_calibrationState(new MonoLensDistortionCalibrationState)
	, m_distortionView(distortionView)
{
	auto markerConfig= markerObjectSystem->getTypedDefinition();
	m_patternFinder= new CalibrationPatternFinder_Charuco(
		distortionView, markerConfig->getCharucoRows(), markerConfig->getCharucoCols(),
		markerConfig->getCharucoSquareLengthMM(), markerConfig->getCharucoMarkerLengthMM(),
		markerConfig->getCharucoDictionaryType());

	frameWidth= distortionView->getFrameWidth();
	frameHeight= distortionView->getFrameHeight();

	m_calibrationState->init(m_patternFinder, distortionView->getVideoSourceComponent(), desiredBoardCount);
}

MonoLensDistortionCalibrator::~MonoLensDistortionCalibrator() { delete m_patternFinder; }

void MonoLensDistortionCalibrator::findNewCalibrationPattern(const float minSeperationDist)
{
	m_patternFinder->findNewCalibrationPattern(minSeperationDist);
}

bool MonoLensDistortionCalibrator::captureLastFoundCalibrationPattern()
{
	bool bCaptured= false;
	t_opencv_point2d_list imagePoints;
	t_opencv_pointID_list imagePointIDs;
	cv::Point2f boundingQuad[4];
	if (m_patternFinder->fetchLastFoundCalibrationPattern(imagePoints, imagePointIDs, boundingQuad))
	{
		m_calibrationState->imagePointsList.push_back(imagePoints);
		m_calibrationState->imagePointIDList.push_back(imagePointIDs);

		for (int i= 0; i < 4; ++i)
		{
			m_calibrationState->quadList.push_back(boundingQuad[i]);
		}

		++m_calibrationState->capturedPatternCount;
		bCaptured= true;
	}

	return bCaptured;
}

bool MonoLensDistortionCalibrator::hasSampledAllCalibrationPatterns() const
{
	return m_calibrationState->capturedPatternCount >= m_calibrationState->desiredPatternCount;
}

bool MonoLensDistortionCalibrator::areCurrentImagePointsValid() const
{
	return m_patternFinder->areCurrentImagePointsValid();
}

float MonoLensDistortionCalibrator::computeCalibrationProgress() const
{
	const float samplePercentage=
		(float)m_calibrationState->capturedPatternCount / (float)m_calibrationState->desiredPatternCount;

	return samplePercentage;
}

void MonoLensDistortionCalibrator::resetCalibrationState() { m_calibrationState->resetCalibration(); }

void MonoLensDistortionCalibrator::resetDistortionView()
{
	m_distortionView->applyMonoCameraIntrinsics(&m_calibrationState->originalCameraIntrinsics);
	m_distortionView->setVideoDisplayMode(eVideoDisplayMode::mode_bgr);
}

void MonoLensDistortionCalibrator::computeCameraCalibration()
{
	assert(m_calibrationState->asyncComputeTask == nullptr);

	CalibrationPatternFinder* patternFinder= m_patternFinder;
	MonoLensDistortionCalibrationState* calibrationState= m_calibrationState;

	// Spin up a worker thread to compute the camera calibration.
	// The result of this is polled by getCameraCalibration().
	calibrationState->asyncComputeTaskStatus= MonoLensDistortionCalibrationState::Running;
	calibrationState->asyncComputeTask= new std::thread(
		[patternFinder, calibrationState]
		{
			bool bSuccess= computeMonoLensCameraCalibration(
				calibrationState->frameWidth, calibrationState->frameHeight, calibrationState->calibrationGeometry,
				calibrationState->imagePointsList, calibrationState->imagePointIDList,
				calibrationState->outputCameraIntrinsics, calibrationState->reprojectionError);

			// Signal the main thread that the task is complete
			calibrationState->asyncComputeTaskStatus=
				bSuccess ? MonoLensDistortionCalibrationState::Succeded : MonoLensDistortionCalibrationState::Failed;
		});
}

bool MonoLensDistortionCalibrator::getIsCameraCalibrationComplete() const
{
	auto status= m_calibrationState->asyncComputeTaskStatus.load();

	return status == MonoLensDistortionCalibrationState::Succeded
		   || status == MonoLensDistortionCalibrationState::Failed;
}

int MonoLensDistortionCalibrator::getDesiredPatternCount() const { return m_calibrationState->desiredPatternCount; }

bool MonoLensDistortionCalibrator::getCameraCalibration(MikanMonoIntrinsics* out_mono_intrinsics)
{
	bool bFetchSuccess= false;

	if (m_calibrationState->asyncComputeTaskStatus.load() == MonoLensDistortionCalibrationState::Succeded)
	{
		if (m_calibrationState->asyncComputeTask != nullptr)
		{
			m_calibrationState->asyncComputeTask->join();

			delete m_calibrationState->asyncComputeTask;
			m_calibrationState->asyncComputeTask= nullptr;
		}

		*out_mono_intrinsics= m_calibrationState->outputCameraIntrinsics;
		bFetchSuccess= true;
	}

	return bFetchSuccess;
}

float MonoLensDistortionCalibrator::getReprojectionError() const { return m_calibrationState->reprojectionError; }

void MonoLensDistortionCalibrator::renderCalibrationState()
{
	IMkGraphicsContext* graphicsContext= m_distortionView->getGraphicsContext();

	// Draw the most recently capture chessboard
	m_patternFinder->renderCalibrationPattern2D();

	// Draw the outlines of all of the chess boards
	if (m_calibrationState->quadList.size() > 0)
	{
		drawQuadList2d(graphicsContext, frameWidth, frameHeight,
					   (float*)m_calibrationState->quadList.data(), // cv::point2f is just two floats
					   (int)m_calibrationState->quadList.size(), Colors::Yellow);
	}
}