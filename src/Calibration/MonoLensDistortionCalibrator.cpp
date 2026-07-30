#include "CalibrationPatternFinder_Charuco.h"
#include "Logger.h"
#include "MathUtility.h"
#include "MonoLensDistortionCalibrator.h"
#include "MathTypeConversion.h"

#include <algorithm>
#include <atomic>
#include <thread>

#include "opencv2/opencv.hpp"

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

	// Image point stability state
	float imagePointsStabilityTimer;
	bool areImagePointsStable;

	// Async Calibration Compute
	std::thread* asyncComputeTask;
	std::atomic_int asyncComputeTaskStatus;

	// Result
	MikanMonoIntrinsics outputCameraIntrinsics;
	double reprojectionError;

	void init(CalibrationPatternFinder* patternFinder, int width, int height, int patternCount)
	{
		frameWidth= width;
		frameHeight= height;
		desiredPatternCount= patternCount;

		createDefautMonoIntrinsics(frameWidth, frameHeight, originalCameraIntrinsics);

		patternFinder->getOpenCVLensCalibrationGeometry(&calibrationGeometry);

		resetCalibration();
	}

	void resetCalibration()
	{
		// Reset the capture state
		capturedPatternCount= 0;
		quadList.clear();
		imagePointsList.clear();
		imagePointIDList.clear();

		// Reset the stability state
		imagePointsStabilityTimer= 0.f;
		areImagePointsStable= false;

		// Reset the async task state
		asyncComputeTask= nullptr;
		asyncComputeTaskStatus= NotStarted;

		// Reset the output
		outputCameraIntrinsics= originalCameraIntrinsics;
		reprojectionError= 0.0;
	}
};

//-- MonoDistortionCalibrator ----
MonoLensDistortionCalibrator::MonoLensDistortionCalibrator(int frameWidth, int frameHeight, int charucoCols,
														   int charucoRows, float charucoSquareLengthMM,
														   float charucoMarkerLengthMM,
														   eCharucoDictionaryType charucoDictionaryType,
														   int desiredSampleCount)
	: m_calibrationState(new MonoLensDistortionCalibrationState)
{
	m_patternFinder= new CalibrationPatternFinder_Charuco(frameWidth, frameHeight, charucoCols, charucoRows,
														  charucoSquareLengthMM, charucoMarkerLengthMM,
														  charucoDictionaryType);

	this->frameWidth= (float)frameWidth;
	this->frameHeight= (float)frameHeight;

	m_calibrationState->init(m_patternFinder, frameWidth, frameHeight, desiredSampleCount);
}

MonoLensDistortionCalibrator::~MonoLensDistortionCalibrator()
{
	if (m_calibrationState->asyncComputeTask != nullptr)
	{
		m_calibrationState->asyncComputeTask->join();
		delete m_calibrationState->asyncComputeTask;
	}

	delete m_patternFinder;
	delete m_calibrationState;
}

void MonoLensDistortionCalibrator::update(float deltaSeconds, const cv::Mat* grayscaleFrame,
										  const float minSeperationDist)
{
	// Feed the latest grayscale frame to the pattern finder
	m_patternFinder->setGrayscaleFrame(grayscaleFrame);

	if (hasSampledAllCalibrationPatterns())
		return;

	// Update the pattern capture state
	findNewCalibrationPattern(minSeperationDist);

	// Update the image point stability timer.
	// Once the pattern has been held valid (in a new location) for the stability
	// duration, capture it as a calibration sample.
	if (areCurrentImagePointsValid())
	{
		if (!m_calibrationState->areImagePointsStable)
		{
			m_calibrationState->imagePointsStabilityTimer+= deltaSeconds;
			if (m_calibrationState->imagePointsStabilityTimer >= k_imagePointStabilityDuration)
			{
				m_calibrationState->areImagePointsStable= true;

				// Capturing updates the finder's last-valid points, so the pattern has to
				// move at least minSeperationDist away before the next capture can start
				captureLastFoundCalibrationPattern();
			}
		}
	}
	else
	{
		m_calibrationState->imagePointsStabilityTimer= 0.f;
		m_calibrationState->areImagePointsStable= false;
	}
}

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

bool MonoLensDistortionCalibrator::areCurrentImagePointsStable() const
{
	return m_calibrationState->areImagePointsStable;
}

float MonoLensDistortionCalibrator::computeCalibrationProgress() const
{
	const float samplePercentage=
		(float)m_calibrationState->capturedPatternCount / (float)m_calibrationState->desiredPatternCount;

	return samplePercentage;
}

void MonoLensDistortionCalibrator::resetCalibrationState() { m_calibrationState->resetCalibration(); }

void MonoLensDistortionCalibrator::computeCameraCalibration()
{
	assert(m_calibrationState->asyncComputeTask == nullptr);

	MonoLensDistortionCalibrationState* calibrationState= m_calibrationState;

	// Spin up a worker thread to compute the camera calibration.
	// The result of this is polled by getCameraCalibration().
	calibrationState->asyncComputeTaskStatus= MonoLensDistortionCalibrationState::Running;
	calibrationState->asyncComputeTask= new std::thread(
		[calibrationState]
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
