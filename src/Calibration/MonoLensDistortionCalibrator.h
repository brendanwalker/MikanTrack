#pragma once

#include "CalibrationPatternFinder.h"
#include "MikanVideoSourceTypes.h"

#include <memory>

// Helper use to implement OpenCV camera lens intrinsic/distortion calibration method.
// See https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html for details.
class MonoLensDistortionCalibrator
{
public:
	// How long the pattern must stay valid (in a new location) before a sample is auto-captured
	static constexpr float k_imagePointStabilityDuration= 1.0f; // seconds
	// Minimum average pixel distance the pattern corners must move from the last
	// captured sample to be considered a new board location
	static constexpr float k_defaultMinSeperationDist= 100.f; // pixels

	MonoLensDistortionCalibrator(int frameWidth, int frameHeight, int charucoCols, int charucoRows,
								 float charucoSquareLengthMM, float charucoMarkerLengthMM,
								 eCharucoDictionaryType charucoDictionaryType, int desiredSampleCount);
	virtual ~MonoLensDistortionCalibrator();

	inline class CalibrationPatternFinder_Charuco* getPatternFinder() const { return m_patternFinder; }

	// Per-frame update: feed in the latest grayscale video frame, search it for the
	// calibration pattern, and auto-capture a sample once the pattern has been held
	// steady in a new location for k_imagePointStabilityDuration seconds
	void update(float deltaSeconds, const cv::Mat* grayscaleFrame,
				const float minSeperationDist= k_defaultMinSeperationDist);

	void findNewCalibrationPattern(const float minSeperationDist);
	bool captureLastFoundCalibrationPattern();

	bool hasSampledAllCalibrationPatterns() const;
	bool areCurrentImagePointsValid() const;
	bool areCurrentImagePointsStable() const;
	float computeCalibrationProgress() const;
	void resetCalibrationState();

	void computeCameraCalibration();
	bool getIsCameraCalibrationComplete() const;
	int getDesiredPatternCount() const;
	bool getCameraCalibration(MikanMonoIntrinsics* out_mono_intrinsics);
	float getReprojectionError() const;

protected:
	float frameWidth;
	float frameHeight;

	// Internal Calibration State
	struct MonoLensDistortionCalibrationState* m_calibrationState;

	// Calibration pattern being used
	class CalibrationPatternFinder_Charuco* m_patternFinder;
};
