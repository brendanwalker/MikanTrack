#pragma once

#include "ObjectSystemFwd.h"
#include "ObjectSystemConfigFwd.h"
#include "MikanVideoSourceTypes.h"
#include "MikanCameraTypes.h"

#include <memory>

// Helper use to implement OpenCV camera lens intrinsic/distortion calibration method.
// See https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html for details.
class MonoLensDistortionCalibrator
{
public:
	MonoLensDistortionCalibrator(MarkerObjectSystemPtr markerObjectSystem,
								 class VideoFrameDistortionView* distortionView, int desiredBoardCount);
	virtual ~MonoLensDistortionCalibrator();

	inline class CalibrationPatternFinder* getPatternFinder() const { return m_patternFinder; }
	void findNewCalibrationPattern(const float minSeperationDist);
	bool captureLastFoundCalibrationPattern();

	bool hasSampledAllCalibrationPatterns() const;
	bool areCurrentImagePointsValid() const;
	float computeCalibrationProgress() const;
	void resetCalibrationState();
	void resetDistortionView();

	void computeCameraCalibration();
	bool getIsCameraCalibrationComplete() const;
	int getDesiredPatternCount() const;
	bool getCameraCalibration(MikanMonoIntrinsics* out_mono_intrinsics);
	float getReprojectionError() const;

	void renderCalibrationState();

protected:
	float frameWidth;
	float frameHeight;

	// Internal Calibration State
	struct MonoLensDistortionCalibrationState* m_calibrationState;

	// Video buffer state
	class VideoFrameDistortionView* m_distortionView;

	// Calibration pattern being used
	class CalibrationPatternFinder* m_patternFinder;
};