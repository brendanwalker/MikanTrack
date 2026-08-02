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
	static constexpr float k_defaultMinSeperationDist= 10.f; // pixels

	// Focal length is only observable from PERSPECTIVE (tilted boards). A
	// capture set of fronto-parallel boards fits distortion to sub-pixel
	// error while fx sits wherever the initializer left it (seen live: a
	// solved hfov of exactly 90 deg). At least this many samples must show
	// real keystone before the set is complete; flat captures stop being
	// accepted once their quota (desired - tilted) is full.
	static constexpr int k_minTiltedSampleCount= 4;
	// A board tilted ~20 deg at arm's length shows ~1.12; fronto-parallel ~1.0
	static constexpr float k_tiltKeystoneThreshold= 1.10f;

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
	// UI hint: the flat-board quota is full - only tilted boards are accepted now
	bool wantsTiltedSample() const;
	int getTiltedSampleCount() const;
	// 0..1 progress of the current "hold steady" window; capture fires at 1
	// and the fraction restarts for the next sample
	float getStabilityFraction() const;
	float computeCalibrationProgress() const;
	void resetCalibrationState();

	void computeCameraCalibration();
	bool getIsCameraCalibrationComplete() const;
	int getDesiredPatternCount() const;
	bool getCameraCalibration(MikanMonoIntrinsics* out_mono_intrinsics);
	float getReprojectionError() const;

protected:
	// Keystone factor (>= 1) of the current detection: max length ratio
	// between opposite outer edges of the board. Requires a FULL board
	// detection (corner index == charuco id), which the finder enforces.
	float computeCurrentPatternKeystone() const;

	float frameWidth;
	float frameHeight;
	int m_cornerCols= 0;
	int m_cornerRows= 0;

	// Internal Calibration State
	struct MonoLensDistortionCalibrationState* m_calibrationState;

	// Calibration pattern being used
	class CalibrationPatternFinder_Charuco* m_patternFinder;
};
