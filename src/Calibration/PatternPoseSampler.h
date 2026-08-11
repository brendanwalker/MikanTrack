#pragma once

#include <map>
#include <memory>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"
#include "opencv2/core/types.hpp"

#include "CalibrationPatternFinder.h"
#include "MikanVideoSourceTypes.h"

// Samples a calibration pattern's pose in one camera, for extrinsics.
//
// Works with any CalibrationPatternFinder (charuco today), because the pose
// solve is driven by the finder's point IDs and object geometry rather than by
// a specific pattern's corner layout.
//
// WHY THE OBSERVATIONS ARE POOLED, not the poses: the pattern sits still while
// several frames are captured, so every frame measures the same corners with
// independent detection noise. Averaging each corner's PIXEL position across
// the captures and solving ONCE beats solving per frame and averaging the
// resulting transforms - pose averaging mixes a nonlinear estimate through a
// quaternion mean, while pixel averaging attacks the noise where it enters.
// It also yields one meaningful reprojection error for the final pose, which
// is the number that says whether the calibration is any good.
class PatternPoseSampler
{
public:
	PatternPoseSampler(const MikanMonoIntrinsics& cameraIntrinsics, CalibrationPatternFinderPtr patternFinder,
					   int desiredSampleCount);

	CalibrationPatternFinder* getPatternFinder() const { return m_patternFinder.get(); }

	// The caller owns the grayscale frame buffer and updates the pointer each
	// frame; captureSample searches the current frame
	void setGrayscaleFrame(const cv::Mat* grayscaleFrame);

	bool hasFinishedSampling() const;
	float getCalibrationProgress() const;
	void resetCalibrationState();

	// Detects the pattern in the current frame and accumulates its corner
	// observations. Returns false when the pattern was not found (the caller
	// keeps trying); a successful call advances the sample count.
	bool captureSample();
	// Whether the most recent captureSample call found the pattern (for the
	// UI's "hold the board still where both cameras see it" feedback)
	bool hasValidDetection() const { return m_bLastDetectionValid; }

	// Solves one pose from the per-corner averaged observations. Valid once
	// hasFinishedSampling(). outCameraFromPattern maps pattern-local points
	// into GL camera space, meters (same convention the wizard consumed from
	// the old aruco sampler).
	bool computeCalibratedPatternPose(glm::dmat4& outCameraFromPattern);

	// Quality of the solve above: mean reprojection error of the averaged
	// corners, in pixels, and how many distinct corners it was solved from
	float getMeanReprojectionErrorPx() const { return m_meanReprojectionErrorPx; }
	int getObservedCornerCount() const { return (int)m_observations.size(); }

	// Per-corner averaged observations, for cross-camera validation: the same
	// corner ID seen by two cameras is a correspondence, which is what makes
	// triangulating the board possible.
	void getAveragedObservations(std::vector<int>& outPointIDs,
								 std::vector<cv::Point2f>& outImagePoints) const;
	// Object-space position of a corner ID (millimeters, pattern local)
	bool getObjectPoint(int pointID, cv::Point3f& outObjectPointMM) const;

private:
	struct CornerObservation
	{
		cv::Point2d pixelSum{0.0, 0.0};
		int sampleCount= 0;
	};

	MikanMonoIntrinsics m_cameraIntrinsics;
	CalibrationPatternFinderPtr m_patternFinder;
	int m_desiredSampleCount= 1;

	int m_capturedSampleCount= 0;
	bool m_bLastDetectionValid= false;
	// Keyed by pattern point ID, so a partially visible board still
	// contributes whatever corners it showed
	std::map<int, CornerObservation> m_observations;

	float m_meanReprojectionErrorPx= 0.f;
};
