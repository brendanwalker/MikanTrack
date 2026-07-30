#pragma once

#include "CalibrationPatternFinder.h"
#include "MikanMathTypes.h"
#include "MikanVideoSourceTypes.h"

#include <memory>

#include "glm/ext/quaternion_double.hpp"
#include "glm/ext/vector_double3.hpp"
#include "glm/ext/matrix_double4x4.hpp"

class ArucoMarkerPoseSampler
{
public:
	ArucoMarkerPoseSampler(const MikanMonoIntrinsics& cameraIntrinsics, int frameWidth, int frameHeight,
						   float markerLengthMM, int desiredArucoId, eCharucoDictionaryType arucoDictionaryType,
						   int desiredSampleCount);
	virtual ~ArucoMarkerPoseSampler();

	inline class CalibrationPatternFinder_Aruco* getPatternFinder() const { return m_markerFinder; }

	// The caller owns the grayscale frame buffer and updates the pointer each frame;
	// computeApertureRelativeMarkerXform searches the current frame for the marker
	void setGrayscaleFrame(const cv::Mat* grayscaleFrame);

	bool hasFinishedSampling() const;
	float getCalibrationProgress() const;
	void resetCalibrationState();

	bool computeApertureRelativeMarkerXform();
	bool hasValidApertureRelativeMarkerXform() const;
	void sampleLastApertureRelativeMarkerXform();

	// Computes the average of the sampled marker poses.
	// The returned transform is the marker's pose in camera space (OpenGL convention,
	// units in meters): it maps marker-local coordinates into camera space
	// ("camera-from-marker"), so its translation column is the marker's position
	// relative to the camera aperture.
	bool computeCalibratedMarkerPose(glm::dmat4& outCameraSpaceMarkerXform);
	bool computeCalibratedMarkerPose(MikanQuatd& outRotation, MikanVector3d& outTranslation);

protected:
	float frameWidth;
	float frameHeight;

	// Internal Calibration State
	struct ArucoMarkerPoseSamplerState* m_calibrationState;

	// Calibration pattern being used
	class CalibrationPatternFinder_Aruco* m_markerFinder;
};
