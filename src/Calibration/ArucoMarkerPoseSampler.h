#pragma once

#include "ComponentFwd.h"
#include "MikanMathTypes.h"
#include "ObjectSystemConfigFwd.h"
#include <memory>

#include "glm/ext/quaternion_double.hpp"
#include "glm/ext/vector_double3.hpp"
#include "glm/ext/matrix_double4x4.hpp"

class ArucoMarkerPoseSampler
{
public:
	// Samples poses of the stage's origin marker
	ArucoMarkerPoseSampler(CameraComponentPtr cameraComponent, class VideoFrameDistortionView* distortionView,
						   int desiredSampleCount);

	// Samples poses of an explicit marker definition (e.g., a utility marker)
	ArucoMarkerPoseSampler(CameraComponentPtr cameraComponent, class VideoFrameDistortionView* distortionView,
						   int desiredSampleCount, MarkerDefinitionConstPtr markerDefinition);
	virtual ~ArucoMarkerPoseSampler();

	inline class CalibrationPatternFinder_Aruco* getPatternFinder() const { return m_markerFinder; }

	bool hasFinishedSampling() const;
	float getCalibrationProgress() const;
	void resetCalibrationState();

	bool computeApertureRelativeMarkerXform();
	bool hasValidApertureRelativeMarkerXform() const;
	void sampleLastApertureRelativeMarkerXform();
	bool computeCalibratedMarkerPose(MikanQuatd& outRotation, MikanVector3d& outTranslation);

	void renderApertureSpaceCalibrationState();

protected:
	float frameWidth;
	float frameHeight;

	// Internal Calibration State
	struct ArucoMarkerPoseSamplerState* m_calibrationState;

	// Tracked camera used for calibration
	CameraComponentPtr m_calibrationCamera;

	// Calibration pattern being used
	class CalibrationPatternFinder_Aruco* m_markerFinder;
};