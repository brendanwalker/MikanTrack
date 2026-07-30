#include "CalibrationRenderHelpers.h"
#include "CalibrationPatternFinder_Aruco.h"
#include "CameraComponent.h"
#include "CameraMath.h"
#include "Colors.h"
#include "MikanLineRenderer.h"
#include "MikanTextRenderer.h"
#include "MathGLM.h"
#include "ArucoMarkerPoseSampler.h"
#include "MathTypeConversion.h"
#include "MathOpenCV.h"
#include "MathUtility.h"
#include "TextStyle.h"
#include "VideoFrameDistortionView.h"

#include <algorithm>
#include <atomic>
#include <thread>

struct ArucoMarkerPoseSamplerState
{
	// Static Input
	MikanMonoIntrinsics inputCameraIntrinsics;
	ProjectConfigConstPtr profileConfig;
	int desiredSampleCount;

	// Computed every frame
	glm::dmat4 cameraToMarkerXform;
	bool hasValidCapture;

	// Sample State
	int capturedSampleCount;
	std::vector<cv::Quatd> cv_apertureOffsetQuats;
	std::vector<cv::Vec3d> cv_apertureOffsetPositions; // meters

	// Result
	glm::dquat rotationOffset;
	glm::dvec3 translationOffset;

	void init(CameraComponentPtr cameraComponent, int sampleCount)
	{
		// Get the current camera intrinsics being used by the video source
		MikanVideoSourceIntrinsics cameraIntrinsics;
		cameraComponent->getApertureIntrinsics(cameraIntrinsics);
		assert(cameraIntrinsics.intrinsics_type == MikanIntrinsicsType::MONO_CAMERA_INTRINSICS);

		inputCameraIntrinsics= cameraIntrinsics.getMonoIntrinsics();
		desiredSampleCount= sampleCount;

		resetCalibration();
	}

	void resetCalibration()
	{
		// Reset the capture state
		cameraToMarkerXform= glm::dmat4(1.0);
		hasValidCapture= false;

		capturedSampleCount= 0;
		cv_apertureOffsetQuats.clear();
		cv_apertureOffsetPositions.clear();

		// Reset the output
		rotationOffset= glm::dquat();
		translationOffset= glm::dvec3(0.0);
	}
};

//-- ArucoMarkerPoseSampler ----
ArucoMarkerPoseSampler::ArucoMarkerPoseSampler(CameraComponentPtr cameraComponent,
											   VideoFrameDistortionView* distortionView, int desiredSampleCount)
	: m_calibrationState(new ArucoMarkerPoseSamplerState)
	, m_calibrationCamera(cameraComponent)
	, m_markerFinder(new CalibrationPatternFinder_Aruco(cameraComponent, distortionView))
{
	frameWidth= distortionView->getFrameWidth();
	frameHeight= distortionView->getFrameHeight();

	// Private calibration state
	m_calibrationState->init(cameraComponent, desiredSampleCount);
}

ArucoMarkerPoseSampler::ArucoMarkerPoseSampler(CameraComponentPtr cameraComponent,
											   VideoFrameDistortionView* distortionView, int desiredSampleCount,
											   MarkerDefinitionConstPtr markerDefinition)
	: m_calibrationState(new ArucoMarkerPoseSamplerState)
	, m_calibrationCamera(cameraComponent)
	, m_markerFinder(new CalibrationPatternFinder_Aruco(cameraComponent, distortionView, markerDefinition))
{
	frameWidth= distortionView->getFrameWidth();
	frameHeight= distortionView->getFrameHeight();

	m_calibrationState->init(cameraComponent, desiredSampleCount);
}

ArucoMarkerPoseSampler::~ArucoMarkerPoseSampler()
{
	delete m_markerFinder;
	delete m_calibrationState;
}

bool ArucoMarkerPoseSampler::hasFinishedSampling() const
{
	return m_calibrationState->capturedSampleCount >= m_calibrationState->desiredSampleCount;
}

float ArucoMarkerPoseSampler::getCalibrationProgress() const
{
	const float samplePercentage=
		(float)m_calibrationState->capturedSampleCount / (float)m_calibrationState->desiredSampleCount;

	return samplePercentage;
}

void ArucoMarkerPoseSampler::resetCalibrationState() { m_calibrationState->resetCalibration(); }

bool ArucoMarkerPoseSampler::computeApertureRelativeMarkerXform()
{
	// Mark the last capture as invalid
	m_calibrationState->hasValidCapture= false;

	// Look for the calibration pattern in the latest video frame
	glm::dmat4 apertureToPatternXform;
	if (!m_markerFinder->estimateNewCalibrationPatternPose(apertureToPatternXform))
	{
		return false;
	}

	// Save the aperture-relative transform to the calibration state
	m_calibrationState->cameraToMarkerXform= apertureToPatternXform;
	m_calibrationState->hasValidCapture= true;

	return true;
}

bool ArucoMarkerPoseSampler::hasValidApertureRelativeMarkerXform() const { return m_calibrationState->hasValidCapture; }

void ArucoMarkerPoseSampler::sampleLastApertureRelativeMarkerXform()
{
	if (!m_calibrationState->hasValidCapture)
		return;

	// Extract the rotation and translation offsets from the aperture-relative marker transform
	glm::dvec3 glm_translationOffset= m_calibrationState->cameraToMarkerXform[3];
	glm::dquat glm_rotationOffset= glm::quat_cast(m_calibrationState->cameraToMarkerXform);

	// Store as OpenCV types for averaging operation in computeCalibratedMarkerPose
	cv::Vec3d cv_translationOffset= glm_dvec3_to_cv_vec3d(glm_translationOffset);
	cv::Quatd cv_rotationOffset= glm_dquat_to_cv_quatd(glm_rotationOffset);

	m_calibrationState->cv_apertureOffsetQuats.push_back(cv_rotationOffset);
	m_calibrationState->cv_apertureOffsetPositions.push_back(cv_translationOffset);
	m_calibrationState->capturedSampleCount++;
}

bool ArucoMarkerPoseSampler::computeCalibratedMarkerPose(MikanQuatd& outRotation, MikanVector3d& outTranslation)
{
	cv::Vec3d cv_cameraOffsetPosition;
	cv::Quatd cv_cameraOffsetQuat;

	if (hasFinishedSampling()
		&& opencv_quaternion_compute_average(m_calibrationState->cv_apertureOffsetQuats, cv_cameraOffsetQuat)
		&& opencv_vec3d_compute_average(m_calibrationState->cv_apertureOffsetPositions, cv_cameraOffsetPosition))
	{
		outRotation= cv_quatd_to_MikanQuatd(cv_cameraOffsetQuat);
		outTranslation= cv_vec3d_to_MikanVector3d(cv_cameraOffsetPosition);

		return true;
	}

	return false;
}

void ArucoMarkerPoseSampler::renderApertureSpaceCalibrationState()
{
	IMkGraphicsContext* graphicsContext= m_calibrationCamera->getGraphicsContext();

	// Draw the most recently captured marker pattern in camera space
	m_markerFinder->renderCalibrationPattern2D();

	// Draw the camera-relative transform of the pattern (computed from solvePnP)
	if (m_calibrationState->hasValidCapture)
	{
		const glm::mat4 cameraToMarkerXform= glm::mat4(m_calibrationState->cameraToMarkerXform);

		drawTransformedAxes(graphicsContext, cameraToMarkerXform, 0.1f);

		TextStyle style= getDefaultTextStyle();
		drawTextAtWorldPosition(graphicsContext, style, glm_mat4_get_position(cameraToMarkerXform), L"Marker");
	}
}
