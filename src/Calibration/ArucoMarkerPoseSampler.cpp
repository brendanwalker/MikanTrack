#include "CalibrationPatternFinder_Aruco.h"
#include "CameraMath.h"
#include "MathGLM.h"
#include "ArucoMarkerPoseSampler.h"
#include "MathTypeConversion.h"
#include "MathOpenCV.h"
#include "MathUtility.h"

#include <algorithm>

#include "opencv2/opencv.hpp"

struct ArucoMarkerPoseSamplerState
{
	// Static Input
	MikanMonoIntrinsics inputCameraIntrinsics;
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

	void init(const MikanMonoIntrinsics& cameraIntrinsics, int sampleCount)
	{
		inputCameraIntrinsics= cameraIntrinsics;
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
ArucoMarkerPoseSampler::ArucoMarkerPoseSampler(const MikanMonoIntrinsics& cameraIntrinsics, int frameWidth,
											   int frameHeight, float markerLengthMM, int desiredArucoId,
											   eCharucoDictionaryType arucoDictionaryType, int desiredSampleCount)
	: m_calibrationState(new ArucoMarkerPoseSamplerState)
	, m_markerFinder(new CalibrationPatternFinder_Aruco(frameWidth, frameHeight, markerLengthMM, desiredArucoId,
														arucoDictionaryType))
{
	this->frameWidth= (float)frameWidth;
	this->frameHeight= (float)frameHeight;

	// The finder needs the mono intrinsics to estimate the marker pose via solvePnP
	m_markerFinder->setCameraIntrinsics(cameraIntrinsics);

	// Private calibration state
	m_calibrationState->init(cameraIntrinsics, desiredSampleCount);
}

ArucoMarkerPoseSampler::~ArucoMarkerPoseSampler()
{
	delete m_markerFinder;
	delete m_calibrationState;
}

void ArucoMarkerPoseSampler::setGrayscaleFrame(const cv::Mat* grayscaleFrame)
{
	m_markerFinder->setGrayscaleFrame(grayscaleFrame);
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

bool ArucoMarkerPoseSampler::computeCalibratedMarkerPose(glm::dmat4& outCameraSpaceMarkerXform)
{
	MikanQuatd rotation;
	MikanVector3d translation;
	if (!computeCalibratedMarkerPose(rotation, translation))
	{
		return false;
	}

	// Compose the camera-space marker transform (maps marker-local points into camera space)
	const glm::dquat glm_rotation= MikanQuatd_to_glm_dquat(rotation);
	const glm::dvec3 glm_translation= MikanVector3d_to_glm_dvec3(translation);
	outCameraSpaceMarkerXform= glm::translate(glm::dmat4(1.0), glm_translation) * glm::mat4_cast(glm_rotation);

	return true;
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
