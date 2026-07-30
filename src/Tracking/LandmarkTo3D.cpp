#include "LandmarkTo3D.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

#include "Logger.h"

static constexpr float kMinDepthMeters= 0.05f;
static constexpr float kElbowDepthRangeMeters= 0.5f;
static constexpr float kDefaultDtSeconds= 1.f / 60.f;
// Anatomical forearm length as a multiple of the wrist->middle-MCP distance
// (forearm ~26cm vs palm ~8cm)
static constexpr float kForearmToHandScaleRatio= 3.2f;

void LandmarkTo3D::configure(
	const MikanMonoIntrinsics& intrinsics,
	double refLengthMeters,
	bool smoothingEnabled,
	float smoothingMinCutoff,
	float smoothingBeta)
{
	// MikanMatrix3d is column-major: fx= x0, fy= y1, cx= z0, cy= z1.
	// Landmarks arrive in undistorted image space, so use the undistorted matrix.
	const MikanMatrix3d& cameraMatrix= intrinsics.undistorted_camera_matrix;
	m_fx= (float)cameraMatrix.x0;
	m_fy= (float)cameraMatrix.y1;
	m_cx= (float)cameraMatrix.z0;
	m_cy= (float)cameraMatrix.z1;
	m_refLengthMeters= (float)refLengthMeters;

	m_bSmoothingEnabled= smoothingEnabled;
	m_filterBank.configure(smoothingMinCutoff, smoothingBeta, 1.f);
	m_filterBank.resetAll();
	for (OneEuroFilterVec3& filter : m_worldElbowFilters)
	{
		filter.configure(smoothingMinCutoff, smoothingBeta, 1.f);
		filter.reset();
	}
	m_lastTimestampMs= -1.0;
	m_bSideWasTracked[0]= false;
	m_bSideWasTracked[1]= false;

	m_bConfigured= m_fx > 1e-3f && m_fy > 1e-3f && m_refLengthMeters > 1e-4f;
	if (!m_bConfigured)
	{
		MIKAN_MT_LOG_WARNING("LandmarkTo3D::configure")
			<< "Invalid intrinsics/hand scale (fx=" << m_fx
			<< " fy=" << m_fy << " refLength=" << m_refLengthMeters << ")";
	}
}

glm::vec3 LandmarkTo3D::backProject(float u, float v, float z) const
{
	// P= z * K^-1 * [u v 1]^T, OpenCV convention (+X right, +Y down, +Z forward)
	return glm::vec3(
		(u - m_cx) * z / m_fx,
		(v - m_cy) * z / m_fy,
		z);
}

void LandmarkTo3D::process(TrackingFrameResult& ioResult)
{
	if (!m_bConfigured)
		return;

	// frame delta for the one-euro filters (clamped against timestamp glitches)
	float dtSeconds= kDefaultDtSeconds;
	if (m_lastTimestampMs >= 0.0 && ioResult.timestampMs > m_lastTimestampMs)
		dtSeconds= std::clamp((float)((ioResult.timestampMs - m_lastTimestampMs) / 1000.0), 1.f / 240.f, 0.25f);
	m_lastTimestampMs= ioResult.timestampMs;
	m_lastDtSeconds= dtSeconds;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		TrackedHand& hand= ioResult.hands[sideIndex];
		TrackedArm& arm= ioResult.arms[sideIndex];

		if (hand.tracked)
		{
			// fresh acquisition: drop stale filter state
			if (!m_bSideWasTracked[sideIndex])
			{
				m_filterBank.resetSide((eHandSide)sideIndex);
				m_worldElbowFilters[sideIndex].reset();
			}

			processHand(hand, dtSeconds);
		}
		m_bSideWasTracked[sideIndex]= hand.tracked && hand.hasCameraSpace;

		if (arm.valid)
			processArm(arm, hand, (eHandSide)sideIndex, dtSeconds);
	}
}

void LandmarkTo3D::processHand(TrackedHand& hand, float dtSeconds)
{
	const int wristIndex= (int)eHandLandmark::WRIST;
	const int middleMcpIndex= (int)eHandLandmark::MIDDLE_MCP;

	// -- scale + foreshortening from the model (metric) landmarks --
	const glm::vec3 modelSegment= hand.modelPoints[wristIndex] - hand.modelPoints[middleMcpIndex];
	const float l3d= glm::length(modelSegment);
	const float l2dModel= glm::length(glm::vec2(modelSegment));
	if (l3d < 1e-4f)
		return;
	const float foreshortening= l3d / std::max(l2dModel, 1e-6f);

	// -- wrist depth from the calibrated reference length --
	const glm::vec2 wristPixel= glm::vec2(hand.imagePoints[wristIndex]);
	const glm::vec2 middleMcpPixel= glm::vec2(hand.imagePoints[middleMcpIndex]);
	const float pixelDist= std::max(glm::length(wristPixel - middleMcpPixel), 1e-3f);
	const float zWrist= m_fx * m_refLengthMeters / (pixelDist * foreshortening);
	if (!std::isfinite(zWrist) || zWrist <= 0.f)
		return;

	// -- per-landmark depth from the model z profile, rescaled to real size --
	const float metersPerModelUnit= m_refLengthMeters / l3d;
	const float modelWristZ= hand.modelPoints[wristIndex].z;

	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		const float zOffset= (hand.modelPoints[i].z - modelWristZ) * metersPerModelUnit;
		const float z= std::max(zWrist + zOffset, kMinDepthMeters);

		glm::vec3 cameraPoint= backProject(hand.imagePoints[i].x, hand.imagePoints[i].y, z);
		if (m_bSmoothingEnabled)
			cameraPoint= m_filterBank.landmarkFilter(hand.side, i).filter(cameraPoint, dtSeconds);

		hand.cameraPoints[i]= cameraPoint;
	}

	hand.hasCameraSpace= true;
}

void LandmarkTo3D::processArm(TrackedArm& arm, const TrackedHand& hand, eHandSide side, float dtSeconds)
{
	// The wrist depth is the arm's only scale reference; without a tracked
	// hand in camera space the arm stays image-space only
	if (!hand.tracked || !hand.hasCameraSpace)
		return;

	const glm::vec3& wristCamera= hand.cameraPoints[(int)eHandLandmark::WRIST];
	const float zWrist= wristCamera.z;

	// pose z hint (elbow relative to wrist, meters) when available
	const float zOffset= arm.hasElbowZHint ? arm.elbowZOffsetFromWrist : 0.f;
	const float zElbow= std::max(
		std::clamp(zWrist + zOffset, zWrist - kElbowDepthRangeMeters, zWrist + kElbowDepthRangeMeters),
		kMinDepthMeters);

	glm::vec3 elbowCamera= backProject(arm.elbowPixel.x, arm.elbowPixel.y, zElbow);
	if (m_bSmoothingEnabled)
		elbowCamera= m_filterBank.elbowFilter(side).filter(elbowCamera, dtSeconds);

	arm.elbowCamera= elbowCamera;
	arm.wristCamera= wristCamera;
	arm.hasCameraSpace= true;
}

void LandmarkTo3D::refineFallbackArms(TrackingFrameResult& ioResult, const glm::dmat4& markerFromCamera)
{
	if (!m_bConfigured)
		return;

	const glm::dmat4 cameraFromMarker= glm::inverse(markerFromCamera);

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		TrackedArm& arm= ioResult.arms[sideIndex];
		const TrackedHand& hand= ioResult.hands[sideIndex];

		// Only refine hand-derived fallback elbows; measured pose elbows keep
		// their own (image + z hint) estimate
		if (!arm.valid || !arm.fromFallback || !hand.tracked || !hand.hasWorldSpace)
			continue;

		// Forearm direction from the hand's world-space orientation:
		// knuckle centroid -> wrist, extended up the forearm
		const glm::vec3& wristWorld= hand.worldPoints[(int)eHandLandmark::WRIST];
		const glm::vec3 mcpCentroid=
			(hand.worldPoints[(int)eHandLandmark::INDEX_MCP] + hand.worldPoints[(int)eHandLandmark::MIDDLE_MCP] +
			 hand.worldPoints[(int)eHandLandmark::RING_MCP] + hand.worldPoints[(int)eHandLandmark::PINKY_MCP]) *
			0.25f;

		const glm::vec3 handSegment= wristWorld - mcpCentroid;
		const float handSegmentLength= glm::length(handSegment);
		if (handSegmentLength < 1e-4f)
			continue;

		const glm::vec3 forearmDir= handSegment / handSegmentLength;
		const float forearmLength= m_refLengthMeters * kForearmToHandScaleRatio;

		glm::vec3 elbowWorld= wristWorld + forearmDir * forearmLength;

		// Table clamp: the marker plane is world z=0 and the arm can't be
		// below the table the hand is resting on
		elbowWorld.z= std::max(elbowWorld.z, 0.f);

		if (m_bSmoothingEnabled)
			elbowWorld= m_worldElbowFilters[sideIndex].filter(elbowWorld, m_lastDtSeconds);

		arm.elbowWorld= elbowWorld;
		arm.wristWorld= wristWorld;
		arm.hasWorldSpace= true;

		// Back-fill camera space + pixel position for the overlay
		const glm::vec3 elbowCamera= glm::vec3(cameraFromMarker * glm::dvec4(elbowWorld, 1.0));
		arm.elbowCamera= elbowCamera;
		arm.hasCameraSpace= true;
		if (elbowCamera.z > kMinDepthMeters)
		{
			arm.elbowPixel= glm::vec2(
				m_fx * elbowCamera.x / elbowCamera.z + m_cx,
				m_fy * elbowCamera.y / elbowCamera.z + m_cy);
		}
	}
}
