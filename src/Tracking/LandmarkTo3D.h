#pragma once

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/ext/vector_float3.hpp"

#include "MikanVideoSourceTypes.h"
#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// Lifts image-space hand/arm landmarks into camera-space meters using the
// calibrated intrinsics and a calibrated hand scale (wrist -> middle-MCP
// length).
//
// Method (per hand):
//   - foreshortening correction from the model/world landmarks:
//       l3d= |model wrist - model middleMCP| (meters)
//       l2dModel= |xy(model wrist - model middleMCP)| (2D projection in model space)
//       c= l3d / max(l2dModel, eps)   (>= 1 when the segment tilts toward the camera)
//   - wrist depth: Zwrist= fx * refLengthMeters / (pixelDist(wrist, middleMCP) * c)
//   - per landmark: Zi= Zwrist + (modelZi - modelZwrist) * (refLengthMeters / l3d),
//     then back-project its pixel at Zi: P= Zi * K^-1 * [u v 1]^T
//   - optional one-euro filtering of the 3D points
//
// Conventions: input pixel coordinates are assumed to be in UNDISTORTED image
// space (the vision thread undistorts frames before inference), so the
// undistorted camera matrix is used. Camera space is the OpenCV convention:
// +X right, +Y down, +Z forward, meters.
//
// Elbows: back-projected at the wrist depth plus the pose model's metric
// z hint when available, clamped to [Zwrist - 0.5, Zwrist + 0.5]. An arm
// without an associated tracked hand has no scale reference and keeps
// hasCameraSpace= false.
class LandmarkTo3D
{
public:
	void configure(
		const MikanMonoIntrinsics& intrinsics,
		double refLengthMeters,
		bool smoothingEnabled,
		float smoothingMinCutoff,
		float smoothingBeta);

	// Fills cameraPoints/hasCameraSpace on the frame's hands and arms
	void process(TrackingFrameResult& ioResult);

	// Recomputes fallback (hand-derived) elbows in world space, after
	// applyWorldTransform has run. The forearm is extended from the hand's
	// world-space orientation with an anatomical length derived from the
	// calibrated hand scale, and the elbow is clamped to at-or-above the
	// table plane (world z >= 0) - a hand resting on the table physically
	// has its elbow at about table height, so the clamp is usually the
	// right answer, not just a guard. Camera-space and pixel positions are
	// back-filled for the overlay.
	void refineFallbackArms(TrackingFrameResult& ioResult, const glm::dmat4& markerFromCamera);

private:
	glm::vec3 backProject(float u, float v, float z) const;
	void processHand(TrackedHand& hand, float dtSeconds);
	void processArm(TrackedArm& arm, const TrackedHand& hand, eHandSide side, float dtSeconds);

	bool m_bConfigured= false;
	float m_fx= 0.f;
	float m_fy= 0.f;
	float m_cx= 0.f;
	float m_cy= 0.f;
	float m_refLengthMeters= 0.08f;

	bool m_bSmoothingEnabled= true;
	HandOneEuroBank m_filterBank;

	// World-space elbow filters for the refined fallback path (separate from
	// the camera-space elbow filters so the two estimates don't share state)
	std::array<OneEuroFilterVec3, 2> m_worldElbowFilters;

	double m_lastTimestampMs= -1.0;
	float m_lastDtSeconds= 1.f / 60.f;
	bool m_bSideWasTracked[2]= {false, false};
};
