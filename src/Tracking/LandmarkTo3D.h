#pragma once

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/ext/vector_float3.hpp"

#include "MikanVideoSourceTypes.h"
#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// Lifts image-space hand/arm landmarks into camera-space meters using the
// calibrated intrinsics and a calibrated hand scale (wrist -> middle-MCP
// length). Two depth estimators, selected via setPnpConfig:
//
// PnP solve (default): the landmark model's per-frame metric hand
// (modelPoints, rescaled so wrist->middleMCP matches the calibrated hand
// scale) is used as a DYNAMIC solvePnP object model against the 2D image
// landmarks - articulation is already baked into the model by the network,
// so PnP only recovers the global rigid pose. 21 correspondences (or the 6
// quasi-rigid palm points) instead of one bone: landmark jitter averages
// across the hand instead of riding on a single segment, foreshortening is
// handled by the full projective model, and per-landmark independent depth
// jitter collapses into one shared translation. Warm-started from the
// previous frame's pose. cameraPoints = R * obj + t.
//
// Legacy two-point estimator (A/B comparison):
//   - foreshortening correction from the model/world landmarks:
//       c= l3d / max(l2dModel, eps)
//   - wrist depth: Zwrist= fx * refLengthMeters / (pixelDist(wrist, middleMCP) * c)
//   - per landmark: back-project its pixel at Zwrist + model z offset
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

	// Depth estimator selection (config-switchable for A/B comparison).
	// bPalmOnly restricts the PnP correspondences to the 6 quasi-rigid palm
	// points (wrist, thumb CMC, 4 finger MCPs) - useful if occluded-fingertip
	// 2D/3D inconsistency drags the full solve.
	void setPnpConfig(bool bUsePnp, bool bPalmOnly)
	{
		m_bUsePnpDepth= bUsePnp;
		m_bPnpPalmOnly= bPalmOnly;
	}

	// Fills cameraPoints/hasCameraSpace on the frame's hands and arms
	void process(TrackingFrameResult& ioResult);

	// Live hand-scale override (stereo auto-scale); does not touch filter state
	void setRefLengthMeters(float refLengthMeters)
	{
		if (refLengthMeters > 1e-4f)
			m_refLengthMeters= refLengthMeters;
	}
	float getRefLengthMeters() const { return m_refLengthMeters; }

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
	// PnP path; returns false when the solve fails (caller falls back to the
	// legacy estimator for that frame)
	bool processHandPnp(TrackedHand& hand, float dtSeconds);
	void processHandLegacy(TrackedHand& hand, float dtSeconds);
	void processArm(TrackedArm& arm, const TrackedHand& hand, eHandSide side, float dtSeconds);

	bool m_bConfigured= false;
	float m_fx= 0.f;
	float m_fy= 0.f;
	float m_cx= 0.f;
	float m_cy= 0.f;
	float m_refLengthMeters= 0.08f;

	bool m_bUsePnpDepth= true;
	bool m_bPnpPalmOnly= false;

	// Warm-start state for the iterative PnP solve (per side, axis-angle +
	// translation in OpenCV camera convention). Also the future vision
	// measurement for IMU (EKF) fusion: a rigid 6-DoF wrist pose per frame.
	std::array<std::array<double, 3>, 2> m_pnpRvec{};
	std::array<std::array<double, 3>, 2> m_pnpTvec{};
	bool m_bPnpPoseValid[2]= {false, false};

	bool m_bSmoothingEnabled= true;
	HandOneEuroBank m_filterBank;

	// World-space elbow filters for the refined fallback path (separate from
	// the camera-space elbow filters so the two estimates don't share state)
	std::array<OneEuroFilterVec3, 2> m_worldElbowFilters;

	double m_lastTimestampMs= -1.0;
	float m_lastDtSeconds= 1.f / 60.f;
	bool m_bSideWasTracked[2]= {false, false};
};
