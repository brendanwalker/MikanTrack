#pragma once

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/ext/vector_float3.hpp"

#include "HandPoseModel.h"
#include "MikanVideoSourceTypes.h"
#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// Lifts image-space hand landmarks into camera-space meters using the
// calibrated intrinsics and the hand scale (wrist -> middle-MCP length,
// seeded by config and live-refined from stereo/depth measurement).
//
// Two 3D sources, best first:
// - Hardware depth (RealSense): metric measurements per landmark, supplied
//   by the vision thread (see HandDepthMeasurement below).
// - Monocular PnP: the landmark model's per-frame metric hand (modelPoints,
//   rescaled so wrist->middleMCP matches the hand scale) as a DYNAMIC
//   solvePnP object model against the 2D landmarks - articulation is baked
//   into the model, so PnP only recovers the global rigid pose. Warm-started
//   from the previous frame. cameraPoints = R * obj + t.
//
// Conventions: input pixel coordinates are assumed to be in UNDISTORTED image
// space (the vision thread undistorts frames before inference), so the
// undistorted camera matrix is used. Camera space is the OpenCV convention:
// +X right, +Y down, +Z forward, meters.
//
// Also fills the parametric HandPose (palm transform + finger angles +
// skeleton) - see HandPoseModel.
// Per-hand hardware depth measurements (RealSense): metric camera-space
// points for whichever landmarks the depth sensor resolved. Sampled by the
// vision thread (which owns the distortion mapping); consumed here as a
// replacement for the PnP palm transform.
struct HandDepthMeasurement
{
	bool bValid[HAND_LANDMARK_COUNT]= {};
	std::array<glm::vec3, HAND_LANDMARK_COUNT> cameraPoints{};
	int validCount= 0;
};

class LandmarkTo3D
{
public:
	void configure(
		const MikanMonoIntrinsics& intrinsics,
		double refLengthMeters);

	// Fills cameraPoints/hasCameraSpace on the frame's hands and arms.
	// depthMeasurements (optional, indexed by eHandSide) supplies hardware
	// depth: when a hand's palm is sufficiently depth-resolved, the metric
	// measurements replace the monocular PnP/legacy estimate entirely.
	void process(TrackingFrameResult& ioResult,
				 const std::array<HandDepthMeasurement, 2>* depthMeasurements= nullptr);

	// Live hand-scale override (stereo auto-scale); does not touch filter state
	void setRefLengthMeters(float refLengthMeters)
	{
		if (refLengthMeters > 1e-4f)
			m_refLengthMeters= refLengthMeters;
	}
	float getRefLengthMeters() const { return m_refLengthMeters; }

	// Per-side rest angles for THIS camera, subtracted from every later
	// measurement so the captured pose reports zeros. Without them the raw
	// angles (relative to the flat-hand default) are reported as-is.
	void setRestAngles(eHandSide side, const std::array<FingerAngles, FINGER_COUNT>& restAngles)
	{
		m_restAngles[(int)side]= restAngles;
		m_bHasRestAngles[(int)side]= true;
	}
	void clearRestAngles()
	{
		m_bHasRestAngles[0]= false;
		m_bHasRestAngles[1]= false;
	}

private:
	glm::vec3 backProject(float u, float v, float z) const;
	void processHand(TrackedHand& hand);
	// Hardware-depth path: builds cameraPoints from measured metric points
	// (surface-to-joint offset applied), back-projecting unresolved landmarks
	// at their parent joint's depth. Returns false when the palm isn't
	// sufficiently resolved (caller falls back to PnP/legacy).
	bool processHandDepth(TrackedHand& hand, const HandDepthMeasurement& measurement);
	// PnP solve; returns false when it fails (the frame simply produces no
	// camera space - fusion's staleness window absorbs the gap)
	bool processHandPnp(TrackedHand& hand);
	// Fills the parametric HandPose (camera-space palm transform, finger
	// angles, skeleton) from a camera-space-tracked hand
	void fillHandPose(const TrackedHand& hand, HandPose& outPose);

	// Remembered palmar side, per hand. Separate memories per SPACE: the
	// stored normal is expressed in the point set's own frame, and camera
	// space and model space are unrelated.
	HandPoseModel::PalmarSideMemory m_cameraPalmarMemory[2];
	HandPoseModel::PalmarSideMemory m_modelPalmarMemory[2];
	// Mean pixel error between the hand's 2D landmarks and the FK hand
	// rebuilt from the extracted pose, projected back into the image
	float computeFkReprojectionError(const TrackedHand& hand, const HandPose& pose) const;

	bool m_bConfigured= false;
	float m_fx= 0.f;
	float m_fy= 0.f;
	float m_cx= 0.f;
	float m_cy= 0.f;
	float m_refLengthMeters= 0.08f;

	std::array<std::array<FingerAngles, FINGER_COUNT>, 2> m_restAngles{};
	bool m_bHasRestAngles[2]= {false, false};

	// Warm-start state for the iterative PnP solve (per side, axis-angle +
	// translation in OpenCV camera convention). Also the future vision
	// measurement for IMU (EKF) fusion: a rigid 6-DoF wrist pose per frame.
	std::array<std::array<double, 3>, 2> m_pnpRvec{};
	std::array<std::array<double, 3>, 2> m_pnpTvec{};
	bool m_bPnpPoseValid[2]= {false, false};

	bool m_bSideWasTracked[2]= {false, false};
};
