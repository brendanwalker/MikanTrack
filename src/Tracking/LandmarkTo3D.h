#pragma once

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/ext/vector_float3.hpp"

#include "HandPoseModel.h"
#include "MikanVideoSourceTypes.h"
#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// Lifts image-space hand landmarks into camera-space meters using the
// calibrated intrinsics and the hand scale (wrist -> middle-MCP length,
// seeded by config and live-refined from stereo measurement).
//
// The 3D source is monocular PnP: a per-frame metric hand as a DYNAMIC
// solvePnP object model against the 2D landmarks - articulation is baked
// into the model, so PnP only recovers the global rigid pose. Warm-started
// from the previous frame. cameraPoints = R * obj + t. The object model is
// the user's CALIBRATED skeleton posed by the landmark model's angles when
// one has been measured, and otherwise the landmark model's own metric hand
// rescaled so wrist->middleMCP matches the hand scale. That distinction
// matters: the model's proportions are not the user's, and rescaling on one
// bone leaves an object with too little spread, which PnP can only fit by
// placing it too close.
//
// Conventions: input pixel coordinates are assumed to be in UNDISTORTED image
// space (the vision thread undistorts frames before inference), so the
// undistorted camera matrix is used. Camera space is the OpenCV convention:
// +X right, +Y down, +Z forward, meters.
//
// Also fills the parametric HandPose (palm transform + finger angles +
// skeleton) - see HandPoseModel.
class LandmarkTo3D
{
public:
	void configure(
		const MikanMonoIntrinsics& intrinsics,
		double refLengthMeters);

	// Fills cameraPoints/hasCameraSpace on the frame's hands and arms
	void process(TrackingFrameResult& ioResult);

	// Returns every cross-frame member to a freshly constructed state (palmar
	// side memories, PnP warm starts, tracked flags). Called when a tracking
	// recording starts so live matches replay's fresh instances bit-exactly.
	void resetTransientState()
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			m_cameraPalmarMemory[sideIndex].reset();
			m_modelPalmarMemory[sideIndex].reset();
			m_pnpRvec[sideIndex]= {};
			m_pnpTvec[sideIndex]= {};
			m_bPnpPoseValid[sideIndex]= false;
			m_bSideWasTracked[sideIndex]= false;
			m_bSideWasDetected[sideIndex]= false;
		}
	}

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

	// The user's measured hand geometry (HandBoneCalibrator). When set it
	// replaces the landmark model's proportions in both places they are used:
	// the PnP object model is rebuilt from it by forward kinematics, and the
	// reported skeleton is it. The model's ARTICULATION is still what drives
	// both - only the bone geometry changes hands.
	// The two object models are expressed in different frames (palm vs the
	// model's own), so the warm start is dropped whenever that changes under
	// the solver.
	void setCalibratedSkeleton(eHandSide side, const HandSkeleton& skeleton)
	{
		m_calibratedSkeleton[(int)side]= skeleton;
		m_bHasCalibratedSkeleton[(int)side]= true;
		m_bPnpPoseValid[(int)side]= false;
	}
	void clearCalibratedSkeleton()
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			m_bHasCalibratedSkeleton[sideIndex]= false;
			m_bPnpPoseValid[sideIndex]= false;
		}
	}
	bool hasCalibratedSkeleton(eHandSide side) const { return m_bHasCalibratedSkeleton[(int)side]; }

private:
	glm::vec3 backProject(float u, float v, float z) const;
	void processHand(TrackedHand& hand);
	// PnP solve; returns false when it fails (the frame simply produces no
	// camera space - fusion's staleness window absorbs the gap)
	bool processHandPnp(TrackedHand& hand);
	// The PnP object model, in the palm frame, rebuilt from the calibrated
	// skeleton and the landmark model's angles. False when that hand has no
	// calibration (caller falls back to the model's own metric hand).
	bool buildCalibratedObjectPoints(const TrackedHand& hand,
									 std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints);
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


	std::array<HandSkeleton, 2> m_calibratedSkeleton{};
	bool m_bHasCalibratedSkeleton[2]= {false, false};

	// Warm-start state for the iterative PnP solve (per side, axis-angle +
	// translation in OpenCV camera convention). Also the future vision
	// measurement for IMU (EKF) fusion: a rigid 6-DoF wrist pose per frame.
	std::array<std::array<double, 3>, 2> m_pnpRvec{};
	std::array<std::array<double, 3>, 2> m_pnpTvec{};
	bool m_bPnpPoseValid[2]= {false, false};

	// Camera-space success last frame (tracked AND solved) - gates the PnP
	// warm start on reacquisition
	bool m_bSideWasTracked[2]= {false, false};
	// 2D tracker had this side last frame - a tracked->untracked transition
	// resets the palmar side memories
	bool m_bSideWasDetected[2]= {false, false};
};
