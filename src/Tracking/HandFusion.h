#pragma once

#include <array>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"

#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// One camera's latest processed tracking output, tagged for fusion.
// result is UNSMOOTHED (smoothing happens after fusion) and world-space
// (hands/arms have hasWorldSpace) when the camera has extrinsics.
struct CameraFrameResult
{
	int cameraIndex= -1;
	bool valid= false;         // has produced at least one processed frame
	double timestampMs= 0.0;   // capture timestamp (shared steady_clock base)
	bool hasExtrinsics= false;
	// World-from-camera transform (OpenCV camera convention -> Z-up marker
	// world), as stored in ExtrinsicsConfig. Column 3 is the camera's world
	// position, which is all fusion needs from it.
	glm::dmat4 markerFromCamera{1.0};
	TrackingFrameResult result;
};

struct HandFusionConfig
{
	double stalenessWindowMs= 66.0;
	float softmaxTemperature= 8.f;
	float wristMatchMaxDistM= 0.25f;
	float presenceThreshold= 0.5f;

	bool smoothingEnabled= true;
	float smoothingMinCutoff= 1.f;
	float smoothingBeta= 0.05f;
};

// Fuses per-camera world-space hand results into one TrackingFrameResult:
// per-landmark softmax-weighted averaging, weighted by a per-camera
// visibility score (presence x how face-on the palm is to that camera).
// Validated approach from WannaKhrop/multicamera-hand-tracking.
//
// Left/Right assignment happens HERE, not per camera: observations from all
// cameras are clustered by world wrist proximity (one cluster = one physical
// hand), then clusters are assigned sides by weighted per-camera classifier
// votes plus temporal continuity of the fused tracks. A camera that sees only
// one hand routinely mislabels it (nothing to disambiguate against) - trusting
// per-camera labels collapses both physical hands into one fusion slot.
//
// A single fresh candidate passes through exactly (the N=1 identity path).
// Owns the post-fusion one-euro smoothing (per-camera LandmarkTo3D must run
// with smoothing disabled to avoid double-filtering).
class HandFusion
{
public:
	void configure(const HandFusionConfig& config);

	// Fuses the candidates that are valid, fresh (within the staleness window
	// of nowTimestampMs) and world-tracked. outFused is world-space only
	// (frameWidth/frameHeight are 0; image-space fields come from the best
	// camera and are advisory).
	void fuse(const std::vector<const CameraFrameResult*>& candidates,
			  double nowTimestampMs,
			  TrackingFrameResult& outFused);

	// Diagnostics: per-side index of the camera that dominated the last fuse
	// (-1 when the side wasn't tracked)
	int getDominantCamera(eHandSide side) const { return m_dominantCamera[(int)side]; }

	// Stereo hand-scale estimation: when two cameras observe the same physical
	// hand, the two view rays through its wrist triangulate the true depth,
	// which implies a correction factor for the configured hand scale (each
	// camera's depth estimate scales linearly with the assumed wrist->knuckle
	// length). Returns true and the correction from the last fuse when a valid
	// two-camera observation was available (factor ~1 = scale is correct,
	// <1 = configured hand scale is too large).
	bool getStereoScaleSample(float& outCorrectionFactor) const
	{
		outCorrectionFactor= m_stereoScaleCorrection;
		return m_bStereoScaleFresh;
	}

	// -- Pure scoring helpers (exposed for the --test-fusion self test) -----

	// Palm plane normal from the world landmarks (unnormalized direction;
	// sign is irrelevant for the |dot| visibility factor)
	static glm::vec3 palmNormalWorld(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& worldPoints);

	// Visibility factor in [0.05, 1.05]: |dot(palmNormal, viewRay)| measures
	// how face-on the palm is to the camera; the epsilon keeps edge-on views
	// from zeroing out entirely (they still see landmark positions, poorly)
	static float visibilityFactor(const glm::vec3& palmNormal, const glm::vec3& landmarkWorld,
								  const glm::vec3& cameraPosWorld);

private:
	struct HandCandidate
	{
		const CameraFrameResult* camera= nullptr;
		const TrackedHand* hand= nullptr;
		const TrackedArm* arm= nullptr;
		float baseScore= 0.f; // presence x mean visibility (candidate ranking)
		// per-camera classifier side vote weight (presence x label confidence)
		float sideVoteWeight= 0.f;
		std::array<float, HAND_LANDMARK_COUNT> landmarkScore{};
	};

	// One physical hand: all cameras' observations of it
	struct HandCluster
	{
		std::vector<HandCandidate> candidates;
		glm::vec3 wristWorld{0.f}; // best candidate's wrist (cluster anchor)
		float bestScore= 0.f;
	};

	// Affinity of a cluster for a side: classifier votes + temporal continuity
	float sideAffinity(const HandCluster& cluster, eHandSide side) const;
	void updateStereoScale(const HandCluster& cluster);
	void fuseCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, TrackedArm& outArm);
	void applySmoothing(TrackingFrameResult& ioFused);

	HandFusionConfig m_config;

	// Post-fusion smoothing state
	HandOneEuroBank m_filterBank;
	std::array<OneEuroFilterVec3, 2> m_elbowWorldFilters;
	double m_lastTimestampMs= -1.0;
	bool m_bSideWasTracked[2]= {false, false};

	// Temporal side-assignment prior: last fused wrist position per side
	glm::vec3 m_lastFusedWrist[2]= {glm::vec3(0.f), glm::vec3(0.f)};
	bool m_bLastFusedWristValid[2]= {false, false};

	int m_dominantCamera[2]= {-1, -1};

	float m_stereoScaleCorrection= 1.f;
	bool m_bStereoScaleFresh= false;
};
