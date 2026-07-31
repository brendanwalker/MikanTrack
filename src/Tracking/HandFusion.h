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
		std::array<float, HAND_LANDMARK_COUNT> landmarkScore{};
	};

	void fuseSide(eHandSide side, std::vector<HandCandidate>& candidates, TrackedHand& outHand, TrackedArm& outArm);
	void applySmoothing(TrackingFrameResult& ioFused);

	HandFusionConfig m_config;

	// Post-fusion smoothing state
	HandOneEuroBank m_filterBank;
	std::array<OneEuroFilterVec3, 2> m_elbowWorldFilters;
	double m_lastTimestampMs= -1.0;
	bool m_bSideWasTracked[2]= {false, false};

	int m_dominantCamera[2]= {-1, -1};
};
