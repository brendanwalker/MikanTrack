#pragma once

#include <array>
#include <map>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"

#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// One camera's latest processed tracking output, tagged for fusion.
// result is UNSMOOTHED (smoothing happens after fusion) and world-space
// (hands/poses have world data) when the camera has extrinsics.
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
	float wristMatchMaxDistM= 0.25f;
	float presenceThreshold= 0.5f;

	// A camera's observation is dropped entirely when its confidence
	// (presence x stability) falls below this. 0 = never drop, rely on the
	// soft weighting alone.
	float minCameraConfidence= 0.f;
	// Palm jitter (constant-velocity residual) at which an observation is
	// considered half as trustworthy. Larger = more tolerant of noise, but
	// also of genuinely fast motion, which produces real residuals.
	float jitterReferenceM= 0.015f;

	// Spatial side prior for users who never cross their hands: which world
	// axis (marker frame, origin = marker center) points toward where the
	// RIGHT hand lives. 0=off, 1=+X, 2=-X, 3=+Y, 4=-Y.
	int spatialSidePriorAxis= 0;

	bool smoothingEnabled= true;
	float smoothingMinCutoff= 1.f;
	float smoothingBeta= 0.05f;
};

// Introspection into the last fuse() call's clustering + side assignment,
// captured for diagnostic dumps (which cameras saw what, how observations
// clustered, and WHY each cluster got its side)
struct FusionDiagnostics
{
	struct Observation
	{
		int cameraIndex= -1;
		int labeledSide= -1; // that camera's own L/R vote (0=L, 1=R)
		float weight= 0.f;   // confidence x palm visibility (blend weight)
		float confidence= 0.f;
		float stability= 0.f;  // measured-jitter factor inside confidence
		float jitterMm= 0.f;   // the raw constant-velocity residual behind it
		float sideVoteWeight= 0.f;
		glm::vec3 palmWorld{0.f};
	};

	struct Cluster
	{
		std::vector<Observation> observations;
		glm::vec3 palmWorld{0.f};
		float bestWeight= 0.f;
		// Side-affinity components, indexed [side][0=vote,1=temporal,2=spatial]
		float affinity[2][3]{};
		int assignedSide= -1; // -1 = dropped (more clusters than hands)
	};

	int totalObservations= 0;
	std::vector<Cluster> clusters; // includes dropped clusters
};

// Fuses per-camera PARAMETRIC hand poses into one TrackingFrameResult:
// visibility-weighted blending of the palm transform (position average +
// quaternion blend) and the finger angles. Poses and angles compose across
// disagreeing observations - blending raw landmarks (the earlier approach)
// distorted bones whenever two cameras saw different articulation.
//
// Left/Right assignment happens HERE, not per camera: observations from all
// cameras are clustered by world palm proximity (one cluster = one physical
// hand), then clusters are assigned sides by weighted per-camera classifier
// votes plus temporal continuity of the fused tracks. A camera that sees only
// one hand routinely mislabels it (nothing to disambiguate against).
//
// A single fresh candidate passes through exactly (the N=1 identity path).
// Owns the post-fusion one-euro smoothing (position + quaternion components
// + 20 finger angles per side).
class HandFusion
{
public:
	void configure(const HandFusionConfig& config);

	// Fuses the candidates that are valid, fresh (within the staleness window
	// of nowTimestampMs) and world-tracked. outFused's poses are world-space;
	// its hands are the best camera's landmark data (advisory, for debug).
	void fuse(const std::vector<const CameraFrameResult*>& candidates,
			  double nowTimestampMs,
			  TrackingFrameResult& outFused);

	// Diagnostics: per-side index of the camera that dominated the last fuse
	// (-1 when the side wasn't tracked)
	int getDominantCamera(eHandSide side) const { return m_dominantCamera[(int)side]; }

	// Full clustering/side-assignment introspection for the last fuse()
	// (call from the fusing thread only)
	const FusionDiagnostics& getLastDiagnostics() const { return m_lastDiagnostics; }

	// Stereo hand-scale estimation: when two cameras observe the same physical
	// hand, the two view rays through its palm triangulate the true depth,
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

	// Visibility factor in [0.05, 1.05]: how face-on the palm is to the camera
	static float visibilityFactor(const glm::quat& palmOrientationWorld, const glm::vec3& palmPositionWorld,
								  const glm::vec3& cameraPosWorld);

	// Stability factor in (0,1]: ref^2 / (ref^2 + jitter^2), a soft
	// inverse-variance weight over the measured palm jitter
	static float stabilityFactor(float jitterM, float jitterReferenceM);

private:
	struct HandCandidate
	{
		const CameraFrameResult* camera= nullptr;
		const TrackedHand* hand= nullptr; // landmark data (votes, stereo scale, debug)
		const HandPose* pose= nullptr;    // the parametric observation being fused
		float confidence= 0.f;            // presence x measured stability
		float stability= 0.f;
		float jitterM= 0.f;
		float weight= 0.f;                // confidence x palm visibility (blend weight)
		float sideVoteWeight= 0.f;        // presence x classifier decisiveness
		// Classifier opinion in [-1 (left), +1 (right)], from the flip-adjusted
		// rightProb - NOT from the (displaceable) per-camera side label
		float signedVote= 0.f;
	};

	// One physical hand: all cameras' observations of it
	struct HandCluster
	{
		std::vector<HandCandidate> candidates;
		glm::vec3 palmWorld{0.f};        // best candidate's palm position (cluster anchor)
		glm::vec3 anchorCameraPos{0.f};  // that candidate's camera position (for ray matching)
		float anchorSignedVote= 0.f;
		float bestWeight= 0.f;
	};

	// Cost of merging an observation into a cluster (lateral-aware position
	// distance + handedness-vote coherence); >= kPairCostInf means "never"
	float pairCost(const HandCandidate& observation, const HandCluster& cluster) const;
	// Joint per-camera assignment of observations onto clusters (see .cpp)
	void clusterObservations(std::vector<HandCandidate>& observations, std::vector<HandCluster>& outClusters) const;

	struct AffinityBreakdown
	{
		float vote= 0.f;
		float temporal= 0.f;
		float spatial= 0.f;
		float total() const { return vote + temporal + spatial; }
	};

	// Affinity of a cluster for a side: classifier votes + temporal continuity
	// + optional spatial prior
	AffinityBreakdown sideAffinity(const HandCluster& cluster, eHandSide side) const;
	void updateStereoScale(const HandCluster& cluster);
	void fuseCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, HandPose& outPose);
	void applySmoothing(TrackingFrameResult& ioFused);

	HandFusionConfig m_config;

	// Post-fusion smoothing state (per side): palm position, palm quaternion
	// components (hemisphere-aligned before filtering), 20 finger angles
	std::array<OneEuroFilterVec3, 2> m_positionFilters;
	std::array<std::array<OneEuroFilter, 4>, 2> m_quaternionFilters;
	std::array<std::array<OneEuroFilter, FINGER_COUNT * 4>, 2> m_angleFilters;
	glm::quat m_lastFilteredQuat[2]= {glm::quat(1, 0, 0, 0), glm::quat(1, 0, 0, 0)};
	double m_lastTimestampMs= -1.0;
	bool m_bSideWasTracked[2]= {false, false};

	// Temporal side-assignment prior: last fused palm position per side
	glm::vec3 m_lastFusedPalm[2]= {glm::vec3(0.f), glm::vec3(0.f)};
	bool m_bLastFusedPalmValid[2]= {false, false};

	// Single-cluster hysteresis: while only one hand is tracked, its side
	// assignment sticks unless decisively contradicted (prevents the L/R
	// flip-flop on near-tied affinities that poisons the temporal prior)
	int m_lastSoloSide= -1;

	// Per (camera, that camera's own side slot) palm-jitter tracker. Keyed by
	// the camera's OWN label because that is what indexes its poses array; a
	// label swap just costs one reset. Jitter is the constant-velocity
	// residual |p(t) - 2p(t-1) + p(t-2)|, so smooth motion reads ~0 and only
	// genuine noise (or real acceleration) registers.
	struct JitterTracker
	{
		glm::vec3 previousPalm{0.f};
		glm::vec3 previousPalm2{0.f};
		int samples= 0;
		double lastTimestampMs= -1.0;
		float jitterEmaM= 0.f;
	};
	std::map<int, JitterTracker> m_jitterTrackers;

	// Updates the tracker for one observation and returns its jitter estimate
	float updateJitter(int cameraIndex, int cameraSideIndex, const glm::vec3& palmWorld, double timestampMs);

	int m_dominantCamera[2]= {-1, -1};

	float m_stereoScaleCorrection= 1.f;
	bool m_bStereoScaleFresh= false;

	FusionDiagnostics m_lastDiagnostics;
};
