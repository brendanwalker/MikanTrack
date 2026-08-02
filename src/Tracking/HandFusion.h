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
	// world), as stored in ExtrinsicsConfig.
	glm::dmat4 markerFromCamera{1.0};
	// Undistorted pinhole intrinsics (the space TrackedHand::imagePoints live
	// in) - what landmark triangulation needs to turn pixels into world rays
	bool hasIntrinsics= false;
	float fx= 0.f;
	float fy= 0.f;
	float cx= 0.f;
	float cy= 0.f;
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

	// Split smoothing: palm transform (position + quaternion) vs finger
	// angles. Palm latency is user-visible, angle latency isn't.
	bool smoothingEnabled= true;
	float palmMinCutoff= 3.f;
	float palmBeta= 0.1f;
	float angleMinCutoff= 0.75f;
	float angleBeta= 0.02f;

	// Stereo landmark triangulation: when two cameras observe the same hand,
	// triangulate all 21 landmarks from the 2D image points and extract the
	// pose from the result - the network's (view-dependent, noisy) monocular
	// depth never enters. Falls back to the per-camera poses when off,
	// unavailable, or vetoed by the residual gate.
	bool triangulationEnabled= true;
	// A triangulated hand whose RMS reprojection residual exceeds this is a
	// wrong cross-camera pairing (two different physical hands) - reject it
	float triangulationMaxResidualPx= 25.f;
	// Residual at which a triangulated pose is half as trustworthy (folds
	// into the fused confidence, same shape as the jitter stability factor)
	float residualReferencePx= 8.f;

	// Legacy A/B switch for the NON-triangulated multi-camera path: true =
	// weighted blend of orientation + finger angles across cameras (the old
	// behavior). False (default) = select one source camera with hysteresis.
	// Blending estimators that disagree by tens of degrees with time-varying
	// weights manufactures low-frequency wander the smoothing can't remove;
	// picking the best single view does not. (Palm POSITION still blends -
	// positions compose, and the disagreement there is cm-scale.)
	bool blendArticulation= false;

	// Rest-pose zero for the TRIANGULATED path: raw stereo angles minus these
	// read zero in the user's captured rest pose. Separate from the
	// per-camera rest offsets (those fold in each camera's own model bias,
	// which triangulation does not have).
	std::array<std::array<FingerAngles, FINGER_COUNT>, 2> fusedRestAngles{};
	bool bHasFusedRestAngles[2]= {false, false};
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

		// Stereo triangulation outcome for this cluster's fused pose
		bool triangulated= false;      // pose came from landmark triangulation
		bool triVetoed= false;         // residual gate rejected the pairing
		float triResidualRmsPx= 0.f;   // RMS reprojection residual, both views
		float triResidualMaxPx= 0.f;   // worst single landmark
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

	// Rest-pose capture support: the RAW (pre rest-offset, pre smoothing)
	// triangulated finger angles from the last fuse() for this side; false
	// when that side wasn't stereo-triangulated. Fusing thread only.
	bool getLastRawTriangulatedAngles(eHandSide side,
									  std::array<FingerAngles, FINGER_COUNT>& outAngles) const
	{
		if (!m_bRawTriAnglesValid[(int)side])
			return false;
		outAngles= m_rawTriAngles[(int)side];
		return true;
	}

	// -- Pure scoring helpers (exposed for the --test-fusion self test) -----

	// Visibility factor in [0.05, 1.05]: how face-on the palm is to the camera
	static float visibilityFactor(const glm::quat& palmOrientationWorld, const glm::vec3& palmPositionWorld,
								  const glm::vec3& cameraPosWorld);

	// Stability factor in (0,1]: ref^2 / (ref^2 + jitter^2), a soft
	// inverse-variance weight over the measured palm jitter
	static float stabilityFactor(float jitterM, float jitterReferenceM);

	// Residual factor in (0,1]: same soft inverse-variance shape over the
	// triangulation reprojection residual - a direct measurement of how well
	// the stereo pose explains what both cameras actually saw
	static float residualFactor(float residualRmsPx, float residualReferencePx)
	{
		return stabilityFactor(residualRmsPx, residualReferencePx);
	}

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

		// Triangulation outcome (mirrored into FusionDiagnostics after fusing)
		bool triangulated= false;
		bool triVetoed= false;
		float triResidualRmsPx= 0.f;
		float triResidualMaxPx= 0.f;
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
	// Stereo landmark triangulation for a >=2-camera cluster. On success fills
	// outHand.worldPoints + the pose (palm frame, angles) from the
	// triangulated geometry and returns true; records the residual (and any
	// veto) on the cluster either way.
	bool triangulateCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, HandPose& outPose);
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

	// Raw triangulated angles of the last fuse (rest-pose capture source)
	std::array<std::array<FingerAngles, FINGER_COUNT>, 2> m_rawTriAngles{};
	bool m_bRawTriAnglesValid[2]= {false, false};

	// Articulation-source selection state (non-triangulated multi-camera
	// path): incumbent camera per side + challenger streak for hysteresis
	int m_articulationSource[2]= {-1, -1};
	int m_articulationChallenger[2]= {-1, -1};
	int m_articulationChallengerFrames[2]= {0, 0};

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
