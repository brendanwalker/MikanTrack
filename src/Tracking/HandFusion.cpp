#include "HandFusion.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"
#include "Logger.h"
#include "SpaceTransforms.h"

// Temporal side-continuity: full-strength attraction within this distance of
// the side's last fused palm, fading to nothing at 2x
static constexpr float kTemporalFullDistM= 0.15f;
// Weighting of temporal continuity vs classifier votes in side assignment
// (continuity dominates - a tracked hand must not swap sides because one
// camera's classifier flickered)
static constexpr float kTemporalWeight= 2.f;
// Stereo scale triangulation guards
static constexpr float kMinRayAngleCos= 0.985f; // rays closer than ~10 deg apart are degenerate
static constexpr float kScaleCorrectionMin= 0.7f;
static constexpr float kScaleCorrectionMax= 1.4f;
// Ray-aware clustering: max plausible depth error along a camera's view ray
// (depth scales with hand-scale/PnP error; lateral accuracy is much better)
static constexpr float kRayDepthSlackM= 0.5f;
// Spatial side prior: full strength this far along the chosen axis
static constexpr float kSpatialPriorFullDistM= 0.15f;
// Below temporal continuity (2.0) so a tracked hand never swaps sides mid-
// flight, but strong enough to beat a single camera's decisive mislabel
static constexpr float kSpatialPriorWeight= 1.f;
// Clustering pair costs (meter-equivalent units)
static constexpr float kPairCostInf= 1e6f;      // "never merge"
static constexpr float kVoteCoherenceCostM= 0.04f; // vote agreement discount / disagreement bump
static constexpr float kVoteVetoDecisiveness= 0.8f; // both this decisive + opposed -> veto merge
static constexpr float kVoteVetoMinDistM= 0.06f;    // ...unless practically the same point
// Single-cluster side stickiness (affinity units, on top of votes/temporal)
static constexpr float kSoloSideStickiness= 0.75f;
// Articulation-source hysteresis (non-triangulated multi-camera path): a
// challenger camera must beat the incumbent's weight by this margin for this
// many consecutive fuses before the orientation/angle source switches
static constexpr float kArticulationSwitchMargin= 1.2f;
static constexpr int kArticulationSwitchFrames= 5;
// Triangulated angle + palm-depth hold across brief mono fallbacks (see
// m_lastTriAngles / m_lastTriPalmWorld)
static constexpr double kTriAngleHoldMs= 300.0;
// Rescue-pool floor: a low-presence hand may still contribute usable 2D
// landmarks as the second triangulation view (the residual is the judge),
// but below this even the image points are guesses
static constexpr float kTriRescueMinPresence= 0.15f;
// Side-assignment refusal: a cluster whose winning affinity total is negative
// is being assigned by ELIMINATION against the evidence - measured across
// three recordings, healthy assignments never scored below +2.05 (p01) while
// every negative-affinity assignment was a mislabeled or phantom cluster that
// stepped the fused hand 256-782 mm. Refusal only applies while the side was
// RECENTLY tracked: after a real dropout the temporal prior is stale
// testimony (the hand may genuinely be anywhere), so reacquisition keeps
// today's behavior and a refusal deadlock is structurally impossible.
static constexpr float kAssignmentRefuseThreshold= 0.f;
static constexpr double kAssignmentRefuseWindowMs= 250.0;

void HandFusion::configure(const HandFusionConfig& config)
{
	m_config= config;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_positionFilters[sideIndex].configure(config.palmMinCutoff, config.palmBeta, 1.f);
		m_positionFilters[sideIndex].reset();
		for (OneEuroFilter& filter : m_quaternionFilters[sideIndex])
		{
			filter.configure(config.palmMinCutoff, config.palmBeta, 1.f);
			filter.reset();
		}
		for (OneEuroFilter& filter : m_angleFilters[sideIndex])
		{
			filter.configure(config.angleMinCutoff, config.angleBeta, 1.f);
			filter.reset();
		}
		m_lastFilteredQuat[sideIndex]= glm::quat(1, 0, 0, 0);
		m_bSideWasTracked[sideIndex]= false;
		m_bLastFusedPalmValid[sideIndex]= false;
		m_lastTrackedMs[sideIndex]= -1e12;
		m_articulationSource[sideIndex]= -1;
		m_articulationChallenger[sideIndex]= -1;
		m_articulationChallengerFrames[sideIndex]= 0;
		m_lastTriTimestampMs[sideIndex]= -1e12;
	}
	m_lastTimestampMs= -1.0;
	m_lastSoloSide= -1;
	m_jitterTrackers.clear();
	m_stereoScaleCorrection= 1.f;
	m_bStereoScaleFresh= false;

	m_estimator.configure(config.estimator);
}

void HandFusion::resetTransientState()
{
	configure(m_config);

	// configure() resets the valid/initialized flags but leaves these VALUES
	// behind. Replay runs on freshly constructed (zeroed) instances, so live
	// must zero them too - a flag-guarded stale value that never influences
	// output is still fine, but a stale palmar normal or held tri angle set
	// would diverge the first frames after a hand reacquisition.
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_triPalmarMemory[sideIndex].reset();
		m_rawTriAngles[sideIndex]= {};
		m_bRawTriAnglesValid[sideIndex]= false;
		m_lastTriAngles[sideIndex]= {};
		m_lastTriSkeleton[sideIndex]= HandSkeleton();
		m_lastTriPalmWorld[sideIndex]= glm::vec3(0.f);
		m_lastFusedPalm[sideIndex]= glm::vec3(0.f);
		m_dominantCamera[sideIndex]= -1;
		m_estimatorSkeleton[sideIndex]= HandSkeleton();
		m_bEstimatorSkeletonValid[sideIndex]= false;
		m_bEstimatorProducedPose[sideIndex]= false;
		m_estimatorJitterM[sideIndex]= 0.f;
	}
	m_estimator.resetAll();
	m_fuseTimestampMs= 0.0;
	m_lastDiagnostics= FusionDiagnostics();
}

float HandFusion::stabilityFactor(float jitterM, float jitterReferenceM)
{
	// Soft inverse-variance weight: 1 at zero jitter, 0.5 at the reference,
	// falling off quadratically beyond it
	const float reference= std::max(jitterReferenceM, 1e-4f);
	const float ratio= jitterM / reference;
	return 1.f / (1.f + ratio * ratio);
}

float HandFusion::updateJitter(int cameraIndex, int cameraSideIndex, const glm::vec3& palmWorld,
							   double timestampMs)
{
	JitterTracker& tracker= m_jitterTrackers[cameraIndex * 2 + cameraSideIndex];

	// The same camera result is re-fused whenever ANOTHER camera delivers a
	// frame; only advance the history on genuinely new samples (a repeated
	// position would otherwise read as perfect stability)
	if (timestampMs == tracker.lastTimestampMs)
		return tracker.jitterEmaM;

	// Reacquisition after a gap: the old history says nothing about the new
	// track, and the position jump would register as enormous jitter
	const bool bStale= tracker.lastTimestampMs >= 0.0 &&
		(timestampMs - tracker.lastTimestampMs) > 4.0 * m_config.stalenessWindowMs;
	if (bStale)
		tracker.samples= 0;

	if (tracker.samples >= 2)
	{
		// Constant-velocity residual: zero for smooth motion of any speed,
		// large for frame-to-frame noise
		const glm::vec3 residual= palmWorld - 2.f * tracker.previousPalm + tracker.previousPalm2;
		const float jitter= glm::length(residual);

		constexpr float kJitterEmaAlpha= 0.15f;
		tracker.jitterEmaM= tracker.samples > 2
			? tracker.jitterEmaM * (1.f - kJitterEmaAlpha) + jitter * kJitterEmaAlpha
			: jitter;
	}

	tracker.previousPalm2= tracker.previousPalm;
	tracker.previousPalm= palmWorld;
	tracker.lastTimestampMs= timestampMs;
	tracker.samples= std::min(tracker.samples + 1, 3);

	// Until there is enough history, assume the observation is good (a fresh
	// track must not be penalized into invisibility)
	return tracker.samples >= 3 ? tracker.jitterEmaM : 0.f;
}

float HandFusion::visibilityFactor(const glm::quat& palmOrientationWorld, const glm::vec3& palmPositionWorld,
								   const glm::vec3& cameraPosWorld)
{
	// Palm +Z is the palmar normal (see HandPose); face-on = normal aligned
	// with the view ray. |cos|: the back of the hand is as informative for
	// pose/angles as the palm - edge-on (the clap case) is the failure mode.
	const glm::vec3 palmNormal= palmOrientationWorld * glm::vec3(0.f, 0.f, 1.f);
	const glm::vec3 viewRay= palmPositionWorld - cameraPosWorld;
	const float viewLength= glm::length(viewRay);
	if (viewLength < 1e-6f)
		return 0.05f;

	return 0.05f + fabsf(glm::dot(palmNormal, viewRay / viewLength));
}

// Lateral-aware distance between two observations of (possibly) the same
// physical hand. Depth along the observing camera's view ray is the noisy
// dimension (it scales linearly with hand-scale / PnP error), so when one
// point lies close to the view ray through the other (same image position,
// different depth), the perpendicular ray distance replaces the euclidean
// one. The ray substitution uses a tighter gate so hands genuinely lined up
// along a ray (one behind the other) keep their full distance.
static float lateralAwareDistance(const glm::vec3& palmA, const glm::vec3& cameraPosA,
								  const glm::vec3& palmB, const glm::vec3& cameraPosB,
								  float maxDistM)
{
	float dist= glm::length(palmA - palmB);

	const float rayGateM= maxDistM * 0.5f;
	auto viewRayLateral= [rayGateM](const glm::vec3& point, const glm::vec3& rayOrigin,
									const glm::vec3& through) -> float {
		glm::vec3 dir= through - rayOrigin;
		const float observedDepth= glm::length(dir);
		if (observedDepth < 1e-4f)
			return kPairCostInf;
		dir/= observedDepth;

		const glm::vec3 toPoint= point - rayOrigin;
		const float along= glm::dot(toPoint, dir);
		if (along <= 0.f || fabsf(along - observedDepth) > kRayDepthSlackM)
			return kPairCostInf;

		const float lateral= glm::length(toPoint - dir * along);
		return lateral <= rayGateM ? lateral : kPairCostInf;
	};

	dist= std::min(dist, viewRayLateral(palmB, cameraPosA, palmA));
	dist= std::min(dist, viewRayLateral(palmA, cameraPosB, palmB));
	return dist;
}

float HandFusion::pairCost(const HandCandidate& observation, const HandCluster& cluster) const
{
	const glm::vec3& palm= observation.pose->palmPositionWorld;
	const glm::vec3 cameraPos= cameraPositionWorld(observation.camera->markerFromCamera);

	const float euclid= glm::length(palm - cluster.palmWorld);
	const float dist=
		lateralAwareDistance(palm, cameraPos, cluster.palmWorld, cluster.anchorCameraPos, m_config.wristMatchMaxDistM);
	if (dist > m_config.wristMatchMaxDistM)
		return kPairCostInf;

	// Handedness-vote coherence. Two observations whose classifiers DECISIVELY
	// disagree are almost never the same physical hand - when hands are close
	// together, position cannot tell them apart (depth noise is comparable to
	// the hand separation; confirmed live in the 2026-08-01 clap dump where
	// each camera correctly tracked a DIFFERENT hand and position-only
	// clustering kept merging them, cancelling both votes).
	const float voteProduct= observation.signedVote * cluster.anchorSignedVote;
	if (voteProduct < 0.f &&
		fabsf(observation.signedVote) > kVoteVetoDecisiveness &&
		fabsf(cluster.anchorSignedVote) > kVoteVetoDecisiveness &&
		euclid > kVoteVetoMinDistM)
	{
		return kPairCostInf;
	}

	return std::max(dist - kVoteCoherenceCostM * voteProduct, 0.f);
}

// Groups observations into clusters (one cluster = one physical hand).
//
// Correspondence when hands are close together CANNOT be decided greedily by
// nearest-position: depth noise along each camera's view ray is comparable to
// the hand separation, so an anchor's nearest cross-camera neighbor is
// routinely the WRONG hand (seen live: 2.6cm to the wrong hand vs 5.7cm to
// the right one). Instead, observations are grouped per camera and each
// camera's observations are assigned onto the existing clusters JOINTLY -
// every injective mapping (at most 2 observations per camera) is scored with
// pairCost and the cheapest total wins.
void HandFusion::clusterObservations(std::vector<HandCandidate>& observations,
									 std::vector<HandCluster>& outClusters) const
{
	outClusters.clear();

	// Group per camera, best camera (by its strongest observation) first
	struct CameraGroup
	{
		float bestWeight= 0.f;
		std::vector<HandCandidate> observations;
	};
	std::vector<CameraGroup> groups;
	std::sort(observations.begin(), observations.end(),
			  [](const HandCandidate& a, const HandCandidate& b) { return a.weight > b.weight; });
	for (const HandCandidate& observation : observations)
	{
		CameraGroup* group= nullptr;
		for (CameraGroup& existing : groups)
		{
			if (!existing.observations.empty() &&
				existing.observations[0].camera->cameraIndex == observation.camera->cameraIndex)
				group= &existing;
		}
		if (group == nullptr)
		{
			groups.emplace_back();
			group= &groups.back();
		}
		group->observations.push_back(observation);
		group->bestWeight= std::max(group->bestWeight, observation.weight);
	}
	std::sort(groups.begin(), groups.end(),
			  [](const CameraGroup& a, const CameraGroup& b) { return a.bestWeight > b.bestWeight; });

	auto appendToCluster= [](HandCluster& cluster, const HandCandidate& observation) {
		cluster.candidates.push_back(observation);
		if (observation.weight > cluster.bestWeight)
		{
			cluster.palmWorld= observation.pose->palmPositionWorld;
			cluster.anchorCameraPos= cameraPositionWorld(observation.camera->markerFromCamera);
			cluster.anchorSignedVote= observation.signedVote;
			cluster.bestWeight= observation.weight;
		}
	};
	auto newCluster= [&outClusters](const HandCandidate& observation) {
		HandCluster cluster;
		cluster.candidates.push_back(observation);
		cluster.palmWorld= observation.pose->palmPositionWorld;
		cluster.anchorCameraPos= cameraPositionWorld(observation.camera->markerFromCamera);
		cluster.anchorSignedVote= observation.signedVote;
		cluster.bestWeight= observation.weight;
		outClusters.push_back(cluster);
	};

	// Cost of NOT merging: keeps single-observation semantics identical to the
	// old per-observation gate (merge exactly when cost < gate)
	const float newClusterCost= m_config.wristMatchMaxDistM;

	for (const CameraGroup& group : groups)
	{
		if (outClusters.empty())
		{
			for (const HandCandidate& observation : group.observations)
				newCluster(observation);
			continue;
		}

		// Enumerate injective mappings observation -> cluster index (or -1 for
		// a new cluster); <=2 observations x few clusters = tiny search
		const int clusterCount= (int)outClusters.size();
		if (group.observations.size() == 1)
		{
			const HandCandidate& observation= group.observations[0];
			int bestTarget= -1;
			float bestCost= newClusterCost;
			for (int target= 0; target < clusterCount; ++target)
			{
				const float cost= pairCost(observation, outClusters[target]);
				if (cost < bestCost)
				{
					bestCost= cost;
					bestTarget= target;
				}
			}
			if (bestTarget >= 0)
				appendToCluster(outClusters[bestTarget], observation);
			else
				newCluster(observation);
		}
		else
		{
			// Two observations from one camera are two DIFFERENT physical
			// hands - they may never share a cluster
			int bestTarget0= -1, bestTarget1= -1;
			float bestCost= 2.f * newClusterCost;
			for (int target0= -1; target0 < clusterCount; ++target0)
			{
				for (int target1= -1; target1 < clusterCount; ++target1)
				{
					if (target0 == target1 && target0 != -1)
						continue;

					const float cost0=
						target0 < 0 ? newClusterCost : pairCost(group.observations[0], outClusters[target0]);
					const float cost1=
						target1 < 0 ? newClusterCost : pairCost(group.observations[1], outClusters[target1]);
					if (cost0 + cost1 < bestCost)
					{
						bestCost= cost0 + cost1;
						bestTarget0= target0;
						bestTarget1= target1;
					}
				}
			}
			if (bestTarget0 >= 0)
				appendToCluster(outClusters[bestTarget0], group.observations[0]);
			else
				newCluster(group.observations[0]);
			if (bestTarget1 >= 0)
				appendToCluster(outClusters[bestTarget1], group.observations[1]);
			else
				newCluster(group.observations[1]);
		}
	}
}

HandFusion::AffinityBreakdown HandFusion::sideAffinity(const HandCluster& cluster, eHandSide side) const
{
	AffinityBreakdown affinity;

	// Classifier votes: each camera's flip-adjusted handedness score, weighted
	// by presence and decisiveness (the per-camera side LABEL is not evidence
	// - slot bookkeeping displaces it)
	for (const HandCandidate& candidate : cluster.candidates)
	{
		const eHandSide votedSide= candidate.signedVote >= 0.f ? eHandSide::Right : eHandSide::Left;
		affinity.vote+= candidate.sideVoteWeight * (votedSide == side ? 1.f : -1.f);
	}

	// Temporal continuity: attraction to where this side's fused hand was last
	if (m_bLastFusedPalmValid[(int)side])
	{
		const float dist= glm::length(cluster.palmWorld - m_lastFusedPalm[(int)side]);
		affinity.temporal= kTemporalWeight *
			std::clamp(1.f - (dist - kTemporalFullDistM) / kTemporalFullDistM, -1.f, 1.f);
	}

	// Spatial prior: hands that never cross stay on their own side of the
	// board, so position across the world's lateral axis is side evidence.
	// The axis is a FIXED CONVENTION, not a setting: the calibration board is
	// printed with FORWARD/RIGHT labels and defines the world frame, so the
	// right hand is always toward -Y (see k_worldRightAxis).
	{
		const float along= glm::dot(cluster.palmWorld, k_worldRightAxis);
		const float prior= std::clamp(along / kSpatialPriorFullDistM, -1.f, 1.f);
		affinity.spatial= kSpatialPriorWeight * (side == eHandSide::Right ? prior : -prior);
	}

	return affinity;
}

void HandFusion::updateStereoScale(const HandCluster& cluster)
{
	// Need two observations of the same palm from two different cameras
	if (cluster.candidates.size() < 2)
		return;

	const HandCandidate& a= cluster.candidates[0];
	const HandCandidate& b= cluster.candidates[1];
	if (a.camera->cameraIndex == b.camera->cameraIndex)
		return;

	const glm::vec3 cameraPosA= cameraPositionWorld(a.camera->markerFromCamera);
	const glm::vec3 cameraPosB= cameraPositionWorld(b.camera->markerFromCamera);
	const glm::vec3 palmA= a.pose->palmPositionWorld;
	const glm::vec3 palmB= b.pose->palmPositionWorld;

	// Each camera's palm estimate lies on the view ray through the true palm;
	// the estimated depth scales linearly with the configured hand scale, but
	// the ray DIRECTION doesn't. Two-ray midpoint triangulation recovers the
	// true depth independent of the configured scale.
	const float depthA= glm::length(palmA - cameraPosA);
	const float depthB= glm::length(palmB - cameraPosB);
	if (depthA < 1e-4f || depthB < 1e-4f)
		return;

	const glm::vec3 rayA= (palmA - cameraPosA) / depthA;
	const glm::vec3 rayB= (palmB - cameraPosB) / depthB;

	// Degenerate when the rays are nearly parallel (cameras too close together)
	if (fabsf(glm::dot(rayA, rayB)) > kMinRayAngleCos)
		return;

	// Closest-point parameters along the two rays (standard midpoint method)
	const glm::vec3 baseline= cameraPosB - cameraPosA;
	const float dotAB= glm::dot(rayA, rayB);
	const float denominator= 1.f - dotAB * dotAB;
	if (denominator < 1e-6f)
		return;

	const float tA= (glm::dot(rayA, baseline) - dotAB * glm::dot(rayB, baseline)) / denominator;
	const float tB= (dotAB * glm::dot(rayA, baseline) - glm::dot(rayB, baseline)) / denominator;
	if (tA <= 0.f || tB <= 0.f)
		return;

	// True-depth / estimated-depth per camera; average the two
	const float correction= 0.5f * (tA / depthA + tB / depthB);
	if (!std::isfinite(correction))
		return;

	m_stereoScaleCorrection= std::clamp(correction, kScaleCorrectionMin, kScaleCorrectionMax);
	m_bStereoScaleFresh= true;
}

// Stereo landmark triangulation. Uses the best two observations from
// DIFFERENT cameras: each 2D landmark back-projects to a world ray through
// its camera's optical center, and the two rays' closest-point midpoint is
// the world landmark. The network's monocular depth (modelPoints z / the PnP
// translation) is never consulted - which is the point: that depth is the
// view-dependent, noisy part of the per-camera estimate (measured live:
// 15-25cm error along the view ray, 25-41 deg articulation disagreement).
//
// The reprojection residual doubles as a correspondence test: two DIFFERENT
// physical hands wrongly merged into one cluster triangulate to points that
// project nowhere near the observed pixels, so a large RMS vetoes the pair.
bool HandFusion::triangulatePairPoints(const HandCandidate& obsA, const HandCandidate& obsB,
									   std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints,
									   float& outResidualRmsPx, float& outResidualMaxPx)
{
	struct View
	{
		const HandCandidate* obs;
		glm::vec3 cameraPos;
		glm::dmat4 cameraFromWorld;
	};
	std::array<View, 2> views;
	for (int v= 0; v < 2; ++v)
	{
		const HandCandidate* obs= v == 0 ? &obsA : &obsB;
		views[v].obs= obs;
		views[v].cameraPos= cameraPositionWorld(obs->camera->markerFromCamera);
		views[v].cameraFromWorld= glm::inverse(obs->camera->markerFromCamera);
	}

	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		glm::vec3 rayDir[2];
		for (int v= 0; v < 2; ++v)
		{
			const CameraFrameResult* camera= views[v].obs->camera;
			const glm::vec3& px= views[v].obs->hand->imagePoints[i];
			rayDir[v]= pixelRayDirWorld(
				camera->markerFromCamera,
				camera->fx, camera->fy, camera->cx, camera->cy,
				glm::vec2(px));
		}

		// Closest-point parameters along the two rays (midpoint method)
		const glm::vec3 baseline= views[1].cameraPos - views[0].cameraPos;
		const float dotAB= glm::dot(rayDir[0], rayDir[1]);
		const float denominator= 1.f - dotAB * dotAB;
		if (denominator < 1e-6f || fabsf(dotAB) > kMinRayAngleCos)
			return false; // near-parallel rays: no depth information

		const float tA= (glm::dot(rayDir[0], baseline) - dotAB * glm::dot(rayDir[1], baseline)) / denominator;
		const float tB= (dotAB * glm::dot(rayDir[0], baseline) - glm::dot(rayDir[1], baseline)) / denominator;
		if (tA <= 0.f || tB <= 0.f)
			return false; // intersection behind a camera: bogus correspondence

		outPoints[i]= 0.5f * (views[0].cameraPos + rayDir[0] * tA + views[1].cameraPos + rayDir[1] * tB);
	}

	// Reprojection residual across both views - the correspondence test
	float residualSquaredSum= 0.f;
	float residualMax= 0.f;
	for (const View& view : views)
	{
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			const glm::dvec4 camPoint= view.cameraFromWorld * glm::dvec4(glm::dvec3(outPoints[i]), 1.0);
			if (camPoint.z < 1e-3)
				return false;

			const CameraFrameResult* camera= view.obs->camera;
			const glm::vec2 projected(
				(float)(camera->fx * camPoint.x / camPoint.z + camera->cx),
				(float)(camera->fy * camPoint.y / camPoint.z + camera->cy));
			const float residual= glm::length(projected - glm::vec2(view.obs->hand->imagePoints[i]));
			residualSquaredSum+= residual * residual;
			residualMax= std::max(residualMax, residual);
		}
	}
	outResidualRmsPx= sqrtf(residualSquaredSum / (2.f * HAND_LANDMARK_COUNT));
	outResidualMaxPx= residualMax;
	return true;
}

void HandFusion::rescueSoloClusters(std::vector<HandCandidate>& rescuePool,
									std::vector<HandCluster>& clusters) const
{
	if (!m_config.triangulationEnabled)
		return;

	for (size_t clusterIndex= 0; clusterIndex < clusters.size(); ++clusterIndex)
	{
		HandCluster& cluster= clusters[clusterIndex];
		if (cluster.candidates.size() != 1)
			continue;
		const HandCandidate& anchor= cluster.candidates[0];
		if (!anchor.camera->hasIntrinsics || !anchor.hand->tracked)
			continue;

		// Candidate partners: observations stranded in OTHER solo clusters
		// (mono position too far off to merge) plus the below-presence pool.
		// Decisively-opposed handedness votes are never paired; the
		// reprojection residual makes the final call either way.
		auto voteCompatible= [&](const HandCandidate& partner) {
			return !(partner.signedVote * anchor.signedVote < 0.f &&
					 fabsf(partner.signedVote) > kVoteVetoDecisiveness &&
					 fabsf(anchor.signedVote) > kVoteVetoDecisiveness);
		};
		auto usable= [&](const HandCandidate& partner) {
			return partner.camera->cameraIndex != anchor.camera->cameraIndex &&
				partner.camera->hasIntrinsics && partner.hand->tracked && voteCompatible(partner);
		};

		float bestRms= m_config.triangulationMaxResidualPx;
		size_t bestCluster= SIZE_MAX;
		int bestPool= -1;

		std::array<glm::vec3, HAND_LANDMARK_COUNT> probePoints;
		float rms= 0.f, maxPx= 0.f;
		for (size_t otherIndex= 0; otherIndex < clusters.size(); ++otherIndex)
		{
			if (otherIndex == clusterIndex || clusters[otherIndex].candidates.size() != 1)
				continue;
			const HandCandidate& partner= clusters[otherIndex].candidates[0];
			if (!usable(partner))
				continue;
			if (triangulatePairPoints(anchor, partner, probePoints, rms, maxPx) && rms < bestRms)
			{
				bestRms= rms;
				bestCluster= otherIndex;
				bestPool= -1;
			}
		}
		for (int poolIndex= 0; poolIndex < (int)rescuePool.size(); ++poolIndex)
		{
			const HandCandidate& partner= rescuePool[poolIndex];
			if (!usable(partner))
				continue;
			if (triangulatePairPoints(anchor, partner, probePoints, rms, maxPx) && rms < bestRms)
			{
				bestRms= rms;
				bestCluster= SIZE_MAX;
				bestPool= poolIndex;
			}
		}

		if (bestCluster != SIZE_MAX)
		{
			HandCluster& donor= clusters[bestCluster];
			cluster.candidates.push_back(donor.candidates[0]);
			clusters.erase(clusters.begin() + bestCluster);
			if (bestCluster < clusterIndex)
				--clusterIndex; // erased before us: our index shifted down
		}
		else if (bestPool >= 0)
		{
			cluster.candidates.push_back(rescuePool[bestPool]);
			rescuePool.erase(rescuePool.begin() + bestPool);
		}
		// anchor/bestWeight stay with the original observation - the rescued
		// partner's mono pose is exactly the data we don't trust
	}
}

bool HandFusion::triangulateCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand,
									HandPose& outPose)
{
	// Score the PAIR, not the two cameras. Depth error from a stereo pair goes
	// as 1/sin(parallax), so two well-scored cameras that happen to sit close
	// together reconstruct worse than a lesser pair with a wide baseline. That
	// error lands on the small out-of-plane finger geometry that decides the
	// palmar side, so a short-baseline pair does not just add noise - it flips
	// the palm frame 180 degrees whenever the fingers straighten.
	//
	// The per-camera term here is presence alone. The blend weight is the
	// wrong ranking for this: it folds in monocular-depth stability, which is
	// exactly the noise triangulation discards, and palm visibility, which
	// works BACKWARDS for a pair. A camera seeing the palm edge-on is the one
	// resolving the depth its face-on partner cannot, so multiplying two
	// face-on scores selects for redundant viewpoints. Measured on recording
	// 2026-08-14_17-23-42: weighting by visibility keeps picking the 46-degree
	// pair (12.4 mm reconstruction error, palm frame flipping) over the
	// 73-degree one (7.3 mm, never flips).
	struct PairView
	{
		const HandCandidate* candidate;
		glm::vec3 dirToHand;
	};
	std::vector<PairView> views;
	views.reserve(cluster.candidates.size());
	for (const HandCandidate& candidate : cluster.candidates)
	{
		if (!candidate.camera->hasIntrinsics || !candidate.hand->tracked)
			continue;
		const glm::vec3 toHand= cluster.palmWorld - cameraPositionWorld(candidate.camera->markerFromCamera);
		if (glm::dot(toHand, toHand) < 1e-8f)
			continue;
		views.push_back({&candidate, glm::normalize(toHand)});
	}

	const HandCandidate* obsA= nullptr;
	const HandCandidate* obsB= nullptr;
	float bestPairScore= 0.f;
	float bestParallaxCos= 1.f;
	for (size_t i= 0; i < views.size(); ++i)
	{
		for (size_t j= i + 1; j < views.size(); ++j)
		{
			if (views[i].candidate->camera->cameraIndex == views[j].candidate->camera->cameraIndex)
				continue;

			// sin of the subtended angle: penalizes near-parallel AND
			// near-antiparallel views, both of which lose depth
			const float parallaxSin= glm::length(glm::cross(views[i].dirToHand, views[j].dirToHand));
			const float score=
				views[i].candidate->pose->presence * views[j].candidate->pose->presence * parallaxSin;
			if (score > bestPairScore)
			{
				bestPairScore= score;
				bestParallaxCos= glm::dot(views[i].dirToHand, views[j].dirToHand);
				obsA= views[i].candidate;
				obsB= views[j].candidate;
			}
		}
	}
	if (obsA == nullptr || obsB == nullptr)
		return false;

	// obsA is the better-viewed of the two: downstream (hand scale, skeleton)
	// asks for "the best candidate", which used to be true by construction
	if (obsB->weight > obsA->weight)
		std::swap(obsA, obsB);

	cluster.triCameraA= obsA->camera->cameraIndex;
	cluster.triCameraB= obsB->camera->cameraIndex;
	cluster.triParallaxDeg= glm::degrees(acosf(std::clamp(bestParallaxCos, -1.f, 1.f)));

	std::array<glm::vec3, HAND_LANDMARK_COUNT> triPoints;
	float residualRms= 0.f;
	float residualMax= 0.f;
	if (!triangulatePairPoints(*obsA, *obsB, triPoints, residualRms, residualMax))
		return false;
	cluster.triResidualRmsPx= residualRms;
	cluster.triResidualMaxPx= residualMax;

	if (residualRms > m_config.triangulationMaxResidualPx)
	{
		cluster.triVetoed= true;
		return false;
	}

	// Pose from the triangulated geometry. The skeleton stays the best
	// candidate's (metric via the calibrated hand scale, and the wire
	// contract wants a stable skeleton) - only the palm frame and the
	// angles come from the stereo landmarks.
	const glm::mat4 palmFrame=
		HandPoseModel::computePalmFrame(triPoints, side, &m_triPalmarMemory[(int)side]);
	outPose.palmPositionWorld= glm::vec3(palmFrame[3]);
	outPose.palmOrientationWorld= glm::quat_cast(glm::mat3(palmFrame));
	outPose.hasWorldPose= true;

	std::array<FingerAngles, FINGER_COUNT> rawAngles{};
	HandPoseModel::computeFingerAngles(triPoints, side, outPose.skeleton.neutralDirInPalm, rawAngles,
									   &m_triPalmarMemory[(int)side]);
	m_rawTriAngles[(int)side]= rawAngles;
	m_bRawTriAnglesValid[(int)side]= true;

	// Fused rest offset (captured from a triangulated rest pose): NOT the
	// per-camera offsets - those correct each camera's own model bias, which
	// stereo geometry doesn't have
	if (m_config.bHasFusedRestAngles[(int)side])
	{
		const std::array<FingerAngles, FINGER_COUNT>& rest= m_config.fusedRestAngles[(int)side];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			rawAngles[finger].lateral-= rest[finger].lateral;
			rawAngles[finger].proximal-= rest[finger].proximal;
			rawAngles[finger].intermediate-= rest[finger].intermediate;
			rawAngles[finger].distal-= rest[finger].distal;
		}
	}
	outPose.fingers= rawAngles;

	float maxPresence= 0.f;
	for (const HandCandidate& candidate : cluster.candidates)
		maxPresence= std::max(maxPresence, candidate.pose->presence);
	outPose.presence= maxPresence;
	outPose.stereoTriangulated= true;

	// Triangulated confidence = presence x TRI-palm stability x residual
	// factor. The per-camera stability (mono-depth jitter) is deliberately
	// NOT in here: triangulation eliminated exactly that noise, and the
	// 19-41-32 dump showed it dragging confidence to 0.33 on solid stereo
	// frames because one camera's mono depth was wild. The residual measures
	// this frame's quality; the tri-palm jitter tracker measures the actual
	// output noise over time.
	const double pairTimestampMs=
		std::max(obsA->camera->timestampMs, obsB->camera->timestampMs);
	const float triJitterM=
		updateJitter(-1, (int)side, outPose.palmPositionWorld, pairTimestampMs);
	const float triStability= stabilityFactor(triJitterM, m_config.jitterReferenceM);
	outPose.confidence= std::clamp(
		maxPresence * triStability * residualFactor(residualRms, m_config.residualReferencePx), 0.f, 1.f);

	// Overlays/debug see the triangulated geometry
	outHand.worldPoints= triPoints;
	outHand.hasWorldSpace= true;

	// The triangulated wrist->middle-MCP bone IS the true hand scale; the
	// best candidate's world bone carries the currently-assumed scale (the
	// PnP object model is rescaled to it), so the ratio is the correction
	if (outHand.tracked)
	{
		const glm::vec3& wrist= triPoints[(int)eHandLandmark::WRIST];
		const glm::vec3& middleMcp= triPoints[(int)eHandLandmark::MIDDLE_MCP];
		const float triBone= glm::length(middleMcp - wrist);

		const TrackedHand* bestHand= obsA->weight >= obsB->weight ? obsA->hand : obsB->hand;
		if (bestHand->hasWorldSpace)
		{
			const float assumedBone= glm::length(
				bestHand->worldPoints[(int)eHandLandmark::MIDDLE_MCP] -
				bestHand->worldPoints[(int)eHandLandmark::WRIST]);
			if (assumedBone > 1e-4f && std::isfinite(triBone))
			{
				m_stereoScaleCorrection=
					std::clamp(triBone / assumedBone, kScaleCorrectionMin, kScaleCorrectionMax);
				m_bStereoScaleFresh= true;
			}
		}
	}

	// Arm the angle + palm-depth holds for brief tri->mono fallbacks
	// (streamed convention: post rest-offset)
	m_lastTriAngles[(int)side]= outPose.fingers;
	m_lastTriSkeleton[(int)side]= outPose.skeleton;
	m_lastTriPalmWorld[(int)side]= outPose.palmPositionWorld;
	m_lastTriTimestampMs[(int)side]= m_fuseTimestampMs;

	cluster.triangulated= true;
	return true;
}

void HandFusion::fuse(const std::vector<const CameraFrameResult*>& candidates, double nowTimestampMs,
					  TrackingFrameResult& outFused)
{
	outFused= TrackingFrameResult();
	outFused.timestampMs= nowTimestampMs;
	m_fuseTimestampMs= nowTimestampMs;
	m_bStereoScaleFresh= false;
	m_bRawTriAnglesValid[0]= false;
	m_bRawTriAnglesValid[1]= false;
	m_bEstimatorProducedPose[0]= false;
	m_bEstimatorProducedPose[1]= false;

	// Carry frame bookkeeping from the freshest contributing camera
	const CameraFrameResult* freshest= nullptr;

	// Gather ALL parametric hand observations from fresh, world-tracked
	// camera results. The per-camera side label is only a VOTE here.
	// Observations too weak to cluster/vote (low presence or low confidence)
	// go into the rescue pool instead: their 2D landmarks may still serve as
	// the second triangulation view for a solo cluster.
	std::vector<HandCandidate> observations;
	std::vector<HandCandidate> rescuePool;
	for (const CameraFrameResult* camera : candidates)
	{
		if (camera == nullptr || !camera->valid)
			continue;
		if (nowTimestampMs - camera->timestampMs > m_config.stalenessWindowMs)
			continue;

		if (freshest == nullptr || camera->timestampMs > freshest->timestampMs)
			freshest= camera;

		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const TrackedHand& hand= camera->result.hands[sideIndex];
			const HandPose& pose= camera->result.poses[sideIndex];
			if (!pose.tracked || !pose.hasWorldPose || pose.presence < kTriRescueMinPresence)
				continue;

			HandCandidate candidate;
			candidate.camera= camera;
			candidate.hand= &hand;
			candidate.pose= &pose;

			// Vote DIRECTION comes from the flip-adjusted classifier score,
			// never from the per-camera side label (labels get displaced by
			// slot bookkeeping and then vote decisively for the wrong side)
			candidate.signedVote= (hand.rightProb - 0.5f) * 2.f;
			candidate.sideVoteWeight= hand.presence * std::max(fabsf(candidate.signedVote), 0.2f);

			if (pose.presence < m_config.presenceThreshold)
			{
				rescuePool.push_back(candidate);
				continue;
			}

			// Confidence = presence x measured stability. Presence alone is
			// NOT a usable quality signal: an edge-on hand scores 0.87 while
			// its depth solve swings by centimeters frame to frame (measured
			// live 2026-08-01: 41mm median jitter at presence 0.85-0.95).
			candidate.jitterM=
				updateJitter(camera->cameraIndex, sideIndex, pose.palmPositionWorld, camera->timestampMs);
			candidate.stability= stabilityFactor(candidate.jitterM, m_config.jitterReferenceM);
			candidate.confidence= pose.presence * candidate.stability;

			if (candidate.confidence < m_config.minCameraConfidence)
			{
				rescuePool.push_back(candidate);
				continue;
			}

			// Blend weight additionally folds in geometric conditioning (how
			// face-on the palm is), which ranks cameras but isn't meaningful
			// as an absolute trust value
			const glm::vec3 cameraPos= cameraPositionWorld(camera->markerFromCamera);
			candidate.weight= candidate.confidence *
				visibilityFactor(pose.palmOrientationWorld, pose.palmPositionWorld, cameraPos);

			observations.push_back(candidate);
		}
	}

	if (freshest != nullptr)
	{
		outFused.frameIndex= freshest->result.frameIndex;
		outFused.captureFps= freshest->result.captureFps;
		outFused.inferenceMs= freshest->result.inferenceMs;
	}

	// Cluster observations: one cluster = one physical hand, regardless of
	// what each camera called it (joint per-camera assignment, see
	// clusterObservations for why greedy nearest-position fails here)
	std::vector<HandCluster> clusters;
	clusterObservations(observations, clusters);

	// Solo clusters get a chance to pair up via image geometry (mono depth
	// error breaks the position-based gates exactly when a hand is hard to
	// see; the reprojection residual is the reliable correspondence signal)
	rescueSoloClusters(rescuePool, clusters);

	// Keep the two best clusters (there are only two physical hands)
	std::sort(clusters.begin(), clusters.end(),
			  [](const HandCluster& a, const HandCluster& b) { return a.bestWeight > b.bestWeight; });

	// Capture clustering diagnostics (including clusters about to be dropped)
	m_lastDiagnostics= FusionDiagnostics();
	m_lastDiagnostics.totalObservations= (int)observations.size();
	for (const HandCluster& cluster : clusters)
	{
		FusionDiagnostics::Cluster diagCluster;
		diagCluster.palmWorld= cluster.palmWorld;
		diagCluster.bestWeight= cluster.bestWeight;
		for (const HandCandidate& candidate : cluster.candidates)
		{
			FusionDiagnostics::Observation observation;
			observation.cameraIndex= candidate.camera->cameraIndex;
			observation.labeledSide= (int)candidate.hand->side;
			observation.weight= candidate.weight;
			observation.confidence= candidate.confidence;
			observation.stability= candidate.stability;
			observation.jitterMm= candidate.jitterM * 1000.f;
			observation.sideVoteWeight= candidate.sideVoteWeight;
			observation.palmWorld= candidate.pose->palmPositionWorld;
			diagCluster.observations.push_back(observation);
		}
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const AffinityBreakdown affinity= sideAffinity(cluster, (eHandSide)sideIndex);
			diagCluster.affinity[sideIndex][0]= affinity.vote;
			diagCluster.affinity[sideIndex][1]= affinity.temporal;
			diagCluster.affinity[sideIndex][2]= affinity.spatial;
		}
		m_lastDiagnostics.clusters.push_back(diagCluster);
	}

	if (clusters.size() > 2)
		clusters.resize(2);

	// Assign sides to clusters by joint affinity (votes + temporal continuity
	// + optional spatial prior). A winning assignment whose RAW affinity is
	// still negative is elimination, not evidence - refused while the side
	// was recently tracked (see kAssignmentRefuseThreshold).
	m_dominantCamera[0]= -1;
	m_dominantCamera[1]= -1;
	auto tryAssign= [&](int clusterIndex, int sideIndex, float rawAffinityTotal) {
		const bool bRecentlyTracked=
			nowTimestampMs - m_lastTrackedMs[sideIndex] < kAssignmentRefuseWindowMs;
		if (rawAffinityTotal < kAssignmentRefuseThreshold && bRecentlyTracked)
		{
			m_lastDiagnostics.clusters[clusterIndex].assignmentRefused= true;
			return;
		}
		fuseCluster((eHandSide)sideIndex, clusters[clusterIndex], outFused.hands[sideIndex],
					outFused.poses[sideIndex]);
		m_lastDiagnostics.clusters[clusterIndex].assignedSide= sideIndex;
	};
	if (clusters.size() == 2)
	{
		const float affinity0Left= sideAffinity(clusters[0], eHandSide::Left).total();
		const float affinity0Right= sideAffinity(clusters[0], eHandSide::Right).total();
		const float affinity1Left= sideAffinity(clusters[1], eHandSide::Left).total();
		const float affinity1Right= sideAffinity(clusters[1], eHandSide::Right).total();

		const int firstSide= affinity0Left + affinity1Right >= affinity0Right + affinity1Left
			? (int)eHandSide::Left
			: (int)eHandSide::Right;
		tryAssign(0, firstSide,
				  firstSide == (int)eHandSide::Left ? affinity0Left : affinity0Right);
		tryAssign(1, 1 - firstSide,
				  firstSide == (int)eHandSide::Left ? affinity1Right : affinity1Left);
	}
	else if (clusters.size() == 1)
	{
		const float rawLeft= sideAffinity(clusters[0], eHandSide::Left).total();
		const float rawRight= sideAffinity(clusters[0], eHandSide::Right).total();

		// Hysteresis: right after a clap both sides' temporal priors sit at
		// the same spot and votes can cancel, leaving the assignment to
		// numeric noise - which flip-flops L/R every frame and destroys the
		// temporal prior for the reacquisition. Stick with the incumbent
		// side unless decisively contradicted. (Refusal judges the RAW
		// affinity: the stickiness is inertia, not evidence.)
		float affinityLeft= rawLeft;
		float affinityRight= rawRight;
		if (m_lastSoloSide == (int)eHandSide::Left)
			affinityLeft+= kSoloSideStickiness;
		else if (m_lastSoloSide == (int)eHandSide::Right)
			affinityRight+= kSoloSideStickiness;

		const int sideIndex= affinityLeft >= affinityRight ? (int)eHandSide::Left : (int)eHandSide::Right;
		tryAssign(0, sideIndex, sideIndex == (int)eHandSide::Left ? rawLeft : rawRight);
	}

	// Mirror triangulation outcomes into the diagnostics (fuseCluster runs
	// after the diagnostics snapshot above; cluster order is preserved)
	for (size_t i= 0; i < clusters.size() && i < m_lastDiagnostics.clusters.size(); ++i)
	{
		FusionDiagnostics::Cluster& diagCluster= m_lastDiagnostics.clusters[i];
		diagCluster.triangulated= clusters[i].triangulated;
		diagCluster.triVetoed= clusters[i].triVetoed;
		diagCluster.triResidualRmsPx= clusters[i].triResidualRmsPx;
		diagCluster.triResidualMaxPx= clusters[i].triResidualMaxPx;
		diagCluster.triCameraA= clusters[i].triCameraA;
		diagCluster.triCameraB= clusters[i].triCameraB;
		diagCluster.triParallaxDeg= clusters[i].triParallaxDeg;
		diagCluster.estimatorUsed= clusters[i].estimatorUsed;
		diagCluster.estimatorCameraCount= clusters[i].estimatorCameraCount;
		diagCluster.estimatorIterations= clusters[i].estimatorIterations;
		diagCluster.estimatorResidualBeforePx= clusters[i].estimatorResidualBeforePx;
		diagCluster.estimatorResidualAfterPx= clusters[i].estimatorResidualAfterPx;
		diagCluster.estimatorReseeded= clusters[i].estimatorReseeded;
		diagCluster.estimatorHeldBadFit= clusters[i].estimatorHeldBadFit;
	}

	// Track the solo-side incumbent for the hysteresis above
	{
		const bool bLeftTracked= outFused.poses[0].tracked;
		const bool bRightTracked= outFused.poses[1].tracked;
		if (bLeftTracked != bRightTracked)
			m_lastSoloSide= bLeftTracked ? (int)eHandSide::Left : (int)eHandSide::Right;
		else
			m_lastSoloSide= -1;
	}

	// Update the temporal side prior from this frame's assignments
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (outFused.poses[sideIndex].tracked)
		{
			m_lastFusedPalm[sideIndex]= outFused.poses[sideIndex].palmPositionWorld;
			m_bLastFusedPalmValid[sideIndex]= true;
			m_lastTrackedMs[sideIndex]= nowTimestampMs;
		}
		// (keep the last position when untracked - it decays naturally via distance)
	}

	applySmoothing(outFused);
}

void HandFusion::fuseCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, HandPose& outPose)
{
	// The classic extraction always runs: with the estimator off it IS the
	// output; with it on it advances the shared bookkeeping and provides the
	// seed/fallback pose the estimator refines
	fuseClusterClassic(side, cluster, outHand, outPose);

	if (m_config.estimatorEnabled)
		applyEstimator(side, cluster, outHand, outPose);
}

void HandFusion::fuseClusterClassic(eHandSide side, HandCluster& cluster, TrackedHand& outHand, HandPose& outPose)
{
	std::vector<HandCandidate>& candidates= cluster.candidates;

	// Best candidate provides the advisory landmark data + skeleton
	const HandCandidate& best= *std::max_element(
		candidates.begin(), candidates.end(),
		[](const HandCandidate& a, const HandCandidate& b) { return a.weight < b.weight; });
	m_dominantCamera[(int)side]= best.camera->cameraIndex;

	outHand= *best.hand;
	outHand.side= side;
	outPose= *best.pose;
	outPose.side= side;
	outPose.visibility= best.weight;

	// Fused confidence = the best single view of this hand (a second, worse
	// view can only add information, never make the estimate less trustworthy)
	outPose.confidence= 0.f;
	for (const HandCandidate& candidate : candidates)
		outPose.confidence= std::max(outPose.confidence, candidate.confidence);
	outPose.confidence= std::clamp(outPose.confidence, 0.f, 1.f);

	// Primary path: triangulate the 21 landmarks across two cameras and
	// extract the pose from real stereo geometry - no monocular depth, no
	// cross-camera blending of disagreeing articulation.
	if (m_config.triangulationEnabled && candidates.size() >= 2 &&
		triangulateCluster(side, cluster, outHand, outPose))
	{
		outPose.tracked= true;
		outPose.hasWorldPose= true;
		return;
	}

	// A residual veto means the cluster's observations are probably two
	// DIFFERENT physical hands - blending them would manufacture a hand
	// between them. Keep the best single observation instead.
	if (cluster.triVetoed)
	{
		applyTriAngleHold(side, outPose);
		applyTriPositionHold(side, best, outPose);
		outPose.tracked= true;
		outPose.hasWorldPose= true;
		return;
	}

	// Fallback multi-camera path (triangulation off or unavailable)
	if (candidates.size() >= 2)
	{
		// Palm POSITION blends in both modes: positions compose, and the
		// cross-camera disagreement there is cm-scale, not tens of degrees
		float weightSum= 0.f;
		glm::vec3 blendedPosition(0.f);
		float maxPresence= 0.f;
		for (const HandCandidate& candidate : candidates)
		{
			weightSum+= candidate.weight;
			blendedPosition+= candidate.pose->palmPositionWorld * candidate.weight;
			maxPresence= std::max(maxPresence, candidate.pose->presence);
		}
		if (weightSum > 1e-6f)
		{
			outPose.palmPositionWorld= blendedPosition / weightSum;
			outPose.presence= maxPresence;
		}

		// Select one camera as the orientation/angle source, with
		// hysteresis: the incumbent keeps the job until a challenger beats
		// its weight by a decisive margin for several consecutive fuses.
		// (Without it, weight noise would flip the source frame to frame -
		// reintroducing the switching wander that selection exists to kill.)
		const int sideIndex= (int)side;
		const HandCandidate* incumbent= nullptr;
		for (const HandCandidate& candidate : candidates)
		{
			if (candidate.camera->cameraIndex == m_articulationSource[sideIndex])
				incumbent= &candidate;
		}

		const HandCandidate* source= &best;
		if (incumbent == nullptr)
		{
			// No incumbent in this cluster: adopt the best immediately
			m_articulationChallengerFrames[sideIndex]= 0;
			m_articulationChallenger[sideIndex]= -1;
		}
		else if (best.camera->cameraIndex != incumbent->camera->cameraIndex &&
				 best.weight > incumbent->weight * kArticulationSwitchMargin)
		{
			if (m_articulationChallenger[sideIndex] == best.camera->cameraIndex)
				++m_articulationChallengerFrames[sideIndex];
			else
			{
				m_articulationChallenger[sideIndex]= best.camera->cameraIndex;
				m_articulationChallengerFrames[sideIndex]= 1;
			}

			if (m_articulationChallengerFrames[sideIndex] < kArticulationSwitchFrames)
				source= incumbent;
		}
		else
		{
			m_articulationChallengerFrames[sideIndex]= 0;
			m_articulationChallenger[sideIndex]= -1;
			source= incumbent;
		}
		m_articulationSource[sideIndex]= source->camera->cameraIndex;
		m_dominantCamera[sideIndex]= source->camera->cameraIndex;

		outPose.palmOrientationWorld= source->pose->palmOrientationWorld;
		outPose.fingers= source->pose->fingers;
		outPose.skeleton= source->pose->skeleton;
		outHand= *source->hand;
		outHand.side= side;

		// A two-camera observation of one physical hand also triangulates the
		// true hand scale
		updateStereoScale(cluster);
	}
	else
	{
		m_articulationSource[(int)side]= best.camera->cameraIndex;
		m_articulationChallengerFrames[(int)side]= 0;
		m_articulationChallenger[(int)side]= -1;
	}

	// Any non-triangulated outcome: bridge brief tri dropouts with the last
	// triangulated angles and palm depth instead of snapping to the mono
	// estimate. (For the blended multi-camera position, best's view ray is an
	// approximation - it dominated the blend weight.)
	applyTriAngleHold(side, outPose);
	applyTriPositionHold(side, best, outPose);

	outPose.tracked= true;
	outPose.hasWorldPose= true;
}

void HandFusion::updateEstimatorSkeleton(eHandSide side, const HandSkeleton& observed)
{
	const int sideIndex= (int)side;
	HandSkeleton& skeleton= m_estimatorSkeleton[sideIndex];
	if (!m_bEstimatorSkeletonValid[sideIndex])
	{
		skeleton= observed;
		m_bEstimatorSkeletonValid[sideIndex]= true;
		return;
	}

	// Slow blend: a calibrated skeleton is constant so this converges to it
	// immediately; an uncalibrated one is re-measured per frame and must not
	// jitter the fit geometry (a moving skeleton reads as moving angles)
	constexpr float kSkeletonEmaAlpha= 0.05f;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		skeleton.baseInPalm[finger]=
			glm::mix(skeleton.baseInPalm[finger], observed.baseInPalm[finger], kSkeletonEmaAlpha);
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			skeleton.phalanxLengths[finger][phalanx]+=
				kSkeletonEmaAlpha *
				(observed.phalanxLengths[finger][phalanx] - skeleton.phalanxLengths[finger][phalanx]);
	}
	// Neutral directions are ALWAYS the flat-hand default derived from the
	// bases (the wire convention), so rebuild rather than blend them
	skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
}

bool HandFusion::makeEstimatorSeed(eHandSide side, const HandCluster& cluster, const HandPose& classicPose,
								   HandStateEstimator::Pose& outSeed)
{
	if (!classicPose.hasWorldPose)
		return false;

	const int sideIndex= (int)side;
	outSeed.palmPositionWorld= classicPose.palmPositionWorld;
	outSeed.palmOrientationWorld= classicPose.palmOrientationWorld;

	// The state lives in RAW angles (FK-consistent); the classic pose's
	// fingers carry the rest offset. Best raw source first: this fuse's
	// triangulated angles, else angles re-derived from the best observation's
	// world landmarks, else un-offsetting the streamed angles.
	if (m_bRawTriAnglesValid[sideIndex])
	{
		outSeed.rawAngles= m_rawTriAngles[sideIndex];
		return true;
	}

	const HandCandidate* best= nullptr;
	for (const HandCandidate& candidate : cluster.candidates)
	{
		if (best == nullptr || candidate.weight > best->weight)
			best= &candidate;
	}
	if (best != nullptr && best->hand->hasWorldSpace)
	{
		HandPoseModel::computeFingerAngles(best->hand->worldPoints, side,
										   m_estimatorSkeleton[sideIndex].neutralDirInPalm,
										   outSeed.rawAngles, &m_triPalmarMemory[sideIndex]);
		return true;
	}

	outSeed.rawAngles= classicPose.fingers;
	if (m_config.bHasFusedRestAngles[sideIndex])
	{
		const std::array<FingerAngles, FINGER_COUNT>& rest= m_config.fusedRestAngles[sideIndex];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			outSeed.rawAngles[finger].lateral+= rest[finger].lateral;
			outSeed.rawAngles[finger].proximal+= rest[finger].proximal;
			outSeed.rawAngles[finger].intermediate+= rest[finger].intermediate;
			outSeed.rawAngles[finger].distal+= rest[finger].distal;
		}
	}
	return true;
}

void HandFusion::applyEstimator(eHandSide side, HandCluster& cluster, TrackedHand& outHand,
								HandPose& outPose)
{
	const int sideIndex= (int)side;

	const HandCandidate* best= nullptr;
	for (const HandCandidate& candidate : cluster.candidates)
	{
		if (best == nullptr || candidate.weight > best->weight)
			best= &candidate;
	}
	if (best == nullptr)
		return;

	// The per-camera poses already carry the calibrated skeleton when one
	// exists, so the best pose's skeleton is calibrated-when-present
	updateEstimatorSkeleton(side, best->pose->skeleton);

	// Measurement rows: every tracked observation with intrinsics. A residual-
	// vetoed cluster is probably two different physical hands, so only the
	// best observation enters (the same call the classic path makes).
	std::vector<HandStateEstimator::Observation> observations;
	observations.reserve(cluster.candidates.size());
	double freshestTimestampMs= -1e12;
	float maxPresence= 0.f;
	for (const HandCandidate& candidate : cluster.candidates)
	{
		if (cluster.triVetoed && &candidate != best)
			continue;
		if (!candidate.camera->hasIntrinsics || !candidate.hand->tracked)
			continue;

		HandStateEstimator::Observation observation;
		observation.cameraIndex= candidate.camera->cameraIndex;
		observation.timestampMs= candidate.camera->timestampMs;
		observation.presence= candidate.pose->presence;
		observation.markerFromCamera= candidate.camera->markerFromCamera;
		observation.fx= candidate.camera->fx;
		observation.fy= candidate.camera->fy;
		observation.cx= candidate.camera->cx;
		observation.cy= candidate.camera->cy;
		observation.imagePoints= &candidate.hand->imagePoints;
		observations.push_back(observation);

		freshestTimestampMs= std::max(freshestTimestampMs, candidate.camera->timestampMs);
		maxPresence= std::max(maxPresence, candidate.pose->presence);
	}
	if (observations.empty())
		return;

	// Seed only matters when the state is cold (cold start or after a
	// divergence drop); building it costs an angle extraction, so skip it
	// while the state is alive
	HandStateEstimator::Pose seed;
	bool bHaveSeed= false;
	if (!m_estimator.hasState(side))
		bHaveSeed= makeEstimatorSeed(side, cluster, outPose, seed);

	HandStateEstimator::Pose fitted;
	const HandStateEstimator::UpdateResult update= m_estimator.update(
		side, observations, m_fuseTimestampMs, bHaveSeed ? &seed : nullptr,
		m_estimatorSkeleton[sideIndex], fitted);

	cluster.estimatorUsed= update.bUpdated;
	cluster.estimatorCameraCount= update.cameraCount;
	cluster.estimatorIterations= update.iterations;
	cluster.estimatorResidualBeforePx= update.residualBeforePx;
	cluster.estimatorResidualAfterPx= update.residualAfterPx;
	cluster.estimatorReseeded= update.bReseeded;
	cluster.estimatorHeldBadFit= update.bHeldBadFit;

	// Divergence (or no state and no seed): the classic pose stands this
	// fuse and the next one reseeds
	if (!update.bUpdated)
		return;

	outPose.palmPositionWorld= fitted.palmPositionWorld;
	outPose.palmOrientationWorld= fitted.palmOrientationWorld;
	outPose.hasWorldPose= true;
	outPose.tracked= true;
	outPose.skeleton= m_estimatorSkeleton[sideIndex];
	outPose.stereoTriangulated= update.cameraCount >= 2;
	outPose.presence= maxPresence;

	// Rest capture reads raw multi-view angles exactly as it does from the
	// triangulated path
	if (update.cameraCount >= 2)
	{
		m_rawTriAngles[sideIndex]= fitted.rawAngles;
		m_bRawTriAnglesValid[sideIndex]= true;
	}

	// Streamed convention: rest offset applied at output, state stays raw
	std::array<FingerAngles, FINGER_COUNT> streamedAngles= fitted.rawAngles;
	if (m_config.bHasFusedRestAngles[sideIndex])
	{
		const std::array<FingerAngles, FINGER_COUNT>& rest= m_config.fusedRestAngles[sideIndex];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			streamedAngles[finger].lateral-= rest[finger].lateral;
			streamedAngles[finger].proximal-= rest[finger].proximal;
			streamedAngles[finger].intermediate-= rest[finger].intermediate;
			streamedAngles[finger].distal-= rest[finger].distal;
		}
	}
	outPose.fingers= streamedAngles;

	// Confidence = presence x measured output stability x fit residual factor
	// (same shape as the triangulated path's; the estimator gets its own
	// jitter tracker key so the classic path's tracker history stays intact)
	if (update.cameraCount > 0)
		m_estimatorJitterM[sideIndex]=
			updateJitter(-2, sideIndex, fitted.palmPositionWorld, freshestTimestampMs);
	const float stability= stabilityFactor(m_estimatorJitterM[sideIndex], m_config.jitterReferenceM);
	outPose.confidence= std::clamp(
		maxPresence * stability *
			residualFactor(update.residualAfterPx, m_config.residualReferencePx),
		0.f, 1.f);

	// Overlays and downstream consumers see the fitted geometry
	HandStateEstimator::predictWorldLandmarks(fitted, m_estimatorSkeleton[sideIndex], outHand.worldPoints);
	outHand.hasWorldSpace= true;

	m_bEstimatorProducedPose[sideIndex]= true;
}

void HandFusion::applyTriAngleHold(eHandSide side, HandPose& outPose) const
{
	const double sinceTriMs= m_fuseTimestampMs - m_lastTriTimestampMs[(int)side];
	if (sinceTriMs < 0.0 || sinceTriMs > kTriAngleHoldMs)
		return;

	outPose.fingers= m_lastTriAngles[(int)side];
	outPose.skeleton= m_lastTriSkeleton[(int)side];
}

void HandFusion::applyTriPositionHold(eHandSide side, const HandCandidate& source, HandPose& outPose) const
{
	const double sinceTriMs= m_fuseTimestampMs - m_lastTriTimestampMs[(int)side];
	if (sinceTriMs < 0.0 || sinceTriMs > kTriAngleHoldMs)
		return;
	if (!outPose.hasWorldPose)
		return;

	// Mono depth error lives along the observing camera's view ray while the
	// lateral measurement stays good, so remove exactly the along-ray
	// innovation against the last triangulated palm. A sustained mono
	// stretch adopts the mono depth when the hold window expires, mirroring
	// the angle hold above.
	const glm::vec3 cameraPosWorld= cameraPositionWorld(source.camera->markerFromCamera);
	const glm::vec3 toPalm= outPose.palmPositionWorld - cameraPosWorld;
	const float rayLength= glm::length(toPalm);
	if (rayLength < 1e-4f)
		return;
	const glm::vec3 ray= toPalm / rayLength;
	const float alongRayInnovation=
		glm::dot(outPose.palmPositionWorld - m_lastTriPalmWorld[(int)side], ray);
	outPose.palmPositionWorld-= ray * alongRayInnovation;
}

void HandFusion::applySmoothing(TrackingFrameResult& ioFused)
{
	if (!m_config.smoothingEnabled)
		return;

	float dtSeconds= 1.f / 60.f;
	if (m_lastTimestampMs >= 0.0 && ioFused.timestampMs > m_lastTimestampMs)
		dtSeconds= std::clamp((float)((ioFused.timestampMs - m_lastTimestampMs) / 1000.0), 1.f / 240.f, 0.25f);
	m_lastTimestampMs= ioFused.timestampMs;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		HandPose& pose= ioFused.poses[sideIndex];

		if (pose.tracked && pose.hasWorldPose)
		{
			// fresh acquisition: drop stale filter state
			if (!m_bSideWasTracked[sideIndex])
			{
				// ...including the remembered palmar side, which describes
				// where the hand WAS
				m_triPalmarMemory[sideIndex].reset();
				m_positionFilters[sideIndex].reset();
				for (OneEuroFilter& filter : m_quaternionFilters[sideIndex])
					filter.reset();
				for (OneEuroFilter& filter : m_angleFilters[sideIndex])
					filter.reset();
				m_lastFilteredQuat[sideIndex]= pose.palmOrientationWorld;
			}

			// An estimator-produced pose is already temporally regularized by
			// its own prior - cascading the one-euro on top would only add
			// lag. Keep the filters reset so a fallback frame starts clean
			// instead of filtering against a stale history.
			if (m_bEstimatorProducedPose[sideIndex])
			{
				m_positionFilters[sideIndex].reset();
				for (OneEuroFilter& filter : m_quaternionFilters[sideIndex])
					filter.reset();
				for (OneEuroFilter& filter : m_angleFilters[sideIndex])
					filter.reset();
				m_lastFilteredQuat[sideIndex]= pose.palmOrientationWorld;
				m_bSideWasTracked[sideIndex]= true;
				continue;
			}

			pose.palmPositionWorld= m_positionFilters[sideIndex].filter(pose.palmPositionWorld, dtSeconds);

			// Quaternion: hemisphere-align against the last filtered value,
			// filter components, renormalize
			glm::quat quat= pose.palmOrientationWorld;
			if (glm::dot(quat, m_lastFilteredQuat[sideIndex]) < 0.f)
				quat= -quat;
			glm::quat filtered(
				m_quaternionFilters[sideIndex][0].filter(quat.w, dtSeconds),
				m_quaternionFilters[sideIndex][1].filter(quat.x, dtSeconds),
				m_quaternionFilters[sideIndex][2].filter(quat.y, dtSeconds),
				m_quaternionFilters[sideIndex][3].filter(quat.z, dtSeconds));
			filtered= glm::normalize(filtered);
			pose.palmOrientationWorld= filtered;
			m_lastFilteredQuat[sideIndex]= filtered;

			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				FingerAngles& angles= pose.fingers[finger];
				angles.lateral= m_angleFilters[sideIndex][finger * 4 + 0].filter(angles.lateral, dtSeconds);
				angles.proximal= m_angleFilters[sideIndex][finger * 4 + 1].filter(angles.proximal, dtSeconds);
				angles.intermediate= m_angleFilters[sideIndex][finger * 4 + 2].filter(angles.intermediate, dtSeconds);
				angles.distal= m_angleFilters[sideIndex][finger * 4 + 3].filter(angles.distal, dtSeconds);
			}
		}

		m_bSideWasTracked[sideIndex]= pose.tracked && pose.hasWorldPose;
	}
}
