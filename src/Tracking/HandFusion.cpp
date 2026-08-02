#include "HandFusion.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"
#include "Logger.h"

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
// Triangulated-angle hold across brief mono fallbacks (see m_lastTriAngles)
static constexpr double kTriAngleHoldMs= 300.0;
// Rescue-pool floor: a low-presence hand may still contribute usable 2D
// landmarks as the second triangulation view (the residual is the judge),
// but below this even the image points are guesses
static constexpr float kTriRescueMinPresence= 0.15f;

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
	const glm::vec3 cameraPos= glm::vec3(observation.camera->markerFromCamera[3]);

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
			cluster.anchorCameraPos= glm::vec3(observation.camera->markerFromCamera[3]);
			cluster.anchorSignedVote= observation.signedVote;
			cluster.bestWeight= observation.weight;
		}
	};
	auto newCluster= [&outClusters](const HandCandidate& observation) {
		HandCluster cluster;
		cluster.candidates.push_back(observation);
		cluster.palmWorld= observation.pose->palmPositionWorld;
		cluster.anchorCameraPos= glm::vec3(observation.camera->markerFromCamera[3]);
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

	// Spatial prior (opt-in): hands that never cross stay on their own side of
	// the marker, so position along the configured axis is side evidence
	if (m_config.spatialSidePriorAxis != 0)
	{
		static const glm::vec3 kRightAxes[]= {
			{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}};
		const glm::vec3 rightAxis= kRightAxes[std::clamp(m_config.spatialSidePriorAxis, 0, 4)];
		const float along= glm::dot(cluster.palmWorld, rightAxis);
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

	const glm::vec3 cameraPosA= glm::vec3(a.camera->markerFromCamera[3]);
	const glm::vec3 cameraPosB= glm::vec3(b.camera->markerFromCamera[3]);
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
		glm::mat3 rotation;        // camera -> world
		glm::dmat4 cameraFromWorld;
	};
	std::array<View, 2> views;
	for (int v= 0; v < 2; ++v)
	{
		const HandCandidate* obs= v == 0 ? &obsA : &obsB;
		views[v].obs= obs;
		views[v].cameraPos= glm::vec3(obs->camera->markerFromCamera[3]);
		views[v].rotation= glm::mat3(glm::mat4(obs->camera->markerFromCamera));
		views[v].cameraFromWorld= glm::inverse(obs->camera->markerFromCamera);
	}

	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		glm::vec3 rayDir[2];
		for (int v= 0; v < 2; ++v)
		{
			const CameraFrameResult* camera= views[v].obs->camera;
			const glm::vec3& px= views[v].obs->hand->imagePoints[i];
			// Undistorted pinhole back-projection, OpenCV camera convention
			const glm::vec3 dirCamera((px.x - camera->cx) / camera->fx,
									  (px.y - camera->cy) / camera->fy, 1.f);
			rayDir[v]= glm::normalize(views[v].rotation * dirCamera);
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
	// Best two candidates from distinct cameras with usable image geometry
	const HandCandidate* obsA= nullptr;
	const HandCandidate* obsB= nullptr;
	for (const HandCandidate& candidate : cluster.candidates)
	{
		if (!candidate.camera->hasIntrinsics || !candidate.hand->tracked)
			continue;
		if (obsA == nullptr || candidate.weight > obsA->weight)
		{
			if (obsA != nullptr && obsA->camera->cameraIndex != candidate.camera->cameraIndex)
				obsB= obsA;
			obsA= &candidate;
		}
		else if (candidate.camera->cameraIndex != obsA->camera->cameraIndex &&
				 (obsB == nullptr || candidate.weight > obsB->weight))
		{
			obsB= &candidate;
		}
	}
	if (obsA == nullptr || obsB == nullptr)
		return false;

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
	const glm::mat4 palmFrame= HandPoseModel::computePalmFrame(triPoints, side);
	outPose.palmPositionWorld= glm::vec3(palmFrame[3]);
	outPose.palmOrientationWorld= glm::quat_cast(glm::mat3(palmFrame));
	outPose.hasWorldPose= true;

	std::array<FingerAngles, FINGER_COUNT> rawAngles{};
	HandPoseModel::computeFingerAngles(triPoints, side, outPose.skeleton.neutralDirInPalm, rawAngles);
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

	// Arm the angle hold for brief tri->mono fallbacks (streamed convention:
	// post rest-offset)
	m_lastTriAngles[(int)side]= outPose.fingers;
	m_lastTriSkeleton[(int)side]= outPose.skeleton;
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
			const glm::vec3 cameraPos= glm::vec3(camera->markerFromCamera[3]);
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
	// + optional spatial prior)
	m_dominantCamera[0]= -1;
	m_dominantCamera[1]= -1;
	if (clusters.size() == 2)
	{
		const float assignLR= sideAffinity(clusters[0], eHandSide::Left).total() +
			sideAffinity(clusters[1], eHandSide::Right).total();
		const float assignRL= sideAffinity(clusters[0], eHandSide::Right).total() +
			sideAffinity(clusters[1], eHandSide::Left).total();

		const int firstSide= assignLR >= assignRL ? (int)eHandSide::Left : (int)eHandSide::Right;
		const int secondSide= 1 - firstSide;
		fuseCluster((eHandSide)firstSide, clusters[0], outFused.hands[firstSide], outFused.poses[firstSide]);
		fuseCluster((eHandSide)secondSide, clusters[1], outFused.hands[secondSide], outFused.poses[secondSide]);
		m_lastDiagnostics.clusters[0].assignedSide= firstSide;
		m_lastDiagnostics.clusters[1].assignedSide= secondSide;
	}
	else if (clusters.size() == 1)
	{
		float affinityLeft= sideAffinity(clusters[0], eHandSide::Left).total();
		float affinityRight= sideAffinity(clusters[0], eHandSide::Right).total();

		// Hysteresis: right after a clap both sides' temporal priors sit at
		// the same spot and votes can cancel, leaving the assignment to
		// numeric noise - which flip-flops L/R every frame and destroys the
		// temporal prior for the reacquisition. Stick with the incumbent
		// side unless decisively contradicted.
		if (m_lastSoloSide == (int)eHandSide::Left)
			affinityLeft+= kSoloSideStickiness;
		else if (m_lastSoloSide == (int)eHandSide::Right)
			affinityRight+= kSoloSideStickiness;

		const eHandSide side= affinityLeft >= affinityRight ? eHandSide::Left : eHandSide::Right;
		fuseCluster(side, clusters[0], outFused.hands[(int)side], outFused.poses[(int)side]);
		m_lastDiagnostics.clusters[0].assignedSide= (int)side;
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
		}
		// (keep the last position when untracked - it decays naturally via distance)
	}

	applySmoothing(outFused);
}

void HandFusion::fuseCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, HandPose& outPose)
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

		if (m_config.blendArticulation)
		{
			// Legacy A/B path: weighted blend of orientation + finger angles.
			// Poses/angles compose - but blending two estimators that disagree
			// by tens of degrees with time-varying weights manufactures
			// low-frequency wander inside the smoothing passband.
			glm::quat blendedQuat(0.f, 0.f, 0.f, 0.f);
			std::array<FingerAngles, FINGER_COUNT> blendedAngles{};

			const glm::quat referenceQuat= best.pose->palmOrientationWorld;
			for (const HandCandidate& candidate : candidates)
			{
				const float weight= candidate.weight;

				// Hemisphere-align before component-wise blending
				glm::quat quat= candidate.pose->palmOrientationWorld;
				if (glm::dot(quat, referenceQuat) < 0.f)
					quat= -quat;
				blendedQuat+= quat * weight;

				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					blendedAngles[finger].lateral+= candidate.pose->fingers[finger].lateral * weight;
					blendedAngles[finger].proximal+= candidate.pose->fingers[finger].proximal * weight;
					blendedAngles[finger].intermediate+= candidate.pose->fingers[finger].intermediate * weight;
					blendedAngles[finger].distal+= candidate.pose->fingers[finger].distal * weight;
				}
			}

			if (weightSum > 1e-6f)
			{
				outPose.palmOrientationWorld= glm::normalize(blendedQuat);
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					outPose.fingers[finger].lateral= blendedAngles[finger].lateral / weightSum;
					outPose.fingers[finger].proximal= blendedAngles[finger].proximal / weightSum;
					outPose.fingers[finger].intermediate= blendedAngles[finger].intermediate / weightSum;
					outPose.fingers[finger].distal= blendedAngles[finger].distal / weightSum;
				}
			}
		}
		else
		{
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
		}

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
	// triangulated angles instead of snapping to the mono articulation
	applyTriAngleHold(side, outPose);

	outPose.tracked= true;
	outPose.hasWorldPose= true;
}

void HandFusion::applyTriAngleHold(eHandSide side, HandPose& outPose) const
{
	const double sinceTriMs= m_fuseTimestampMs - m_lastTriTimestampMs[(int)side];
	if (sinceTriMs < 0.0 || sinceTriMs > kTriAngleHoldMs)
		return;

	outPose.fingers= m_lastTriAngles[(int)side];
	outPose.skeleton= m_lastTriSkeleton[(int)side];
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
				m_positionFilters[sideIndex].reset();
				for (OneEuroFilter& filter : m_quaternionFilters[sideIndex])
					filter.reset();
				for (OneEuroFilter& filter : m_angleFilters[sideIndex])
					filter.reset();
				m_lastFilteredQuat[sideIndex]= pose.palmOrientationWorld;
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
