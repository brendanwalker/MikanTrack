#include "HandFusion.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

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

void HandFusion::configure(const HandFusionConfig& config)
{
	m_config= config;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_positionFilters[sideIndex].configure(config.smoothingMinCutoff, config.smoothingBeta, 1.f);
		m_positionFilters[sideIndex].reset();
		for (OneEuroFilter& filter : m_quaternionFilters[sideIndex])
		{
			filter.configure(config.smoothingMinCutoff, config.smoothingBeta, 1.f);
			filter.reset();
		}
		for (OneEuroFilter& filter : m_angleFilters[sideIndex])
		{
			filter.configure(config.smoothingMinCutoff, config.smoothingBeta, 1.f);
			filter.reset();
		}
		m_lastFilteredQuat[sideIndex]= glm::quat(1, 0, 0, 0);
		m_bSideWasTracked[sideIndex]= false;
		m_bLastFusedPalmValid[sideIndex]= false;
	}
	m_lastTimestampMs= -1.0;
	m_stereoScaleCorrection= 1.f;
	m_bStereoScaleFresh= false;
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

// Same-physical-hand test for clustering. Depth along the observing camera's
// view ray is the noisy dimension (it scales linearly with hand-scale / PnP
// error), so a plain euclidean gate splits one hand into two clusters when a
// camera's depth is off - which then fights the real hand for a side. Two
// observations match if they're close in 3D, OR one lies close to the view
// ray through the other (same image position, different depth).
static bool isSameHandObservation(const glm::vec3& palmA, const glm::vec3& cameraPosA,
								  const glm::vec3& palmB, const glm::vec3& cameraPosB,
								  float maxDistM)
{
	if (glm::length(palmA - palmB) <= maxDistM)
		return true;

	// Tighter lateral gate for the ray test: hands genuinely lined up along a
	// view ray (one behind the other) must not merge
	const float rayGateM= maxDistM * 0.5f;
	auto nearViewRay= [rayGateM](const glm::vec3& point, const glm::vec3& rayOrigin, const glm::vec3& through) {
		glm::vec3 dir= through - rayOrigin;
		const float observedDepth= glm::length(dir);
		if (observedDepth < 1e-4f)
			return false;
		dir/= observedDepth;

		const glm::vec3 toPoint= point - rayOrigin;
		const float along= glm::dot(toPoint, dir);
		if (along <= 0.f || fabsf(along - observedDepth) > kRayDepthSlackM)
			return false;

		return glm::length(toPoint - dir * along) <= rayGateM;
	};

	return nearViewRay(palmB, cameraPosA, palmA) || nearViewRay(palmA, cameraPosB, palmB);
}

float HandFusion::sideAffinity(const HandCluster& cluster, eHandSide side) const
{
	// Classifier votes: each camera's assigned side, weighted by presence and
	// how decisive its (smoothed) handedness score was
	float voteScore= 0.f;
	for (const HandCandidate& candidate : cluster.candidates)
		voteScore+= candidate.sideVoteWeight * (candidate.hand->side == side ? 1.f : -1.f);

	// Temporal continuity: attraction to where this side's fused hand was last
	float temporalScore= 0.f;
	if (m_bLastFusedPalmValid[(int)side])
	{
		const float dist= glm::length(cluster.palmWorld - m_lastFusedPalm[(int)side]);
		temporalScore= kTemporalWeight *
			std::clamp(1.f - (dist - kTemporalFullDistM) / kTemporalFullDistM, -1.f, 1.f);
	}

	// Spatial prior (opt-in): hands that never cross stay on their own side of
	// the marker, so position along the configured axis is side evidence
	float spatialScore= 0.f;
	if (m_config.spatialSidePriorAxis != 0)
	{
		static const glm::vec3 kRightAxes[]= {
			{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {-1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}};
		const glm::vec3 rightAxis= kRightAxes[std::clamp(m_config.spatialSidePriorAxis, 0, 4)];
		const float along= glm::dot(cluster.palmWorld, rightAxis);
		const float prior= std::clamp(along / kSpatialPriorFullDistM, -1.f, 1.f);
		spatialScore= kSpatialPriorWeight * (side == eHandSide::Right ? prior : -prior);
	}

	return voteScore + temporalScore + spatialScore;
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

void HandFusion::fuse(const std::vector<const CameraFrameResult*>& candidates, double nowTimestampMs,
					  TrackingFrameResult& outFused)
{
	outFused= TrackingFrameResult();
	outFused.timestampMs= nowTimestampMs;
	m_bStereoScaleFresh= false;

	// Carry frame bookkeeping from the freshest contributing camera
	const CameraFrameResult* freshest= nullptr;

	// Gather ALL parametric hand observations from fresh, world-tracked
	// camera results. The per-camera side label is only a VOTE here.
	std::vector<HandCandidate> observations;
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
			if (!pose.tracked || !pose.hasWorldPose || pose.presence < m_config.presenceThreshold)
				continue;

			HandCandidate candidate;
			candidate.camera= camera;
			candidate.hand= &hand;
			candidate.pose= &pose;

			const glm::vec3 cameraPos= glm::vec3(camera->markerFromCamera[3]);
			candidate.weight= pose.presence *
				visibilityFactor(pose.palmOrientationWorld, pose.palmPositionWorld, cameraPos);

			// Vote weight: presence x how decisive the classifier was
			candidate.sideVoteWeight= hand.presence * std::max(fabsf(hand.handednessScore - 0.5f) * 2.f, 0.2f);

			observations.push_back(candidate);
		}
	}

	if (freshest != nullptr)
	{
		outFused.frameIndex= freshest->result.frameIndex;
		outFused.captureFps= freshest->result.captureFps;
		outFused.inferenceMs= freshest->result.inferenceMs;
	}

	// Cluster observations by world palm proximity: one cluster = one
	// physical hand, regardless of what each camera called it
	std::vector<HandCluster> clusters;
	std::sort(observations.begin(), observations.end(),
			  [](const HandCandidate& a, const HandCandidate& b) { return a.weight > b.weight; });
	for (const HandCandidate& observation : observations)
	{
		const glm::vec3& palm= observation.pose->palmPositionWorld;

		HandCluster* target= nullptr;
		for (HandCluster& cluster : clusters)
		{
			// One observation per camera per cluster (a camera can't see the
			// same physical hand twice)
			bool bCameraAlreadyInCluster= false;
			for (const HandCandidate& member : cluster.candidates)
				bCameraAlreadyInCluster|= member.camera->cameraIndex == observation.camera->cameraIndex;

			if (!bCameraAlreadyInCluster &&
				isSameHandObservation(palm, glm::vec3(observation.camera->markerFromCamera[3]),
									  cluster.palmWorld, cluster.anchorCameraPos,
									  m_config.wristMatchMaxDistM))
			{
				target= &cluster;
				break;
			}
		}

		if (target != nullptr)
		{
			target->candidates.push_back(observation);
		}
		else
		{
			HandCluster cluster;
			cluster.candidates.push_back(observation);
			cluster.palmWorld= palm; // best-weighted member (observations are sorted)
			cluster.anchorCameraPos= glm::vec3(observation.camera->markerFromCamera[3]);
			cluster.bestWeight= observation.weight;
			clusters.push_back(cluster);
		}
	}

	// Keep the two best clusters (there are only two physical hands)
	std::sort(clusters.begin(), clusters.end(),
			  [](const HandCluster& a, const HandCluster& b) { return a.bestWeight > b.bestWeight; });
	if (clusters.size() > 2)
		clusters.resize(2);

	// Assign sides to clusters by joint affinity (votes + temporal continuity)
	m_dominantCamera[0]= -1;
	m_dominantCamera[1]= -1;
	if (clusters.size() == 2)
	{
		const float assignLR= sideAffinity(clusters[0], eHandSide::Left) + sideAffinity(clusters[1], eHandSide::Right);
		const float assignRL= sideAffinity(clusters[0], eHandSide::Right) + sideAffinity(clusters[1], eHandSide::Left);

		const int firstSide= assignLR >= assignRL ? (int)eHandSide::Left : (int)eHandSide::Right;
		const int secondSide= 1 - firstSide;
		fuseCluster((eHandSide)firstSide, clusters[0], outFused.hands[firstSide], outFused.poses[firstSide]);
		fuseCluster((eHandSide)secondSide, clusters[1], outFused.hands[secondSide], outFused.poses[secondSide]);
	}
	else if (clusters.size() == 1)
	{
		const eHandSide side=
			sideAffinity(clusters[0], eHandSide::Left) >= sideAffinity(clusters[0], eHandSide::Right)
				? eHandSide::Left
				: eHandSide::Right;
		fuseCluster(side, clusters[0], outFused.hands[(int)side], outFused.poses[(int)side]);
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

	// Weighted blend of the palm transform + finger angles across cameras.
	// Poses/angles compose - unlike raw landmark blending, disagreeing
	// articulation degrades gracefully instead of distorting bones.
	if (candidates.size() >= 2)
	{
		float weightSum= 0.f;
		glm::vec3 blendedPosition(0.f);
		glm::quat blendedQuat(0.f, 0.f, 0.f, 0.f);
		std::array<FingerAngles, FINGER_COUNT> blendedAngles{};

		const glm::quat referenceQuat= best.pose->palmOrientationWorld;
		float maxPresence= 0.f;
		for (const HandCandidate& candidate : candidates)
		{
			const float weight= candidate.weight;
			weightSum+= weight;
			maxPresence= std::max(maxPresence, candidate.pose->presence);

			blendedPosition+= candidate.pose->palmPositionWorld * weight;

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
			outPose.palmPositionWorld= blendedPosition / weightSum;
			outPose.palmOrientationWorld= glm::normalize(blendedQuat);
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				outPose.fingers[finger].lateral= blendedAngles[finger].lateral / weightSum;
				outPose.fingers[finger].proximal= blendedAngles[finger].proximal / weightSum;
				outPose.fingers[finger].intermediate= blendedAngles[finger].intermediate / weightSum;
				outPose.fingers[finger].distal= blendedAngles[finger].distal / weightSum;
			}
			outPose.presence= maxPresence;
		}

		// A two-camera observation of one physical hand also triangulates the
		// true hand scale
		updateStereoScale(cluster);
	}

	outPose.tracked= true;
	outPose.hasWorldPose= true;
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
