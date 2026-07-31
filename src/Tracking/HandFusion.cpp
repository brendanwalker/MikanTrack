#include "HandFusion.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

#include "Logger.h"

// Temporal side-continuity: full-strength attraction within this distance of
// the side's last fused wrist, fading to nothing at 2x
static constexpr float kTemporalFullDistM= 0.15f;
// Weighting of temporal continuity vs classifier votes in side assignment
// (continuity dominates - a tracked hand must not swap sides because one
// camera's classifier flickered)
static constexpr float kTemporalWeight= 2.f;
// Stereo scale triangulation guards
static constexpr float kMinRayAngleCos= 0.985f; // rays closer than ~10 deg apart are degenerate
static constexpr float kScaleCorrectionMin= 0.7f;
static constexpr float kScaleCorrectionMax= 1.4f;

void HandFusion::configure(const HandFusionConfig& config)
{
	m_config= config;

	m_filterBank.configure(config.smoothingMinCutoff, config.smoothingBeta, 1.f);
	m_filterBank.resetAll();
	for (OneEuroFilterVec3& filter : m_elbowWorldFilters)
	{
		filter.configure(config.smoothingMinCutoff, config.smoothingBeta, 1.f);
		filter.reset();
	}
	m_lastTimestampMs= -1.0;
	m_bSideWasTracked[0]= false;
	m_bSideWasTracked[1]= false;
	m_bLastFusedWristValid[0]= false;
	m_bLastFusedWristValid[1]= false;
	m_stereoScaleCorrection= 1.f;
	m_bStereoScaleFresh= false;
}

glm::vec3 HandFusion::palmNormalWorld(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& worldPoints)
{
	const glm::vec3& wrist= worldPoints[(int)eHandLandmark::WRIST];
	const glm::vec3 toIndex= worldPoints[(int)eHandLandmark::INDEX_MCP] - wrist;
	const glm::vec3 toPinky= worldPoints[(int)eHandLandmark::PINKY_MCP] - wrist;

	return glm::cross(toIndex, toPinky);
}

float HandFusion::visibilityFactor(const glm::vec3& palmNormal, const glm::vec3& landmarkWorld,
								   const glm::vec3& cameraPosWorld)
{
	const float normalLength= glm::length(palmNormal);
	const glm::vec3 viewRay= landmarkWorld - cameraPosWorld;
	const float viewLength= glm::length(viewRay);
	if (normalLength < 1e-6f || viewLength < 1e-6f)
		return 0.05f;

	// |cos| between the palm plane normal and the view ray: 1 = palm face-on
	// to the camera (best case), 0 = palm edge-on (the clap failure mode)
	const float faceOn= fabsf(glm::dot(palmNormal, viewRay)) / (normalLength * viewLength);

	return 0.05f + faceOn;
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
	if (m_bLastFusedWristValid[(int)side])
	{
		const float dist= glm::length(cluster.wristWorld - m_lastFusedWrist[(int)side]);
		temporalScore= kTemporalWeight *
			std::clamp(1.f - (dist - kTemporalFullDistM) / kTemporalFullDistM, -1.f, 1.f);
	}

	return voteScore + temporalScore;
}

void HandFusion::updateStereoScale(const HandCluster& cluster)
{
	// Need two observations of the same wrist from two different cameras
	if (cluster.candidates.size() < 2)
		return;

	const HandCandidate& a= cluster.candidates[0];
	const HandCandidate& b= cluster.candidates[1];
	if (a.camera->cameraIndex == b.camera->cameraIndex)
		return;

	const glm::vec3 cameraPosA= glm::vec3(a.camera->markerFromCamera[3]);
	const glm::vec3 cameraPosB= glm::vec3(b.camera->markerFromCamera[3]);
	const glm::vec3 wristA= a.hand->worldPoints[(int)eHandLandmark::WRIST];
	const glm::vec3 wristB= b.hand->worldPoints[(int)eHandLandmark::WRIST];

	// Each camera's wrist estimate lies on the view ray through the true
	// wrist; the estimated depth scales linearly with the configured hand
	// scale, but the ray DIRECTION doesn't. Two-ray midpoint triangulation
	// recovers the true depth independent of the configured scale.
	const float depthA= glm::length(wristA - cameraPosA);
	const float depthB= glm::length(wristB - cameraPosB);
	if (depthA < 1e-4f || depthB < 1e-4f)
		return;

	const glm::vec3 rayA= (wristA - cameraPosA) / depthA;
	const glm::vec3 rayB= (wristB - cameraPosB) / depthB;

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

	// Gather ALL hand observations from fresh, world-tracked camera results.
	// The per-camera side label is only a VOTE here - see clustering below.
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
			if (!hand.tracked || !hand.hasWorldSpace || hand.presence < m_config.presenceThreshold)
				continue;

			HandCandidate candidate;
			candidate.camera= camera;
			candidate.hand= &hand;
			candidate.arm= &camera->result.arms[sideIndex];

			// Vote weight: presence x how decisive the classifier was
			candidate.sideVoteWeight= hand.presence * std::max(fabsf(hand.handednessScore - 0.5f) * 2.f, 0.2f);

			// Per-landmark visibility scores
			const glm::vec3 cameraPos= glm::vec3(camera->markerFromCamera[3]);
			const glm::vec3 palmNormal= palmNormalWorld(hand.worldPoints);
			float scoreSum= 0.f;
			for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			{
				const float score= hand.presence * visibilityFactor(palmNormal, hand.worldPoints[i], cameraPos);
				candidate.landmarkScore[i]= score;
				scoreSum+= score;
			}
			candidate.baseScore= scoreSum / (float)HAND_LANDMARK_COUNT;

			observations.push_back(candidate);
		}
	}

	if (freshest != nullptr)
	{
		outFused.frameIndex= freshest->result.frameIndex;
		outFused.captureFps= freshest->result.captureFps;
		outFused.inferenceMs= freshest->result.inferenceMs;
	}

	// Cluster observations by world wrist proximity: one cluster = one
	// physical hand, regardless of what each camera called it
	std::vector<HandCluster> clusters;
	std::sort(observations.begin(), observations.end(),
			  [](const HandCandidate& a, const HandCandidate& b) { return a.baseScore > b.baseScore; });
	for (const HandCandidate& observation : observations)
	{
		const glm::vec3& wrist= observation.hand->worldPoints[(int)eHandLandmark::WRIST];

		HandCluster* target= nullptr;
		for (HandCluster& cluster : clusters)
		{
			// One observation per camera per cluster (a camera can't see the
			// same physical hand twice)
			bool bCameraAlreadyInCluster= false;
			for (const HandCandidate& member : cluster.candidates)
				bCameraAlreadyInCluster|= member.camera->cameraIndex == observation.camera->cameraIndex;

			if (!bCameraAlreadyInCluster && glm::length(wrist - cluster.wristWorld) <= m_config.wristMatchMaxDistM)
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
			cluster.wristWorld= wrist; // best-scoring member (observations are sorted)
			cluster.bestScore= observation.baseScore;
			clusters.push_back(cluster);
		}
	}

	// Keep the two best clusters (there are only two physical hands)
	std::sort(clusters.begin(), clusters.end(),
			  [](const HandCluster& a, const HandCluster& b) { return a.bestScore > b.bestScore; });
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
		fuseCluster((eHandSide)firstSide, clusters[0], outFused.hands[firstSide], outFused.arms[firstSide]);
		fuseCluster((eHandSide)secondSide, clusters[1], outFused.hands[secondSide], outFused.arms[secondSide]);
	}
	else if (clusters.size() == 1)
	{
		const eHandSide side=
			sideAffinity(clusters[0], eHandSide::Left) >= sideAffinity(clusters[0], eHandSide::Right)
				? eHandSide::Left
				: eHandSide::Right;
		fuseCluster(side, clusters[0], outFused.hands[(int)side], outFused.arms[(int)side]);
	}

	// Update the temporal side prior from this frame's assignments
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (outFused.hands[sideIndex].tracked)
		{
			m_lastFusedWrist[sideIndex]= outFused.hands[sideIndex].worldPoints[(int)eHandLandmark::WRIST];
			m_bLastFusedWristValid[sideIndex]= true;
		}
		// (keep the last position when untracked - it decays naturally via distance)
	}

	applySmoothing(outFused);
}

void HandFusion::fuseCluster(eHandSide side, HandCluster& cluster, TrackedHand& outHand, TrackedArm& outArm)
{
	std::vector<HandCandidate>& candidates= cluster.candidates;

	// Best candidate provides the advisory (image/model/camera space) fields
	const HandCandidate& best= *std::max_element(
		candidates.begin(), candidates.end(),
		[](const HandCandidate& a, const HandCandidate& b) { return a.baseScore < b.baseScore; });
	m_dominantCamera[(int)side]= best.camera->cameraIndex;

	outHand= *best.hand;
	outHand.side= side;

	// Per-landmark softmax blend of the world points across cameras
	if (candidates.size() >= 2)
	{
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			// subtract the max score before exponentiation for numeric stability
			float maxScore= -1e9f;
			for (const HandCandidate& candidate : candidates)
				maxScore= std::max(maxScore, candidate.landmarkScore[i]);

			glm::vec3 blended(0.f);
			float weightSum= 0.f;
			for (const HandCandidate& candidate : candidates)
			{
				const float weight= expf(m_config.softmaxTemperature * (candidate.landmarkScore[i] - maxScore));
				blended+= candidate.hand->worldPoints[i] * weight;
				weightSum+= weight;
			}

			outHand.worldPoints[i]= blended / weightSum;
		}

		float maxPresence= 0.f;
		for (const HandCandidate& candidate : candidates)
			maxPresence= std::max(maxPresence, candidate.hand->presence);
		outHand.presence= maxPresence;

		// A two-camera observation of one physical hand also triangulates the
		// true hand scale
		updateStereoScale(cluster);
	}

	outHand.hasWorldSpace= true;
	outHand.tracked= true;

	// Arms: the elbow is a coarse geometric estimate - take the best camera's
	// whole arm rather than blending two clamped guesses
	if (best.arm != nullptr && best.arm->valid)
	{
		outArm= *best.arm;
	}
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
		TrackedHand& hand= ioFused.hands[sideIndex];
		TrackedArm& arm= ioFused.arms[sideIndex];

		if (hand.tracked && hand.hasWorldSpace)
		{
			// fresh acquisition: drop stale filter state
			if (!m_bSideWasTracked[sideIndex])
			{
				m_filterBank.resetSide((eHandSide)sideIndex);
				m_elbowWorldFilters[sideIndex].reset();
			}

			for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			{
				hand.worldPoints[i]=
					m_filterBank.landmarkFilter((eHandSide)sideIndex, i).filter(hand.worldPoints[i], dtSeconds);
			}

			if (arm.valid && arm.hasWorldSpace)
			{
				arm.elbowWorld= m_elbowWorldFilters[sideIndex].filter(arm.elbowWorld, dtSeconds);
				arm.wristWorld= hand.worldPoints[(int)eHandLandmark::WRIST];
			}
		}

		m_bSideWasTracked[sideIndex]= hand.tracked && hand.hasWorldSpace;
	}
}
