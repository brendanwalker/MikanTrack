#include "HandFusion.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

#include "Logger.h"

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

void HandFusion::fuse(const std::vector<const CameraFrameResult*>& candidates, double nowTimestampMs,
					  TrackingFrameResult& outFused)
{
	outFused= TrackingFrameResult();
	outFused.timestampMs= nowTimestampMs;

	// Carry frame bookkeeping from the freshest contributing camera
	const CameraFrameResult* freshest= nullptr;

	// Gather per-side hand candidates from fresh, world-tracked camera results
	std::vector<HandCandidate> sideCandidates[2];
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

			sideCandidates[sideIndex].push_back(candidate);
		}
	}

	if (freshest != nullptr)
	{
		outFused.frameIndex= freshest->result.frameIndex;
		outFused.captureFps= freshest->result.captureFps;
		outFused.inferenceMs= freshest->result.inferenceMs;
	}

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_dominantCamera[sideIndex]= -1;
		if (!sideCandidates[sideIndex].empty())
		{
			fuseSide((eHandSide)sideIndex, sideCandidates[sideIndex],
					 outFused.hands[sideIndex], outFused.arms[sideIndex]);
		}
	}

	applySmoothing(outFused);
}

void HandFusion::fuseSide(eHandSide side, std::vector<HandCandidate>& candidates, TrackedHand& outHand,
						  TrackedArm& outArm)
{
	// Correspondence sanity gate: two cameras whose world wrists are far apart
	// can't be seeing the same physical hand (cross-camera handedness
	// disagreement) - keep only the best-scoring candidate in that case
	if (candidates.size() >= 2)
	{
		bool bConflict= false;
		for (size_t a= 0; a < candidates.size() && !bConflict; ++a)
		{
			for (size_t b= a + 1; b < candidates.size() && !bConflict; ++b)
			{
				const glm::vec3& wristA= candidates[a].hand->worldPoints[(int)eHandLandmark::WRIST];
				const glm::vec3& wristB= candidates[b].hand->worldPoints[(int)eHandLandmark::WRIST];
				bConflict= glm::length(wristA - wristB) > m_config.wristMatchMaxDistM;
			}
		}

		if (bConflict)
		{
			auto bestIt= std::max_element(
				candidates.begin(), candidates.end(),
				[](const HandCandidate& a, const HandCandidate& b) { return a.baseScore < b.baseScore; });
			HandCandidate best= *bestIt;
			candidates.clear();
			candidates.push_back(best);
		}
	}

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
