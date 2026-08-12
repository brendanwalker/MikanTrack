#include "HandTrackingPipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "glm/geometric.hpp"

#include "Logger.h"

static float boxIou(
	const glm::vec2& aMin, const glm::vec2& aMax,
	const glm::vec2& bMin, const glm::vec2& bMax)
{
	const float interW= std::max(0.f, std::min(aMax.x, bMax.x) - std::max(aMin.x, bMin.x));
	const float interH= std::max(0.f, std::min(aMax.y, bMax.y) - std::max(aMin.y, bMin.y));
	const float interArea= interW * interH;

	const float areaA= std::max(0.f, aMax.x - aMin.x) * std::max(0.f, aMax.y - aMin.y);
	const float areaB= std::max(0.f, bMax.x - bMin.x) * std::max(0.f, bMax.y - bMin.y);
	const float unionArea= areaA + areaB - interArea;

	return unionArea > 0.f ? interArea / unionArea : 0.f;
}

static bool pointInBox(const glm::vec2& p, const glm::vec2& boxMin, const glm::vec2& boxMax)
{
	return p.x >= boxMin.x && p.x <= boxMax.x && p.y >= boxMin.y && p.y <= boxMax.y;
}

bool HandTrackingPipeline::startup(const HandTrackingPipelineConfig& config)
{
	m_config= config;

	const std::string& dir= m_config.modelDir;
	const std::string& ep= m_config.preferredEp;

	// Palm + hand landmark models are required
	if (!m_palmDetector.load(dir + "/palm_detection.onnx", ep))
		return false;
	if (!m_handLandmarkModel.load(dir + "/hand_landmark.onnx", ep))
		return false;

	for (HandSlot& slot : m_slots)
		slot.deactivate();
	m_frameIndex= -1;
	m_framesSinceDetector= 0;

	MIKAN_MT_LOG_INFO("HandTrackingPipeline::startup")
		<< "Pipeline ready (EP: " << getActiveExecutionProvider() << ")";
	return true;
}

void HandTrackingPipeline::setConfig(const HandTrackingPipelineConfig& config)
{
	if (config.preferredEp != m_config.preferredEp || config.modelDir != m_config.modelDir)
	{
		MIKAN_MT_LOG_WARNING("HandTrackingPipeline::setConfig")
			<< "preferredEp/modelDir changes require a pipeline restart - ignored";
	}

	// keep the session-bound settings, take everything else
	HandTrackingPipelineConfig newConfig= config;
	newConfig.preferredEp= m_config.preferredEp;
	newConfig.modelDir= m_config.modelDir;
	m_config= newConfig;
}

const char* HandTrackingPipeline::getActiveExecutionProvider() const
{
	return m_palmDetector.isLoaded() ? m_palmDetector.activeEp() : "none";
}

int HandTrackingPipeline::countActiveSlots() const
{
	int count= 0;
	for (const HandSlot& slot : m_slots)
	{
		if (slot.active)
			count++;
	}
	return count;
}

void HandTrackingPipeline::process(const cv::Mat& bgrFrame, TrackingFrameResult& outResult)
{
	const auto startTime= std::chrono::steady_clock::now();

	outResult.palmDetections.clear();
	for (TrackedHand& hand : outResult.hands)
		hand= TrackedHand();
	for (HandPose& pose : outResult.poses)
		pose= HandPose();

	if (bgrFrame.empty() || !m_palmDetector.isLoaded())
		return;

	m_frameIndex++;

	// Reacquisition window: a drop in active slots arms the relaxed detector
	// threshold for a short window (recall for a hand we KNOW just vanished;
	// strict precision otherwise). Drift-guard runs with both slots active
	// always stay strict, so a fully-tracked frame gains no false positives.
	const int activeSlots= countActiveSlots();
	if (activeSlots < m_lastActiveSlotCount)
		m_lastSlotLossFrame= m_frameIndex;
	m_lastActiveSlotCount= activeSlots;

	const bool bReacquiring= activeSlots < 2 && m_lastSlotLossFrame >= 0 &&
		m_frameIndex - m_lastSlotLossFrame <= m_config.relaxedDetectorWindowFrames;
	m_palmDetector.setScoreThreshold(
		bReacquiring ? m_config.palmScoreThresholdRelaxed : m_config.palmScoreThreshold);

	// Cross-camera seeds go BEFORE the detector: another camera's tracked hand
	// projected into this image is better evidence than a blind detection, and
	// running second meant the detector claimed the free slot first - usually
	// with a false positive that died two frames later - leaving the seed with
	// nowhere to go (measured: 74 of 75 seeds during real gaps hit
	// skippedNoFreeSlot). A wrong seed costs one frame: runHandLandmarkStage
	// drops a speculative seed the same frame it fails the presence threshold.
	applySearchHints();

	// Recount AFTER seeding so a slot a seed just claimed suppresses the
	// detector run it would have made redundant. The count above stays where
	// it is: a slot LOSS still has to arm the relaxed reacquisition window,
	// whether or not a seed refills it in the same frame.
	const bool runDetector=
		countActiveSlots() < 2 ||
		m_framesSinceDetector >= std::max(m_config.detectorIntervalFrames, 1);
	if (runDetector)
		runPalmDetectionStage(bgrFrame, outResult);
	else
		m_framesSinceDetector++;

	runHandLandmarkStage(bgrFrame);

	killDuplicateSlots();

	resolveHandedness(bgrFrame.cols);
	publishHands(outResult);

	// slot ROI debug boxes ride along with the raw detector boxes
	for (const HandSlot& slot : m_slots)
	{
		if (slot.active)
			outResult.palmDetections.push_back(slot.lastRoiDebug);
	}

	const auto endTime= std::chrono::steady_clock::now();
	outResult.inferenceMs=
		std::chrono::duration<float, std::milli>(endTime - startTime).count();
}

void HandTrackingPipeline::runPalmDetectionStage(const cv::Mat& bgrFrame, TrackingFrameResult& outResult)
{
	m_palmDetector.detect(bgrFrame, m_palmDetections);
	m_framesSinceDetector= 0;

	for (const PalmDetection& detection : m_palmDetections)
		outResult.palmDetections.push_back(PalmDetector::toDebugBox(detection));

	// Match detections against active slots (dedupe), spawn new slots from
	// unmatched ones. m_palmDetections is sorted score-descending.
	for (const PalmDetection& detection : m_palmDetections)
	{
		bool matchesExistingSlot= false;
		for (const HandSlot& slot : m_slots)
		{
			if (!slot.active)
				continue;

			// palm boxes are smaller than the slot's hand box, so also treat
			// a detection whose wrist lands inside the slot box as a match
			if (boxIou(detection.boxMin, detection.boxMax, slot.roiBoxMin, slot.roiBoxMax) >
					m_config.slotDedupeIouThreshold ||
				pointInBox(detection.keypoints[0], slot.roiBoxMin, slot.roiBoxMax))
			{
				matchesExistingSlot= true;
				break;
			}
		}
		if (matchesExistingSlot)
			continue;

		for (HandSlot& slot : m_slots)
		{
			if (slot.active)
				continue;

			slot.deactivate();
			slot.active= true;
			slot.hasPendingDetection= true;
			slot.pendingDetection= detection;
			slot.roiBoxMin= detection.boxMin;
			slot.roiBoxMax= detection.boxMax;
			break;
		}
	}
}

// Synthesizes a PalmDetection at a hinted image location. Only the wrist->
// middle-MCP keypoint pair (crop rotation) and the keypoint bounding box
// (crop extent) feed the landmark model's crop math, so approximate anatomy
// is fine - the first landmark pass corrects the ROI or rejects it.
static PalmDetection synthesizeDetectionFromHint(const HandSearchHint& hint)
{
	const glm::vec2 c= hint.centerPx;
	const glm::vec2 d= hint.dirPx;
	const glm::vec2 p(-d.y, d.x);
	const float s= hint.palmSizePx;

	PalmDetection detection;
	detection.keypoints[0]= c - d * (0.5f * s);                   // wrist
	detection.keypoints[1]= c + d * (0.5f * s) + p * (0.35f * s); // index MCP
	detection.keypoints[2]= c + d * (0.5f * s);                   // middle MCP
	detection.keypoints[3]= c + d * (0.45f * s) - p * (0.3f * s); // ring MCP
	detection.keypoints[4]= c + d * (0.3f * s) - p * (0.45f * s); // pinky MCP
	detection.keypoints[5]= c - d * (0.3f * s) + p * (0.4f * s);  // thumb CMC
	detection.keypoints[6]= c + p * (0.5f * s);                   // thumb MCP
	detection.boxMin= c - glm::vec2(0.55f * s);
	detection.boxMax= c + glm::vec2(0.55f * s);
	detection.score= 0.5f;
	detection.rotationRadians= PalmDetector::computeRotation(detection.keypoints[0], detection.keypoints[2]);
	return detection;
}

bool HandTrackingPipeline::isSeedRedundant(const glm::vec2& centerPx, const std::vector<HandBox>& activeBoxes)
{
	// Margin for the projection error the hint carries in from the other
	// camera. Wide enough to cover it, narrow enough that a genuine second
	// hand still gets seeded; a duplicate that slips through is caught by the
	// slotDuplicate guard downstream.
	constexpr float k_inflation= 1.25f;

	for (const HandBox& box : activeBoxes)
	{
		const glm::vec2 center= (box.min + box.max) * 0.5f;
		const glm::vec2 halfExtent= (box.max - box.min) * (0.5f * k_inflation);
		if (pointInBox(centerPx, center - halfExtent, center + halfExtent))
			return true;
	}
	return false;
}

void HandTrackingPipeline::applySearchHints()
{
	for (const HandSearchHint& hint : m_searchHints)
	{
		if (hint.palmSizePx < 8.f)
		{
			m_seedStats.skippedTooSmall++;
			continue;
		}

		m_activeBoxes.clear();
		for (const HandSlot& slot : m_slots)
		{
			if (slot.active)
				m_activeBoxes.push_back({slot.roiBoxMin, slot.roiBoxMax});
		}
		if (isSeedRedundant(hint.centerPx, m_activeBoxes))
		{
			m_seedStats.skippedRedundant++;
			continue;
		}

		bool bSeeded= false;
		for (HandSlot& slot : m_slots)
		{
			if (slot.active)
				continue;

			slot.deactivate();
			slot.active= true;
			slot.hasPendingDetection= true;
			slot.bSeededFromHint= true;
			slot.pendingDetection= synthesizeDetectionFromHint(hint);
			slot.roiBoxMin= slot.pendingDetection.boxMin;
			slot.roiBoxMax= slot.pendingDetection.boxMax;
			bSeeded= true;
			break;
		}

		if (bSeeded)
			m_seedStats.applied++;
		else
			m_seedStats.skippedNoFreeSlot++;
	}
	m_searchHints.clear();
}

void HandTrackingPipeline::runHandLandmarkStage(const cv::Mat& bgrFrame)
{
	for (HandSlot& slot : m_slots)
	{
		if (!slot.active)
			continue;

		// New slots use the palm detection; tracked slots derive the ROI from
		// last frame's landmarks (detector-free tracking)
		const HandRoi roi=
			slot.hasPendingDetection
			? HandRoi::fromPalmDetection(slot.pendingDetection)
			: HandRoi::fromLandmarks(slot.imagePoints);

		const bool bSpeculativeSeed= slot.hasPendingDetection && slot.bSeededFromHint;

		m_handLandmarkModel.estimate(bgrFrame, roi, m_handResult);
		if (!m_handResult.valid ||
			(bSpeculativeSeed && m_handResult.confidence < m_config.handPresenceThreshold))
		{
			if (bSpeculativeSeed)
				m_seedStats.rejectedByModel++;
			slot.deactivate();
			continue;
		}
		if (bSpeculativeSeed)
			m_seedStats.accepted++;
		slot.bSeededFromHint= false;

		slot.hasPendingDetection= false;
		slot.framesSinceDetection++;
		slot.presence= m_handResult.confidence;
		slot.imagePoints= m_handResult.imagePoints;
		slot.modelPoints= m_handResult.modelPoints;
		slot.roiBoxMin= m_handResult.handBoxMin;
		slot.roiBoxMax= m_handResult.handBoxMax;
		slot.lastRoiDebug= m_handResult.usedRoi;

		// smooth the raw handedness score a little for stability
		slot.handednessScore=
			slot.sideInitialized
			? slot.handednessScore * 0.7f + m_handResult.handedness * 0.3f
			: m_handResult.handedness;

		if (slot.presence < m_config.handPresenceThreshold)
		{
			slot.lowPresenceFrames++;
			if (slot.lowPresenceFrames >= m_config.handPresenceLostFrames)
				slot.deactivate();
		}
		else
		{
			slot.lowPresenceFrames= 0;
		}
	}
}

void HandTrackingPipeline::killDuplicateSlots()
{
	// After hands overlap (a clap), both slots' landmark ROIs can converge on
	// the SAME physical hand and track it indefinitely - the pipeline thinks
	// both hands are accounted for, so the separated hand has no free slot to
	// be detected into. Sustained near-identical hand boxes mean one physical
	// hand: kill the lower-presence duplicate so the detector (which runs
	// every frame while a slot is free) can re-acquire the other hand.
	HandSlot& a= m_slots[0];
	HandSlot& b= m_slots[1];
	if (!a.active || !b.active)
	{
		m_duplicateOverlapFrames= 0;
		return;
	}

	const float iou= boxIou(a.roiBoxMin, a.roiBoxMax, b.roiBoxMin, b.roiBoxMax);
	if (iou <= m_config.slotDuplicateIouThreshold)
	{
		m_duplicateOverlapFrames= 0;
		return;
	}

	m_duplicateOverlapFrames++;
	if (m_duplicateOverlapFrames >= m_config.slotDuplicateKillFrames)
	{
		HandSlot& loser= a.presence <= b.presence ? a : b;
		loser.deactivate();
		m_duplicateOverlapFrames= 0;
	}
}

void HandTrackingPipeline::resolveHandedness(int frameWidth)
{
	(void)frameWidth;

	for (HandSlot& slot : m_slots)
	{
		if (!slot.active)
			continue;

		// model handedness: per mp_handpose.py, raw score 1 == right hand in
		// (mirrored selfie) model terms; flip for unmirrored feeds
		float rightProb= slot.handednessScore;
		if (m_config.flipHandedness)
			rightProb= 1.f - rightProb;
		slot.rightProb= rightProb;
		eHandSide candidate= rightProb > 0.5f ? eHandSide::Right : eHandSide::Left;

		// temporal stickiness: keep the assigned side unless contradicted for
		// handednessSwitchFrames consecutive frames
		if (!slot.sideInitialized)
		{
			slot.side= candidate;
			slot.sideInitialized= true;
			slot.sideContradictionFrames= 0;
		}
		else if (candidate != slot.side)
		{
			slot.sideContradictionFrames++;
			if (slot.sideContradictionFrames >= m_config.handednessSwitchFrames)
			{
				slot.side= candidate;
				slot.sideContradictionFrames= 0;
			}
		}
		else
		{
			slot.sideContradictionFrames= 0;
		}
	}
}

int HandTrackingPipeline::preferredSlotOrder(float rightProbA, float presenceA, float rightProbB,
											 float presenceB)
{
	// The loser of a side collision is displaced to the OTHER side, so this
	// decides both hands' labels. Order by classifier DECISIVENESS, not
	// presence: presence measures tracking quality and says nothing about
	// which hand is which. (Live 2026-08-01: both slots scored "right"
	// (0.98 and 0.64) at presence 0.978 vs 0.991 - a 1.3% presence
	// difference, pure noise, displaced the decisively-right hand to Left
	// and stuck there for seconds.)
	const float decisivenessA= fabsf(rightProbA - 0.5f);
	const float decisivenessB= fabsf(rightProbB - 0.5f);
	if (decisivenessB != decisivenessA)
		return decisivenessB > decisivenessA ? 1 : 0;

	return presenceB > presenceA ? 1 : 0;
}

void HandTrackingPipeline::publishHands(TrackingFrameResult& outResult)
{
	// The more decisive slot claims its side first; at most one hand per side
	int order[2]= {0, 1};
	if (preferredSlotOrder(m_slots[0].rightProb, m_slots[0].presence, m_slots[1].rightProb,
						   m_slots[1].presence) == 1)
		std::swap(order[0], order[1]);

	for (int i= 0; i < 2; ++i)
	{
		const int slotIndex= order[i];
		const HandSlot& slot= m_slots[slotIndex];
		if (!slot.active || !slot.sideInitialized)
			continue;

		eHandSide side= slot.side;
		if (outResult.hands[(int)side].tracked)
		{
			// side collision: lower-presence slot takes the free side
			side= (side == eHandSide::Left) ? eHandSide::Right : eHandSide::Left;
			if (outResult.hands[(int)side].tracked)
				continue;
		}

		TrackedHand& hand= outResult.hands[(int)side];
		hand.tracked= true;
		hand.side= side;
		hand.slotId= slotIndex;
		hand.presence= slot.presence;
		hand.handednessScore= slot.handednessScore;
		hand.rightProb= slot.rightProb;
		hand.imagePoints= slot.imagePoints;
		hand.modelPoints= slot.modelPoints;
		hand.hasCameraSpace= false;
		hand.hasWorldSpace= false;
	}
}
