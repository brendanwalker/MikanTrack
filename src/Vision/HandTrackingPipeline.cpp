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
	for (TrackedArm& arm : outResult.arms)
		arm= TrackedArm();

	if (bgrFrame.empty() || !m_palmDetector.isLoaded())
		return;

	m_frameIndex++;

	// Palm detector: only when a slot is free, or periodically as drift guard
	const bool runDetector=
		countActiveSlots() < 2 ||
		m_framesSinceDetector >= std::max(m_config.detectorIntervalFrames, 1);
	if (runDetector)
		runPalmDetectionStage(bgrFrame, outResult);
	else
		m_framesSinceDetector++;

	runHandLandmarkStage(bgrFrame);

	resolveHandedness(bgrFrame.cols);
	publishHands(outResult);
	publishArms(outResult);

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

		m_handLandmarkModel.estimate(bgrFrame, roi, m_handResult);
		if (!m_handResult.valid)
		{
			slot.deactivate();
			continue;
		}

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

void HandTrackingPipeline::publishHands(TrackingFrameResult& outResult)
{
	// higher-presence slot claims its side first; at most one hand per side
	int order[2]= {0, 1};
	if (m_slots[1].presence > m_slots[0].presence)
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
		hand.imagePoints= slot.imagePoints;
		hand.modelPoints= slot.modelPoints;
		hand.hasCameraSpace= false;
		hand.hasWorldSpace= false;
	}
}

void HandTrackingPipeline::publishArms(TrackingFrameResult& outResult)
{
	// Elbow/forearm estimation from hand geometry: extend up the forearm
	// direction from the palm orientation. When extrinsics are calibrated,
	// LandmarkTo3D::refineFallbackArms recomputes this in world space with an
	// anatomical forearm length and a table-plane clamp.
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		TrackedArm& arm= outResult.arms[sideIndex];
		const TrackedHand& hand= outResult.hands[sideIndex];

		if (!hand.tracked)
			continue;

		const glm::vec2 wrist= glm::vec2(hand.imagePoints[(int)eHandLandmark::WRIST]);
		const glm::vec2 mcpCentroid=
			(glm::vec2(hand.imagePoints[(int)eHandLandmark::INDEX_MCP]) +
			 glm::vec2(hand.imagePoints[(int)eHandLandmark::MIDDLE_MCP]) +
			 glm::vec2(hand.imagePoints[(int)eHandLandmark::RING_MCP]) +
			 glm::vec2(hand.imagePoints[(int)eHandLandmark::PINKY_MCP])) * 0.25f;

		const glm::vec2 toWrist= wrist - mcpCentroid;
		const float toWristLength= glm::length(toWrist);
		if (toWristLength > 1e-3f)
		{
			const glm::vec2 dir= toWrist / toWristLength;
			const float handLength=
				glm::length(wrist - glm::vec2(hand.imagePoints[(int)eHandLandmark::MIDDLE_MCP]));

			arm.valid= true;
			arm.fromFallback= true;
			arm.confidence= 0.1f;
			arm.wristPixel= wrist;
			arm.elbowPixel= wrist + dir * (m_config.armFallbackElbowScale * handLength);
		}
	}
}
