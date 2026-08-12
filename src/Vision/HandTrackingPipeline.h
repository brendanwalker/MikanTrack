#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "HandLandmarkModel.h"
#include "PalmDetector.h"
#include "TrackingTypes.h"

struct HandTrackingPipelineConfig
{
	// "directml" | "cpu" (session recreation requires a pipeline restart, so
	// setConfig ignores changes to this and modelDir)
	std::string preferredEp= "directml";
	// model directory relative to the working directory (exe dir)
	std::string modelDir= "models";

	// MediaPipe's handedness classifier assumes a mirrored/selfie view; with
	// an unmirrored (rear/webcam) feed the label must be flipped
	bool flipHandedness= true;

	// run the palm detector at least every N frames even with 2 active slots
	// (drift guard); it always runs when fewer than 2 slots are active
	int detectorIntervalFrames= 30;

	float palmScoreThreshold= 0.5f;     // mp_palmdet.py default
	// Relaxed detection cutoff used while REACQUIRING: a slot is free AND a
	// slot was lost within relaxedDetectorWindowFrames. Recall when a known
	// hand vanished, precision otherwise - safe because the fusion gates
	// (clustering, residual veto, temporal/spatial priors) catch the extra
	// false positives a lower cutoff lets through.
	float palmScoreThresholdRelaxed= 0.25f;
	int relaxedDetectorWindowFrames= 90; // ~1.5s at 60fps
	float palmNmsIouThreshold= 0.3f;    // mp_palmdet.py default
	float handPresenceThreshold= 0.5f;  // slot keepalive threshold
	int handPresenceLostFrames= 2;      // consecutive low-presence frames before slot deactivates
	float slotDedupeIouThreshold= 0.3f; // palm detection vs active slot IoU
	int handednessSwitchFrames= 15;     // consecutive contradictions before a slot flips side

	// Duplicate-slot guard: after a clap/overlap both slots can end up
	// tracking the SAME physical hand (their landmark ROIs converge), which
	// blocks the free slot the separated hand needs. Sustained near-identical
	// hand boxes -> kill the lower-presence slot.
	float slotDuplicateIouThreshold= 0.6f;
	int slotDuplicateKillFrames= 6;
};

// Cross-camera search hint: another camera tracks a hand this camera lost;
// its fused world pose projected into this camera's image seeds a direct
// landmark-model attempt (no palm detection needed)
struct HandSearchHint
{
	glm::vec2 centerPx{0.f};   // projected palm center
	glm::vec2 dirPx{0.f, -1.f}; // projected wrist->fingers direction (unit)
	float palmSizePx= 0.f;     // projected wrist->middle-MCP distance
};

// One camera's landmark ROI box, for the seed redundancy test
struct HandBox
{
	glm::vec2 min{0.f};
	glm::vec2 max{0.f};
};

// Where cross-camera seeds go, cumulative since startup. Seeding runs UPSTREAM
// of the tracking recording (recordings capture post-model results), so it
// cannot be replayed offline and a counter is the only way to see which gate
// is consuming the hints.
struct HandSeedStats
{
	int candidates= 0;        // fused hands the vision thread tried to seed here
	int skippedProjection= 0; // behind the camera or outside the image
	int offered= 0;           // hints handed to the pipeline
	int skippedTooSmall= 0;   // projected palm too small to crop around
	int skippedRedundant= 0;  // landed on a hand this camera already tracks
	int skippedNoFreeSlot= 0;
	int applied= 0;           // a slot was seeded
	int rejectedByModel= 0;   // first landmark pass found no confident hand
	int accepted= 0;          // survived that pass and became a tracked hand
};

// Orchestrates the MediaPipe-style tracking graph on the inference thread:
//   - two hand slots with landmark-driven ROI reuse (palm detector only runs
//     when a slot is free or every detectorIntervalFrames as a drift guard)
//   - handedness resolution with temporal stickiness
// Fills the image-space fields of TrackingFrameResult; camera/world space and
// the parametric hand poses are filled later by the Tracking module
// (LandmarkTo3D / SpaceTransforms). Elbow estimation was removed entirely -
// there isn't enough information in a hand-only view for a useful estimate;
// clients solve arms with IK from the palm pose.
//
// NOTE: BlazePose elbow measurement was removed - its person detector never
// fires on top-down/overhead views, so the pose stage never produced usable
// elbows on the rigs this app targets (see git history for the experiment).
//
// THREAD AFFINITY: startup/process/shutdown must all happen on the same
// (inference) thread — the ONNX sessions live on it.
class HandTrackingPipeline
{
public:
	HandTrackingPipeline()= default;

	// Loads the palm detection + hand landmark models
	bool startup(const HandTrackingPipelineConfig& config);

	HandTrackingPipelineConfig getConfig() const { return m_config; }
	void setConfig(const HandTrackingPipelineConfig& config);

	// "DirectML" | "CPU" | "none"
	const char* getActiveExecutionProvider() const;

	// Runs one frame; fills hands/arms image-space fields, debug detection
	// boxes and inferenceMs on outResult
	void process(const cv::Mat& bgrFrame, TrackingFrameResult& outResult);

	// Which of two contending slots claims its preferred side first (0 = A,
	// 1 = B). The loser is displaced to the free side, so this decides the
	// published L/R labels whenever both slots want the same side. Exposed
	// for the --test-fusion self test.
	static int preferredSlotOrder(float rightProbA, float presenceA, float rightProbB, float presenceB);

	// Queues cross-camera search hints for the next process() call (same
	// thread as process; hints landing on an already-tracked hand are ignored)
	void setSearchHints(const std::vector<HandSearchHint>& hints)
	{
		m_searchHints= hints;
		m_seedStats.offered+= (int)hints.size();
	}

	// Whether a hint duplicates a hand this camera already tracks. POSITIONAL
	// on purpose: a camera can track the right physical hand under the wrong
	// side label, which is exactly what happens after a hand leaves and
	// re-enters, so a label comparison would both miss real duplicates and
	// suppress real seeds. Boxes are inflated about their centre first, since
	// the projected centre carries the OTHER camera's depth error - a range
	// error along its view ray arrives here as a lateral offset.
	// Exposed for the --test-seeding self test.
	static bool isSeedRedundant(const glm::vec2& centerPx, const std::vector<HandBox>& activeBoxes);

	// Cross-camera seeding accounting. The vision thread owns the two counters
	// that describe hints it never handed over; everything after that is
	// counted here, so consumers read one struct.
	const HandSeedStats& getSeedStats() const { return m_seedStats; }
	void noteSeedCandidate(bool bProjectedIntoFrame)
	{
		m_seedStats.candidates++;
		if (!bProjectedIntoFrame)
			m_seedStats.skippedProjection++;
	}

private:
	struct HandSlot
	{
		bool active= false;
		// a palm detection is pending until the first landmark pass consumes it
		bool hasPendingDetection= false;
		// pending detection came from a cross-camera hint (speculative): if the
		// first landmark pass doesn't find a confident hand, drop immediately
		bool bSeededFromHint= false;
		PalmDetection pendingDetection;

		std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints{};
		std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints{};
		glm::vec2 roiBoxMin{0.f}; // hand box used for detection dedupe + ROI seed
		glm::vec2 roiBoxMax{0.f};
		DetectionBox lastRoiDebug;

		float presence= 0.f;
		float handednessScore= 0.5f; // smoothed raw model score
		float rightProb= 0.5f;       // flip-adjusted P(right hand)
		int lowPresenceFrames= 0;
		int framesSinceDetection= 0;

		bool sideInitialized= false;
		eHandSide side= eHandSide::Left;
		int sideContradictionFrames= 0;

		void deactivate()
		{
			active= false;
			hasPendingDetection= false;
			bSeededFromHint= false;
			presence= 0.f;
			handednessScore= 0.5f;
			rightProb= 0.5f;
			lowPresenceFrames= 0;
			framesSinceDetection= 0;
			sideInitialized= false;
			sideContradictionFrames= 0;
		}
	};

	void runPalmDetectionStage(const cv::Mat& bgrFrame, TrackingFrameResult& outResult);
	void applySearchHints();
	void runHandLandmarkStage(const cv::Mat& bgrFrame);
	void killDuplicateSlots();
	void resolveHandedness(int frameWidth);
	void publishHands(TrackingFrameResult& outResult);

	int countActiveSlots() const;

	HandTrackingPipelineConfig m_config;

	PalmDetector m_palmDetector;
	HandLandmarkModel m_handLandmarkModel;

	std::array<HandSlot, 2> m_slots;

	int64_t m_frameIndex= -1;
	int m_framesSinceDetector= 0;
	int m_duplicateOverlapFrames= 0;

	// Reacquisition window bookkeeping: when the active slot count drops, the
	// palm detector runs with the relaxed score threshold for a short window
	int m_lastActiveSlotCount= 0;
	int64_t m_lastSlotLossFrame= -1;

	std::vector<HandSearchHint> m_searchHints;
	HandSeedStats m_seedStats;
	// scratch for isSeedRedundant, so the per-frame path allocates nothing
	std::vector<HandBox> m_activeBoxes;

	// scratch
	std::vector<PalmDetection> m_palmDetections;
	HandLandmarkResult m_handResult;
};
