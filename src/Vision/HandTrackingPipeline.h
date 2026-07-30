#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "HandLandmarkModel.h"
#include "PalmDetector.h"
#include "PoseDetector.h"
#include "PoseLandmarkModel.h"
#include "TrackingTypes.h"

struct HandTrackingPipelineConfig
{
	// "directml" | "cpu" (session recreation requires a pipeline restart, so
	// setConfig ignores changes to this and modelDir)
	std::string preferredEp= "directml";
	// model directory relative to the working directory (exe dir)
	std::string modelDir= "models";

	bool usePoseModel= true;
	// MediaPipe's handedness classifier assumes a mirrored/selfie view; with
	// an unmirrored (rear/webcam) feed the label must be flipped
	bool flipHandedness= true;
	// Overhead-rig experiment: BlazePose's person detector is trained on
	// upright people and essentially never fires on a top-down view. When
	// enabled, the pose ROI is seeded from the tracked hand positions
	// (extended up the forearm direction) instead of running the person
	// detector. The pose-confidence gate and arm fallback still apply.
	bool poseHandSeededRoi= false;

	// run the palm detector at least every N frames even with 2 active slots
	// (drift guard); it always runs when fewer than 2 slots are active
	int detectorIntervalFrames= 30;
	// run the person-detect/pose-landmark stage every N frames
	int poseFrameDivider= 2;

	float palmScoreThreshold= 0.5f;     // mp_palmdet.py default
	float palmNmsIouThreshold= 0.3f;    // mp_palmdet.py default
	float personScoreThreshold= 0.5f;   // mp_persondet.py default
	float personNmsIouThreshold= 0.3f;  // mp_persondet.py default
	float handPresenceThreshold= 0.5f;  // slot keepalive threshold
	int handPresenceLostFrames= 2;      // consecutive low-presence frames before slot deactivates
	float poseConfidenceThreshold= 0.5f; // mp_pose.py default
	float slotDedupeIouThreshold= 0.3f; // palm detection vs active slot IoU
	float poseWristAssocFrameWidthFrac= 0.15f; // pose-wrist to hand-wrist association distance
	float elbowVisibilityThreshold= 0.5f;
	int handednessSwitchFrames= 15;     // consecutive contradictions before a slot flips side
	float armFallbackElbowScale= 2.2f;  // elbow= wrist + dir * scale * |wrist - middleMCP|
	float poseWristPlausibilityPx= 80.f; // pose wrist must be this close to the hand wrist
};

// Orchestrates the MediaPipe-style tracking graph on the inference thread:
//   - two hand slots with landmark-driven ROI reuse (palm detector only runs
//     when a slot is free or every detectorIntervalFrames as a drift guard)
//   - optional person-detect + pose-landmark stage on a frame divider, with
//     aux-landmark ROI reuse, used for elbows and handedness cross-checks
//   - handedness resolution with temporal stickiness
//   - forearm fallback from hand geometry when the pose elbow is unavailable
// Fills the image-space fields of TrackingFrameResult; camera/world space is
// filled later by the Tracking module (LandmarkTo3D / SpaceTransforms).
//
// THREAD AFFINITY: startup/process/shutdown must all happen on the same
// (inference) thread — the ONNX sessions live on it.
class HandTrackingPipeline
{
public:
	HandTrackingPipeline()= default;

	// Loads all models. Palm + hand landmark models are required; pose models
	// are optional (pose stage disabled with a warning if they fail).
	bool startup(const HandTrackingPipelineConfig& config);

	HandTrackingPipelineConfig getConfig() const { return m_config; }
	void setConfig(const HandTrackingPipelineConfig& config);

	// "DirectML" | "CPU" | "none"
	const char* getActiveExecutionProvider() const;

	// Runs one frame; fills hands/arms image-space fields, debug detection
	// boxes and inferenceMs on outResult
	void process(const cv::Mat& bgrFrame, TrackingFrameResult& outResult);

private:
	struct HandSlot
	{
		bool active= false;
		// a palm detection is pending until the first landmark pass consumes it
		bool hasPendingDetection= false;
		PalmDetection pendingDetection;

		std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints{};
		std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints{};
		glm::vec2 roiBoxMin{0.f}; // hand box used for detection dedupe + ROI seed
		glm::vec2 roiBoxMax{0.f};
		DetectionBox lastRoiDebug;

		float presence= 0.f;
		float handednessScore= 0.5f; // smoothed raw model score
		int lowPresenceFrames= 0;
		int framesSinceDetection= 0;

		bool sideInitialized= false;
		eHandSide side= eHandSide::Left;
		int sideContradictionFrames= 0;

		void deactivate()
		{
			active= false;
			hasPendingDetection= false;
			presence= 0.f;
			handednessScore= 0.5f;
			lowPresenceFrames= 0;
			framesSinceDetection= 0;
			sideInitialized= false;
			sideContradictionFrames= 0;
		}
	};

	void runPalmDetectionStage(const cv::Mat& bgrFrame, TrackingFrameResult& outResult);
	void runHandLandmarkStage(const cv::Mat& bgrFrame);
	void runPoseStage(const cv::Mat& bgrFrame, TrackingFrameResult& outResult);
	void resolveHandedness(int frameWidth);
	void publishHands(TrackingFrameResult& outResult);
	void publishArms(TrackingFrameResult& outResult);

	int countActiveSlots() const;
	bool isPoseResultFresh() const;

	HandTrackingPipelineConfig m_config;

	PalmDetector m_palmDetector;
	HandLandmarkModel m_handLandmarkModel;
	PoseDetector m_poseDetector;
	PoseLandmarkModel m_poseLandmarkModel;
	bool m_bPoseModelsLoaded= false;

	std::array<HandSlot, 2> m_slots;

	int64_t m_frameIndex= -1;
	int m_framesSinceDetector= 0;

	// pose tracking state
	bool m_bPoseRoiTracked= false;
	PoseRoi m_trackedPoseRoi;
	PoseLandmarkResult m_lastPoseResult;
	int64_t m_lastPoseFrameIndex= -1000;

	// scratch
	std::vector<PalmDetection> m_palmDetections;
	std::vector<PersonDetection> m_personDetections;
	HandLandmarkResult m_handResult;
};
