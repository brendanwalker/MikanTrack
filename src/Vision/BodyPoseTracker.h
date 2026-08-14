#pragma once

#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "BodyPoseTypes.h"
#include "PoseDetector.h"
#include "RtmPoseBodyModel.h"

struct BodyPoseTrackerConfig
{
	// Models run every Nth processed frame
	int frameDivider= 2;
	// Re-run the person detector every Nth MODEL frame no matter how
	// confident the landmark model is. Without this the person box is only
	// ever rebuilt from the previous keypoints, so a drifting or wrongly
	// scaled box feeds itself: measured live, landmark confidence never once
	// fell under its reset gate (minimum 0.52 over 338 model frames) while
	// the geometry was visibly wrong. The hand pipeline carries the same
	// guard for the same reason.
	int detectorIntervalFrames= 20;
	// A keypoint below this does not contribute to the tracked box
	float keypointScoreThreshold= 0.3f;
};

// Per-camera body-pose stage: person detection, landmark inference, and the
// detect-vs-track cadence between them. Opt-in per camera, so cameras without
// it pay nothing.
//
// The landmark model is RTMPose, which is TOP-DOWN: this class supplies the
// person box and each keypoint is predicted independently with its own score,
// so joints outside the frame degrade to low confidence instead of being
// invented. BlazePose was tried first and lost decisively on this rig - same
// frames, same session: 99% of frames usable against 70%, and 6-8 px of 2D
// jitter per joint against 53-83 px. It owned its own crop (a hip-centered
// square reaching a "full body" point) which has no correct answer for a
// person truncated at a desk, and it always emitted one coherent full-body
// skeleton, so it fabricated the unseen legs and dragged the arms with them.
//
// The person DETECTOR from that family is still here and still useful: only
// its keypoints are read, never its box, which is a face box.
class BodyPoseTracker
{
public:
	// Loads the person detector and the landmark model from modelDir. Missing
	// models fail soft: returns false, tracker stays a no-op.
	bool load(const std::string& modelDir, const std::string& preferredEp,
			  const BodyPoseTrackerConfig& config);
	bool isLoaded() const { return m_bLoaded; }
	const char* activeEp() const { return m_landmarkModel.activeEp(); }

	void setConfig(const BodyPoseTrackerConfig& config);

	void process(const cv::Mat& bgrFrame, BodyPoseObservation& outObservation);

	// Drops box tracking, the held observation, and the frame counter
	void reset();

private:
	// Person box for this model frame, from tracking or from the detector
	eBodyBoxSource acquireBox(const cv::Mat& bgrFrame, bool bForceDetect,
							  glm::vec2& outBoxMin, glm::vec2& outBoxMax);

	bool m_bLoaded= false;
	BodyPoseTrackerConfig m_config;

	PoseDetector m_detector;
	RtmPoseBodyModel m_landmarkModel;

	bool m_bBoxTracked= false;
	glm::vec2 m_trackedBoxMin{0.f};
	glm::vec2 m_trackedBoxMax{0.f};

	BodyPoseObservation m_lastObservation;
	int64_t m_frameIndex= 0;
	int64_t m_modelFrameIndex= 0;
	int64_t m_lastDetectModelFrame= -1000;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<PersonDetection> m_detections;
	RtmPoseResult m_landmarkResult;
};
