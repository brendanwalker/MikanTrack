#pragma once

#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "BodyPoseTypes.h"
#include "PoseDetector.h"
#include "PoseLandmarkModel.h"
#include "RtmPoseBodyModel.h"

// Which landmark model runs behind the per-camera body-pose stage.
//
// BlazePose owns its own region of interest (a hip-centered square reaching a
// "full body" point) and always emits one coherent 33-point skeleton. For a
// person truncated at a desk there is no correct crop for that convention,
// and the crop is rebuilt each frame from the model's own fabricated lower
// body - measured live, its reconstruction ran 40% short in the legs and its
// per-frame metric scale swung nearly 3x.
//
// RtmPose is top-down: this class supplies the person box, and each keypoint
// is predicted independently with its own score, so joints outside the frame
// degrade to low confidence instead of being invented. It emits 2D only,
// which is all BodyPoseSolver consumes.
enum class eBodyPoseBackend : int
{
	BlazePose= 0,
	RtmPose= 1,
};

const char* bodyPoseBackendName(eBodyPoseBackend backend);

struct BodyPoseTrackerConfig
{
	eBodyPoseBackend backend= eBodyPoseBackend::RtmPose;
	// Models run every Nth processed frame
	int frameDivider= 2;
	// Re-run the person detector every Nth MODEL frame no matter how
	// confident the landmark model is. Without this the region of interest is
	// only ever rebuilt from the previous landmarks, so a drifting or
	// wrongly-scaled crop feeds itself: measured live, the landmark
	// confidence never once fell under its 0.5 reset gate (minimum 0.52 over
	// 338 model frames) while the geometry was visibly wrong. The hand
	// pipeline carries the same guard for the same reason.
	int detectorIntervalFrames= 20;
	// A keypoint below this does not contribute to the tracked box
	float keypointScoreThreshold= 0.3f;
};

// Per-camera body-pose stage: person detection, landmark inference, and the
// detect-vs-track cadence between them. Opt-in per camera, so cameras without
// it pay nothing.
class BodyPoseTracker
{
public:
	// Loads the person detector plus the configured landmark backend from
	// modelDir. Missing models fail soft: returns false, tracker stays a
	// no-op.
	bool load(const std::string& modelDir, const std::string& preferredEp,
			  const BodyPoseTrackerConfig& config);
	bool isLoaded() const { return m_bLoaded; }
	const char* activeEp() const;
	eBodyPoseBackend getBackend() const { return m_config.backend; }

	// Cadence and thresholds only; changing the backend needs a reload
	void setConfig(const BodyPoseTrackerConfig& config);

	void process(const cv::Mat& bgrFrame, BodyPoseObservation& outObservation);

	// Drops ROI tracking, the held observation, and the frame counter
	void reset();

private:
	// Person box for this model frame, from tracking or from the detector
	eBodyBoxSource acquireBox(const cv::Mat& bgrFrame, bool bForceDetect,
							  glm::vec2& outBoxMin, glm::vec2& outBoxMax);
	bool runBlazePose(const cv::Mat& bgrFrame, bool bForceDetect, BodyPoseObservation& outObservation);
	bool runRtmPose(const cv::Mat& bgrFrame, bool bForceDetect, BodyPoseObservation& outObservation);

	bool m_bLoaded= false;
	BodyPoseTrackerConfig m_config;

	PoseDetector m_detector;
	PoseLandmarkModel m_blazePoseModel;
	RtmPoseBodyModel m_rtmPoseModel;

	// BlazePose tracking state (its own aux-landmark region of interest)
	bool m_bRoiTracked= false;
	PoseRoi m_trackedRoi;
	// RtmPose tracking state (a plain box grown from the last keypoints)
	bool m_bBoxTracked= false;
	glm::vec2 m_trackedBoxMin{0.f};
	glm::vec2 m_trackedBoxMax{0.f};

	BodyPoseObservation m_lastObservation;
	int64_t m_frameIndex= 0;
	int64_t m_modelFrameIndex= 0;
	int64_t m_lastDetectModelFrame= -1000;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<PersonDetection> m_detections;
	PoseLandmarkResult m_blazePoseResult;
	RtmPoseResult m_rtmPoseResult;
};
