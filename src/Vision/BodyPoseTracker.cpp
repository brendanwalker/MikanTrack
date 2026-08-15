#include "BodyPoseTracker.h"

#include <algorithm>

#include "glm/geometric.hpp"

#include "Logger.h"

// Grow the box around the tracked keypoints. The model applies its own 1.25
// padding on top; this only has to cover motion between model frames.
static constexpr float kTrackedBoxExpansion= 1.15f;
// A tracked box smaller than this (px) is treated as collapsed
static constexpr float kMinBoxSize= 24.f;

// COCO 17 -> the 33-slot BlazePose layout, which stays the canonical index
// space for observations so every downstream consumer (solver, overlay,
// recordings, dumps) is backend-independent - and so recordings made before
// the backend swap still load. Slots this model does not fill are reported
// through BodyPoseObservation::providedMask rather than being invented.
static constexpr int k_cocoToPoseLandmark[COCO_KEYPOINT_COUNT]= {
	0,  // nose
	2,  // left eye
	5,  // right eye
	7,  // left ear
	8,  // right ear
	11, // left shoulder
	12, // right shoulder
	13, // left elbow
	14, // right elbow
	15, // left wrist
	16, // right wrist
	23, // left hip
	24, // right hip
	25, // left knee
	26, // right knee
	27, // left ankle
	28, // right ankle
};

bool BodyPoseTracker::load(const std::string& modelDir, const std::string& preferredEp,
						   const BodyPoseTrackerConfig& config)
{
	setConfig(config);

	m_bLoaded= m_detector.load(modelDir + "/person_detection.onnx", preferredEp) &&
		m_landmarkModel.load(modelDir + "/rtmpose_body.onnx", preferredEp);

	if (!m_bLoaded)
	{
		MIKAN_MT_LOG_WARNING("BodyPoseTracker::load")
			<< "Body pose models missing or unloadable in " << modelDir
			<< " - body pose stage disabled (re-run InitialSetup_x64.bat)";
	}
	reset();
	return m_bLoaded;
}

void BodyPoseTracker::setConfig(const BodyPoseTrackerConfig& config)
{
	m_config= config;
	m_config.frameDivider= std::max(config.frameDivider, 1);
	m_config.detectorIntervalFrames= std::max(config.detectorIntervalFrames, 1);
}

void BodyPoseTracker::reset()
{
	m_bBoxTracked= false;
	m_lastObservation= BodyPoseObservation();
	m_frameIndex= 0;
	m_modelFrameIndex= 0;
	m_lastDetectModelFrame= -1000;
}

eBodyBoxSource BodyPoseTracker::acquireBox(const cv::Mat& bgrFrame, bool bForceDetect,
										   glm::vec2& outBoxMin, glm::vec2& outBoxMax)
{
	if (!bForceDetect && m_bBoxTracked)
	{
		outBoxMin= m_trackedBoxMin;
		outBoxMax= m_trackedBoxMax;
		return eBodyBoxSource::Tracked;
	}

	// Counted whether or not it fires. The person detector is trained on
	// whole people, so on a subject truncated at a desk it misses often; if a
	// miss did not count as an attempt, every later frame would re-run it and
	// stay latched in detection until one happened to succeed.
	m_lastDetectModelFrame= m_modelFrameIndex;
	m_detector.detect(bgrFrame, m_detections);

	for (const PersonDetection& detection : m_detections)
	{
		// The detection's own box is a FACE box, not a person box - feeding
		// it to a top-down model cropped the arms away (measured: landmark
		// scores collapsed from ~0.7 to ~0.3 on exactly the frames the
		// detector supplied the box). The person extent lives in the
		// keypoints: a square about the hip center reaching the full-body
		// point, which is the crop convention this detector was built for.
		const glm::vec2& hipCenter= detection.keypoints[0];
		const glm::vec2& fullBodyPoint= detection.keypoints[1];
		const float radius= glm::length(fullBodyPoint - hipCenter);
		if (radius < kMinBoxSize)
			continue;

		// Clipped to the image: for a person truncated at a desk that square
		// runs off the bottom, and a crop mostly outside the frame is mostly
		// padding
		const glm::vec2 imageMax((float)bgrFrame.cols, (float)bgrFrame.rows);
		const glm::vec2 clippedMin= glm::max(hipCenter - glm::vec2(radius), glm::vec2(0.f));
		const glm::vec2 clippedMax= glm::min(hipCenter + glm::vec2(radius), imageMax);
		if ((clippedMax.x - clippedMin.x) < kMinBoxSize || (clippedMax.y - clippedMin.y) < kMinBoxSize)
			continue;

		outBoxMin= clippedMin;
		outBoxMax= clippedMax;
		return eBodyBoxSource::Detector;
	}

	// A failed re-detect must not cost a working track: the periodic detect
	// is a drift guard, not a requirement
	if (m_bBoxTracked)
	{
		outBoxMin= m_trackedBoxMin;
		outBoxMax= m_trackedBoxMax;
		return eBodyBoxSource::Tracked;
	}

	// Nothing to go on. A top-down model still reads a centered subject from
	// the whole frame, which beats reporting nothing until the detector fires.
	outBoxMin= glm::vec2(0.f, 0.f);
	outBoxMax= glm::vec2((float)bgrFrame.cols, (float)bgrFrame.rows);
	return eBodyBoxSource::FullFrame;
}

void BodyPoseTracker::process(const cv::Mat& bgrFrame, BodyPoseObservation& outObservation)
{
	const int64_t frameIndex= m_frameIndex++;
	if ((frameIndex % (int64_t)m_config.frameDivider) != 0)
	{
		// Divided-out frame: hold the last model result. Recording taps sit
		// downstream, so this cadence is baked into recorded observations and
		// replay never needs to know the divider.
		outObservation= m_lastObservation;
		return;
	}

	outObservation= BodyPoseObservation();
	if (m_bLoaded && !bgrFrame.empty())
	{
		// Drift guard: periodically rebuild the person box from the image
		// rather than from the previous keypoints, whatever the model claims
		// about its own confidence
		const bool bForceDetect=
			(m_modelFrameIndex - m_lastDetectModelFrame) >= (int64_t)m_config.detectorIntervalFrames;

		glm::vec2 boxMin(0.f);
		glm::vec2 boxMax(0.f);
		const eBodyBoxSource boxSource= acquireBox(bgrFrame, bForceDetect, boxMin, boxMax);

		m_landmarkModel.estimate(bgrFrame, boxMin, boxMax, m_landmarkResult);
		if (m_landmarkResult.valid)
		{
			// Next frame's box is the confident keypoints' extent. No
			// full-body assumption is involved, so a truncated subject simply
			// produces a smaller box instead of a fabricated one.
			glm::vec2 trackedMin(FLT_MAX);
			glm::vec2 trackedMax(-FLT_MAX);
			int confidentCount= 0;
			for (int keypoint= 0; keypoint < COCO_KEYPOINT_COUNT; ++keypoint)
			{
				if (m_landmarkResult.scores[keypoint] < m_config.keypointScoreThreshold)
					continue;
				trackedMin= glm::min(trackedMin, m_landmarkResult.points[keypoint]);
				trackedMax= glm::max(trackedMax, m_landmarkResult.points[keypoint]);
				confidentCount++;
			}

			if (confidentCount >= 3 && (trackedMax.x - trackedMin.x) >= kMinBoxSize &&
				(trackedMax.y - trackedMin.y) >= kMinBoxSize)
			{
				const glm::vec2 boxCenter= (trackedMin + trackedMax) * 0.5f;
				const glm::vec2 halfExtent= (trackedMax - trackedMin) * 0.5f * kTrackedBoxExpansion;
				m_trackedBoxMin= boxCenter - halfExtent;
				m_trackedBoxMax= boxCenter + halfExtent;
				m_bBoxTracked= true;
			}
			else
			{
				m_bBoxTracked= false;
			}

			outObservation.valid= true;
			outObservation.boxSource= boxSource;
			outObservation.modelFrameIndex= m_modelFrameIndex;

			// SimCC peaks are not probabilities; they are reported as-is and
			// clamped so downstream visibility gates keep their [0,1] meaning
			float scoreSum= 0.f;
			for (int keypoint= 0; keypoint < COCO_KEYPOINT_COUNT; ++keypoint)
			{
				const int landmark= k_cocoToPoseLandmark[keypoint];
				const float score= std::clamp(m_landmarkResult.scores[keypoint], 0.f, 1.f);
				outObservation.imagePoints[landmark]= glm::vec3(m_landmarkResult.points[keypoint], 0.f);
				outObservation.visibility[landmark]= score;
				outObservation.providedMask|= 1ull << landmark;
				scoreSum+= score;
			}
			outObservation.confidence= scoreSum / (float)COCO_KEYPOINT_COUNT;
		}
		else
		{
			m_bBoxTracked= false;
		}

		m_modelFrameIndex++;
	}

	m_lastObservation= outObservation;
}
