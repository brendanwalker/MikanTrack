#include "CameraContext.h"

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/matrix.hpp"

#include "AppConfig.h"
#include "BodyPoseTracker.h"
#include "CVVideoFrameProcessor.h"
#include "FrameRecorder.h"
#include "HandRoiQuality.h"
#include "HandTrackingPipeline.h"
#include "LandmarkTo3D.h"
#include "Logger.h"
#include "SpaceTransforms.h"
#include "SteadyClock.h"
#include "VideoCaptureSystem.h"
#include "VideoFrame.h"

CameraContext::CameraContext(int cameraIndex, const AppConfig* config)
	: m_cameraIndex(cameraIndex)
	, m_config(config)
{
	m_lastResult.cameraIndex= cameraIndex;
}

// Out of line so the unique_ptr members can hold forward-declared types
CameraContext::~CameraContext()= default;

bool CameraContext::fetchPreviewFrame(VisionPreviewFrame& outFrame)
{
	std::lock_guard<std::mutex> lock(m_previewMutex);
	if (!m_bPreviewFresh)
		return false;

	m_previewFrame.bgr.copyTo(outFrame.bgr);
	outFrame.result= m_previewFrame.result;
	outFrame.valid= true;
	m_bPreviewFresh= false;
	return true;
}

void CameraContext::configure(const AppConfig& config)
{
	const CameraProfile& profile= config.camera(m_cameraIndex);

	// ML pipeline
	if (m_pipeline == nullptr)
	{
		HandTrackingPipelineConfig pipelineConfig;
		pipelineConfig.flipHandedness= config.tracking.flipHandedness;
		pipelineConfig.detectorIntervalFrames= config.tracking.detectorIntervalFrames;
		pipelineConfig.palmScoreThresholdRelaxed= config.tracking.palmScoreThresholdRelaxed;
		pipelineConfig.preferredEp= config.tracking.onnxEp;

		m_pipeline= std::make_unique<HandTrackingPipeline>();
		if (m_pipeline->startup(pipelineConfig))
		{
			m_activeEp= m_pipeline->getActiveExecutionProvider();
		}
		else
		{
			MIKAN_MT_LOG_ERROR("CameraContext")
				<< "HandTrackingPipeline startup failed for camera " << m_cameraIndex
				<< " - tracking disabled";
			m_pipeline= nullptr;
			m_activeEp= "none";
		}
	}
	else
	{
		HandTrackingPipelineConfig pipelineConfig= m_pipeline->getConfig();
		pipelineConfig.flipHandedness= config.tracking.flipHandedness;
		pipelineConfig.detectorIntervalFrames= config.tracking.detectorIntervalFrames;
		pipelineConfig.palmScoreThresholdRelaxed= config.tracking.palmScoreThresholdRelaxed;
		m_pipeline->setConfig(pipelineConfig);
	}

	// 3D projection (needs that camera's calibrated intrinsics).
	// Smoothing is always disabled here - the fused output is smoothed
	// after fusion instead (avoids double-filtering).
	if (profile.intrinsics.present)
	{
		if (m_landmarkTo3D == nullptr)
			m_landmarkTo3D= std::make_unique<LandmarkTo3D>();
		m_landmarkTo3D->configure(
			profile.intrinsics.intrinsics,
			config.handScale.refLengthMeters);

		// Measured hand geometry, when there is any. Not per camera: bones
		// are a physical property, so every camera gets the same skeleton.
		m_landmarkTo3D->clearCalibratedSkeleton();
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			if (config.handSkeleton.present[sideIndex])
			{
				m_landmarkTo3D->setCalibratedSkeleton(
					(eHandSide)sideIndex, config.handSkeleton.skeleton[sideIndex]);
			}
		}

		// Undistortion for the ML input + preview. ALWAYS rebuilt on a
		// config change: gating on frame dimensions alone kept the OLD
		// undistortion maps alive after a recalibration at the same
		// resolution (live symptom: wildly zoomed preview from the
		// previous bad intrinsics until an app restart).
		m_undistorter= std::make_unique<CVVideoFrameProcessor>(
			profile.intrinsics.intrinsics,
			(int)profile.intrinsics.intrinsics.pixel_width,
			(int)profile.intrinsics.intrinsics.pixel_height);
	}
	else
	{
		m_landmarkTo3D= nullptr;
		m_undistorter= nullptr;
	}

	// Opt-in body-pose stage. Load failure (missing models) leaves the
	// tracker allocated but unloaded so a refresh doesn't retry-spam.
	if (profile.bodyPose.enabled)
	{
		BodyPoseTrackerConfig trackerConfig;
		trackerConfig.frameDivider= profile.bodyPose.poseFrameDivider;
		trackerConfig.detectorIntervalFrames= profile.bodyPose.detectorIntervalFrames;

		if (m_bodyPoseTracker == nullptr)
		{
			m_bodyPoseTracker= std::make_unique<BodyPoseTracker>();
			m_bodyPoseTracker->load("models", config.tracking.onnxEp, trackerConfig);
		}
		else
		{
			m_bodyPoseTracker->setConfig(trackerConfig);
		}
	}
	else
	{
		m_bodyPoseTracker= nullptr;
	}

	// Invalidate the last result so stale calibration state can't leak
	// through a config change
	m_lastResult= CameraFrameResult();
	m_lastResult.cameraIndex= m_cameraIndex;
}

void CameraContext::resetTransientState()
{
	if (m_landmarkTo3D != nullptr)
		m_landmarkTo3D->resetTransientState();
	m_lastResult= CameraFrameResult();
	m_lastResult.cameraIndex= m_cameraIndex;
	m_bPendingRecordFresh= false;
}

void CameraContext::setRefLengthMeters(float refLengthMeters)
{
	if (m_landmarkTo3D != nullptr)
		m_landmarkTo3D->setRefLengthMeters(refLengthMeters);
}

bool CameraContext::consumePendingRecordInput(RecordedCameraInput& outInput)
{
	if (!m_bPendingRecordFresh)
		return false;

	outInput= m_pendingRecordInput;
	m_bPendingRecordFresh= false;
	return true;
}

const HandSeedStats* CameraContext::getSeedStats() const
{
	return m_pipeline != nullptr ? &m_pipeline->getSeedStats() : nullptr;
}

void CameraContext::seedSearchHints(const TrackingFrameResult& lastFused, float autoScaleFactor)
{
	if (m_pipeline == nullptr || !m_bTrackingEnabled)
		return;

	const CameraProfile& profile= m_config->camera(m_cameraIndex);
	if (!profile.intrinsics.present || !profile.extrinsics.present)
		return;

	if (m_hintCooldownFrames > 0)
	{
		m_hintCooldownFrames--;
		return;
	}

	const glm::dmat4 cameraFromWorld= glm::inverse(profile.extrinsics.markerFromCamera);
	const MikanMatrix3d& cameraMatrix= profile.intrinsics.intrinsics.undistorted_camera_matrix;
	const double fx= cameraMatrix.x0, fy= cameraMatrix.y1;
	const double cx= cameraMatrix.z0, cy= cameraMatrix.z1;
	const double width= profile.intrinsics.intrinsics.pixel_width;
	const double height= profile.intrinsics.intrinsics.pixel_height;

	// Projects a world point into this camera's (undistorted) image; false
	// when behind or implausibly close to the camera
	auto projectPoint= [&](const glm::vec3& world, glm::vec2& outPx, double& outDepth) {
		const glm::dvec4 cameraPt= cameraFromWorld * glm::dvec4(glm::dvec3(world), 1.0);
		if (cameraPt.z < 0.05)
			return false;
		outPx= glm::vec2((float)(fx * cameraPt.x / cameraPt.z + cx), (float)(fy * cameraPt.y / cameraPt.z + cy));
		outDepth= cameraPt.z;
		return true;
	};

	std::vector<HandSearchHint> hints;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& fusedPose= lastFused.poses[sideIndex];
		if (!fusedPose.tracked || !fusedPose.hasWorldPose)
			continue;

		// Whether this camera already has the hand is decided POSITIONALLY,
		// downstream in applySearchHints. It used to be decided here by
		// comparing side LABELS, which fails exactly when it matters: labels
		// get displaced when hands leave and re-enter, and a camera holding
		// one hand under the wrong label then had one side skipped here and
		// the other suppressed there, so neither was seeded and reacquisition
		// waited a full palm-detector interval.
		glm::vec2 centerPx;
		double depth= 0.0;
		const bool bProjected=
			projectPoint(fusedPose.palmPositionWorld, centerPx, depth) &&
			centerPx.x >= 0.f && centerPx.x < (float)width &&
			centerPx.y >= 0.f && centerPx.y < (float)height;
		m_pipeline->noteSeedCandidate(bProjected);
		if (!bProjected)
			continue;

		// Palm +X points toward the fingers; its projection orients the crop
		const glm::vec3 fingersDirWorld= fusedPose.palmOrientationWorld * glm::vec3(1.f, 0.f, 0.f);
		glm::vec2 aheadPx;
		double unusedDepth= 0.0;
		if (!projectPoint(fusedPose.palmPositionWorld + fingersDirWorld * 0.05f, aheadPx, unusedDepth))
			continue;
		glm::vec2 dirPx= aheadPx - centerPx;
		const float dirLength= glm::length(dirPx);

		const float refLengthMeters=
			(float)(m_config->handScale.refLengthMeters * (double)autoScaleFactor);

		HandSearchHint hint;
		hint.centerPx= centerPx;
		hint.dirPx= dirLength > 1e-3f ? dirPx / dirLength : glm::vec2(0.f, -1.f);
		hint.palmSizePx= (float)(fx * (double)refLengthMeters / depth);
		hints.push_back(hint);
	}

	if (!hints.empty())
	{
		m_pipeline->setSearchHints(hints);
		m_hintCooldownFrames= 2; // retry every ~3 frames while unseen
	}
}

bool CameraContext::process(VideoCaptureSystem* videoCapture, const TrackingFrameResult* seedFromFused,
								float autoScaleFactor, bool bRecordingInputs, FrameRecorder* frameRecorder)
{
	VideoFrameBlock* block= videoCapture->tryPopFrame(m_cameraIndex);
	if (block == nullptr)
		return false;

	// Per-step wall time, published with the result so a slow step names
	// itself in the tracking panel and the dump instead of only tripping the
	// vision thread's hitch watchdog
	TrackingFrameResult::CaptureTimings timings;
	const double popMs= steadyNowMs();
	double stepMarkMs= popMs;
	auto lapMs= [&stepMarkMs]() {
		const double nowMs= steadyNowMs();
		const float elapsedMs= (float)(nowMs - stepMarkMs);
		stepMarkMs= nowMs;
		return elapsedMs;
	};
	timings.queueAgeMs= (float)(popMs - block->timestampMs);

	// FPS estimate from frame timestamps
	if (m_lastFrameTimestampMs > 0.0 && block->timestampMs > m_lastFrameTimestampMs)
	{
		const float instFps= (float)(1000.0 / (block->timestampMs - m_lastFrameTimestampMs));
		m_captureFps= m_captureFps > 0.f ? m_captureFps * 0.9f + instFps * 0.1f : instFps;
	}
	m_lastFrameTimestampMs= block->timestampMs;

	// Raw -> BGR
	VideoCaptureSystem::convertFrameToBGR(*block, m_bgrScratch);
	const int64_t frameIndex= block->frameIndex;
	const double timestampMs= block->timestampMs;
	videoCapture->releaseFrame(m_cameraIndex, block);
	timings.convertMs= lapMs();

	if (m_bgrScratch.empty())
		return false;

	const CameraProfile& profile= m_config->camera(m_cameraIndex);

	// Undistort when calibrated (ML + preview both use the undistorted image)
	cv::Mat* activeFrame= &m_bgrScratch;
	if (m_bUndistortEnabled &&
		m_undistorter != nullptr &&
		m_bgrScratch.cols == m_undistorter->getFrameWidth() &&
		m_bgrScratch.rows == m_undistorter->getFrameHeight())
	{
		m_undistorter->processColorFrame(m_bgrScratch, m_undistortedScratch);
		activeFrame= &m_undistortedScratch;
	}
	m_lastActiveFrame= activeFrame;
	timings.undistortMs= lapMs();

	TrackingFrameResult result;
	result.frameIndex= frameIndex;
	result.timestampMs= timestampMs;
	result.frameWidth= activeFrame->cols;
	result.frameHeight= activeFrame->rows;
	result.captureFps= m_captureFps;

	// Luminance-oscillation diagnostics run on every processed frame (cheap
	// decimated mean), so flicker is measurable even before a hand shows up
	m_flickerTracker.addFrame(*activeFrame, timestampMs);
	result.lumaInstability= m_flickerTracker.getInstability();
	result.lumaFlickerHz= m_flickerTracker.getDominantHz();
	timings.flickerMs= lapMs();

	bool bProducedTracking= false;
	if (m_bTrackingEnabled && m_pipeline != nullptr)
	{
		// Seed here rather than once per vision-thread iteration: the pipeline
		// only consumes hints inside process(), and setSearchHints REPLACES the
		// queue, so hints handed over on an iteration where this camera had no
		// frame were overwritten unseen. Measured before this moved: 4425
		// offered against 1659 that ever reached a decision.
		if (seedFromFused != nullptr)
			seedSearchHints(*seedFromFused, autoScaleFactor);

		m_pipeline->process(*activeFrame, result);

		// Raw frame capture, taken here so what lands on disk is EXACTLY what
		// the models consumed (undistorted, same pixels), which is what makes
		// an offline model comparison meaningful
		if (frameRecorder != nullptr)
			frameRecorder->enqueueFrame(m_cameraIndex, result.frameIndex, *activeFrame);

		// The hand pipeline reports its own inferenceMs; restart the lap so
		// the body-pose stage below is measured on its own
		lapMs();

		// Opt-in body-pose stage, same undistorted frame the hand pipeline
		// consumed (so its imagePoints share the undistorted camera matrix)
		if (bRecordingInputs)
			m_pendingRecordInput.bHaveBodyPose= false;
		if (m_bodyPoseTracker != nullptr && m_bodyPoseTracker->isLoaded())
		{
			m_bodyPoseTracker->process(*activeFrame, result.body);

			// Recording tap: the stage's cadence (divider re-emits) is baked
			// into the observation, so replay never re-runs the pose models
			if (bRecordingInputs)
			{
				m_pendingRecordInput.bHaveBodyPose= true;
				m_pendingRecordInput.body= result.body;
			}
		}
		timings.bodyPoseMs= lapMs();

		// Lighting/exposure diagnostics on the exact image the model consumed
		for (TrackedHand& hand : result.hands)
			HandRoiQuality::analyzeHand(*activeFrame, hand);
		timings.roiQualityMs= lapMs();

		// Recording tap, part 1: the pipeline-output hands, exactly as the
		// LandmarkTo3D call below consumes them
		if (bRecordingInputs)
		{
			m_pendingRecordInput.refLengthMeters= 0.f;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const TrackedHand& hand= result.hands[sideIndex];
				RecordedHandInput& outHand= m_pendingRecordInput.hands[sideIndex];
				outHand= RecordedHandInput();
				outHand.tracked= hand.tracked;
				if (!hand.tracked)
					continue;
				outHand.side= (int)hand.side;
				outHand.slotId= hand.slotId;
				outHand.presence= hand.presence;
				outHand.handednessScore= hand.handednessScore;
				outHand.rightProb= hand.rightProb;
				outHand.imagePoints= hand.imagePoints;
				outHand.modelPoints= hand.modelPoints;
				outHand.imageQuality= hand.imageQuality;
			}
		}

		// Image space -> camera space (needs intrinsics + hand scale)
		if (m_landmarkTo3D != nullptr)
		{
			// Recording tap, part 2: the hand scale in effect for this exact
			// process() call (the auto-scale EMA feedback loop becomes a
			// recorded input)
			if (bRecordingInputs)
			{
				m_pendingRecordInput.refLengthMeters=
					m_landmarkTo3D->getRefLengthMeters();
			}

			m_landmarkTo3D->process(result);

			// Camera space -> marker/world space (needs extrinsics)
			if (profile.extrinsics.present)
				applyWorldTransform(result, profile.extrinsics.markerFromCamera);
		}
		timings.liftMs= lapMs();

		bProducedTracking= true;
	}

	// Recording tap, part 3: frame metadata + the fresh flag. A popped frame
	// with tracking disabled records as fresh-but-invalid (replay advances
	// that camera's mirror timestamp without running LandmarkTo3D).
	if (bRecordingInputs)
	{
		RecordedCameraInput& record= m_pendingRecordInput;
		if (!bProducedTracking)
			record= RecordedCameraInput();
		record.cameraIndex= m_cameraIndex;
		record.timestampMs= timestampMs;
		record.valid= bProducedTracking;
		record.frameIndex= frameIndex;
		record.frameWidth= result.frameWidth;
		record.frameHeight= result.frameHeight;
		record.captureFps= result.captureFps;
		record.inferenceMs= result.inferenceMs;
		record.lumaInstability= result.lumaInstability;
		record.lumaFlickerHz= result.lumaFlickerHz;
		m_bPendingRecordFresh= true;
	}

	// Store the fusion input (the result itself follows once its timings are
	// complete)
	m_lastResult.cameraIndex= m_cameraIndex;
	m_lastResult.valid= bProducedTracking;
	m_lastResult.timestampMs= timestampMs;
	m_lastResult.hasExtrinsics= profile.extrinsics.present;
	m_lastResult.markerFromCamera= profile.extrinsics.markerFromCamera;
	// Undistorted pinhole for landmark triangulation (imagePoints space)
	m_lastResult.hasIntrinsics= profile.intrinsics.present;
	if (profile.intrinsics.present)
	{
		const MikanMatrix3d& cameraMatrix= profile.intrinsics.intrinsics.undistorted_camera_matrix;
		m_lastResult.fx= (float)cameraMatrix.x0;
		m_lastResult.fy= (float)cameraMatrix.y1;
		m_lastResult.cx= (float)cameraMatrix.z0;
		m_lastResult.cy= (float)cameraMatrix.z1;
	}

	// Publish this camera's preview (latest-wins)
	{
		std::lock_guard<std::mutex> lock(m_previewMutex);
		activeFrame->copyTo(m_previewFrame.bgr);
		// Timed inside the lock so the published copy carries its own cost;
		// the result copy that follows is small next to the frame
		timings.publishMs= lapMs();
		timings.totalMs= (float)(stepMarkMs - popMs);
		result.captureTimings= timings;
		m_previewFrame.result= result;
		m_previewFrame.valid= true;
		m_bPreviewFresh= true;
	}
	m_lastResult.result= result;

	return bProducedTracking;
}
