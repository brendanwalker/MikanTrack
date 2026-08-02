#include "VisionThread.h"

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/matrix.hpp"

#include "AppConfig.h"
#include "CVVideoFrameProcessor.h"
#include "HandTrackingPipeline.h"
#include "LandmarkTo3D.h"
#include "Logger.h"
#include "OscStreamer.h"
#include "SpaceTransforms.h"
#include "ThreadUtils.h"
#include "VideoCaptureSystem.h"
#include "VideoFrame.h"

VisionThread::VisionThread(VideoCaptureSystem* videoCapture, AppConfig* config)
	: m_videoCapture(videoCapture)
	, m_config(config)
{
}

VisionThread::~VisionThread()
{
	stop();
}

void VisionThread::start()
{
	if (m_bRunning)
		return;

	// Size the context list on the main thread BEFORE the thread spawns
	// (contexts are empty shells here; ORT sessions are created on the vision
	// thread in refreshConfigOnThread). The list is never resized while the
	// thread runs, so main-thread accessors can index it without locking.
	m_cameras.clear();
	for (size_t i= 0; i < m_config->cameraCount(); ++i)
	{
		auto context= std::make_unique<CameraContext>();
		context->cameraIndex= (int)i;
		m_cameras.push_back(std::move(context));
	}

	m_bConfigRefreshRequested= true;
	m_bRunning= true;
	m_thread= std::thread([this]() { threadLoop(); });
}

void VisionThread::stop()
{
	if (!m_bRunning)
		return;

	m_bRunning= false;
	if (m_thread.joinable())
		m_thread.join();
}

void VisionThread::setTrackingEnabled(int cameraIndex, bool bEnabled)
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_cameras.size())
		m_cameras[cameraIndex]->bTrackingEnabled= bEnabled;
}

bool VisionThread::isTrackingEnabled(int cameraIndex) const
{
	return cameraIndex >= 0 && cameraIndex < (int)m_cameras.size() && m_cameras[cameraIndex]->bTrackingEnabled;
}

void VisionThread::setUndistortEnabled(int cameraIndex, bool bEnabled)
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_cameras.size())
		m_cameras[cameraIndex]->bUndistortEnabled= bEnabled;
}

bool VisionThread::isUndistortEnabled(int cameraIndex) const
{
	return cameraIndex >= 0 && cameraIndex < (int)m_cameras.size() && m_cameras[cameraIndex]->bUndistortEnabled;
}

bool VisionThread::fetchPreviewFrame(int cameraIndex, VisionPreviewFrame& outFrame)
{
	if (cameraIndex < 0 || cameraIndex >= (int)m_cameras.size())
		return false;

	CameraContext& context= *m_cameras[cameraIndex];
	std::lock_guard<std::mutex> lock(context.previewMutex);
	if (!context.bPreviewFresh)
		return false;

	context.previewFrame.bgr.copyTo(outFrame.bgr);
	outFrame.result= context.previewFrame.result;
	outFrame.valid= true;
	context.bPreviewFresh= false;
	return true;
}

bool VisionThread::fetchFusedResult(TrackingFrameResult& outResult)
{
	std::lock_guard<std::mutex> lock(m_fusedMutex);
	if (!m_bFusedFresh)
		return false;

	outResult= m_fusedResult;
	m_bFusedFresh= false;
	return true;
}

bool VisionThread::fetchRestPoseCapture(std::vector<RestPoseCapture>& outCaptures)
{
	std::lock_guard<std::mutex> lock(m_restPoseMutex);
	if (!m_bRestPoseReady)
		return false;

	outCaptures= m_capturedRestPose;
	m_bRestPoseReady= false;
	return true;
}

void VisionThread::requestDiagnosticDump(const std::string& dumpDir)
{
	{
		std::lock_guard<std::mutex> lock(m_dumpMutex);
		m_requestedDumpDir= dumpDir;
	}
	m_bDumpRequested= true;
}

std::string VisionThread::getLastDumpPath()
{
	std::lock_guard<std::mutex> lock(m_dumpMutex);
	return m_lastDumpPath;
}

void VisionThread::performDiagnosticDump(const TrackingFrameResult& latestOutput)
{
	std::string dumpDir;
	{
		std::lock_guard<std::mutex> lock(m_dumpMutex);
		dumpDir= m_requestedDumpDir;
	}
	if (dumpDir.empty())
		return;

	std::vector<DiagCameraSnapshot> snapshots;
	for (const std::unique_ptr<CameraContext>& contextPtr : m_cameras)
	{
		const CameraContext& context= *contextPtr;

		DiagCameraSnapshot snapshot;
		snapshot.lastResult= &context.lastResult;
		snapshot.frame= context.lastActiveFrame;
		snapshot.deviceFps= m_videoCapture->getDeviceFrameRate(context.cameraIndex);
		snapshot.droppedFrames= m_videoCapture->getDroppedFrameCount(context.cameraIndex);
		snapshot.activeEp= context.activeEp.load();
		snapshot.trackingEnabled= context.bTrackingEnabled;
		snapshots.push_back(snapshot);
	}

	const bool bOk= m_diagnostics.write(dumpDir, snapshots, latestOutput, m_config->toJsonString());
	if (bOk)
	{
		std::lock_guard<std::mutex> lock(m_dumpMutex);
		m_lastDumpPath= dumpDir;
	}

	if (bOk)
		MIKAN_MT_LOG_INFO("VisionThread") << "Diagnostic dump written to " << dumpDir;
	else
		MIKAN_MT_LOG_ERROR("VisionThread") << "Diagnostic dump to " << dumpDir << " failed (partial output possible)";
}

float VisionThread::getObservationConfidence(int cameraIndex, eHandSide side) const
{
	if (cameraIndex < 0 || cameraIndex >= k_maxReportedCameras)
		return -1.f;

	return m_observationConfidence[cameraIndex * 2 + (int)side].load();
}

const char* VisionThread::getActiveExecutionProvider(int cameraIndex) const
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_cameras.size())
		return m_cameras[cameraIndex]->activeEp.load();

	return "none";
}

void VisionThread::refreshConfigOnThread()
{
	// (Re)build the per-camera context internals from the current config.
	// Runs on the vision thread, so ORT sessions live and die here. The
	// context LIST is fixed for the thread's lifetime (see start()); a camera
	// count change requires a thread restart from the app layer.
	if (m_config->cameraCount() != m_cameras.size())
	{
		MIKAN_MT_LOG_WARNING("VisionThread")
			<< "Camera count changed (" << m_cameras.size() << " -> " << m_config->cameraCount()
			<< ") - restart the vision thread to apply";
	}

	for (std::unique_ptr<CameraContext>& contextPtr : m_cameras)
	{
		CameraContext& context= *contextPtr;
		if (context.cameraIndex >= (int)m_config->cameraCount())
			continue;
		const CameraProfile& profile= m_config->camera(context.cameraIndex);

		// ML pipeline
		if (context.pipeline == nullptr)
		{
			HandTrackingPipelineConfig pipelineConfig;
			pipelineConfig.flipHandedness= m_config->tracking.flipHandedness;
			pipelineConfig.detectorIntervalFrames= m_config->tracking.detectorIntervalFrames;
			pipelineConfig.preferredEp= m_config->tracking.onnxEp;

			context.pipeline= std::make_unique<HandTrackingPipeline>();
			if (context.pipeline->startup(pipelineConfig))
			{
				context.activeEp= context.pipeline->getActiveExecutionProvider();
			}
			else
			{
				MIKAN_MT_LOG_ERROR("VisionThread")
					<< "HandTrackingPipeline startup failed for camera " << context.cameraIndex
					<< " - tracking disabled";
				context.pipeline= nullptr;
				context.activeEp= "none";
			}
		}
		else
		{
			HandTrackingPipelineConfig pipelineConfig= context.pipeline->getConfig();
			pipelineConfig.flipHandedness= m_config->tracking.flipHandedness;
			pipelineConfig.detectorIntervalFrames= m_config->tracking.detectorIntervalFrames;
			context.pipeline->setConfig(pipelineConfig);
		}

			// 3D projection (needs that camera's calibrated intrinsics).
		// Smoothing is always disabled here - the fused output is smoothed
		// after fusion instead (avoids double-filtering).
		if (profile.intrinsics.present)
		{
			if (context.landmarkTo3D == nullptr)
				context.landmarkTo3D= std::make_unique<LandmarkTo3D>();
			context.landmarkTo3D->configure(
				profile.intrinsics.intrinsics,
				m_config->handScale.refLengthMeters,
				false, // smoothing post-fusion
				m_config->tracking.palmMinCutoff,
				m_config->tracking.palmBeta);
			context.landmarkTo3D->setPnpConfig(m_config->tracking.usePnpDepth, m_config->tracking.pnpPalmOnly);

			// Undistortion for the ML input + preview
			if (context.undistorter == nullptr ||
				context.undistorter->getFrameWidth() != (int)profile.intrinsics.intrinsics.pixel_width ||
				context.undistorter->getFrameHeight() != (int)profile.intrinsics.intrinsics.pixel_height)
			{
				context.undistorter= std::make_unique<CVVideoFrameProcessor>(
					profile.intrinsics.intrinsics,
					(int)profile.intrinsics.intrinsics.pixel_width,
					(int)profile.intrinsics.intrinsics.pixel_height);
			}
		}
		else
		{
			context.landmarkTo3D= nullptr;
			context.undistorter= nullptr;
		}

		// This camera's own rest angles. Per camera: the model landmarks are
		// view-dependent, so removing each camera's bias is what makes their
		// angles agree well enough for fusion to blend them.
		if (context.landmarkTo3D != nullptr)
		{
			context.landmarkTo3D->clearRestAngles();
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				if (profile.restAngles.present[sideIndex])
					context.landmarkTo3D->setRestAngles((eHandSide)sideIndex,
														profile.restAngles.angles[sideIndex]);
			}
		}

		// Invalidate the last result so stale calibration state can't leak
		// through a config change
		context.lastResult= CameraFrameResult();
		context.lastResult.cameraIndex= context.cameraIndex;
	}

	// Config hand scale is the baseline the stereo correction applies to;
	// a refresh (e.g. after saving a new calibrated scale) resets the EMA
	m_autoScaleFactor= 1.f;

	// Fusion
	HandFusionConfig fusionConfig;
	fusionConfig.stalenessWindowMs= m_config->fusion.stalenessWindowMs;
	fusionConfig.wristMatchMaxDistM= m_config->fusion.wristMatchMaxDistM;
	fusionConfig.spatialSidePriorAxis= m_config->fusion.spatialSidePriorAxis;
	fusionConfig.minCameraConfidence= m_config->fusion.minCameraConfidence;
	fusionConfig.jitterReferenceM= m_config->fusion.jitterReferenceMm * 0.001f;
	fusionConfig.smoothingEnabled= m_config->tracking.smoothingEnabled;
	fusionConfig.palmMinCutoff= m_config->tracking.palmMinCutoff;
	fusionConfig.palmBeta= m_config->tracking.palmBeta;
	fusionConfig.angleMinCutoff= m_config->tracking.angleMinCutoff;
	fusionConfig.angleBeta= m_config->tracking.angleBeta;
	m_fusion.configure(fusionConfig);

	// OSC
	if (m_oscStreamer == nullptr)
	{
		m_oscStreamer= std::make_unique<OscStreamer>();
		if (!m_oscStreamer->startup())
		{
			MIKAN_MT_LOG_ERROR("VisionThread") << "OscStreamer startup failed - OSC output disabled";
			m_oscStreamer= nullptr;
		}
	}
	if (m_oscStreamer != nullptr)
	{
		OscStreamerConfig oscConfig;
		oscConfig.enabled= m_config->osc.enabled;
		oscConfig.targetIp= m_config->osc.targetIp;
		oscConfig.targetPort= (uint16_t)m_config->osc.targetPort;
		oscConfig.maxRateHz= (float)m_config->osc.maxRateHz;
		oscConfig.minConfidence= m_config->osc.minConfidence;
		m_oscStreamer->setConfig(oscConfig);
	}
}

void VisionThread::seedSearchHints(CameraContext& context, const TrackingFrameResult& lastFused)
{
	if (context.pipeline == nullptr || !context.bTrackingEnabled)
		return;

	const CameraProfile& profile= m_config->camera(context.cameraIndex);
	if (!profile.intrinsics.present || !profile.extrinsics.present)
		return;

	if (context.hintCooldownFrames > 0)
	{
		context.hintCooldownFrames--;
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

		// Only seed hands THIS camera is missing (some other camera sees it)
		if (context.lastResult.valid && context.lastResult.result.poses[sideIndex].tracked)
			continue;

		glm::vec2 centerPx;
		double depth= 0.0;
		if (!projectPoint(fusedPose.palmPositionWorld, centerPx, depth))
			continue;
		if (centerPx.x < 0.f || centerPx.x >= (float)width || centerPx.y < 0.f || centerPx.y >= (float)height)
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
			(float)(m_config->handScale.refLengthMeters * (double)m_autoScaleFactor.load());

		HandSearchHint hint;
		hint.centerPx= centerPx;
		hint.dirPx= dirLength > 1e-3f ? dirPx / dirLength : glm::vec2(0.f, -1.f);
		hint.palmSizePx= (float)(fx * (double)refLengthMeters / depth);
		hints.push_back(hint);
	}

	if (!hints.empty())
	{
		context.pipeline->setSearchHints(hints);
		context.hintCooldownFrames= 2; // retry every ~3 frames while unseen
	}
}

bool VisionThread::processCameraFrame(CameraContext& context)
{
	VideoFrameBlock* block= m_videoCapture->tryPopFrame(context.cameraIndex);
	if (block == nullptr)
		return false;

	// FPS estimate from frame timestamps
	if (context.lastFrameTimestampMs > 0.0 && block->timestampMs > context.lastFrameTimestampMs)
	{
		const float instFps= (float)(1000.0 / (block->timestampMs - context.lastFrameTimestampMs));
		context.captureFps= context.captureFps > 0.f ? context.captureFps * 0.9f + instFps * 0.1f : instFps;
	}
	context.lastFrameTimestampMs= block->timestampMs;

	// Raw -> BGR
	VideoCaptureSystem::convertFrameToBGR(*block, context.bgrScratch);
	const int64_t frameIndex= block->frameIndex;
	const double timestampMs= block->timestampMs;
	m_videoCapture->releaseFrame(context.cameraIndex, block);

	if (context.bgrScratch.empty())
		return false;

	const CameraProfile& profile= m_config->camera(context.cameraIndex);

	// Undistort when calibrated (ML + preview both use the undistorted image)
	cv::Mat* activeFrame= &context.bgrScratch;
	if (context.bUndistortEnabled &&
		context.undistorter != nullptr &&
		context.bgrScratch.cols == context.undistorter->getFrameWidth() &&
		context.bgrScratch.rows == context.undistorter->getFrameHeight())
	{
		context.undistorter->processColorFrame(context.bgrScratch, context.undistortedScratch);
		activeFrame= &context.undistortedScratch;
	}
	context.lastActiveFrame= activeFrame;

	TrackingFrameResult result;
	result.frameIndex= frameIndex;
	result.timestampMs= timestampMs;
	result.frameWidth= activeFrame->cols;
	result.frameHeight= activeFrame->rows;
	result.captureFps= context.captureFps;

	bool bProducedTracking= false;
	if (context.bTrackingEnabled && context.pipeline != nullptr)
	{
		context.pipeline->process(*activeFrame, result);

		// Image space -> camera space (needs intrinsics + hand scale)
		if (context.landmarkTo3D != nullptr)
		{
			context.landmarkTo3D->process(result);

			// Camera space -> marker/world space (needs extrinsics)
			if (profile.extrinsics.present)
				applyWorldTransform(result, profile.extrinsics.markerFromCamera);
		}

		bProducedTracking= true;
	}

	// Store the fusion input
	context.lastResult.cameraIndex= context.cameraIndex;
	context.lastResult.valid= bProducedTracking;
	context.lastResult.timestampMs= timestampMs;
	context.lastResult.hasExtrinsics= profile.extrinsics.present;
	context.lastResult.markerFromCamera= profile.extrinsics.markerFromCamera;
	context.lastResult.result= result;

	// Publish this camera's preview (latest-wins)
	{
		std::lock_guard<std::mutex> lock(context.previewMutex);
		activeFrame->copyTo(context.previewFrame.bgr);
		context.previewFrame.result= result;
		context.previewFrame.valid= true;
		context.bPreviewFresh= true;
	}

	return bProducedTracking;
}

void VisionThread::threadLoop()
{
	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread started";

	std::vector<const CameraFrameResult*> fusionCandidates;

	// Previous iteration's fused world result, used to seed cross-camera
	// search hints (vision-thread-local; the published copy is mutex-guarded)
	TrackingFrameResult lastFusedForHints;

	// Latest published output (world OR camera space) for diagnostic dumps
	TrackingFrameResult lastOutputResult;

	while (m_bRunning)
	{
		if (m_bConfigRefreshRequested.exchange(false))
			refreshConfigOnThread();

		// Process whichever cameras have a new frame (sequential; DirectML
		// serializes on one GPU queue anyway)
		bool bAnyNewResult= false;
		float inferenceMsSum= 0.f;
		double newestTimestampMs= 0.0;
		for (std::unique_ptr<CameraContext>& context : m_cameras)
		{
			if (m_cameras.size() > 1 && m_config->tracking.crossCameraSeeding)
				seedSearchHints(*context, lastFusedForHints);

			if (processCameraFrame(*context))
			{
				bAnyNewResult= true;
				inferenceMsSum+= context->lastResult.result.inferenceMs;
			}
			newestTimestampMs= std::max(newestTimestampMs, context->lastResult.timestampMs);
		}

		if (!bAnyNewResult)
		{
			// Still service dump requests while idle (cameras may be stopped)
			if (m_bDumpRequested.exchange(false))
				performDiagnosticDump(lastOutputResult);

			std::this_thread::sleep_for(std::chrono::milliseconds(2));
			continue;
		}

		m_lastInferenceMs= inferenceMsSum;

		// Fuse the cameras' world-space results (single fresh candidate
		// passes through exactly; stale/uncalibrated cameras are excluded)
		fusionCandidates.clear();
		bool bAnyWorldCandidate= false;
		for (const std::unique_ptr<CameraContext>& context : m_cameras)
		{
			fusionCandidates.push_back(&context->lastResult);
			bAnyWorldCandidate|= context->lastResult.valid && context->lastResult.hasExtrinsics;
		}

		TrackingFrameResult outputResult;
		if (bAnyWorldCandidate)
		{
			m_fusion.fuse(fusionCandidates, newestTimestampMs, outputResult);
			lastFusedForHints= outputResult;
			m_dominantCamera[0]= m_fusion.getDominantCamera(eHandSide::Left);
			m_dominantCamera[1]= m_fusion.getDominantCamera(eHandSide::Right);

			// Publish per-camera observation confidence for the UI readout
			for (std::atomic<float>& slot : m_observationConfidence)
				slot= -1.f;
			for (const FusionDiagnostics::Cluster& cluster : m_fusion.getLastDiagnostics().clusters)
			{
				if (cluster.assignedSide < 0)
					continue;
				for (const FusionDiagnostics::Observation& observation : cluster.observations)
				{
					if (observation.cameraIndex >= 0 && observation.cameraIndex < k_maxReportedCameras)
						m_observationConfidence[observation.cameraIndex * 2 + cluster.assignedSide]=
							observation.confidence;
				}
			}

			// Stereo auto hand-scale: slow EMA over the triangulated
			// correction, applied live to every camera's 3D projection
			float scaleSample= 1.f;
			if (m_config->tracking.autoHandScaleFromStereo && m_fusion.getStereoScaleSample(scaleSample))
			{
				constexpr float kScaleEmaAlpha= 0.02f;
				const float ema= m_autoScaleFactor.load() * (1.f - kScaleEmaAlpha) + scaleSample * kScaleEmaAlpha;
				m_autoScaleFactor= ema;

				const float effectiveRefLength= (float)m_config->handScale.refLengthMeters * ema;
				for (const std::unique_ptr<CameraContext>& context : m_cameras)
				{
					if (context->landmarkTo3D != nullptr)
						context->landmarkTo3D->setRefLengthMeters(effectiveRefLength);
				}
			}
		}
		else
		{
			// No calibrated camera: preserve the single-camera camera-space
			// behavior (OSC announces space=camera) using camera 0's result
			outputResult= m_cameras.empty() ? TrackingFrameResult() : m_cameras[0]->lastResult.result;
			lastFusedForHints= TrackingFrameResult(); // camera-space - can't project
			m_dominantCamera[0]= -1;
			m_dominantCamera[1]= -1;
		}

		if (m_oscStreamer != nullptr)
			m_oscStreamer->sendFrame(outputResult);

		// Publish the fused result (latest-wins)
		{
			std::lock_guard<std::mutex> lock(m_fusedMutex);
			m_fusedResult= outputResult;
			m_bFusedFresh= true;
		}
		lastOutputResult= outputResult;

		// Rest-pose capture: EVERY camera records what it currently reports for
		// each hand, so each one's own view-dependent bias is removed
		if (m_bRestPoseCaptureRequested.exchange(false))
		{
			std::vector<RestPoseCapture> captures;
			for (const std::unique_ptr<CameraContext>& context : m_cameras)
			{
				RestPoseCapture capture;
				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					const TrackedHand& hand= context->lastResult.result.hands[sideIndex];
					if (!context->lastResult.valid || !hand.tracked)
						continue;

					HandPoseModel::captureRestAngles(hand.modelPoints, hand.side, capture.angles[sideIndex]);
					capture.bCaptured[sideIndex]= true;
				}
				captures.push_back(capture);
			}

			std::lock_guard<std::mutex> lock(m_restPoseMutex);
			m_capturedRestPose= std::move(captures);
			m_bRestPoseReady= true;
		}

		// Diagnostic history (compact copies - cheap enough for every frame)
		{
			const int dominant[2]= {m_dominantCamera[0].load(), m_dominantCamera[1].load()};
			m_diagnostics.record(fusionCandidates, outputResult,
								 bAnyWorldCandidate ? m_fusion.getLastDiagnostics() : FusionDiagnostics(),
								 dominant, m_autoScaleFactor.load());
		}

		if (m_bDumpRequested.exchange(false))
			performDiagnosticDump(lastOutputResult);
	}

	// ORT sessions must be destroyed on this thread
	m_cameras.clear();
	m_oscStreamer= nullptr;

	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread stopped";
}
