#include "VisionThread.h"

#include "glm/ext/matrix_double4x4.hpp"

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
				m_config->tracking.smoothingMinCutoff,
				m_config->tracking.smoothingBeta);

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

		// Invalidate the last result so stale calibration state can't leak
		// through a config change
		context.lastResult= CameraFrameResult();
		context.lastResult.cameraIndex= context.cameraIndex;
	}

	// Fusion
	HandFusionConfig fusionConfig;
	fusionConfig.stalenessWindowMs= m_config->fusion.stalenessWindowMs;
	fusionConfig.softmaxTemperature= m_config->fusion.softmaxTemperature;
	fusionConfig.wristMatchMaxDistM= m_config->fusion.wristMatchMaxDistM;
	fusionConfig.smoothingEnabled= m_config->tracking.smoothingEnabled;
	fusionConfig.smoothingMinCutoff= m_config->tracking.smoothingMinCutoff;
	fusionConfig.smoothingBeta= m_config->tracking.smoothingBeta;
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
		m_oscStreamer->setConfig(oscConfig);
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
			{
				applyWorldTransform(result, profile.extrinsics.markerFromCamera);

				// Recompute fallback elbows in world space with the
				// table-plane clamp (fixes below-the-table elbows)
				context.landmarkTo3D->refineFallbackArms(result, profile.extrinsics.markerFromCamera);
			}
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
			if (processCameraFrame(*context))
			{
				bAnyNewResult= true;
				inferenceMsSum+= context->lastResult.result.inferenceMs;
			}
			newestTimestampMs= std::max(newestTimestampMs, context->lastResult.timestampMs);
		}

		if (!bAnyNewResult)
		{
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
			m_dominantCamera[0]= m_fusion.getDominantCamera(eHandSide::Left);
			m_dominantCamera[1]= m_fusion.getDominantCamera(eHandSide::Right);
		}
		else
		{
			// No calibrated camera: preserve the single-camera camera-space
			// behavior (OSC announces space=camera) using camera 0's result
			outputResult= m_cameras.empty() ? TrackingFrameResult() : m_cameras[0]->lastResult.result;
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
	}

	// ORT sessions must be destroyed on this thread
	m_cameras.clear();
	m_oscStreamer= nullptr;

	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread stopped";
}
