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

bool VisionThread::fetchPreviewFrame(VisionPreviewFrame& outFrame)
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

const char* VisionThread::getActiveExecutionProvider() const
{
	return m_pipeline != nullptr ? m_pipeline->getActiveExecutionProvider() : "none";
}

void VisionThread::refreshConfigOnThread()
{
	// (Re)build pipeline pieces from the current config. Runs on the vision
	// thread, so ORT sessions live and die here.
	if (m_pipeline == nullptr)
	{
		HandTrackingPipelineConfig pipelineConfig;
		pipelineConfig.flipHandedness= m_config->tracking.flipHandedness;
		pipelineConfig.usePoseModel= m_config->tracking.usePoseModel;
		pipelineConfig.detectorIntervalFrames= m_config->tracking.detectorIntervalFrames;
		pipelineConfig.poseFrameDivider= m_config->tracking.poseFrameDivider;
		pipelineConfig.preferredEp= m_config->tracking.onnxEp;

		m_pipeline= std::make_unique<HandTrackingPipeline>();
		if (!m_pipeline->startup(pipelineConfig))
		{
			MIKAN_MT_LOG_ERROR("VisionThread") << "HandTrackingPipeline startup failed - tracking disabled";
			m_pipeline= nullptr;
		}
	}
	else
	{
		HandTrackingPipelineConfig pipelineConfig= m_pipeline->getConfig();
		pipelineConfig.flipHandedness= m_config->tracking.flipHandedness;
		pipelineConfig.usePoseModel= m_config->tracking.usePoseModel;
		pipelineConfig.detectorIntervalFrames= m_config->tracking.detectorIntervalFrames;
		pipelineConfig.poseFrameDivider= m_config->tracking.poseFrameDivider;
		m_pipeline->setConfig(pipelineConfig);
	}

	// 3D projection (needs calibrated intrinsics)
	if (m_config->intrinsics.present)
	{
		if (m_landmarkTo3D == nullptr)
			m_landmarkTo3D= std::make_unique<LandmarkTo3D>();
		m_landmarkTo3D->configure(
			m_config->intrinsics.intrinsics,
			m_config->handScale.refLengthMeters,
			m_config->tracking.smoothingEnabled,
			m_config->tracking.smoothingMinCutoff,
			m_config->tracking.smoothingBeta);

		// Undistortion for the ML input + preview
		if (m_undistorter == nullptr ||
			m_undistorter->getFrameWidth() != (int)m_config->intrinsics.intrinsics.pixel_width ||
			m_undistorter->getFrameHeight() != (int)m_config->intrinsics.intrinsics.pixel_height)
		{
			m_undistorter= std::make_unique<CVVideoFrameProcessor>(
				m_config->intrinsics.intrinsics,
				(int)m_config->intrinsics.intrinsics.pixel_width,
				(int)m_config->intrinsics.intrinsics.pixel_height);
		}
	}
	else
	{
		m_landmarkTo3D= nullptr;
		m_undistorter= nullptr;
	}

	// OSC
	if (m_oscStreamer == nullptr)
		m_oscStreamer= std::make_unique<OscStreamer>();
	OscStreamerConfig oscConfig;
	oscConfig.enabled= m_config->osc.enabled;
	oscConfig.targetIp= m_config->osc.targetIp;
	oscConfig.targetPort= m_config->osc.targetPort;
	oscConfig.maxRateHz= m_config->osc.maxRateHz;
	m_oscStreamer->configure(oscConfig);
}

void VisionThread::threadLoop()
{
	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread started";

	cv::Mat bgrFrame;
	cv::Mat undistortedFrame;

	while (m_bRunning)
	{
		if (m_bConfigRefreshRequested.exchange(false))
			refreshConfigOnThread();

		VideoFrameBlock* block= m_videoCapture->tryPopFrame();
		if (block == nullptr)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(2));
			continue;
		}

		// FPS estimate from frame timestamps
		if (m_lastFrameTimestampMs > 0.0 && block->timestampMs > m_lastFrameTimestampMs)
		{
			const float instFps= (float)(1000.0 / (block->timestampMs - m_lastFrameTimestampMs));
			m_captureFps= m_captureFps > 0.f ? m_captureFps * 0.9f + instFps * 0.1f : instFps;
		}
		m_lastFrameTimestampMs= block->timestampMs;

		// Raw -> BGR
		VideoCaptureSystem::convertFrameToBGR(*block, bgrFrame);
		const int64_t frameIndex= block->frameIndex;
		const double timestampMs= block->timestampMs;
		m_videoCapture->releaseFrame(block);

		if (bgrFrame.empty())
			continue;

		// Undistort when calibrated (ML + preview both use the undistorted image)
		cv::Mat* activeFrame= &bgrFrame;
		if (m_undistorter != nullptr &&
			bgrFrame.cols == m_undistorter->getFrameWidth() &&
			bgrFrame.rows == m_undistorter->getFrameHeight())
		{
			m_undistorter->processColorFrame(bgrFrame, undistortedFrame);
			activeFrame= &undistortedFrame;
		}

		TrackingFrameResult result;
		result.frameIndex= frameIndex;
		result.timestampMs= timestampMs;
		result.frameWidth= activeFrame->cols;
		result.frameHeight= activeFrame->rows;
		result.captureFps= m_captureFps;

		if (m_bTrackingEnabled && m_pipeline != nullptr)
		{
			m_pipeline->process(*activeFrame, result);
			m_lastInferenceMs= result.inferenceMs;

			// Image space -> camera space (needs intrinsics + hand scale)
			if (m_landmarkTo3D != nullptr)
			{
				m_landmarkTo3D->process(result);

				// Camera space -> marker/world space (needs extrinsics)
				if (m_config->extrinsics.present)
					applyWorldTransform(result, m_config->extrinsics.markerFromCamera);
			}

			if (m_oscStreamer != nullptr)
				m_oscStreamer->sendFrame(result);
		}

		// Publish preview (latest-wins)
		{
			std::lock_guard<std::mutex> lock(m_previewMutex);
			activeFrame->copyTo(m_previewFrame.bgr);
			m_previewFrame.result= result;
			m_previewFrame.valid= true;
			m_bPreviewFresh= true;
		}
	}

	// ORT sessions must be destroyed on this thread
	m_pipeline= nullptr;
	m_landmarkTo3D= nullptr;
	m_oscStreamer= nullptr;
	m_undistorter= nullptr;

	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread stopped";
}
