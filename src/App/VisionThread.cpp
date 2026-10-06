#include "VisionThread.h"

#include "AppConfig.h"
#include "AvatarRetarget.h"
#include "AvatarSkeleton.h"
#include "FaceHead.h"
#include "FaceService.h"
#include "Logger.h"
#include "OscStreamer.h"
#include "SteadyClock.h"
#include "ThreadUtils.h"
#include "VideoCaptureSystem.h"

// A loop iteration longer than this starves every camera at once (frames keep
// arriving, find no free block, and are dropped). Well above a healthy
// iteration - two cameras of inference plus fusion runs in the low tens of ms.
static constexpr double k_hitchThresholdMs= 50.0;

const char* VisionThread::getPhaseName(eVisionPhase phase)
{
	switch (phase)
	{
	case eVisionPhase::ConfigRefresh: return "config";
	case eVisionPhase::Capture: return "capture+inference";
	case eVisionPhase::Imu: return "imu";
	case eVisionPhase::Fusion: return "fusion";
	case eVisionPhase::Face: return "face";
	case eVisionPhase::Osc: return "osc";
	case eVisionPhase::Diagnostics: return "diagnostics";
	default: return "none";
	}
}

void VisionThread::reportHitchIfSlow(double totalMs, const double* phaseMs)
{
	if (totalMs < k_hitchThresholdMs)
		return;

	eVisionPhase worstPhase= eVisionPhase::None;
	double worstMs= 0.0;
	for (int phaseIndex= 1; phaseIndex < (int)eVisionPhase::Count; ++phaseIndex)
	{
		if (phaseMs[phaseIndex] > worstMs)
		{
			worstMs= phaseMs[phaseIndex];
			worstPhase= (eVisionPhase)phaseIndex;
		}
	}

	m_hitchCount.fetch_add(1, std::memory_order_relaxed);
	m_lastHitchMs= (float)totalMs;
	m_lastHitchPhase= (int)worstPhase;

	MIKAN_MT_LOG_WARNING("VisionThread")
		<< "Frame loop hitch " << (int)totalMs << " ms (all cameras drop frames for this long) - worst: "
		<< getPhaseName(worstPhase) << " " << (int)worstMs << " ms [config "
		<< (int)phaseMs[(int)eVisionPhase::ConfigRefresh] << " capture "
		<< (int)phaseMs[(int)eVisionPhase::Capture] << " imu " << (int)phaseMs[(int)eVisionPhase::Imu]
		<< " fusion " << (int)phaseMs[(int)eVisionPhase::Fusion] << " face "
		<< (int)phaseMs[(int)eVisionPhase::Face] << " osc "
		<< (int)phaseMs[(int)eVisionPhase::Osc] << " diag "
		<< (int)phaseMs[(int)eVisionPhase::Diagnostics] << "]";
}

VisionThread::VisionThread(VideoCaptureSystem* videoCapture, ImuService* imuService, FaceService* faceService,
						   AppConfig* config)
	: m_videoCapture(videoCapture)
	, m_config(config)
	, m_imuService(imuService)
	, m_faceService(faceService)
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
		m_cameras.push_back(std::make_unique<CameraContext>((int)i, m_config));
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
		m_cameras[cameraIndex]->setTrackingEnabled(bEnabled);
}

bool VisionThread::isTrackingEnabled(int cameraIndex) const
{
	return cameraIndex >= 0 && cameraIndex < (int)m_cameras.size() && m_cameras[cameraIndex]->isTrackingEnabled();
}

void VisionThread::setUndistortEnabled(int cameraIndex, bool bEnabled)
{
	if (cameraIndex >= 0 && cameraIndex < (int)m_cameras.size())
		m_cameras[cameraIndex]->setUndistortEnabled(bEnabled);
}

bool VisionThread::isUndistortEnabled(int cameraIndex) const
{
	return cameraIndex >= 0 && cameraIndex < (int)m_cameras.size() && m_cameras[cameraIndex]->isUndistortEnabled();
}

bool VisionThread::fetchPreviewFrame(int cameraIndex, VisionPreviewFrame& outFrame)
{
	if (cameraIndex < 0 || cameraIndex >= (int)m_cameras.size())
		return false;

	return m_cameras[cameraIndex]->fetchPreviewFrame(outFrame);
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

bool VisionThread::fetchRestPoseCapture(RestPoseCapture& outFused)
{
	std::lock_guard<std::mutex> lock(m_restPoseMutex);
	if (!m_bRestPoseReady)
		return false;

	outFused= m_capturedFusedRest;
	m_bRestPoseReady= false;
	return true;
}

bool VisionThread::fetchBoneCalibration(BoneCalibrationCapture& outCapture)
{
	std::lock_guard<std::mutex> lock(m_boneCalibrationMutex);
	if (!m_bBoneCalibrationReady)
		return false;

	outCapture= m_capturedBones;
	m_bBoneCalibrationReady= false;
	return true;
}

bool VisionThread::fetchImuMountingCapture(ImuMountingCapture& outCapture)
{
	std::lock_guard<std::mutex> lock(m_imuMutex);
	if (!m_bImuMountingReady)
		return false;

	outCapture= m_capturedImuMounting;
	m_bImuMountingReady= false;
	return true;
}

ImuSideStatus VisionThread::getImuSideStatus(eHandSide side) const
{
	std::lock_guard<std::mutex> lock(m_imuMutex);
	return m_imuStatus[(int)side];
}

void VisionThread::requestRecordingStart(const std::string& filePath)
{
	{
		std::lock_guard<std::mutex> lock(m_recordingMutex);
		m_requestedRecordingPath= filePath;
	}
	m_bRecordingStartRequested= true;
}

void VisionThread::requestRecordingStop()
{
	m_bRecordingStopRequested= true;
}

std::string VisionThread::getLastRecordingPath()
{
	std::lock_guard<std::mutex> lock(m_recordingMutex);
	return m_lastRecordingPath;
}

void VisionThread::handleRecordingStartOnThread()
{
	std::string filePath;
	{
		std::lock_guard<std::mutex> lock(m_recordingMutex);
		filePath= m_requestedRecordingPath;
	}
	if (filePath.empty())
		return;

	// Reset the transient tracking state so replay's freshly constructed
	// instances start from the identical zero state as live. Also invalidate
	// every camera's fusion input: a pre-recording lastResult would feed the
	// first fuses with data the file cannot reproduce. m_autoScaleFactor is
	// deliberately NOT reset - the effective ref length is recorded per frame
	// as a plain input, and resetting the EMA would blip the live hand scale.
	m_fusion.resetTransientState();
	m_bodyPoseSolver.reset();
	for (std::unique_ptr<CameraContext>& context : m_cameras)
	{
		context->resetTransientState();
	}
	m_recordingSeq= 0;

	RecordingHeader header;
	header.formatVersion= TrackingRecording::k_formatVersion;
	header.appConfigJsonText= m_config->toJsonString();
	header.cameraCount= (int)m_cameras.size();

	if (m_recorder == nullptr)
		m_recorder= std::make_unique<TrackingRecorder>();
	m_recorder->start(filePath, TrackingRecording::headerToJson(header));

	// Raw frames are opt-in and local: a landmark recording is abstract
	// enough to share, frames are video of the room
	if (m_config->recording.recordRawFrames)
	{
		if (m_frameRecorder == nullptr)
			m_frameRecorder= std::make_unique<FrameRecorder>();
		m_frameRecorder->start(TrackingRecording::makeFrameDirectoryPath(filePath),
							   m_config->recording.jpegQuality);
	}
}

void VisionThread::finalizeRecordingOnThread(bool bAborted, const std::string& reason)
{
	if (m_recorder == nullptr || !m_recorder->isRecording())
		return;

	m_recorder->stop(bAborted, reason);
	if (m_frameRecorder != nullptr)
		m_frameRecorder->stop();
	std::lock_guard<std::mutex> lock(m_recordingMutex);
	m_lastRecordingPath= m_recorder->getFilePath();
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
		snapshot.lastResult= &context.getLastResult();
		snapshot.frame= context.getLastActiveFrame();
		snapshot.deviceFps= m_videoCapture->getDeviceFrameRate(context.getCameraIndex());
		snapshot.droppedFrames= m_videoCapture->getDroppedFrameCount(context.getCameraIndex());
		snapshot.callbackCopyMs= m_videoCapture->getCallbackCopyMs(context.getCameraIndex());
		snapshot.queuedFrames= m_videoCapture->getQueuedFrameCount(context.getCameraIndex());
		snapshot.activeEp= context.getActiveExecutionProvider();
		snapshot.trackingEnabled= context.isTrackingEnabled();
		snapshot.seedStats= context.getSeedStats();
		snapshots.push_back(snapshot);
	}

	// Raw IMU samples ride along so the sensor axis convention can be settled
	// by replaying candidate mappings against a real capture
	std::vector<DiagImuRawSample> rawImu[2];
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		std::vector<ImuSample> samples;
		m_imuService->getRawSampleHistory((eHandSide)sideIndex, samples);
		rawImu[sideIndex].reserve(samples.size());
		for (const ImuSample& sample : samples)
		{
			DiagImuRawSample raw;
			raw.timestampMs= sample.timestampMs;
			raw.acceleration= sample.acceleration;
			raw.angularVelocity= sample.angularVelocity;
			rawImu[sideIndex].push_back(raw);
		}
	}

	DiagImuCapture lastCapture[2];
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const MountingCaptureResult& capture= m_imuService->getLastCapture((eHandSide)sideIndex);
		DiagImuCapture& out= lastCapture[sideIndex];
		out.present= capture.bCaptured;
		out.forearmToSensor= capture.forearmToSensor;
		out.motionUsable= capture.bMotionUsable;
		out.poseSamples= capture.poseSamples;
		out.poseSpreadDegrees= capture.poseSpreadDegrees;
		out.axisDominance= capture.axisDominance;
		out.twistProgress= capture.twistProgress;
		out.twistReversal= capture.twistReversal;
		out.curlDominance= capture.curlDominance;
		out.curlProgress= capture.curlProgress;
		out.curlReversal= capture.curlReversal;
		out.curlStrokes= capture.curlStrokes;
		out.hingeSpreadDegrees= capture.hingeSpreadDegrees;
		out.interAxisAngleDegrees= capture.interAxisAngleDegrees;
		out.lengthMeasured= capture.bLengthMeasured;
		out.forearmLengthMeters= capture.forearmLengthMeters;
		out.lengthFitCorrelation= capture.lengthFitCorrelation;
		out.palmarSource= (int)capture.palmarSource;
	}

	const bool bOk= m_diagnostics.write(dumpDir, snapshots, latestOutput, m_config->toJsonString(),
										rawImu, lastCapture);
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
		return m_cameras[cameraIndex]->getActiveExecutionProvider();

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

	m_faceAnchor= m_config->face.headAnchorPresent ? m_config->face.headAnchor : glm::quat(1.f, 0.f, 0.f, 0.f);

	// A full bone calibration parks the auto hand-scale, so its factor is
	// pinned back to 1 rather than left wherever this session's EMA had
	// wandered to - it is still what the Hand Scale readout multiplies by.
	if (m_config->handSkeleton.present[0] && m_config->handSkeleton.present[1])
		m_autoScaleFactor= 1.f;

	for (std::unique_ptr<CameraContext>& context : m_cameras)
	{
		if (context->getCameraIndex() < (int)m_config->cameraCount())
			context->configure(*m_config);
	}

	// Config hand scale is the baseline the stereo correction applies to;
	// a refresh (e.g. after saving a new calibrated scale) resets the EMA
	m_autoScaleFactor= 1.f;

	// Fusion (one shared mapping - see makeHandFusionConfig)
	m_fusion.configure(makeHandFusionConfig(*m_config));

	// Wrist IMU
	{
		ImuServiceConfig imuConfig;
		imuConfig.enabled= m_config->imu.enabled;
		imuConfig.visionYawSigma= m_config->imu.visionYawSigma;
		imuConfig.swapSides= m_config->imu.swapSides;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			imuConfig.mountingPresent[sideIndex]= m_config->imu.mountingPresent[sideIndex];
			imuConfig.forearmToSensor[sideIndex]= m_config->imu.forearmToSensor[sideIndex];
		}
		m_imuService->setConfig(imuConfig);
	}

	// OSC
	applyOscConfigOnThread();
}

void VisionThread::applyOscConfigOnThread()
{
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
		const BodyDimensions bodyDimensions= makeBodyDimensions(*m_config);

		OscStreamerConfig oscConfig;
		oscConfig.enabled= m_config->osc.enabled;
		oscConfig.targetIp= m_config->osc.targetIp;
		oscConfig.port= (uint16_t)m_config->osc.port;
		oscConfig.maxRateHz= (float)m_config->osc.maxRateHz;
		oscConfig.minConfidence= m_config->osc.minConfidence;
		oscConfig.holdOnDropoutMs= m_config->osc.holdOnDropoutMs;
		// The VMC bone offsets must agree with the lengths the body-pose solver
		// placed those joints with, or the streamed skeleton contradicts the
		// pose it is carrying.
		oscConfig.forearmLengthMeters= bodyDimensions.forearmLengthMeters;
		oscConfig.upperArmLengthMeters= bodyDimensions.upperArmLengthMeters;
		oscConfig.shoulderWidthMeters= bodyDimensions.shoulderWidthMeters;
		oscConfig.vmcHeadOffsetMeters= m_config->osc.vmcHeadOffsetMeters;
		oscConfig.vmcFreezeOnLoss= m_config->osc.vmcFreezeOnLoss;
		// The avatar path takes the same lengths the solver placed the joints
		// with, plus the skeleton the main thread handed over
		oscConfig.bodyDimensions= bodyDimensions;
		oscConfig.avatarRetarget= makeAvatarRetargetConfig(*m_config, m_avatar.rig);
		oscConfig.avatarSkeleton= m_avatar.skeleton;
		oscConfig.avatarPath= m_avatarIdentity.path;
		oscConfig.avatarTitle= m_avatarIdentity.title;
		oscConfig.avatarSha256= m_avatarIdentity.sha256;
		oscConfig.faceMap= m_avatarIdentity.faceMap;
		m_oscStreamer->setConfig(oscConfig);
	}
}

// -- The frame loop ---------------------------------------------------------
//
// One iteration is a fixed sequence of stages. The stages are methods rather
// than objects: every handoff to the main thread (mutexes, atomics, fetch
// accessors) stays on VisionThread, and a stage reads and writes the
// iteration's scratch through IterationState. The recording taps are pinned:
// the fused checksum is taken immediately after fuse(), before the IMU fill,
// which is the point replay checksums at.

void VisionThread::threadLoop()
{
	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread started";

	// Per-phase timing for the hitch watchdog, reset each iteration
	double phaseMs[(int)eVisionPhase::Count]= {};

	while (m_bRunning)
	{
		const double iterationStartMs= steadyNowMs();
		for (double& phase : phaseMs)
			phase= 0.0;
		double phaseMarkMs= iterationStartMs;
		auto lapPhase= [&](eVisionPhase phase) {
			const double nowMs= steadyNowMs();
			phaseMs[(int)phase]+= nowMs - phaseMarkMs;
			phaseMarkMs= nowMs;
		};

		IterationState iteration;

		servicePendingRequests();
		lapPhase(eVisionPhase::ConfigRefresh);

		runCaptureStage(iteration);
		lapPhase(eVisionPhase::Capture);

		if (m_faceService != nullptr)
			m_faceService->update(steadyNowMs());
		lapPhase(eVisionPhase::Face);

		if (!iteration.bAnyNewResult)
		{
			if (runFaceOnlyOutput(iteration, steadyNowMs()))
				lapPhase(eVisionPhase::Osc);

			// Still service dump requests while idle (cameras may be stopped)
			if (m_bDumpRequested.exchange(false))
				performDiagnosticDump(m_lastOutputResult);
			lapPhase(eVisionPhase::Diagnostics);

			// The idle sleep below is not a hitch, but a slow config refresh or
			// dump write on this path still starves the cameras
			reportHitchIfSlow(steadyNowMs() - iterationStartMs, phaseMs);

			std::this_thread::sleep_for(std::chrono::milliseconds(2));
			continue;
		}

		runFusionStage(iteration);
		lapPhase(eVisionPhase::Fusion);

		if (iteration.bAnyWorldCandidate)
		{
			runImuStage(iteration);
			lapPhase(eVisionPhase::Imu);

			// The body solve and the fusion bookkeeping belong to the fusion
			// phase
			runBodySolveStage(iteration);
			lapPhase(eVisionPhase::Fusion);
		}

		runOutputStage(iteration);
		lapPhase(eVisionPhase::Osc);

		runCalibrationCaptures();
		runDiagnosticsStage(iteration);
		lapPhase(eVisionPhase::Diagnostics);

		reportHitchIfSlow(steadyNowMs() - iterationStartMs, phaseMs);
	}

	// Finalize an in-flight recording before the contexts are torn down
	finalizeRecordingOnThread(false, "vision thread stopped");

	// ORT sessions must be destroyed on this thread
	m_cameras.clear();
	m_oscStreamer= nullptr;

	MIKAN_MT_LOG_INFO("VisionThread") << "Vision thread stopped";
}

void VisionThread::setAvatarSkeleton(std::shared_ptr<const AvatarSkeleton> skeleton, const AvatarRigSettings& rig)
{
	{
		std::lock_guard<std::mutex> lock(m_avatarMutex);
		m_pendingAvatar.skeleton= std::move(skeleton);
		m_pendingAvatar.rig= rig;
	}
	m_bAvatarSkeletonChanged= true;
}

void VisionThread::setAvatarIdentity(const AvatarIdentity& identity)
{
	{
		std::lock_guard<std::mutex> lock(m_avatarMutex);
		m_pendingAvatarIdentity= identity;
	}
	m_bAvatarIdentityChanged= true;
}

bool VisionThread::fetchFaceAnchorCapture(glm::quat& outAnchor)
{
	std::lock_guard<std::mutex> lock(m_faceAnchorMutex);
	if (!m_bFaceAnchorReady)
		return false;

	outAnchor= m_capturedFaceAnchor;
	m_bFaceAnchorReady= false;
	return true;
}

void VisionThread::servicePendingRequests()
{
	if (m_bConfigRefreshRequested.exchange(false))
	{
		// A refresh wipes fusion inputs and resets fusion state - a hard
		// discontinuity the recording's header snapshot cannot describe
		finalizeRecordingOnThread(false, "config changed");
		refreshConfigOnThread();
	}
	const bool bSkeletonChanged= m_bAvatarSkeletonChanged.exchange(false);
	const bool bIdentityChanged= m_bAvatarIdentityChanged.exchange(false);
	if (bSkeletonChanged || bIdentityChanged)
	{
		{
			std::lock_guard<std::mutex> lock(m_avatarMutex);
			if (bSkeletonChanged)
				m_avatar= m_pendingAvatar;
			if (bIdentityChanged)
				m_avatarIdentity= m_pendingAvatarIdentity;
		}
		applyOscConfigOnThread();
	}
	if (m_bRecordingStopRequested.exchange(false))
		finalizeRecordingOnThread(false, "");
	// Start AFTER any refresh so the header snapshot reflects it
	if (m_bRecordingStartRequested.exchange(false))
		handleRecordingStartOnThread();
}

void VisionThread::runCaptureStage(IterationState& iteration)
{
	// Process whichever cameras have a new frame (sequential; DirectML
	// serializes on one GPU queue anyway)
	// The recording taps inside process() are live only while a recording
	// is active; both recorders can only change state in the request stage
	const bool bRecordingInputs= m_recorder != nullptr && m_recorder->isRecording();
	FrameRecorder* frameRecorder=
		m_frameRecorder != nullptr && m_frameRecorder->isRecording() ? m_frameRecorder.get() : nullptr;

	float inferenceMsSum= 0.f;
	for (std::unique_ptr<CameraContext>& context : m_cameras)
	{
		if (context->process(m_videoCapture, m_cameras.size() > 1 ? &m_lastFusedForHints : nullptr,
							 m_autoScaleFactor.load(), bRecordingInputs, frameRecorder))
		{
			iteration.bAnyNewResult= true;
			inferenceMsSum+= context->getLastResult().result.inferenceMs;
		}
		iteration.newestTimestampMs=
			std::max(iteration.newestTimestampMs, context->getLastResult().timestampMs);
	}

	if (iteration.bAnyNewResult)
		m_lastInferenceMs= inferenceMsSum;
}

void VisionThread::runFusionStage(IterationState& iteration)
{
	// Fuse the cameras' world-space results (single fresh candidate passes
	// through exactly; stale/uncalibrated cameras are excluded)
	for (const std::unique_ptr<CameraContext>& context : m_cameras)
	{
		const CameraFrameResult& lastResult= context->getLastResult();
		iteration.fusionCandidates.push_back(&lastResult);
		iteration.bAnyWorldCandidate|= lastResult.valid && lastResult.hasExtrinsics;
	}

	iteration.bRecordingThisFrame= m_recorder != nullptr && m_recorder->isRecording();

	if (iteration.bAnyWorldCandidate)
	{
		m_fusion.fuse(iteration.fusionCandidates, iteration.newestTimestampMs, iteration.outputResult);

		// Recording: fused output + checksum, PRE-IMU (the forearm fill in the
		// IMU stage mutates the poses; replay checksums at this same point)
		if (iteration.bRecordingThisFrame)
		{
			iteration.recordFrame.bFused= true;
			TrackingRecording::snapshotFusedOutput(iteration.outputResult, iteration.recordFrame.outPoses);
			iteration.recordFrame.checksum= TrackingRecording::computeFusedChecksum(iteration.outputResult);
		}
	}
	else
	{
		// No calibrated camera: preserve the single-camera camera-space
		// behavior (OSC announces space=camera) using camera 0's result
		iteration.outputResult=
			m_cameras.empty() ? TrackingFrameResult() : m_cameras[0]->getLastResult().result;
		m_lastFusedForHints= TrackingFrameResult(); // camera-space - can't project
		m_dominantCamera[0]= -1;
		m_dominantCamera[1]= -1;

		// Recording: passthrough frames checksum too (replay reconstructs
		// them from its own camera-0 mirror, so this still verifies the
		// LandmarkTo3D stage)
		if (iteration.bRecordingThisFrame)
		{
			iteration.recordFrame.bFused= false;
			TrackingRecording::snapshotFusedOutput(iteration.outputResult, iteration.recordFrame.outPoses);
			iteration.recordFrame.checksum= TrackingRecording::computeFusedChecksum(iteration.outputResult);
		}
	}
}

void VisionThread::runImuStage(IterationState& iteration)
{
	// Integrate every buffered inertial sample (they carry their own
	// timestamps, so running at camera rate loses no information), then let
	// the fused palm orientation anchor yaw, then publish the forearm
	// orientation onto the pose.
	TrackingFrameResult& outputResult= iteration.outputResult;

	if (m_bImuMotionRecordingRequested.exchange(false))
	{
		const eMountingMotion motion= (eMountingMotion)m_requestedImuMotionRecording.load();
		if (motion == eMountingMotion::None)
			m_imuService->endMotionRecording();
		else
			m_imuService->beginMotionRecording(motion);
	}
	if (m_bImuBiasCalibrationRequested.exchange(false))
		m_imuService->beginBiasCalibration();
	if (m_bImuBiasCancelRequested.exchange(false))
		m_imuService->cancelBiasCalibration();
	m_imuService->update();

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= outputResult.poses[sideIndex];
		if (pose.tracked && pose.hasWorldPose)
		{
			m_imuService->applyVisionPalmOrientation((eHandSide)sideIndex, pose.palmOrientationWorld);

			// Feeds the mounting average. Runs every tracked frame rather
			// than only during the wizard, so a capture always has a
			// populated window behind it.
			m_imuService->accumulatePoseMounting((eHandSide)sideIndex, pose.palmOrientationWorld,
												 pose.confidence);

			// Health check on the mounting's roll: the wrist cannot rotate
			// about the forearm's long axis, so any axial component of the
			// measured joint is calibration error
			m_imuService->updateWristAxialResidual((eHandSide)sideIndex, pose.palmOrientationWorld,
												   pose.confidence);
		}
	}
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		glm::quat forearmToWorld(1.f, 0.f, 0.f, 0.f);
		if (m_imuService->getForearmOrientation((eHandSide)sideIndex, forearmToWorld))
		{
			// The EKF owns this orientation outright - deliberately no
			// additional smoothing, which would cascade two filters onto one
			// signal
			outputResult.poses[sideIndex].hasForearmPose= true;
			outputResult.poses[sideIndex].forearmOrientationWorld= forearmToWorld;

			// The elbow rides on both the palm position and the forearm
			// direction, so it is only as good as the weaker of the two.
			// Mounting quality is the one a consumer cannot see for itself: a
			// bad mounting leaves the hand looking perfect while the elbow
			// sweeps a cone.
			const ImuSideStatus status= m_imuService->getSideStatus((eHandSide)sideIndex);
			const float mountingQuality= status.forearmAxisConsistency < 0.f
				// Not enough motion to score it yet. Treated as good because
				// the calibration wizard refuses a mounting whose arm axis
				// was not measurable in the first place.
				? 1.f
				: std::clamp(status.forearmAxisConsistency, 0.f, 1.f);
			outputResult.poses[sideIndex].forearmConfidence=
				outputResult.poses[sideIndex].confidence * mountingQuality;
		}
	}

	// Recording: the published forearm output. The IMU EKF is not replayed
	// (fusion never reads it); replay overlays these onto the replayed poses
	// for display.
	if (iteration.bRecordingThisFrame)
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const HandPose& pose= outputResult.poses[sideIndex];
			iteration.recordFrame.imu[sideIndex].hasForearmPose= pose.hasForearmPose;
			iteration.recordFrame.imu[sideIndex].forearmOrientationWorld= pose.forearmOrientationWorld;
			iteration.recordFrame.imu[sideIndex].forearmConfidence= pose.forearmConfidence;
		}
	}

	// Mounting capture: needs a tracked palm AND a converged filter
	if (m_bImuMountingCaptureRequested.exchange(false))
	{
		// No pose argument and no "is the hand tracked right now" test: the
		// capture consumes the averaged window, so what matters is what was
		// seen during the twist, not this instant
		ImuMountingCapture capture;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			m_imuService->captureMounting((eHandSide)sideIndex, capture.sides[sideIndex]);

		std::lock_guard<std::mutex> lock(m_imuMutex);
		m_capturedImuMounting= capture;
		m_bImuMountingReady= true;
	}

	{
		std::lock_guard<std::mutex> lock(m_imuMutex);
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			m_imuStatus[sideIndex]= m_imuService->getSideStatus((eHandSide)sideIndex);
	}
}

void VisionThread::runBodySolveStage(IterationState& iteration)
{
	// Vision body pose: elbows for sides the IMU didn't claim, plus shoulders
	// and head. Runs AFTER the IMU forearm fill (IMU wins) and after the IMU
	// recording tap (so recordings keep pure IMU output and replay can re-run
	// this solver for what-if A/Bs).
	m_bodyPoseSolver.solve(iteration.fusionCandidates, makeBodyDimensions(*m_config), iteration.outputResult);

	// Fusion bookkeeping: the seed for the next iteration's search hints, the
	// dominant camera and per-camera confidence for the UI readout, and the
	// stereo auto hand-scale
	m_lastFusedForHints= iteration.outputResult;
	m_dominantCamera[0]= m_fusion.getDominantCamera(eHandSide::Left);
	m_dominantCamera[1]= m_fusion.getDominantCamera(eHandSide::Right);

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

	// Stereo auto hand-scale: slow EMA over the triangulated correction,
	// applied live to every camera's 3D projection. A calibrated skeleton
	// supersedes it - that measurement IS the hand's geometry, and two
	// mechanisms setting scale at once would only fight. The EMA stays for
	// hands with no calibration.
	const bool bBothSidesCalibrated=
		m_config->handSkeleton.present[0] && m_config->handSkeleton.present[1];
	float scaleSample= 1.f;
	if (!bBothSidesCalibrated && m_fusion.getStereoScaleSample(scaleSample))
	{
		constexpr float kScaleEmaAlpha= 0.02f;
		const float ema= m_autoScaleFactor.load() * (1.f - kScaleEmaAlpha) + scaleSample * kScaleEmaAlpha;
		m_autoScaleFactor= ema;

		const float effectiveRefLength= (float)m_config->handScale.refLengthMeters * ema;
		for (const std::unique_ptr<CameraContext>& context : m_cameras)
			context->setRefLengthMeters(effectiveRefLength);
	}
}

void VisionThread::runOutputStage(IterationState& iteration)
{
	if (iteration.bAnyNewResult)
		m_lastCameraOutputMs= iteration.outputResult.timestampMs;
	applyFaceToOutput(iteration.outputResult, steadyNowMs(), iteration.bAnyNewResult);

	if (m_oscStreamer != nullptr)
		m_oscStreamer->sendFrame(iteration.outputResult);

	// Publish the result (latest-wins)
	{
		std::lock_guard<std::mutex> lock(m_fusedMutex);
		m_fusedResult= iteration.outputResult;
		m_bFusedFresh= true;
	}
	m_lastOutputResult= iteration.outputResult;
}

void VisionThread::applyFaceToOutput(TrackingFrameResult& output, double nowMs, bool bCameraHead)
{
	output.face.present= false;

	FaceSample sample;
	if (m_faceService == nullptr || !m_faceService->isStreaming(nowMs) || !m_faceService->getLatestSample(sample))
		return;
	// The phone lost the face: its values are stale, so the face relaxes and
	// the camera head stands
	if (!sample.faceTracked)
	{
		m_lastFaceSequence= sample.sequence;
		return;
	}

	// A camera frame carrying this sample means no face-only frame repeats it
	m_lastFaceSequence= sample.sequence;
	output.face.present= true;
	static_assert(std::tuple_size<decltype(output.face.blendshapes)>::value == ARKIT_BLENDSHAPE_COUNT,
				  "the output's face pose follows the ARKit table");
	output.face.blendshapes= sample.blendshapes;

	if (!sample.hasHead)
		return;

	// The anchor capture needs the CAMERA head, so it runs before the phone
	// replaces it, and never on a face-only frame (whose head is the phone's)
	if (bCameraHead && output.head.valid && m_bFaceAnchorCaptureRequested.exchange(false))
	{
		std::lock_guard<std::mutex> lock(m_faceAnchorMutex);
		m_capturedFaceAnchor= faceHeadAnchorFromWorld(output.head.orientationWorld, sample.headEulerDegrees);
		m_bFaceAnchorReady= true;
	}

	// The phone measures the head far better than the body-pose stage, so it
	// wins outright while the stream is live. The position stays the camera
	// head's: the phone's is relative to the phone.
	output.head.orientationWorld= faceHeadWorldOrientation(m_faceAnchor, sample.headEulerDegrees);
	output.head.valid= true;
	output.head.confidence= 1.f;
}

bool VisionThread::runFaceOnlyOutput(IterationState& iteration, double nowMs)
{
	FaceSample sample;
	if (m_faceService == nullptr || !m_faceService->getLatestSample(sample) || sample.sequence == m_lastFaceSequence)
		return false;
	m_lastFaceSequence= sample.sequence;
	// A lost face has nothing new to send, beyond the one frame that relaxes
	// a face the last output still carried
	if (!sample.faceTracked && !m_lastOutputResult.face.present)
		return false;

	// Repeats the last output with the new face. The hands it carries stay
	// live while the cameras keep up; once the last camera frame is older
	// than the dropout window they go out untracked, so the streamer's hold
	// and freeze rules see a loss rather than a pose that never ages.
	// The sample's own arrival time keeps the streamer's rate gate pacing the
	// interleaved stream. A camera frame stamped before it then reads as a
	// timestamp regression, which the gate answers by sending that frame.
	iteration.outputResult= m_lastOutputResult;
	iteration.outputResult.timestampMs= sample.timestampMs;
	const double holdMs= (double)m_config->osc.holdOnDropoutMs;
	if (m_lastCameraOutputMs < 0.0 || sample.timestampMs - m_lastCameraOutputMs > holdMs)
	{
		for (HandPose& pose : iteration.outputResult.poses)
			pose.tracked= false;
	}

	runOutputStage(iteration);
	return true;
}

void VisionThread::runCalibrationCaptures()
{
	// Bone calibration: accumulate the stereo-triangulated landmarks over an
	// open window. Only triangulated frames carry measured geometry - a
	// monocular pose is the landmark model's shape wearing a pose, which is
	// exactly what this calibration exists to stop trusting.
	if (m_bBoneCalibrationRequested.exchange(false))
	{
		m_boneCalibrator.reset();
		m_boneCalibrationSamples[0]= 0;
		m_boneCalibrationSamples[1]= 0;
		m_boneCalibrationEndMs= steadyNowMs() + 1000.0 * (double)m_boneCalibrationSeconds.load();
		m_bBoneCalibrationActive= true;
	}
	if (m_bBoneCalibrationCancelRequested.exchange(false) && m_bBoneCalibrationActive.load())
	{
		m_boneCalibrator.reset();
		m_boneCalibrationSamples[0]= 0;
		m_boneCalibrationSamples[1]= 0;
		m_bBoneCalibrationActive= false;
	}
	if (m_bBoneCalibrationActive.load())
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			// Bones are measured from the TRIANGULATED landmarks, asked for
			// explicitly: the streamed pose is built on a skeleton, so
			// calibrating from it would re-measure the skeleton the estimator
			// was already given rather than the user's hand.
			std::array<glm::vec3, HAND_LANDMARK_COUNT> triPoints;
			if (!m_fusion.getLastTriangulatedPoints((eHandSide)sideIndex, triPoints))
				continue;

			m_boneCalibrator.addSample((eHandSide)sideIndex, triPoints);
			m_boneCalibrationSamples[sideIndex]= m_boneCalibrator.getSampleCount((eHandSide)sideIndex);
		}

		if (steadyNowMs() >= m_boneCalibrationEndMs)
		{
			BoneCalibrationCapture capture;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				capture.bCaptured[sideIndex]= m_boneCalibrator.finish(
					(eHandSide)sideIndex, capture.skeleton[sideIndex], capture.quality[sideIndex]);
			}

			{
				std::lock_guard<std::mutex> lock(m_boneCalibrationMutex);
				m_capturedBones= capture;
				m_bBoneCalibrationReady= true;
			}
			m_bBoneCalibrationActive= false;
		}
	}

	// Rest-pose capture: the zero reference is the RAW multi-view angles of
	// the fuse that just ran (stereo-quality only - a monocular pose carries
	// the model's view-dependent bias, which is exactly what a zero reference
	// must not bake in)
	if (m_bRestPoseCaptureRequested.exchange(false))
	{
		RestPoseCapture fusedCapture;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			fusedCapture.bCaptured[sideIndex]=
				m_fusion.getLastRawTriangulatedAngles((eHandSide)sideIndex, fusedCapture.angles[sideIndex]);
		}

		std::lock_guard<std::mutex> lock(m_restPoseMutex);
		m_capturedFusedRest= fusedCapture;
		m_bRestPoseReady= true;
	}
}

void VisionThread::runDiagnosticsStage(IterationState& iteration)
{
	// Diagnostic history (compact copies - cheap enough for every frame)
	{
		const int dominant[2]= {m_dominantCamera[0].load(), m_dominantCamera[1].load()};

		DiagImuState imuStates[2];
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const ImuSideStatus status= m_imuService->getSideStatus((eHandSide)sideIndex);
			DiagImuState& imuState= imuStates[sideIndex];
			imuState.deviceConnected= status.deviceConnected;
			imuState.streaming= status.streaming;
			imuState.calibrated= status.calibrated;
			imuState.orientationValid= status.orientationValid;
			imuState.sampleRateHz= status.sampleRateHz;
			imuState.millisecondsSinceLastSample= status.millisecondsSinceLastSample;
			imuState.forearmAxisConsistency= status.forearmAxisConsistency;
			imuState.armAxisDominance= status.armAxisDominance;
			imuState.twistProgress= status.twistProgress;
			imuState.twistReversal= status.twistReversal;
			imuState.wristAxialTwistDegrees= status.wristAxialTwistDegrees;
			imuState.gyroBiasDegreesPerSecond= status.gyroBiasDegreesPerSecond;
			imuState.biasSaturated= status.biasSaturated;
			imuState.yawSigmaRadians= status.yawSigmaRadians;
			imuState.filterOrientation= status.filterOrientation;
			imuState.tiltSigmaRadians= status.tiltSigmaRadians;
			imuState.gravityAcceptRatio= status.gravityAcceptRatio;
			imuState.visionYawCorrectionDegrees= status.visionYawCorrectionDegrees;
		}

		m_diagnostics.record(iteration.fusionCandidates, iteration.outputResult,
							 iteration.bAnyWorldCandidate ? m_fusion.getLastDiagnostics() : FusionDiagnostics(),
							 dominant, m_autoScaleFactor.load(), imuStates);
	}

	// Recording: assemble this iteration's record (the fresh cameras' staged
	// inputs + the fused output taps above) and hand it to the writer thread
	if (iteration.bRecordingThisFrame)
	{
		RecordedFrame& recordFrame= iteration.recordFrame;
		recordFrame.seq= m_recordingSeq++;
		recordFrame.nowTimestampMs= iteration.newestTimestampMs;
		for (std::unique_ptr<CameraContext>& context : m_cameras)
		{
			RecordedCameraInput input;
			if (context->consumePendingRecordInput(input))
				recordFrame.freshCameras.push_back(std::move(input));
		}
		m_recorder->enqueueFrame(std::move(recordFrame));
	}

	if (m_bDumpRequested.exchange(false))
		performDiagnosticDump(m_lastOutputResult);
}
