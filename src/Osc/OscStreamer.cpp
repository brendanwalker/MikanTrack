#include "OscStreamer.h"

#include "Logger.h"
#include "TrackingTypes.h"

// Per-side OSC address tables, indexed by eHandSide (Left= 0, Right= 1)
static const char* k_handTrackedAddress[2]= {"/mikan/hand/left/tracked", "/mikan/hand/right/tracked"};
static const char* k_handPalmAddress[2]= {"/mikan/hand/left/palm", "/mikan/hand/right/palm"};
static const char* k_handWristAddress[2]= {"/mikan/hand/left/wrist", "/mikan/hand/right/wrist"};
static const char* k_handFingersAddress[2]= {"/mikan/hand/left/fingers", "/mikan/hand/right/fingers"};
static const char* k_handSkeletonAddress[2]= {"/mikan/hand/left/skeleton", "/mikan/hand/right/skeleton"};

static const char* k_frameAddress= "/mikan/frame";
static const char* k_infoAddress= "/mikan/info";

static const char* k_infoWorldSpace=
	"space=marker;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=rad";
static const char* k_infoCameraSpace=
	"space=camera;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=rad";

static void addVec3(OscMessage& message, const glm::vec3& point)
{
	message.addFloat(point.x).addFloat(point.y).addFloat(point.z);
}

OscStreamer::~OscStreamer()
{
	shutdown();
}

bool OscStreamer::startup()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_isRunning)
		return true;

	if (!m_socket.open())
	{
		MIKAN_LOG_ERROR("OscStreamer::startup") << "Failed to open UDP socket";
		return false;
	}

	m_isRunning= true;
	m_lastSendTimestampMs= -1.0;
	m_hasSentInfo= false;
	m_sentInWindow= 0;
	m_statsWindowStart= std::chrono::steady_clock::now();
	m_messagesPerSecond.store(0.f, std::memory_order_relaxed);

	MIKAN_LOG_INFO("OscStreamer::startup")
		<< "Streaming OSC to " << m_config.targetIp << ":" << m_config.targetPort;

	return true;
}

void OscStreamer::shutdown()
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (m_isRunning)
	{
		m_socket.close();
		m_isRunning= false;
		m_messagesPerSecond.store(0.f, std::memory_order_relaxed);
	}
}

OscStreamerConfig OscStreamer::getConfig() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_config;
}

void OscStreamer::setConfig(const OscStreamerConfig& config)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	m_config= config;
	m_hasSentInfo= false; // re-announce info on config change
}

void OscStreamer::sendFrame(const TrackingFrameResult& frame)
{
	std::lock_guard<std::mutex> lock(m_mutex);

	if (!m_isRunning || !m_config.enabled || !m_socket.isOpen())
		return;

	// Rate decimation on frame timestamps: skip the frame if it arrived less
	// than one send interval after the last sent frame. A backwards timestamp
	// jump (video restart) resets the gate.
	if (m_config.maxRateHz > 0.f && m_lastSendTimestampMs >= 0.0 && frame.timestampMs >= m_lastSendTimestampMs)
	{
		const double minIntervalMs= 1000.0 / static_cast<double>(m_config.maxRateHz);
		if ((frame.timestampMs - m_lastSendTimestampMs) < minIntervalMs)
			return;
	}
	m_lastSendTimestampMs= frame.timestampMs;

	// Do we have a marker-anchored world transform this frame?
	bool hasWorldSpace= false;
	for (const HandPose& pose : frame.poses)
	{
		if (pose.tracked && pose.hasWorldPose)
			hasWorldSpace= true;
	}

	m_bundle.clear();
	m_bundle.setTimeTag(k_oscTimeTagImmediate);

	// /mikan/frame ,iif frameId timestampMs fps
	OscMessage& frameMessage= m_bundle.addMessage(k_frameAddress);
	frameMessage.addInt32(static_cast<int32_t>(frame.frameIndex));
	frameMessage.addInt32(static_cast<int32_t>(frame.timestampMs));
	frameMessage.addFloat(frame.captureFps);

	// Skeleton geometry is slowly varying - ride the 1 Hz info cadence
	const ClockTimePoint now= std::chrono::steady_clock::now();
	const bool bSendSkeleton= !m_hasSentInfo || (now - m_lastInfoTime) >= std::chrono::seconds(1);

	for (int sideIndex= 0; sideIndex < static_cast<int>(eHandSide::Count); ++sideIndex)
	{
		appendHandMessages(frame, sideIndex, bSendSkeleton);
	}

	appendInfoMessage(hasWorldSpace, now);

	m_scratchBuffer.clear();
	m_bundle.encode(m_scratchBuffer);

	if (m_socket.sendTo(m_config.targetIp, m_config.targetPort,
						m_scratchBuffer.data(), static_cast<int>(m_scratchBuffer.size())))
	{
		m_sentInWindow++;
	}

	updateSendStats(now);
}

bool OscStreamer::resolveOutputPose(const HandPose& pose, double frameTimestampMs, float minConfidence,
									float holdMs, HeldPoseState& ioHeld, HandPose& outPose)
{
	// Low-confidence hands are withheld like untracked ones: a client that
	// holds its last good pose (or blends to a rest pose) looks far better
	// than one following a jittering estimate.
	const bool bLive= pose.tracked && pose.confidence >= minConfidence;
	if (bLive)
	{
		ioHeld.valid= true;
		ioHeld.timestampMs= frameTimestampMs;
		ioHeld.pose= pose;
		outPose= pose;
		return true;
	}

	// Dropout: bridge with the last good pose while its confidence decays
	// linearly to zero, so brief losses don't slam the client to rest pose
	// and back. A backwards timestamp (video restart) drops the hold.
	if (holdMs > 0.f && ioHeld.valid)
	{
		const double elapsedMs= frameTimestampMs - ioHeld.timestampMs;
		if (elapsedMs >= 0.0 && elapsedMs <= (double)holdMs)
		{
			outPose= ioHeld.pose;
			outPose.confidence= ioHeld.pose.confidence * (float)(1.0 - elapsedMs / (double)holdMs);
			return true;
		}
	}

	ioHeld.valid= false;
	outPose= pose;
	return false;
}

void OscStreamer::appendHandMessages(const TrackingFrameResult& frame, int sideIndex, bool bSendSkeleton)
{
	HandPose pose;
	const bool bSendPose= resolveOutputPose(frame.poses[sideIndex], frame.timestampMs,
											m_config.minConfidence, m_config.holdOnDropoutMs,
											m_heldPose[sideIndex], pose);

	// /mikan/hand/{s}/tracked ,iff tracked(0|1) presence confidence
	OscMessage& trackedMessage= m_bundle.addMessage(k_handTrackedAddress[sideIndex]);
	trackedMessage.addInt32(bSendPose ? 1 : 0);
	trackedMessage.addFloat(pose.presence);
	trackedMessage.addFloat(pose.confidence);

	if (!bSendPose)
		return;

	const bool bWorld= pose.hasWorldPose;
	const glm::vec3& palmPosition= bWorld ? pose.palmPositionWorld : pose.palmPositionCamera;
	const glm::quat& palmOrientation= bWorld ? pose.palmOrientationWorld : pose.palmOrientationCamera;

	// /mikan/hand/{s}/palm ,fffffff -- palm transform: position xyz +
	// quaternion xyzw. Palm frame: +X toward the fingers, +Z out of the
	// palmar surface, +Y right-handed; meters.
	OscMessage& palmMessage= m_bundle.addMessage(k_handPalmAddress[sideIndex]);
	addVec3(palmMessage, palmPosition);
	palmMessage.addFloat(palmOrientation.x)
		.addFloat(palmOrientation.y)
		.addFloat(palmOrientation.z)
		.addFloat(palmOrientation.w);

	// /mikan/hand/{s}/wrist ,iffffffff -- valid + forearm orientation (world,
	// xyzw) + wrist joint rotation (xyzw). The wrist rotation is the palm
	// expressed IN the forearm frame, which is exactly the local rotation a
	// skeleton applies to a hand bone parented to a forearm bone; identity
	// means the palm continues straight along the forearm.
	//
	// Sent unconditionally (with valid=0 and identity quaternions when no
	// IMU is calibrated) so a client can bind the address once instead of
	// handling an address that appears and disappears.
	{
		const bool bHasWrist= pose.hasForearmPose && bWorld;
		const glm::quat forearmOrientation=
			bHasWrist ? pose.forearmOrientationWorld : glm::quat(1.f, 0.f, 0.f, 0.f);
		const glm::quat wristRotation= bHasWrist ? pose.getWristRotation() : glm::quat(1.f, 0.f, 0.f, 0.f);

		OscMessage& wristMessage= m_bundle.addMessage(k_handWristAddress[sideIndex]);
		wristMessage.addInt32(bHasWrist ? 1 : 0);
		wristMessage.addFloat(forearmOrientation.x)
			.addFloat(forearmOrientation.y)
			.addFloat(forearmOrientation.z)
			.addFloat(forearmOrientation.w);
		wristMessage.addFloat(wristRotation.x)
			.addFloat(wristRotation.y)
			.addFloat(wristRotation.z)
			.addFloat(wristRotation.w);
	}

	// /mikan/hand/{s}/fingers ,f x20 -- per finger (thumb..pinky):
	// [lateral, proximalBend, intermediateBend, distalBend] radians,
	// relative to the neutral straight pose
	OscMessage& fingersMessage= m_bundle.addMessage(k_handFingersAddress[sideIndex]);
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const FingerAngles& angles= pose.fingers[finger];
		fingersMessage.addFloat(angles.lateral)
			.addFloat(angles.proximal)
			.addFloat(angles.intermediate)
			.addFloat(angles.distal);
	}

	// /mikan/hand/{s}/skeleton ,f x45 (1 Hz) -- per finger: base position in
	// the palm frame (xyz), phalanx lengths [proximal, intermediate, distal],
	// and the neutral direction in the palm frame (xyz) that the phalanx
	// points when all four of that finger's angles are zero. Everything a
	// client-side forward-kinematics setup needs.
	if (bSendSkeleton)
	{
		OscMessage& skeletonMessage= m_bundle.addMessage(k_handSkeletonAddress[sideIndex]);
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			addVec3(skeletonMessage, pose.skeleton.baseInPalm[finger]);
			skeletonMessage.addFloat(pose.skeleton.phalanxLengths[finger][0])
				.addFloat(pose.skeleton.phalanxLengths[finger][1])
				.addFloat(pose.skeleton.phalanxLengths[finger][2]);
			addVec3(skeletonMessage, pose.skeleton.neutralDirInPalm[finger]);
		}
	}
}

void OscStreamer::appendInfoMessage(bool hasWorldSpace, const ClockTimePoint& now)
{
	// /mikan/info ,ss -- sent at most once per second (wall clock)
	if (m_hasSentInfo && (now - m_lastInfoTime) < std::chrono::seconds(1))
		return;

	OscMessage& infoMessage= m_bundle.addMessage(k_infoAddress);
	infoMessage.addString(hasWorldSpace ? k_infoWorldSpace : k_infoCameraSpace);
	infoMessage.addString(m_config.appVersion);

	m_hasSentInfo= true;
	m_lastInfoTime= now;
}

void OscStreamer::updateSendStats(const ClockTimePoint& now)
{
	const std::chrono::duration<float> windowElapsed= now - m_statsWindowStart;
	if (windowElapsed >= std::chrono::seconds(1))
	{
		m_messagesPerSecond.store(
			static_cast<float>(m_sentInWindow) / windowElapsed.count(),
			std::memory_order_relaxed);
		m_sentInWindow= 0;
		m_statsWindowStart= now;
	}
}
