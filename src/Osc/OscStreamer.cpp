#include "OscStreamer.h"

#include "Logger.h"
#include "TrackingTypes.h"

#include <cstdio>

// Per-side OSC address tables, indexed by eHandSide (Left= 0, Right= 1)
static const char* k_handTrackedAddress[2]= {"/mikan/hand/left/tracked", "/mikan/hand/right/tracked"};
static const char* k_handPalmAddress[2]= {"/mikan/hand/left/palm", "/mikan/hand/right/palm"};
static const char* k_handWristAddress[2]= {"/mikan/hand/left/wrist", "/mikan/hand/right/wrist"};
static const char* k_handElbowAddress[2]= {"/mikan/hand/left/elbow", "/mikan/hand/right/elbow"};
static const char* k_handShoulderAddress[2]= {"/mikan/hand/left/shoulder", "/mikan/hand/right/shoulder"};
static const char* k_handFingersAddress[2]= {"/mikan/hand/left/fingers", "/mikan/hand/right/fingers"};
static const char* k_handSkeletonAddress[2]= {"/mikan/hand/left/skeleton", "/mikan/hand/right/skeleton"};

static const char* k_frameAddress= "/mikan/frame";
static const char* k_headAddress= "/mikan/body/head";
static const char* k_infoAddress= "/mikan/info";

static const char* k_infoWorldSpace=
	"space=marker;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=deg";
static const char* k_infoCameraSpace=
	"space=camera;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=deg";

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
	// Restart the sequence with the socket, so a client that reconnects sees
	// a clean discontinuity rather than a phantom loss of thousands
	m_sendSequence= 0;
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

	// /mikan/frame ,iifi frameId timestampMs fps sendSequence
	//
	// The sequence is appended rather than placed first so that a client
	// reading the original three arguments positionally keeps working.
	// Incremented here, past the rate gate, so it counts bundles SENT rather
	// than frames considered - that is what makes a gap mean packet loss.
	OscMessage& frameMessage= m_bundle.addMessage(k_frameAddress);
	frameMessage.addInt32(static_cast<int32_t>(frame.frameIndex));
	frameMessage.addInt32(static_cast<int32_t>(frame.timestampMs));
	frameMessage.addFloat(frame.captureFps);
	frameMessage.addInt32(m_sendSequence++);

	// Skeleton geometry is slowly varying - ride the 1 Hz info cadence
	const ClockTimePoint now= std::chrono::steady_clock::now();
	const bool bSendSkeleton= !m_hasSentInfo || (now - m_lastInfoTime) >= std::chrono::seconds(1);

	for (int sideIndex= 0; sideIndex < static_cast<int>(eHandSide::Count); ++sideIndex)
	{
		appendHandMessages(frame, sideIndex, bSendSkeleton);
	}

	// /mikan/body/head ,ffffffff -- position xyz + orientation xyzw +
	// confidence. Head frame: +X facing direction, +Y toward the person's
	// left, +Z up. Always sent; confidence 0 means do not use it. Live-only
	// (no dropout hold): the head estimate re-anchors off the wrists each
	// frame, so a held value has nothing measured behind it.
	{
		glm::vec3 headPosition(0.f);
		glm::quat headOrientation(1.f, 0.f, 0.f, 0.f);
		float headConfidence= 0.f;
		resolveHeadOutput(frame.head, headPosition, headOrientation, headConfidence);

		OscMessage& headMessage= m_bundle.addMessage(k_headAddress);
		addVec3(headMessage, headPosition);
		headMessage.addFloat(headOrientation.x)
			.addFloat(headOrientation.y)
			.addFloat(headOrientation.z)
			.addFloat(headOrientation.w);
		headMessage.addFloat(headConfidence);
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
			const float decay= (float)(1.0 - elapsedMs / (double)holdMs);
			outPose= ioHeld.pose;
			outPose.confidence= ioHeld.pose.confidence * decay;
			// The elbow and shoulder confidences decay with it. A consumer
			// gates each on its one number, so leaving them at their last live
			// values would advertise a held pose as freshly measured.
			outPose.forearmConfidence= ioHeld.pose.forearmConfidence * decay;
			outPose.shoulderConfidence= ioHeld.pose.shoulderConfidence * decay;
			return true;
		}
	}

	ioHeld.valid= false;
	outPose= pose;
	return false;
}

void OscStreamer::resolveElbowOutput(const HandPose& pose, bool bPoseSent, float forearmLengthMeters,
									 glm::vec3& outPosition, float& outConfidence)
{
	// The elbow needs a measured forearm direction AND a world-anchored palm
	// to hang it off. Camera-space poses are excluded because the forearm
	// orientation is only ever produced in world space.
	const bool bUsable= bPoseSent && pose.hasForearmPose && pose.hasWorldPose;
	if (!bUsable)
	{
		outPosition= glm::vec3(0.f);
		outConfidence= 0.f;
		return;
	}

	outPosition= pose.getElbowPositionWorld(forearmLengthMeters);
	outConfidence= glm::clamp(pose.forearmConfidence, 0.f, 1.f);
}

void OscStreamer::resolveShoulderOutput(const HandPose& pose, bool bPoseSent,
										glm::vec3& outPosition, float& outConfidence)
{
	// Same contract as the elbow: always produces a value, confidence 0 means
	// do not use this position. The shoulder is only ever solved in world
	// space off a world-anchored wrist.
	const bool bUsable= bPoseSent && pose.hasShoulder && pose.hasWorldPose;
	if (!bUsable)
	{
		outPosition= glm::vec3(0.f);
		outConfidence= 0.f;
		return;
	}

	outPosition= pose.shoulderPositionWorld;
	outConfidence= glm::clamp(pose.shoulderConfidence, 0.f, 1.f);
}

void OscStreamer::resolveHeadOutput(const TrackingFrameResult::HeadPose& head,
									glm::vec3& outPosition, glm::quat& outOrientation, float& outConfidence)
{
	if (!head.valid)
	{
		outPosition= glm::vec3(0.f);
		outOrientation= glm::quat(1.f, 0.f, 0.f, 0.f);
		outConfidence= 0.f;
		return;
	}

	outPosition= head.positionWorld;
	outOrientation= head.orientationWorld;
	outConfidence= glm::clamp(head.confidence, 0.f, 1.f);
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

	// /mikan/hand/{s}/elbow ,ffff -- position xyz + confidence.
	//
	// Emitted before the untracked early-out, so this address arrives EVERY
	// frame no matter what. A consumer holds the last value it received for
	// any address that stops arriving, so an elbow that simply went silent
	// would sit at its last confident value while the hand was gone.
	// Confidence therefore carries validity too: 0 means do not use this
	// position. Every degraded case - no calibrated IMU, poor mounting, a
	// low-confidence hand, a decaying dropout hold, no hand at all - ends in
	// the same question, which is how much to trust this position.
	//
	// Derived here rather than left to the client because the wrist-to-elbow
	// length is the sender's calibrated value.
	{
		glm::vec3 elbowPosition(0.f);
		float elbowConfidence= 0.f;
		resolveElbowOutput(pose, bSendPose, m_config.forearmLengthMeters, elbowPosition, elbowConfidence);

		OscMessage& elbowMessage= m_bundle.addMessage(k_handElbowAddress[sideIndex]);
		addVec3(elbowMessage, elbowPosition);
		elbowMessage.addFloat(elbowConfidence);
	}

	// /mikan/hand/{s}/shoulder ,ffff -- position xyz + confidence. Same
	// always-send, confidence-carries-validity contract as the elbow.
	{
		glm::vec3 shoulderPosition(0.f);
		float shoulderConfidence= 0.f;
		resolveShoulderOutput(pose, bSendPose, shoulderPosition, shoulderConfidence);

		OscMessage& shoulderMessage= m_bundle.addMessage(k_handShoulderAddress[sideIndex]);
		addVec3(shoulderMessage, shoulderPosition);
		shoulderMessage.addFloat(shoulderConfidence);
	}

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

	// Diff log against a client's receive log. Emitted from the same locals
	// that were just encoded, not from the pose, so it cannot drift from what
	// actually went on the wire. frame is the join key with the receiver,
	// which reads it from the /mikan/frame message earlier in this bundle.
	if (m_config.logPalmFrames)
	{
		char palmLine[256];
		snprintf(palmLine, sizeof(palmLine),
				 "PALM SEND frame=%lld side=%s pos=%.6f,%.6f,%.6f quat=%.6f,%.6f,%.6f,%.6f",
				 (long long)frame.frameIndex, sideIndex == 0 ? "left" : "right", palmPosition.x,
				 palmPosition.y, palmPosition.z, palmOrientation.x, palmOrientation.y, palmOrientation.z,
				 palmOrientation.w);
		MIKAN_LOG_INFO("OscPalmSend") << palmLine;
	}

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
	// [lateral, proximalBend, intermediateBend, distalBend] DEGREES, relative
	// to the neutral straight pose.
	//
	// Degrees only on the wire; every angle inside this application is
	// radians, which is what the math wants. The conversion sits here, at the
	// boundary, because the consumers are animation rigs and game engines
	// whose own rotation types are degrees.
	OscMessage& fingersMessage= m_bundle.addMessage(k_handFingersAddress[sideIndex]);
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const FingerAngles& angles= pose.fingers[finger];
		fingersMessage.addFloat(glm::degrees(angles.lateral))
			.addFloat(glm::degrees(angles.proximal))
			.addFloat(glm::degrees(angles.intermediate))
			.addFloat(glm::degrees(angles.distal));
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
