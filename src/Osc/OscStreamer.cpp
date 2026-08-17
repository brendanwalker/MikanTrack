#include "OscStreamer.h"

#include "Logger.h"
#include "TrackingTypes.h"

#include <cstdio>

// Per-side OSC address tables, indexed by eHandSide (Left= 0, Right= 1)
static const char* k_handTrackedAddress[2]= {"/mikan/hand/left/tracked", "/mikan/hand/right/tracked"};
static const char* k_handPalmAddress[2]= {"/mikan/hand/left/palm", "/mikan/hand/right/palm"};
static const char* k_handForearmAddress[2]= {"/mikan/hand/left/forearm", "/mikan/hand/right/forearm"};
static const char* k_handElbowAddress[2]= {"/mikan/hand/left/elbow", "/mikan/hand/right/elbow"};
static const char* k_handShoulderAddress[2]= {"/mikan/hand/left/shoulder", "/mikan/hand/right/shoulder"};
static const char* k_handFingersAddress[2]= {"/mikan/hand/left/fingers", "/mikan/hand/right/fingers"};
static const char* k_handSkeletonAddress[2]= {"/mikan/hand/left/skeleton", "/mikan/hand/right/skeleton"};

static const char* k_frameAddress= "/mikan/frame";
static const char* k_headAddress= "/mikan/body/head";
static const char* k_infoAddress= "/mikan/info";

// VMC protocol addresses (https://protocol.vmc.info)
static const char* k_vmcOkAddress= "/VMC/Ext/OK";
static const char* k_vmcTimeAddress= "/VMC/Ext/T";
static const char* k_vmcRootAddress= "/VMC/Ext/Root/Pos";
static const char* k_vmcBoneAddress= "/VMC/Ext/Bone/Pos";

static const char* k_infoWorldSpace=
	"space=marker;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=deg";
static const char* k_infoCameraSpace=
	"space=camera;units=m;handed=RH;up=Z;palm=x-fingers,z-palmar;angles=deg";

static void addVec3(OscMessage& message, const glm::vec3& point)
{
	message.addFloat(point.x).addFloat(point.y).addFloat(point.z);
}

static void addQuat(OscMessage& message, const glm::quat& rotation)
{
	message.addFloat(rotation.x).addFloat(rotation.y).addFloat(rotation.z).addFloat(rotation.w);
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
	m_startTime= m_statsWindowStart;
	m_messagesPerSecond.store(0.f, std::memory_order_relaxed);
	// A reconnecting client must not inherit a frozen pose from the last one
	m_lastVmcPose[0]= HeldPoseState();
	m_lastVmcPose[1]= HeldPoseState();

	// The target is logged by setConfig instead: startup runs before the app's
	// config reaches the streamer, so anything named here would be a default
	MIKAN_LOG_INFO("OscStreamer::startup") << "UDP socket open";

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

	const bool bTargetChanged= config.outputMode != m_config.outputMode ||
							   config.targetIp != m_config.targetIp ||
							   config.targetPort != m_config.targetPort;

	m_config= config;
	m_hasSentInfo= false; // re-announce info on config change

	if (bTargetChanged)
	{
		MIKAN_LOG_INFO("OscStreamer::setConfig")
			<< "Streaming " << oscOutputModeName(m_config.outputMode) << " to " << m_config.targetIp
			<< ":" << m_config.targetPort;
	}
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

	const ClockTimePoint now= std::chrono::steady_clock::now();

	encodeFrameLocked(frame, now, m_scratchPackets);

	// One frame can be several datagrams. The stats count FRAMES, so the rate
	// readout keeps meaning the same thing whichever format is active.
	bool bSentAny= false;
	for (const std::vector<uint8_t>& packet : m_scratchPackets)
	{
		bSentAny|= m_socket.sendTo(m_config.targetIp, m_config.targetPort,
								   packet.data(), static_cast<int>(packet.size()));
	}
	if (bSentAny)
		m_sentInWindow++;

	updateSendStats(now);
}

void OscStreamer::encodeFrame(const TrackingFrameResult& frame, std::vector<std::vector<uint8_t>>& outPackets)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	encodeFrameLocked(frame, std::chrono::steady_clock::now(), outPackets);
}

void OscStreamer::encodeFrameLocked(const TrackingFrameResult& frame, const ClockTimePoint& now,
									std::vector<std::vector<uint8_t>>& outPackets)
{
	m_bundle.clear();
	m_bundle.setTimeTag(k_oscTimeTagImmediate);

	if (m_config.outputMode == eOscOutputMode::Vmc)
		appendVmcMessages(frame, now);
	else
		appendMikanMessages(frame, now);

	// Pack the frame's messages into as few complete bundles as fit. Splitting
	// here rather than letting IP fragment the datagram matters because a
	// receiver drops a whole bundle for one lost fragment, and because the OSC
	// library several VMC tools are built on defaults to a 2048-byte receive
	// buffer - a single 3.4 KB bundle simply never arrives.
	//
	// VMC ONLY. The Mikan stream stays one bundle per frame: it has a shipping
	// consumer, it fits inside a datagram on the loopback path it actually
	// runs on, and changing how a working wire is chopped up belongs in its
	// own change rather than riding along with this one.
	const size_t maxDatagramBytes=
		m_config.outputMode == eOscOutputMode::Vmc ? k_maxDatagramBytes : SIZE_MAX;

	size_t packetCount= 0;
	// Reuses the outer vector's buffers across frames (the packet count is
	// stable), so this stays allocation-light like the rest of the encode path
	auto emitPacket= [&](size_t firstMessage, size_t count) {
		if (packetCount == outPackets.size())
			outPackets.emplace_back();
		else
			outPackets[packetCount].clear();

		m_bundle.encodeRange(firstMessage, count, outPackets[packetCount]);
		packetCount++;
	};

	const size_t messageCount= m_bundle.getMessageCount();
	size_t firstInPacket= 0;
	size_t packetBytes= OscBundle::k_headerSize;

	for (size_t messageIndex= 0; messageIndex < messageCount; ++messageIndex)
	{
		const size_t messageBytes= m_bundle.getMessageEncodedSize(messageIndex);

		// Flush before adding, so the packet under construction stays legal.
		// A single message over the limit still goes out alone: dropping it
		// would silently lose a bone, and no message this streams comes close.
		if (messageIndex > firstInPacket && packetBytes + messageBytes > maxDatagramBytes)
		{
			emitPacket(firstInPacket, messageIndex - firstInPacket);
			firstInPacket= messageIndex;
			packetBytes= OscBundle::k_headerSize;
		}

		packetBytes+= messageBytes;
	}

	if (messageCount > firstInPacket)
		emitPacket(firstInPacket, messageCount - firstInPacket);

	outPackets.resize(packetCount);
}

void OscStreamer::appendMikanMessages(const TrackingFrameResult& frame, const ClockTimePoint& now)
{
	// Do we have a marker-anchored world transform this frame?
	bool hasWorldSpace= false;
	for (const HandPose& pose : frame.poses)
	{
		if (pose.tracked && pose.hasWorldPose)
			hasWorldSpace= true;
	}

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
}

void OscStreamer::appendVmcMessages(const TrackingFrameResult& frame, const ClockTimePoint& now)
{
	// The confidence gate and the dropout hold run exactly as they do in Mikan
	// mode, then the freeze rule decides what a hand that is still lost looks
	// like on a wire that has no way to say "unmeasured"
	std::array<HandPose, 2> streamedPoses;
	bool bSideValid[2]= {false, false};
	for (int sideIndex= 0; sideIndex < static_cast<int>(eHandSide::Count); ++sideIndex)
	{
		HandPose resolved;
		const bool bPoseSent= resolveOutputPose(frame.poses[sideIndex], frame.timestampMs,
												m_config.minConfidence, m_config.holdOnDropoutMs,
												m_heldPose[sideIndex], resolved);

		bSideValid[sideIndex]= resolveVmcOutputPose(resolved, bPoseSent, m_config.vmcFreezeOnLoss,
													m_lastVmcPose[sideIndex], streamedPoses[sideIndex]);
	}

	VmcRetarget::VmcBodyLengths lengths;
	lengths.shoulderWidthMeters= m_config.shoulderWidthMeters;
	lengths.upperArmLengthMeters= m_config.upperArmLengthMeters;
	lengths.forearmLengthMeters= m_config.forearmLengthMeters;
	lengths.headOffsetMeters= m_config.vmcHeadOffsetMeters;

	VmcRetarget::buildPose(streamedPoses, bSideValid, frame.head, lengths, m_vmcPose);

	// /VMC/Ext/OK ,iiii -- loaded, calibration state, calibration mode,
	// tracking status. Calibration always reads as done in normal mode: this
	// rig calibrates itself against a printed board long before it streams, so
	// there is no receiver-driven calibration step for anyone to wait on.
	{
		const bool bTracking= bSideValid[0] || bSideValid[1];
		OscMessage& okMessage= m_bundle.addMessage(k_vmcOkAddress);
		okMessage.addInt32(1).addInt32(3).addInt32(0).addInt32(bTracking ? 1 : 0);
	}

	// /VMC/Ext/T ,f -- the sender's own clock, which a receiver uses to tell a
	// live stream from a stalled one
	{
		const std::chrono::duration<float> elapsed= now - m_startTime;
		m_bundle.addMessage(k_vmcTimeAddress).addFloat(elapsed.count());
	}

	// /VMC/Ext/Root/Pos ,sfffffff -- identity on purpose. This is an upper-body
	// tracker anchored to a desk marker, so the marker frame is not a place to
	// put an avatar; the receiver keeps whatever root it already has.
	{
		OscMessage& rootMessage= m_bundle.addMessage(k_vmcRootAddress);
		rootMessage.addString("root");
		addVec3(rootMessage, glm::vec3(0.f));
		addQuat(rootMessage, glm::quat(1.f, 0.f, 0.f, 0.f));
	}

	// /VMC/Ext/Bone/Pos ,sfffffff -- one per MEASURED bone. A bone left out
	// stays at the avatar's rest pose on the receiving side, which is exactly
	// the reference the streamed rotations are relative to.
	for (int boneIndex= 0; boneIndex < VmcRetarget::VMC_BONE_COUNT; ++boneIndex)
	{
		const VmcRetarget::VmcBone& bone= m_vmcPose.bones[boneIndex];
		if (!bone.present)
			continue;

		OscMessage& boneMessage= m_bundle.addMessage(k_vmcBoneAddress);
		boneMessage.addString(VmcRetarget::boneName((VmcRetarget::eVmcBone)boneIndex));
		addVec3(boneMessage, bone.localPosition);
		addQuat(boneMessage, bone.localRotation);
	}
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

bool OscStreamer::resolveVmcOutputPose(const HandPose& pose, bool bPoseSent, bool bFreezeOnLoss,
									   HeldPoseState& ioLast, HandPose& outPose)
{
	// A world-anchored palm is the entry requirement: VMC bones are a skeleton,
	// and a camera-space pose has nothing to hang one off.
	if (bPoseSent && pose.hasWorldPose)
	{
		ioLast.valid= true;
		ioLast.pose= pose;
		outPose= pose;
		return true;
	}

	// Past the dropout hold. Going silent here would return that arm to the
	// avatar's rest T-pose, because a receiver holds an unstreamed bone at its
	// reference pose - so an arm that stops moving is both the closer reading
	// of a hand that stopped being measured and the better looking one.
	if (bFreezeOnLoss && ioLast.valid)
	{
		outPose= ioLast.pose;
		return true;
	}

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

bool OscStreamer::resolveForearmOutput(const HandPose& pose, bool bPoseSent,
									   glm::vec3& outPosition, glm::quat& outOrientation)
{
	// Same entry requirement as the elbow, which is derived from this frame:
	// a measured forearm AND a world-anchored palm. The forearm orientation is
	// only ever produced in world space, and its origin is read off the palm.
	const bool bUsable= bPoseSent && pose.hasForearmPose && pose.hasWorldPose;
	if (!bUsable)
	{
		outPosition= glm::vec3(0.f);
		outOrientation= glm::quat(1.f, 0.f, 0.f, 0.f);
		return false;
	}

	// The WRIST JOINT, not the palm center. The frame is defined as the palm
	// frame at a neutral wrist, so anchoring it anywhere else would make its
	// +X stop meaning "one forearm length from the elbow".
	outPosition= pose.getWristPositionWorld();
	outOrientation= pose.forearmOrientationWorld;
	return true;
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

	// /mikan/hand/{s}/forearm ,ifffffff -- valid + position xyz + orientation
	// xyzw, the forearm frame in world space. Its origin is the WRIST JOINT,
	// the one joint of the chain a consumer cannot recover from the palm
	// message alone: the palm transform is centered half a palm forward of it,
	// and that half-palm only arrives with the 1 Hz skeleton. Its +X runs
	// along the forearm toward the hand, so the elbow sits one forearm length
	// back along -X - a consumer rescaling the arm onto its own proportions
	// has both ends of the bone plus the roll between them.
	//
	// Sent unconditionally (with valid=0, the origin and identity, when no
	// forearm is measured) so a client can bind the address once instead of
	// handling an address that appears and disappears.
	{
		glm::vec3 forearmPosition(0.f);
		glm::quat forearmOrientation(1.f, 0.f, 0.f, 0.f);
		const bool bHasForearm=
			resolveForearmOutput(pose, bSendPose, forearmPosition, forearmOrientation);

		OscMessage& forearmMessage= m_bundle.addMessage(k_handForearmAddress[sideIndex]);
		forearmMessage.addInt32(bHasForearm ? 1 : 0);
		addVec3(forearmMessage, forearmPosition);
		forearmMessage.addFloat(forearmOrientation.x)
			.addFloat(forearmOrientation.y)
			.addFloat(forearmOrientation.z)
			.addFloat(forearmOrientation.w);
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
