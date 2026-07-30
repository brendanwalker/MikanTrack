#include "OscStreamer.h"

#include "Logger.h"
#include "TrackingTypes.h"

// Per-side OSC address tables, indexed by eHandSide (Left= 0, Right= 1)
static const char* k_handTrackedAddress[2]= {"/mikan/hand/left/tracked", "/mikan/hand/right/tracked"};
static const char* k_handWristAddress[2]= {"/mikan/hand/left/wrist", "/mikan/hand/right/wrist"};
static const char* k_handPalmAddress[2]= {"/mikan/hand/left/palm", "/mikan/hand/right/palm"};
static const char* k_handLandmarksAddress[2]= {"/mikan/hand/left/landmarks", "/mikan/hand/right/landmarks"};
static const char* k_armElbowAddress[2]= {"/mikan/arm/left/elbow", "/mikan/arm/right/elbow"};
static const char* k_armForearmAddress[2]= {"/mikan/arm/left/forearm", "/mikan/arm/right/forearm"};

static const char* k_frameAddress= "/mikan/frame";
static const char* k_infoAddress= "/mikan/info";

static const char* k_infoWorldSpace= "space=marker;units=m;handed=RH;up=Z";
static const char* k_infoCameraSpace= "space=camera;units=m;handed=RH;up=Z";

/// Pick the best available point set for a hand: world (marker-anchored) if
/// valid, otherwise camera space.
static const std::array<glm::vec3, HAND_LANDMARK_COUNT>& getHandPoints(const TrackedHand& hand)
{
	return hand.hasWorldSpace ? hand.worldPoints : hand.cameraPoints;
}

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
	for (const TrackedHand& hand : frame.hands)
	{
		if (hand.tracked && hand.hasWorldSpace)
			hasWorldSpace= true;
	}
	for (const TrackedArm& arm : frame.arms)
	{
		if (arm.valid && arm.hasWorldSpace)
			hasWorldSpace= true;
	}

	m_bundle.clear();
	m_bundle.setTimeTag(k_oscTimeTagImmediate);

	// /mikan/frame ,iif frameId timestampMs fps
	OscMessage& frameMessage= m_bundle.addMessage(k_frameAddress);
	frameMessage.addInt32(static_cast<int32_t>(frame.frameIndex));
	frameMessage.addInt32(static_cast<int32_t>(frame.timestampMs));
	frameMessage.addFloat(frame.captureFps);

	for (int sideIndex= 0; sideIndex < static_cast<int>(eHandSide::Count); ++sideIndex)
	{
		appendHandMessages(frame, sideIndex);
		appendArmMessages(frame, sideIndex);
	}

	const ClockTimePoint now= std::chrono::steady_clock::now();
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

void OscStreamer::appendHandMessages(const TrackingFrameResult& frame, int sideIndex)
{
	const TrackedHand& hand= frame.hands[sideIndex];

	// /mikan/hand/{s}/tracked ,if tracked(0|1) presence
	OscMessage& trackedMessage= m_bundle.addMessage(k_handTrackedAddress[sideIndex]);
	trackedMessage.addInt32(hand.tracked ? 1 : 0);
	trackedMessage.addFloat(hand.presence);

	if (!hand.tracked)
		return;

	const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points= getHandPoints(hand);

	// /mikan/hand/{s}/wrist ,fff
	addVec3(m_bundle.addMessage(k_handWristAddress[sideIndex]),
			points[static_cast<int>(eHandLandmark::WRIST)]);

	// /mikan/hand/{s}/palm ,fff -- centroid of the four finger MCP knuckles
	const glm::vec3 palmCenter=
		(points[static_cast<int>(eHandLandmark::INDEX_MCP)] +
		 points[static_cast<int>(eHandLandmark::MIDDLE_MCP)] +
		 points[static_cast<int>(eHandLandmark::RING_MCP)] +
		 points[static_cast<int>(eHandLandmark::PINKY_MCP)]) * 0.25f;
	addVec3(m_bundle.addMessage(k_handPalmAddress[sideIndex]), palmCenter);

	// /mikan/hand/{s}/landmarks ,fff x21 -- 63 floats in MediaPipe index order
	OscMessage& landmarksMessage= m_bundle.addMessage(k_handLandmarksAddress[sideIndex]);
	for (int landmarkIndex= 0; landmarkIndex < HAND_LANDMARK_COUNT; ++landmarkIndex)
	{
		addVec3(landmarksMessage, points[landmarkIndex]);
	}
}

void OscStreamer::appendArmMessages(const TrackingFrameResult& frame, int sideIndex)
{
	const TrackedArm& arm= frame.arms[sideIndex];
	if (!arm.valid)
		return;

	const glm::vec3& elbow= arm.hasWorldSpace ? arm.elbowWorld : arm.elbowCamera;
	const glm::vec3& wrist= arm.hasWorldSpace ? arm.wristWorld : arm.wristCamera;

	// /mikan/arm/{s}/elbow ,ffff x y z confidence
	OscMessage& elbowMessage= m_bundle.addMessage(k_armElbowAddress[sideIndex]);
	addVec3(elbowMessage, elbow);
	elbowMessage.addFloat(arm.confidence);

	// /mikan/arm/{s}/forearm ,ffffff elbowXyz wristXyz
	OscMessage& forearmMessage= m_bundle.addMessage(k_armForearmAddress[sideIndex]);
	addVec3(forearmMessage, elbow);
	addVec3(forearmMessage, wrist);
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
