#pragma once

#include "OscOutputMode.h"
#include "OscWriter.h"
#include "TrackingTypes.h"
#include "UdpSocket.h"
#include "VmcRetarget.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct TrackingFrameResult;

struct OscStreamerConfig
{
	bool enabled= true;
	// Which wire format to speak. Mutually exclusive - see eOscOutputMode.
	eOscOutputMode outputMode= eOscOutputMode::Mikan;
	std::string targetIp= "127.0.0.1";
	// The caller picks the port for the active mode; the two formats have
	// different conventional listeners (VMC's is 39539).
	uint16_t targetPort= 8000;
	float maxRateHz= 60.f; // <= 0 disables rate limiting
	// Hands whose fused confidence falls below this are reported untracked
	// and their pose messages are withheld entirely, so a client holds or
	// blends to a rest pose instead of following a jittering estimate.
	// 0 = always send.
	float minConfidence= 0.f;
	// Dropout grace window: after a hand goes untracked (or below
	// minConfidence), keep sending its last good pose with the confidence
	// decaying linearly to zero over this window, and only then report
	// tracked=0. Bridges 2-10 frame dropouts so clients don't slam to their
	// rest-pose blend and back. 0 = report the dropout immediately.
	float holdOnDropoutMs= 250.f;
	// Wrist-to-elbow distance used to place the streamed elbow. Only the
	// length is assumed; the direction is measured, so an error here slides
	// the elbow along the forearm rather than rotating it.
	float forearmLengthMeters= 0.25f;
	// Log each palm transform as it is encoded, so a client's receive log can
	// be diffed against it frame for frame. One line per hand per frame.
	bool logPalmFrames= false;
	std::string appVersion= "MikanTrack";

	// -- VMC mode only ------------------------------------------------------
	// Bone offsets the streamed skeleton carries. A VMC receiver replaces both
	// the rotation AND the translation of every bone it is sent, so these are
	// not optional: the avatar takes these proportions.
	float shoulderWidthMeters= 0.40f;
	float upperArmLengthMeters= 0.30f;
	// Neck -> head bone offset. Nothing on this rig measures it, so it is a
	// setting: too small and the head sinks into the neck, too large and it
	// floats.
	float vmcHeadOffsetMeters= 0.08f;
	// VMC carries no confidence and no tracked flag, so a lost hand can only be
	// expressed as motion. Freezing the last streamed bones reads far better
	// than going silent, which drops the arm back to the avatar's T-pose.
	bool vmcFreezeOnLoss= true;
};

/// Streams per-frame parametric hand poses as OSC 1.0 bundles over UDP
/// unicast (consumed by e.g. Unreal Engine's OSC plugin).
///
/// Bundle layout (positions in world/marker space meters when available,
/// otherwise camera space — the active space is reported via /mikan/info):
///   /mikan/frame ,iifi frameId timestampMs fps sendSequence
///     sendSequence increments by exactly one per bundle actually sent, so a
///     client can measure packet loss. frameId CANNOT do that job: it is the
///     capture index of whichever camera produced the newest result, so with
///     several cameras it repeats, skips, and steps backwards (measured at
///     11.7% backwards steps on a two-camera rig). It is kept because it
///     identifies the capture, not the transmission.
///   per side s in {left,right}:
///     /mikan/hand/{s}/tracked ,iff tracked(0|1) presence confidence
///     /mikan/hand/{s}/elbow ,ffff position xyz + confidence [0,1].
///       Sent EVERY frame, tracked or not, because a consumer holds the last
///       value for any address that stops arriving - an elbow that simply
///       went silent would sit at its last confident value while the hand
///       was gone. Confidence therefore carries validity: 0 means do not use
///       this position. It folds together the hand's own confidence, the
///       forearm source's quality (IMU mounting quality, or measured jitter
///       stability for a vision elbow) and the dropout decay.
///     /mikan/hand/{s}/shoulder ,ffff position xyz + confidence [0,1].
///       Same always-send contract; from the vision body-pose solver.
///     if tracked (confidence below minConfidence reports tracked=0 and
///     withholds everything below):
///       /mikan/hand/{s}/palm ,fffffff position xyz + orientation xyzw
///       /mikan/hand/{s}/forearm ,ifffffff valid(0|1) + position xyz +
///         orientation xyzw (world). The forearm frame's origin is the WRIST
///         JOINT - half a palm back from the palm center, which is the one
///         joint the palm transform alone cannot give a consumer - and its
///         +X points along the forearm toward the hand, so the elbow is one
///         forearm length back along -X. A consumer retargeting the arm onto
///         its own proportions therefore has both ends of the bone without
///         reconstructing either. valid=0 (origin and identity) when no
///         forearm is measured for that hand.
///       /mikan/hand/{s}/fingers ,f x20 per finger (thumb..pinky):
///         lateral, proximalBend, intermediateBend, distalBend (DEGREES -
///         the wire is degrees, everything inside this app is radians)
///       /mikan/hand/{s}/skeleton ,f x45 (1 Hz) per finger: base position in
///         the palm frame xyz + phalanx lengths [prox, inter, distal] +
///         neutral (zero-angle) direction in the palm frame xyz
///   /mikan/body/head ,ffffffff position xyz + orientation xyzw + confidence.
///     Head frame: +X facing, +Y toward the person's left, +Z up. Always
///     sent, confidence 0 carries invalidity; live-only (no dropout hold).
///   /mikan/info ,ss "space=...;units=m;handed=RH;up=Z;...;angles=deg" appVersion
///     (at most once per second)
///
/// In VMC mode the bundle is the VMC protocol instead (see VmcRetarget.h for
/// the retarget and its conventions):
///   /VMC/Ext/OK ,iiii loaded calibrationState calibrationMode trackingStatus
///   /VMC/Ext/T ,f seconds since the socket opened
///   /VMC/Ext/Root/Pos ,sfffffff "root" + identity. Deliberately identity:
///     this is an upper-body tracker anchored to a desk marker, not a
///     room-scale root, so pushing the marker frame onto the avatar would
///     teleport it.
///   /VMC/Ext/Bone/Pos ,sfffffff name + local position xyz + rotation xyzw,
///     once per measured bone (head, clavicles, arms, hands, fingers)
class OscStreamer
{
public:
	OscStreamer()= default;
	~OscStreamer();

	// Non-copyable
	OscStreamer(const OscStreamer&)= delete;
	OscStreamer& operator=(const OscStreamer&)= delete;

	/// Open the UDP socket. @returns true on success
	bool startup();
	void shutdown();

	bool isRunning() const { return m_isRunning; }

	OscStreamerConfig getConfig() const;
	void setConfig(const OscStreamerConfig& config);

	/// Encode and send one bundle for the given frame.
	/// Called from the inference thread; allocation-light after warm-up
	/// (reuses a pooled bundle and a scratch encode buffer).
	void sendFrame(const TrackingFrameResult& frame);

	/// Encode one frame in the active output format without touching the
	/// socket, advancing the same streaming state (sequence, dropout holds).
	/// sendFrame is this plus the rate gate and the send, so a test reads
	/// exactly the bytes a receiver would rather than a reconstruction of them.
	/// One entry per datagram: a frame too large for a single packet is split
	/// into several complete bundles (see k_maxDatagramBytes).
	void encodeFrame(const TrackingFrameResult& frame, std::vector<std::vector<uint8_t>>& outPackets);

	/// Largest datagram VMC mode will put on the wire. Sized to stay inside a
	/// 1500-byte ethernet MTU (so nothing depends on IP fragmentation, where
	/// one lost fragment costs the whole bundle) and well inside the 2048-byte
	/// default receive buffer of Rug.Osc, which several VMC tools are built on.
	/// The Mikan format is deliberately not chunked - see encodeFrameLocked.
	static constexpr size_t k_maxDatagramBytes= 1400;

	/// Bundles sent per second (updated once a second). Safe to poll from the
	/// UI thread.
	float getMessagesPerSecond() const { return m_messagesPerSecond.load(std::memory_order_relaxed); }

	// -- Dropout hold logic (pure; public for the self test) -----
	struct HeldPoseState
	{
		bool valid= false;
		double timestampMs= 0.0;
		HandPose pose;
	};
	/// Decides what (if anything) to send for a hand this frame. A live,
	/// confident pose passes through and refreshes ioHeld; on a dropout the
	/// held pose bridges up to holdMs (confidence decaying linearly to 0);
	/// past the window (or on a timestamp regression) the hold is dropped.
	/// @returns true when outPose should be sent as tracked
	static bool resolveOutputPose(const HandPose& pose, double frameTimestampMs, float minConfidence,
								  float holdMs, HeldPoseState& ioHeld, HandPose& outPose);

	/// Decides what /elbow carries for one hand. Always produces a value,
	/// because that message is sent whether or not the hand is tracked; an
	/// unusable elbow reports confidence 0 rather than going silent, since a
	/// consumer holds the last value for an address that stops arriving.
	/// Static so the self test can exercise the contract without a socket.
	static void resolveElbowOutput(const HandPose& pose, bool bPoseSent, float forearmLengthMeters,
								   glm::vec3& outPosition, float& outConfidence);

	/// Same always-send contract for /shoulder: confidence 0 carries
	/// invalidity, the address never goes silent.
	static void resolveShoulderOutput(const HandPose& pose, bool bPoseSent,
									  glm::vec3& outPosition, float& outConfidence);

	/// Decides what /forearm carries for one hand: the forearm frame's origin
	/// (the WRIST JOINT, not the palm center) and its world orientation.
	/// @returns true when the pose is measured; on false the outputs are the
	/// origin and identity, and the message's valid flag reports it.
	static bool resolveForearmOutput(const HandPose& pose, bool bPoseSent,
									 glm::vec3& outPosition, glm::quat& outOrientation);

	/// Same contract for /mikan/body/head; live-only (no dropout hold).
	static void resolveHeadOutput(const TrackingFrameResult::HeadPose& head,
								  glm::vec3& outPosition, glm::quat& outOrientation, float& outConfidence);

	/// What VMC mode streams for one hand, applied AFTER resolveOutputPose.
	/// VMC has no confidence and no tracked flag, so the only way to say "this
	/// is no longer measured" is to stop moving: past the dropout hold the last
	/// streamed bones freeze rather than going silent, because a silent address
	/// drops that arm back to the avatar's rest T-pose.
	/// @returns true when outPose should be streamed as bones
	static bool resolveVmcOutputPose(const HandPose& pose, bool bPoseSent, bool bFreezeOnLoss,
									 HeldPoseState& ioLast, HandPose& outPose);

private:
	using ClockTimePoint= std::chrono::steady_clock::time_point;

	void encodeFrameLocked(const TrackingFrameResult& frame, const ClockTimePoint& now,
						   std::vector<std::vector<uint8_t>>& outPackets);
	void appendMikanMessages(const TrackingFrameResult& frame, const ClockTimePoint& now);
	void appendVmcMessages(const TrackingFrameResult& frame, const ClockTimePoint& now);
	void appendHandMessages(const TrackingFrameResult& frame, int sideIndex, bool bSendSkeleton);
	void appendInfoMessage(bool hasWorldSpace, const ClockTimePoint& now);
	void updateSendStats(const ClockTimePoint& now);

	mutable std::mutex m_mutex; // guards config, socket, and encode state
	OscStreamerConfig m_config;
	UdpSocket m_socket;
	bool m_isRunning= false;

	// Per-frame encode state (reused to stay allocation-light)
	OscBundle m_bundle;

	// Increments once per frame actually put on the wire; the client's loss counter
	int32_t m_sendSequence= 0;
	std::vector<std::vector<uint8_t>> m_scratchPackets;

	// Dropout hold state per side
	HeldPoseState m_heldPose[2];
	// VMC freeze-on-loss state per side, plus the retarget scratch (reused so
	// the per-frame encode stays allocation-light like the Mikan path)
	HeldPoseState m_lastVmcPose[2];
	VmcRetarget::VmcPose m_vmcPose;
	ClockTimePoint m_startTime;

	// Rate decimation (frame timestamps) and info-message throttling (wall clock)
	double m_lastSendTimestampMs= -1.0;
	bool m_hasSentInfo= false;
	ClockTimePoint m_lastInfoTime;

	// Send-rate stats
	ClockTimePoint m_statsWindowStart;
	int m_sentInWindow= 0;
	std::atomic<float> m_messagesPerSecond{0.f};
};
