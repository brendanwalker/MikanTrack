#pragma once

#include "OscWriter.h"
#include "TrackingTypes.h"
#include "UdpSocket.h"
#include "VmcRetarget.h"
#include "AvatarRetarget.h"
#include "AvatarFaceMap.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct TrackingFrameResult;

struct OscStreamerConfig
{
	bool enabled= true;
	std::string targetIp= "127.0.0.1";
	// The conventional VMC receiver port
	uint16_t port= 39539;
	float maxRateHz= 60.f; // <= 0 disables rate limiting
	// Hands whose fused confidence falls below this stop being measured on the
	// wire: past the dropout hold their bones freeze or go silent (see
	// vmcFreezeOnLoss) instead of following a jittering estimate.
	// 0 = always send.
	float minConfidence= 0.f;
	// Dropout grace window: after a hand goes untracked (or below
	// minConfidence), keep streaming its last good pose for this long before
	// the loss rule applies. Bridges 2-10 frame dropouts so the avatar does
	// not stall and resume on every brief loss. 0 = apply the loss rule
	// immediately.
	float holdOnDropoutMs= 250.f;
	// Wrist-to-elbow distance used to place the streamed elbow. Only the
	// length is assumed; the direction is measured, so an error here slides
	// the elbow along the forearm rather than rotating it.
	float forearmLengthMeters= 0.25f;

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
	// With a loaded avatar the VMC bones come from retargeting the resolved
	// poses onto its skeleton (its proportions, its rest hand) instead of the
	// measured-length chain above. Null keeps the measured-length path.
	std::shared_ptr<const AvatarSkeleton> avatarSkeleton;
	AvatarRetargetConfig avatarRetarget;
	BodyDimensions bodyDimensions;
	// The loaded avatar file, announced as /VMC/Ext/VRM so a receiver on the
	// same machine can load the very file the bones were retargeted onto.
	// An empty path announces nothing.
	std::string avatarPath;
	std::string avatarTitle;
	std::string avatarSha256;
	// The face stream's blendshapes as the avatar's names. Null streams the
	// ARKit names verbatim.
	std::shared_ptr<const AvatarFaceMap> faceMap;
};

/// Streams per-frame tracked poses as the VMC protocol (OSC 1.0 bundles over
/// UDP unicast) for VMC receivers. See VmcRetarget.h for the retarget and its
/// conventions. A frame is split into complete bundles of at most
/// k_maxDatagramBytes each; the messages, in order:
///   /VMC/Ext/OK ,iiii loaded calibrationState calibrationMode trackingStatus
///   /VMC/Ext/T ,f seconds since the socket opened
///   /VMC/Ext/Root/Pos ,sfffffff "root" + identity. Deliberately identity:
///     this is an upper-body tracker anchored to a desk marker, not a
///     room-scale root, so pushing the marker frame onto the avatar would
///     teleport it.
///   /VMC/Ext/Bone/Pos ,sfffffff name + local position xyz + rotation xyzw,
///     once per measured bone (head, clavicles, arms, hands, fingers)
///   while the phone face stream is live (see AvatarFaceMap for the names):
///     /VMC/Ext/Blend/Val ,sf name + value [0,1], once per mapped blendshape
///     /VMC/Ext/Blend/Apply , after the last value. The frame the stream
///       stops sends every value once more at zero, so the face relaxes
///       instead of holding its last expression.
///   /VMC/Ext/VRM ,sss path title sha256 (at most once per second, only with
///     an avatar loaded): the local file path (UTF-8), the avatar's title,
///     and the lowercase hex SHA-256 of the file bytes
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

	/// Encode one frame without touching the socket, advancing the same
	/// streaming state (dropout holds, freeze-on-loss, face and avatar throttles).
	/// sendFrame is this plus the rate gate and the send, so a test reads
	/// exactly the bytes a receiver would rather than a reconstruction of them.
	/// One entry per datagram: a frame too large for a single packet is split
	/// into several complete bundles (see k_maxDatagramBytes).
	void encodeFrame(const TrackingFrameResult& frame, std::vector<std::vector<uint8_t>>& outPackets);

	/// Largest datagram the streamer will put on the wire. Sized to stay inside
	/// a 1500-byte ethernet MTU (so nothing depends on IP fragmentation, where
	/// one lost fragment costs the whole bundle) and well inside the 2048-byte
	/// default receive buffer of Rug.Osc, which several VMC tools are built on.
	static constexpr size_t k_maxDatagramBytes= 1400;

	/// Frames sent per second (updated once a second). Safe to poll from the
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

	/// What the stream carries for one hand, applied AFTER resolveOutputPose.
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
	void appendVmcMessages(const TrackingFrameResult& frame, const ClockTimePoint& now);
	void appendVmcBlendMessages(const TrackingFrameResult& frame);
	void appendVmcAvatarMessage(const ClockTimePoint& now);
	void updateSendStats(const ClockTimePoint& now);

	mutable std::mutex m_mutex; // guards config, socket, and encode state
	OscStreamerConfig m_config;
	UdpSocket m_socket;
	bool m_isRunning= false;

	// Per-frame encode state (reused to stay allocation-light)
	OscBundle m_bundle;
	std::vector<std::vector<uint8_t>> m_scratchPackets;

	// Dropout hold state per side
	HeldPoseState m_heldPose[2];
	// VMC freeze-on-loss state per side, plus the retarget scratch (reused so
	// the per-frame encode stays allocation-light)
	HeldPoseState m_lastVmcPose[2];
	VmcRetarget::VmcPose m_vmcPose;
	AvatarRetarget m_avatarRetarget;
	AvatarPose m_avatarPose;
	ClockTimePoint m_startTime;
	// Face blendshape state: the ARKit fallback map, the evaluate scratch, and
	// whether the last frame carried a face (for the one relaxing frame)
	std::shared_ptr<const AvatarFaceMap> m_rawFaceMap;
	std::vector<float> m_blendValues;
	bool m_bFaceWasPresent= false;
	// /VMC/Ext/VRM throttling (wall clock)
	bool m_hasSentAvatar= false;
	ClockTimePoint m_lastAvatarTime;

	// Rate decimation (frame timestamps)
	double m_lastSendTimestampMs= -1.0;

	// Send-rate stats
	ClockTimePoint m_statsWindowStart;
	int m_sentInWindow= 0;
	std::atomic<float> m_messagesPerSecond{0.f};
};
