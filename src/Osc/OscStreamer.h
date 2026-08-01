#pragma once

#include "OscWriter.h"
#include "UdpSocket.h"

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
	std::string targetIp= "127.0.0.1";
	uint16_t targetPort= 8000;
	float maxRateHz= 60.f; // <= 0 disables rate limiting
	// Hands whose fused confidence falls below this are reported untracked
	// and their pose messages are withheld entirely, so a client holds or
	// blends to a rest pose instead of following a jittering estimate.
	// 0 = always send.
	float minConfidence= 0.f;
	std::string appVersion= "MikanMediaPipe";
};

/// Streams per-frame parametric hand poses as OSC 1.0 bundles over UDP
/// unicast (consumed by e.g. Unreal Engine's OSC plugin).
///
/// Bundle layout (positions in world/marker space meters when available,
/// otherwise camera space — the active space is reported via /mikan/info):
///   /mikan/frame ,iif frameId timestampMs fps
///   per side s in {left,right}:
///     /mikan/hand/{s}/tracked ,iff tracked(0|1) presence confidence
///     if tracked (confidence below minConfidence reports tracked=0 and
///     withholds everything below):
///       /mikan/hand/{s}/palm ,fffffff position xyz + orientation xyzw
///       /mikan/hand/{s}/fingers ,f x20 per finger (thumb..pinky):
///         lateral, proximalBend, intermediateBend, distalBend (radians)
///       /mikan/hand/{s}/skeleton ,f x30 (1 Hz) per finger: base position in
///         the palm frame xyz + phalanx lengths [prox, inter, distal]
///   /mikan/info ,ss "space=...;units=m;handed=RH;up=Z" appVersion
///     (at most once per second)
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

	/// Bundles sent per second (updated once a second). Safe to poll from the
	/// UI thread.
	float getMessagesPerSecond() const { return m_messagesPerSecond.load(std::memory_order_relaxed); }

private:
	using ClockTimePoint= std::chrono::steady_clock::time_point;

	void appendHandMessages(const TrackingFrameResult& frame, int sideIndex, bool bSendSkeleton);
	void appendInfoMessage(bool hasWorldSpace, const ClockTimePoint& now);
	void updateSendStats(const ClockTimePoint& now);

	mutable std::mutex m_mutex; // guards config, socket, and encode state
	OscStreamerConfig m_config;
	UdpSocket m_socket;
	bool m_isRunning= false;

	// Per-frame encode state (reused to stay allocation-light)
	OscBundle m_bundle;
	std::vector<uint8_t> m_scratchBuffer;

	// Rate decimation (frame timestamps) and info-message throttling (wall clock)
	double m_lastSendTimestampMs= -1.0;
	bool m_hasSentInfo= false;
	ClockTimePoint m_lastInfoTime;

	// Send-rate stats
	ClockTimePoint m_statsWindowStart;
	int m_sentInWindow= 0;
	std::atomic<float> m_messagesPerSecond{0.f};
};
