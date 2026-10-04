#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "FaceTypes.h"
#include "UdpSocket.h"

struct FaceServiceConfig
{
	bool enabled= false;
	// The UDP port the phone app streams to, and is asked to stream on
	uint16_t port= 49983;
	// Where the streaming handshake goes; the broadcast address reaches a
	// phone anywhere on the LAN
	std::string phoneAddress= "255.255.255.255";
};

// Snapshot for UI/diagnostics
struct FaceServiceStatus
{
	bool enabled= false;
	bool bound= false;
	// A datagram arrived within the last half second
	bool streaming= false;
	float datagramsPerSecond= 0.f;
	uint64_t unknownNameCount= 0;
	// The most recent blendshape name the ARKit table lacks
	std::string lastUnknownName;
	uint64_t parseFailures= 0;
	FaceSample latest;
};

// Receives iFacialMocap (an iPhone app streaming ARKit blendshapes and head
// rotation over UDP) on one non-blocking socket. No thread of its own: the
// vision thread's update() drains the socket and runs the handshake schedule.
// All state sits behind one mutex, so setConfig and getStatus may be called
// from the UI thread while update runs on the vision thread.
class FaceService
{
public:
	FaceService();
	~FaceService();

	// Marks the service live. Opens no socket: setConfig binds.
	bool startup();
	void shutdown();

	// Thread-safe. Rebinds when the port or enable flag changes and
	// re-handshakes when the phone address changes. Takes effect on a started
	// service; a config set before startup() is applied by it.
	void setConfig(const FaceServiceConfig& config);

	// Vision thread. Drains every waiting datagram (the last one wins) and
	// repeats the handshake while the stream is silent. Cheap when idle.
	// nowMs is steadyNowMs().
	void update(double nowMs);

	// Newest sample; false before the first datagram
	bool getLatestSample(FaceSample& outSample) const;
	// A datagram arrived within the last half second
	bool isStreaming(double nowMs) const;

	FaceServiceStatus getStatus() const;

	// The datagram text into `out`, which keeps its previous values for
	// anything the text omits. The blendshape section comes before the first
	// '=', the bone section after it, entries '|'-separated. A blendshape
	// entry is `name-value` (or `name&value` in the v2 format, which allows
	// negatives) with the value in percent. A bone entry is `name#numbers`:
	// `head#rx,ry,rz,px,py,pz` (all six required), `rightEye#rx,ry,rz`,
	// `leftEye#rx,ry,rz`. Malformed entries are skipped and unknown blendshape
	// names are added to outUnknownNames. Returns false when no entry parsed.
	// Does not touch valid, sequence, or timestampMs.
	// Unknown names are also listed in outUnknownNameList when one is given.
	static bool parseDatagram(const char* text, size_t length, FaceSample& out, uint64_t& outUnknownNames,
							  std::vector<std::string>* outUnknownNameList= nullptr);

	// The broadcast handshake loops back to the bound port on this machine's
	// own interfaces. Such a datagram is not the stream and gets dropped.
	static bool isOwnHandshake(const char* text, size_t length);

	// The handshake text: asks the app to stream to the sender's port
	static const char* getHandshakeText();

private:
	// Callers hold m_mutex
	void applyConfigLocked();
	void closeSocketLocked();
	void sendHandshakeLocked(double nowMs);

	mutable std::mutex m_mutex;
	FaceServiceConfig m_config;
	bool m_bStarted= false;
	UdpSocket m_socket;
	std::array<char, 4096> m_buffer{};

	FaceSample m_latest;
	double m_lastDatagramMs= 0.0;
	double m_lastHandshakeMs= 0.0;
	bool m_bQuietLogged= false;
	bool m_bHandshakeFailureLogged= false;

	uint64_t m_unknownNameCount= 0;
	// Distinct unknown names already logged, capped so a stream of garbage
	// cannot grow it without bound
	static constexpr size_t k_maxLoggedUnknownNames= 64;
	std::set<std::string> m_loggedUnknownNames;
	std::vector<std::string> m_unknownNames;
	std::string m_lastUnknownName;
	uint64_t m_parseFailures= 0;
	double m_rateWindowStartMs= 0.0;
	uint64_t m_rateWindowCount= 0;
	float m_datagramsPerSecond= 0.f;
};
