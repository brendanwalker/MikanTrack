#include "FaceService.h"

#include <charconv>
#include <cstring>
#include <string_view>

#include "Logger.h"
#include "SteadyClock.h"

// The app's handshake: a datagram carrying this text asks it to stream to the
// sender's port. No terminator on the wire.
static const char* const k_handshakeText= "iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719";
static constexpr std::string_view k_handshakePrefix= "iFacialMocap_";
// Sent a few times over, since a datagram may not land
static constexpr int k_handshakeBurst= 5;
// The handshake repeats at this interval while nothing has arrived for as
// long, so a phone that comes up later, or restarts, finds the service on its
// own
static constexpr double k_handshakeIntervalMs= 3000.0;
// The sample counts as live for this long past its datagram
static constexpr double k_streamingWindowMs= 500.0;
static constexpr double k_rateWindowMs= 1000.0;

namespace
{
std::string_view trimView(std::string_view text)
{
	const char* whitespace= " \t\r\n";
	const size_t first= text.find_first_not_of(whitespace);
	if (first == std::string_view::npos)
		return std::string_view();
	const size_t last= text.find_last_not_of(whitespace);
	return text.substr(first, last - first + 1);
}

// The whole token as a float; false for anything else
bool parseFloatToken(std::string_view token, float& outValue)
{
	if (token.empty())
		return false;
	const char* const end= token.data() + token.size();
	const std::from_chars_result result= std::from_chars(token.data(), end, outValue);
	return result.ec == std::errc() && result.ptr == end;
}

// Comma-separated numbers into values, up to count; returns how many parsed
int parseNumbers(std::string_view text, float* values, int count)
{
	int parsed= 0;
	while (parsed < count && !text.empty())
	{
		const size_t comma= text.find(',');
		if (!parseFloatToken(text.substr(0, comma), values[parsed]))
			break;
		++parsed;
		text= comma == std::string_view::npos ? std::string_view() : text.substr(comma + 1);
	}
	return parsed;
}

// Splits off the next '|'-separated entry
std::string_view nextEntry(std::string_view& section)
{
	const size_t bar= section.find('|');
	const std::string_view entry= section.substr(0, bar);
	section= bar == std::string_view::npos ? std::string_view() : section.substr(bar + 1);
	return entry;
}
} // namespace

FaceService::FaceService()= default;

FaceService::~FaceService()
{
	shutdown();
}

bool FaceService::startup()
{
	std::lock_guard<std::mutex> lock(m_mutex);
	m_bStarted= true;
	applyConfigLocked();
	return true;
}

void FaceService::shutdown()
{
	std::lock_guard<std::mutex> lock(m_mutex);
	closeSocketLocked();
	m_bStarted= false;
	m_latest= FaceSample();
	m_unknownNameCount= 0;
	m_parseFailures= 0;
	m_datagramsPerSecond= 0.f;
	m_rateWindowCount= 0;
}

void FaceService::setConfig(const FaceServiceConfig& config)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	const bool bRebind= config.enabled != m_config.enabled || config.port != m_config.port;
	const bool bAddressChanged= config.phoneAddress != m_config.phoneAddress;
	m_config= config;

	if (!m_bStarted)
		return;

	if (bRebind || (m_config.enabled && !m_socket.isOpen()))
	{
		applyConfigLocked();
	}
	else if (bAddressChanged && m_socket.isOpen())
	{
		m_bHandshakeFailureLogged= false;
		sendHandshakeLocked(steadyNowMs());
	}
}

void FaceService::applyConfigLocked()
{
	closeSocketLocked();
	if (!m_config.enabled)
		return;

	if (!m_socket.openBound(m_config.port, true))
	{
		MIKAN_LOG_ERROR("FaceService") << "Failed to bind UDP port " << m_config.port;
		m_socket.close();
		return;
	}

	MIKAN_LOG_INFO("FaceService") << "Listening on UDP " << m_config.port << " for iFacialMocap, handshake to "
								  << m_config.phoneAddress;

	const double nowMs= steadyNowMs();
	m_lastDatagramMs= nowMs;
	m_rateWindowStartMs= nowMs;
	m_rateWindowCount= 0;
	m_bQuietLogged= false;
	m_bHandshakeFailureLogged= false;
	sendHandshakeLocked(nowMs);
}

void FaceService::closeSocketLocked()
{
	m_socket.close();
	m_datagramsPerSecond= 0.f;
}

void FaceService::sendHandshakeLocked(double nowMs)
{
	m_lastHandshakeMs= nowMs;
	if (!m_socket.isOpen())
		return;

	const int length= (int)strlen(k_handshakeText);
	bool bSent= true;
	for (int burst= 0; burst < k_handshakeBurst; ++burst)
		bSent&= m_socket.sendTo(m_config.phoneAddress, m_config.port, k_handshakeText, length);

	if (!bSent && !m_bHandshakeFailureLogged)
	{
		m_bHandshakeFailureLogged= true;
		MIKAN_LOG_WARNING("FaceService") << "Handshake to '" << m_config.phoneAddress
										 << "' failed; point the app at this machine by hand if it stays silent";
	}
}

void FaceService::update(double nowMs)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	if (!m_bStarted || !m_socket.isOpen())
		return;

	// Every datagram waiting, parsed in arrival order so the newest stands; a
	// partial entry in an older one is overwritten by the next
	for (;;)
	{
		std::string source;
		const int received= m_socket.receiveFrom(m_buffer.data(), (int)m_buffer.size(), &source);
		if (received <= 0)
			break;

		if (isOwnHandshake(m_buffer.data(), (size_t)received))
			continue;

		FaceSample next= m_latest;
		m_unknownNames.clear();
		const bool bParsed= parseDatagram(m_buffer.data(), (size_t)received, next, m_unknownNameCount, &m_unknownNames);
		// Each name the table lacks is logged once, so a phone app version
		// that spells a blendshape differently is named rather than counted
		for (const std::string& name : m_unknownNames)
		{
			m_lastUnknownName= name;
			if (m_loggedUnknownNames.size() < k_maxLoggedUnknownNames && m_loggedUnknownNames.insert(name).second)
				MIKAN_LOG_WARNING("FaceService") << "Unknown blendshape name '" << name << "' skipped";
		}
		if (!bParsed)
		{
			++m_parseFailures;
			continue;
		}

		if (!m_latest.valid || m_bQuietLogged)
			MIKAN_LOG_INFO("FaceService") << "iFacialMocap streaming from " << source;

		next.valid= true;
		next.sequence= m_latest.sequence + 1;
		next.timestampMs= nowMs;
		m_latest= next;
		m_lastDatagramMs= nowMs;
		m_bQuietLogged= false;
		++m_rateWindowCount;
	}

	if (nowMs - m_lastDatagramMs >= k_handshakeIntervalMs)
	{
		if (m_latest.valid && !m_bQuietLogged)
		{
			MIKAN_LOG_INFO("FaceService") << "The iFacialMocap stream went quiet; the face holds";
			m_bQuietLogged= true;
		}

		if (nowMs - m_lastHandshakeMs >= k_handshakeIntervalMs)
			sendHandshakeLocked(nowMs);
	}

	const double windowMs= nowMs - m_rateWindowStartMs;
	if (windowMs >= k_rateWindowMs)
	{
		m_datagramsPerSecond= (float)((double)m_rateWindowCount * 1000.0 / windowMs);
		m_rateWindowStartMs= nowMs;
		m_rateWindowCount= 0;
	}
}

bool FaceService::getLatestSample(FaceSample& outSample) const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	if (!m_latest.valid)
		return false;
	outSample= m_latest;
	return true;
}

bool FaceService::isStreaming(double nowMs) const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_latest.valid && nowMs - m_latest.timestampMs < k_streamingWindowMs;
}

FaceServiceStatus FaceService::getStatus() const
{
	std::lock_guard<std::mutex> lock(m_mutex);
	FaceServiceStatus status;
	status.enabled= m_config.enabled;
	status.bound= m_socket.isOpen();
	status.streaming= m_latest.valid && steadyNowMs() - m_latest.timestampMs < k_streamingWindowMs;
	status.datagramsPerSecond= status.streaming ? m_datagramsPerSecond : 0.f;
	status.unknownNameCount= m_unknownNameCount;
	status.lastUnknownName= m_lastUnknownName;
	status.parseFailures= m_parseFailures;
	status.latest= m_latest;
	return status;
}

const char* FaceService::getHandshakeText()
{
	return k_handshakeText;
}

bool FaceService::isOwnHandshake(const char* text, size_t length)
{
	return text != nullptr && std::string_view(text, length).substr(0, k_handshakePrefix.size()) == k_handshakePrefix;
}

bool FaceService::parseDatagram(const char* text, size_t length, FaceSample& out, uint64_t& outUnknownNames,
								std::vector<std::string>* outUnknownNameList)
{
	if (text == nullptr)
		return false;

	std::string_view all= trimView(std::string_view(text, length));
	if (all.empty())
		return false;

	const size_t divide= all.find('=');
	std::string_view shapes= all.substr(0, divide);
	std::string_view bones= divide == std::string_view::npos ? std::string_view() : all.substr(divide + 1);
	bool bParsedAny= false;

	while (!shapes.empty())
	{
		const std::string_view entry= nextEntry(shapes);
		if (entry.empty())
			continue;

		size_t separator= entry.find('&');
		if (separator == std::string_view::npos)
			separator= entry.find('-');
		if (separator == std::string_view::npos)
			continue;

		float value= 0.f;
		if (!parseFloatToken(entry.substr(separator + 1), value))
			continue;

		const std::string name(entry.substr(0, separator));
		// Status fields share the blendshape section: the face-tracked flag,
		// and an app field that carries nothing a face needs
		if (name == "trackingStatus")
		{
			out.faceTracked= value != 0.f;
			bParsedAny= true;
			continue;
		}
		if (name == "hapihapi")
			continue;

		const int index= arkitBlendshapeFromName(name.c_str());
		if (index < 0)
		{
			++outUnknownNames;
			if (outUnknownNameList != nullptr)
				outUnknownNameList->push_back(name);
			continue;
		}

		out.blendshapes[index]= value / 100.f;
		bParsedAny= true;
	}

	while (!bones.empty())
	{
		const std::string_view entry= nextEntry(bones);
		const size_t hash= entry.find('#');
		if (hash == std::string_view::npos)
			continue;

		const std::string_view name= entry.substr(0, hash);
		const std::string_view numbers= entry.substr(hash + 1);
		float values[6]= {};

		if (name == "head")
		{
			if (parseNumbers(numbers, values, 6) == 6)
			{
				out.headEulerDegrees= glm::vec3(values[0], values[1], values[2]);
				out.headPosition= glm::vec3(values[3], values[4], values[5]);
				out.hasHead= true;
				bParsedAny= true;
			}
		}
		else if (name == "rightEye")
		{
			if (parseNumbers(numbers, values, 3) == 3)
			{
				out.rightEyeEulerDegrees= glm::vec3(values[0], values[1], values[2]);
				out.hasEyes= true;
				bParsedAny= true;
			}
		}
		else if (name == "leftEye")
		{
			if (parseNumbers(numbers, values, 3) == 3)
			{
				out.leftEyeEulerDegrees= glm::vec3(values[0], values[1], values[2]);
				out.hasEyes= true;
				bParsedAny= true;
			}
		}
	}

	return bParsedAny;
}
