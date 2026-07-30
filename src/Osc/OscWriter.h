#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Minimal hand-rolled OSC 1.0 encoder (no external dependencies).
//
// Wire format summary (OSC 1.0 spec):
// - Strings are ASCII, null-terminated, padded with additional nulls to a
//   4-byte boundary (a string always occupies at least one null byte).
// - int32/float32 are transmitted big-endian.
// - A message is: <padded address> <padded type-tag string (","+tags)> <args>.
// - A bundle is: "#bundle\0" <8-byte big-endian time tag> then for each
//   element a big-endian int32 byte-size prefix followed by the element bytes.

/// OSC time tag meaning "execute immediately" (LSB set, everything else 0).
constexpr uint64_t k_oscTimeTagImmediate= 0x0000000000000001ull;

namespace OscEncoding
{
	void appendInt32(std::vector<uint8_t>& outBuffer, int32_t value);
	void appendUint32(std::vector<uint8_t>& outBuffer, uint32_t value);
	void appendUint64(std::vector<uint8_t>& outBuffer, uint64_t value);
	void appendFloat32(std::vector<uint8_t>& outBuffer, float value);
	void appendPaddedString(std::vector<uint8_t>& outBuffer, const char* str, size_t length);
	void appendPaddedString(std::vector<uint8_t>& outBuffer, const std::string& str);
}

/// A single OSC message: address pattern + typed arguments.
/// Reusable: call reset() to clear it while keeping buffer capacity.
class OscMessage
{
public:
	OscMessage()= default;
	explicit OscMessage(const char* address) { reset(address); }

	/// Clear the message and assign a new address (keeps internal capacity).
	void reset(const char* address);

	OscMessage& addFloat(float value);
	OscMessage& addInt32(int32_t value);
	OscMessage& addString(const char* value);
	OscMessage& addString(const std::string& value) { return addString(value.c_str()); }

	/// Append the encoded message bytes to outBuffer.
	void encode(std::vector<uint8_t>& outBuffer) const;

	/// Convenience: encode into a freshly allocated buffer.
	std::vector<uint8_t> encode() const;

private:
	std::string m_address;
	std::string m_typeTags= ","; // always starts with ','
	std::vector<uint8_t> m_argData; // args pre-encoded big-endian/padded
};

/// An OSC bundle: a time tag plus a flat list of child messages.
/// (One nesting level only — nested bundles are not supported.)
/// Reusable: clear() resets the message list but keeps the message pool
/// allocated so per-frame encoding stays allocation-light.
class OscBundle
{
public:
	OscBundle()= default;

	void setTimeTag(uint64_t timeTag) { m_timeTag= timeTag; }
	uint64_t getTimeTag() const { return m_timeTag; }

	/// Remove all messages (retains pooled message capacity).
	void clear() { m_usedMessageCount= 0; }

	size_t getMessageCount() const { return m_usedMessageCount; }

	/// Add a message with the given address and return a reference to fill in
	/// arguments. The reference is valid until the next addMessage()/clear().
	OscMessage& addMessage(const char* address);

	/// Append the encoded bundle bytes to outBuffer.
	void encode(std::vector<uint8_t>& outBuffer) const;

	/// Convenience: encode into a freshly allocated buffer.
	std::vector<uint8_t> encode() const;

private:
	uint64_t m_timeTag= k_oscTimeTagImmediate;
	std::vector<OscMessage> m_messagePool;
	size_t m_usedMessageCount= 0;
};
