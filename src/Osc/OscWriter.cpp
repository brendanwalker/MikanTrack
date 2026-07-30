#include "OscWriter.h"

#include <cstring>

// -- OscEncoding -----
namespace OscEncoding
{
	void appendUint32(std::vector<uint8_t>& outBuffer, uint32_t value)
	{
		outBuffer.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
		outBuffer.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
		outBuffer.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
		outBuffer.push_back(static_cast<uint8_t>(value & 0xFF));
	}

	void appendInt32(std::vector<uint8_t>& outBuffer, int32_t value)
	{
		appendUint32(outBuffer, static_cast<uint32_t>(value));
	}

	void appendUint64(std::vector<uint8_t>& outBuffer, uint64_t value)
	{
		appendUint32(outBuffer, static_cast<uint32_t>(value >> 32));
		appendUint32(outBuffer, static_cast<uint32_t>(value & 0xFFFFFFFFull));
	}

	void appendFloat32(std::vector<uint8_t>& outBuffer, float value)
	{
		static_assert(sizeof(float) == sizeof(uint32_t), "float must be 32-bit");
		uint32_t bits= 0;
		std::memcpy(&bits, &value, sizeof(bits));
		appendUint32(outBuffer, bits);
	}

	void appendPaddedString(std::vector<uint8_t>& outBuffer, const char* str, size_t length)
	{
		// A string occupies its characters plus 1 to 4 null bytes so that the
		// total length is a multiple of 4 (there is always at least one null).
		const size_t paddedLength= (length + 4) & ~static_cast<size_t>(3);

		outBuffer.insert(outBuffer.end(),
						 reinterpret_cast<const uint8_t*>(str),
						 reinterpret_cast<const uint8_t*>(str) + length);
		outBuffer.insert(outBuffer.end(), paddedLength - length, 0);
	}

	void appendPaddedString(std::vector<uint8_t>& outBuffer, const std::string& str)
	{
		appendPaddedString(outBuffer, str.c_str(), str.size());
	}
}

// -- OscMessage -----
void OscMessage::reset(const char* address)
{
	m_address= (address != nullptr) ? address : "";
	m_typeTags= ",";
	m_argData.clear();
}

OscMessage& OscMessage::addFloat(float value)
{
	m_typeTags+= 'f';
	OscEncoding::appendFloat32(m_argData, value);
	return *this;
}

OscMessage& OscMessage::addInt32(int32_t value)
{
	m_typeTags+= 'i';
	OscEncoding::appendInt32(m_argData, value);
	return *this;
}

OscMessage& OscMessage::addString(const char* value)
{
	m_typeTags+= 's';
	OscEncoding::appendPaddedString(m_argData, value, std::strlen(value));
	return *this;
}

void OscMessage::encode(std::vector<uint8_t>& outBuffer) const
{
	OscEncoding::appendPaddedString(outBuffer, m_address);
	OscEncoding::appendPaddedString(outBuffer, m_typeTags);
	outBuffer.insert(outBuffer.end(), m_argData.begin(), m_argData.end());
}

std::vector<uint8_t> OscMessage::encode() const
{
	std::vector<uint8_t> buffer;
	encode(buffer);
	return buffer;
}

// -- OscBundle -----
OscMessage& OscBundle::addMessage(const char* address)
{
	if (m_usedMessageCount < m_messagePool.size())
	{
		// Reuse a pooled message (keeps its internal buffer capacity)
		OscMessage& message= m_messagePool[m_usedMessageCount];
		message.reset(address);
		m_usedMessageCount++;
		return message;
	}

	m_messagePool.emplace_back(address);
	m_usedMessageCount++;
	return m_messagePool.back();
}

void OscBundle::encode(std::vector<uint8_t>& outBuffer) const
{
	// "#bundle" + null == exactly 8 bytes, already 4-byte aligned
	OscEncoding::appendPaddedString(outBuffer, "#bundle", 7);
	OscEncoding::appendUint64(outBuffer, m_timeTag);

	for (size_t messageIndex= 0; messageIndex < m_usedMessageCount; ++messageIndex)
	{
		// Reserve a placeholder for the big-endian int32 element size prefix
		const size_t sizeOffset= outBuffer.size();
		OscEncoding::appendInt32(outBuffer, 0);

		const size_t elementStart= outBuffer.size();
		m_messagePool[messageIndex].encode(outBuffer);

		// Patch the size prefix now that the element length is known
		const uint32_t elementSize= static_cast<uint32_t>(outBuffer.size() - elementStart);
		outBuffer[sizeOffset]= static_cast<uint8_t>((elementSize >> 24) & 0xFF);
		outBuffer[sizeOffset + 1]= static_cast<uint8_t>((elementSize >> 16) & 0xFF);
		outBuffer[sizeOffset + 2]= static_cast<uint8_t>((elementSize >> 8) & 0xFF);
		outBuffer[sizeOffset + 3]= static_cast<uint8_t>(elementSize & 0xFF);
	}
}

std::vector<uint8_t> OscBundle::encode() const
{
	std::vector<uint8_t> buffer;
	encode(buffer);
	return buffer;
}
