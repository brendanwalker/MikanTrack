#include "OscWriterTest.h"

#include "Logger.h"
#include "OscWriter.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// All expected byte arrays below are hand-computed from the OSC 1.0 spec:
// - strings: ASCII + null terminator, padded with nulls to a 4-byte boundary
// - int32/float32: big-endian
// - bundle: "#bundle\0" + 8-byte big-endian time tag + per-element
//   big-endian int32 size prefix followed by the element bytes

static std::string toHexString(const uint8_t* data, size_t size)
{
	std::string result;
	char byteText[4];
	for (size_t byteIndex= 0; byteIndex < size; ++byteIndex)
	{
		std::snprintf(byteText, sizeof(byteText), "%02X ", data[byteIndex]);
		result+= byteText;
	}
	return result;
}

static bool checkBytes(const char* testName,
					   const std::vector<uint8_t>& actual,
					   const uint8_t* expected,
					   size_t expectedSize)
{
	if (actual.size() != expectedSize || std::memcmp(actual.data(), expected, expectedSize) != 0)
	{
		MIKAN_LOG_ERROR("runOscWriterSelfTest") << testName << " FAILED";
		MIKAN_LOG_ERROR("runOscWriterSelfTest") << "  expected(" << expectedSize
			<< "): " << toHexString(expected, expectedSize);
		MIKAN_LOG_ERROR("runOscWriterSelfTest") << "  actual  (" << actual.size()
			<< "): " << toHexString(actual.data(), actual.size());
		return false;
	}

	MIKAN_LOG_INFO("runOscWriterSelfTest") << testName << " passed";
	return true;
}

bool runOscWriterSelfTest()
{
	bool allPassed= true;

	// -- Test 1: message "/test" with args (1.0f, 2) --------------------------
	// address "/test" (5 chars) -> null + pad to 8 : 2F 74 65 73 74 00 00 00
	// type tags ",fi"  (3 chars) -> null to 4      : 2C 66 69 00
	// float 1.0f big-endian (0x3F800000)           : 3F 80 00 00
	// int32 2 big-endian                           : 00 00 00 02
	static const uint8_t k_expectedTestMessage[]= {
		0x2F, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00, // "/test\0\0\0"
		0x2C, 0x66, 0x69, 0x00,                         // ",fi\0"
		0x3F, 0x80, 0x00, 0x00,                         // 1.0f
		0x00, 0x00, 0x00, 0x02,                         // 2
	};
	{
		OscMessage message("/test");
		message.addFloat(1.0f).addInt32(2);
		allPassed&= checkBytes("message /test (,fi 1.0 2)",
							   message.encode(), k_expectedTestMessage, sizeof(k_expectedTestMessage));
	}

	// -- Test 2: bundle containing the "/test" message ------------------------
	// "#bundle\0"                                   : 23 62 75 6E 64 6C 65 00
	// time tag "immediate" (uint64 1, big-endian)   : 00 00 00 00 00 00 00 01
	// element size prefix (20 bytes, big-endian)    : 00 00 00 14
	// element bytes                                 : the 20 bytes from Test 1
	{
		static const uint8_t k_expectedBundle[]= {
			0x23, 0x62, 0x75, 0x6E, 0x64, 0x6C, 0x65, 0x00, // "#bundle\0"
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // immediate time tag
			0x00, 0x00, 0x00, 0x14,                         // element size = 20
			0x2F, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00, // "/test\0\0\0"
			0x2C, 0x66, 0x69, 0x00,                         // ",fi\0"
			0x3F, 0x80, 0x00, 0x00,                         // 1.0f
			0x00, 0x00, 0x00, 0x02,                         // 2
		};

		OscBundle bundle;
		bundle.addMessage("/test").addFloat(1.0f).addInt32(2);
		allPassed&= checkBytes("bundle [/test]",
							   bundle.encode(), k_expectedBundle, sizeof(k_expectedBundle));
	}

	// -- Test 3: address padding edge cases (3, 4, 5 char addresses) ----------
	// No args: type tag string is just "," -> ",\0\0\0" : 2C 00 00 00
	{
		// "/ab" (3 chars) -> exactly one null reaches the boundary: 4 bytes
		static const uint8_t k_expectedAddr3[]= {
			0x2F, 0x61, 0x62, 0x00, // "/ab\0"
			0x2C, 0x00, 0x00, 0x00, // ",\0\0\0"
		};
		allPassed&= checkBytes("address len 3 (/ab)",
							   OscMessage("/ab").encode(), k_expectedAddr3, sizeof(k_expectedAddr3));

		// "/abc" (4 chars) -> already aligned, so a FULL extra 4-null word is added
		static const uint8_t k_expectedAddr4[]= {
			0x2F, 0x61, 0x62, 0x63, 0x00, 0x00, 0x00, 0x00, // "/abc\0\0\0\0"
			0x2C, 0x00, 0x00, 0x00,                         // ",\0\0\0"
		};
		allPassed&= checkBytes("address len 4 (/abc)",
							   OscMessage("/abc").encode(), k_expectedAddr4, sizeof(k_expectedAddr4));

		// "/abcd" (5 chars) -> null + 2 pad nulls to reach 8
		static const uint8_t k_expectedAddr5[]= {
			0x2F, 0x61, 0x62, 0x63, 0x64, 0x00, 0x00, 0x00, // "/abcd\0\0\0"
			0x2C, 0x00, 0x00, 0x00,                         // ",\0\0\0"
		};
		allPassed&= checkBytes("address len 5 (/abcd)",
							   OscMessage("/abcd").encode(), k_expectedAddr5, sizeof(k_expectedAddr5));
	}

	// -- Test 4: string argument padding --------------------------------------
	{
		// "/s" + string "abc": arg -> "abc\0" (3 chars + 1 null = aligned)
		static const uint8_t k_expectedStr3[]= {
			0x2F, 0x73, 0x00, 0x00, // "/s\0\0"
			0x2C, 0x73, 0x00, 0x00, // ",s\0\0"
			0x61, 0x62, 0x63, 0x00, // "abc\0"
		};
		{
			OscMessage message("/s");
			message.addString("abc");
			allPassed&= checkBytes("string arg len 3 (abc)",
								   message.encode(), k_expectedStr3, sizeof(k_expectedStr3));
		}

		// "/s" + string "abcd": arg -> "abcd" + full 4-null word
		static const uint8_t k_expectedStr4[]= {
			0x2F, 0x73, 0x00, 0x00,                         // "/s\0\0"
			0x2C, 0x73, 0x00, 0x00,                         // ",s\0\0"
			0x61, 0x62, 0x63, 0x64, 0x00, 0x00, 0x00, 0x00, // "abcd\0\0\0\0"
		};
		{
			OscMessage message("/s");
			message.addString("abcd");
			allPassed&= checkBytes("string arg len 4 (abcd)",
								   message.encode(), k_expectedStr4, sizeof(k_expectedStr4));
		}
	}

	// -- Test 5: negative int32 is encoded as big-endian two's complement -----
	{
		static const uint8_t k_expectedNegative[]= {
			0x2F, 0x6E, 0x00, 0x00, // "/n\0\0"
			0x2C, 0x69, 0x00, 0x00, // ",i\0\0"
			0xFF, 0xFF, 0xFF, 0xFE, // -2
		};
		OscMessage message("/n");
		message.addInt32(-2);
		allPassed&= checkBytes("negative int32 (-2)",
							   message.encode(), k_expectedNegative, sizeof(k_expectedNegative));
	}

	// -- Test 6: multi-message bundle + pool reuse after clear() --------------
	// element 1: "/a" ,i 1   -> 12 bytes; element 2: "/b" ,f 0.5f -> 12 bytes
	{
		static const uint8_t k_expectedTwoMessageBundle[]= {
			0x23, 0x62, 0x75, 0x6E, 0x64, 0x6C, 0x65, 0x00, // "#bundle\0"
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, // immediate time tag
			0x00, 0x00, 0x00, 0x0C,                         // element size = 12
			0x2F, 0x61, 0x00, 0x00,                         // "/a\0\0"
			0x2C, 0x69, 0x00, 0x00,                         // ",i\0\0"
			0x00, 0x00, 0x00, 0x01,                         // 1
			0x00, 0x00, 0x00, 0x0C,                         // element size = 12
			0x2F, 0x62, 0x00, 0x00,                         // "/b\0\0"
			0x2C, 0x66, 0x00, 0x00,                         // ",f\0\0"
			0x3F, 0x00, 0x00, 0x00,                         // 0.5f
		};

		OscBundle bundle;
		// Fill once with junk, then clear and re-fill: encoding must not be
		// affected by pooled message reuse.
		bundle.addMessage("/junk").addString("stale").addFloat(9.f);
		bundle.addMessage("/junk2").addInt32(99);
		bundle.clear();

		bundle.addMessage("/a").addInt32(1);
		bundle.addMessage("/b").addFloat(0.5f);
		allPassed&= checkBytes("two-message bundle after clear()",
							   bundle.encode(), k_expectedTwoMessageBundle, sizeof(k_expectedTwoMessageBundle));
	}

	if (allPassed)
	{
		MIKAN_LOG_INFO("runOscWriterSelfTest") << "All OSC writer self tests passed";
	}
	else
	{
		MIKAN_LOG_ERROR("runOscWriterSelfTest") << "One or more OSC writer self tests FAILED";
	}

	return allPassed;
}
