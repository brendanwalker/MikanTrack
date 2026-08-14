#include "OscWriterTest.h"

#include "Logger.h"
#include "OscStreamer.h"
#include "OscWriter.h"

#include "glm/gtc/quaternion.hpp"
#include "glm/trigonometric.hpp"

#include <algorithm>
#include <cmath>
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

	// -- Dropout hold-and-decay (OscStreamer::resolveOutputPose) --------------
	{
		bool holdPassed= true;

		HandPose live;
		live.tracked= true;
		live.presence= 0.9f;
		live.confidence= 0.8f;
		live.palmPositionWorld= glm::vec3(0.1f, 0.2f, 0.3f);
		live.hasWorldPose= true;

		HandPose lost; // untracked default

		OscStreamer::HeldPoseState held;
		HandPose out;

		live.hasForearmPose= true;
		live.forearmConfidence= 0.6f;

		// live pose passes through and arms the hold
		holdPassed&= OscStreamer::resolveOutputPose(live, 1000.0, 0.f, 250.f, held, out);
		holdPassed&= out.confidence == 0.8f;
		holdPassed&= out.forearmConfidence == 0.6f;

		// dropout at +100ms: held pose, confidence decayed by 100/250
		holdPassed&= OscStreamer::resolveOutputPose(lost, 1100.0, 0.f, 250.f, held, out);
		holdPassed&= fabsf(out.confidence - 0.8f * (1.f - 100.f / 250.f)) < 1e-4f;
		holdPassed&= out.palmPositionWorld == live.palmPositionWorld;
		// The elbow confidence must decay WITH the hand's. A consumer gates
		// the elbow on that one number, so a held pose advertising its last
		// live value would read as freshly measured.
		holdPassed&= fabsf(out.forearmConfidence - 0.6f * (1.f - 100.f / 250.f)) < 1e-4f;

		// still down at +250ms: last held frame (confidence ~0)
		holdPassed&= OscStreamer::resolveOutputPose(lost, 1250.0, 0.f, 250.f, held, out);
		holdPassed&= out.confidence < 1e-4f;

		// past the window: untracked, hold disarmed
		holdPassed&= !OscStreamer::resolveOutputPose(lost, 1251.0, 0.f, 250.f, held, out);
		holdPassed&= !held.valid;

		// reacquisition re-arms; a low-confidence pose gates like a dropout
		holdPassed&= OscStreamer::resolveOutputPose(live, 2000.0, 0.f, 250.f, held, out);
		HandPose lowConfidence= live;
		lowConfidence.confidence= 0.1f;
		holdPassed&= OscStreamer::resolveOutputPose(lowConfidence, 2100.0, 0.5f, 250.f, held, out);
		holdPassed&= out.confidence > 0.f && out.confidence < 0.8f;

		// holdMs=0 reports the dropout immediately (legacy behavior)
		OscStreamer::HeldPoseState heldOff;
		holdPassed&= OscStreamer::resolveOutputPose(live, 3000.0, 0.f, 0.f, heldOff, out);
		holdPassed&= !OscStreamer::resolveOutputPose(lost, 3016.0, 0.f, 0.f, heldOff, out);

		// timestamp regression (video restart) drops the hold
		OscStreamer::HeldPoseState heldRegress;
		holdPassed&= OscStreamer::resolveOutputPose(live, 4000.0, 0.f, 250.f, heldRegress, out);
		holdPassed&= !OscStreamer::resolveOutputPose(lost, 3900.0, 0.f, 250.f, heldRegress, out);

		if (holdPassed)
			MIKAN_LOG_INFO("runOscWriterSelfTest") << "dropout hold-and-decay passed";
		else
			MIKAN_LOG_ERROR("runOscWriterSelfTest") << "dropout hold-and-decay FAILED";
		allPassed&= holdPassed;
	}

	// -- Wrist joint rotation (HandPose::getWristRotation) -------------------
	{
		bool wristPassed= true;

		// Forearm yawed 30 deg; palm additionally flexed 25 deg about the
		// forearm's local X. The wrist rotation must recover exactly that
		// local flex - independent of where the forearm is pointing, which is
		// the whole point of expressing it in the forearm frame.
		const glm::quat forearm= glm::angleAxis(glm::radians(30.f), glm::vec3(0.f, 0.f, 1.f));
		const glm::quat trueWristLocal= glm::angleAxis(glm::radians(25.f), glm::vec3(1.f, 0.f, 0.f));

		HandPose pose;
		pose.hasWorldPose= true;
		pose.hasForearmPose= true;
		pose.forearmOrientationWorld= forearm;
		pose.palmOrientationWorld= forearm * trueWristLocal; // child = parent * local

		const glm::quat recovered= pose.getWristRotation();
		const glm::quat error= glm::inverse(trueWristLocal) * recovered;
		const float errorDegrees=
			glm::degrees(2.f * asinf(std::min(glm::length(glm::vec3(error.x, error.y, error.z)), 1.f)));
		wristPassed&= errorDegrees < 0.01f;

		// A palm aligned with the forearm must read identity (no wrist bend)
		HandPose straight;
		straight.hasWorldPose= true;
		straight.hasForearmPose= true;
		straight.forearmOrientationWorld= forearm;
		straight.palmOrientationWorld= forearm;
		const glm::quat straightRotation= straight.getWristRotation();
		wristPassed&= fabsf(fabsf(straightRotation.w) - 1.f) < 1e-5f;

		if (wristPassed)
			MIKAN_LOG_INFO("runOscWriterSelfTest") << "wrist joint rotation passed";
		else
			MIKAN_LOG_ERROR("runOscWriterSelfTest")
				<< "wrist joint rotation FAILED (err " << errorDegrees << " deg)";
		allPassed&= wristPassed;
	}

	// -- Elbow output (OscStreamer::resolveElbowOutput) ----------------------
	{
		bool elbowPassed= true;

		// Forearm pointing along world +X, so the elbow sits one forearm
		// length back along -X from the WRIST (not the palm center - the palm
		// origin is half a palm forward of the wrist joint)
		HandPose pose;
		pose.tracked= true;
		pose.hasWorldPose= true;
		pose.hasForearmPose= true;
		pose.confidence= 0.9f;
		pose.forearmConfidence= 0.72f;
		pose.palmPositionWorld= glm::vec3(0.5f, 0.f, 1.f);
		pose.palmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
		pose.forearmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
		pose.skeleton.baseInPalm[(int)eFinger::Middle]= glm::vec3(0.04f, 0.f, 0.f);

		glm::vec3 elbow(0.f);
		float confidence= -1.f;
		OscStreamer::resolveElbowOutput(pose, true, 0.25f, elbow, confidence);

		const glm::vec3 expected= pose.getWristPositionWorld() - glm::vec3(0.25f, 0.f, 0.f);
		elbowPassed&= glm::length(elbow - expected) < 1e-5f;
		elbowPassed&= fabsf(confidence - 0.72f) < 1e-5f;

		// A hand with no calibrated IMU still produces output, reporting
		// confidence 0 - the message is sent every frame, so silence is not
		// available as a way to say "unusable"
		HandPose noImu= pose;
		noImu.hasForearmPose= false;
		OscStreamer::resolveElbowOutput(noImu, true, 0.25f, elbow, confidence);
		elbowPassed&= confidence == 0.f;
		elbowPassed&= elbow == glm::vec3(0.f);

		// Same for a hand that is not being sent at all
		OscStreamer::resolveElbowOutput(pose, false, 0.25f, elbow, confidence);
		elbowPassed&= confidence == 0.f;

		// And for a camera-space pose, which has no world frame to hang an
		// elbow off
		HandPose cameraSpace= pose;
		cameraSpace.hasWorldPose= false;
		OscStreamer::resolveElbowOutput(cameraSpace, true, 0.25f, elbow, confidence);
		elbowPassed&= confidence == 0.f;

		// Forearm length only slides the elbow along the forearm axis; it
		// must not rotate it
		glm::vec3 shortElbow(0.f);
		glm::vec3 longElbow(0.f);
		float ignored= 0.f;
		OscStreamer::resolveElbowOutput(pose, true, 0.20f, shortElbow, ignored);
		OscStreamer::resolveElbowOutput(pose, true, 0.30f, longElbow, ignored);
		const glm::vec3 slide= longElbow - shortElbow;
		elbowPassed&= fabsf(slide.y) < 1e-6f && fabsf(slide.z) < 1e-6f;
		elbowPassed&= fabsf(glm::length(slide) - 0.10f) < 1e-5f;

		if (elbowPassed)
			MIKAN_LOG_INFO("runOscWriterSelfTest") << "elbow output passed";
		else
			MIKAN_LOG_ERROR("runOscWriterSelfTest") << "elbow output FAILED";
		allPassed&= elbowPassed;
	}

	// -- Shoulder output (OscStreamer::resolveShoulderOutput) ----------------
	{
		bool shoulderPassed= true;

		HandPose pose;
		pose.tracked= true;
		pose.hasWorldPose= true;
		pose.hasShoulder= true;
		pose.shoulderPositionWorld= glm::vec3(0.2f, -0.1f, 1.4f);
		pose.shoulderConfidence= 0.6f;

		glm::vec3 shoulder(0.f);
		float confidence= -1.f;
		OscStreamer::resolveShoulderOutput(pose, true, shoulder, confidence);
		shoulderPassed&= glm::length(shoulder - pose.shoulderPositionWorld) < 1e-6f;
		shoulderPassed&= fabsf(confidence - 0.6f) < 1e-6f;

		// No solved shoulder, unsent pose, and camera-space pose all report
		// confidence 0 rather than going silent
		HandPose noShoulder= pose;
		noShoulder.hasShoulder= false;
		OscStreamer::resolveShoulderOutput(noShoulder, true, shoulder, confidence);
		shoulderPassed&= confidence == 0.f && shoulder == glm::vec3(0.f);

		OscStreamer::resolveShoulderOutput(pose, false, shoulder, confidence);
		shoulderPassed&= confidence == 0.f;

		HandPose cameraSpace= pose;
		cameraSpace.hasWorldPose= false;
		OscStreamer::resolveShoulderOutput(cameraSpace, true, shoulder, confidence);
		shoulderPassed&= confidence == 0.f;

		if (shoulderPassed)
			MIKAN_LOG_INFO("runOscWriterSelfTest") << "shoulder output passed";
		else
			MIKAN_LOG_ERROR("runOscWriterSelfTest") << "shoulder output FAILED";
		allPassed&= shoulderPassed;
	}

	// -- Head output (OscStreamer::resolveHeadOutput) ------------------------
	{
		bool headPassed= true;

		TrackingFrameResult::HeadPose head;
		head.valid= true;
		head.positionWorld= glm::vec3(0.1f, 0.2f, 1.6f);
		head.orientationWorld= glm::normalize(glm::quat(0.9f, 0.1f, 0.2f, 0.3f));
		head.confidence= 0.8f;

		glm::vec3 position(0.f);
		glm::quat orientation(1.f, 0.f, 0.f, 0.f);
		float confidence= -1.f;
		OscStreamer::resolveHeadOutput(head, position, orientation, confidence);
		headPassed&= glm::length(position - head.positionWorld) < 1e-6f;
		headPassed&= fabsf(glm::dot(orientation, head.orientationWorld)) > 1.f - 1e-6f;
		headPassed&= fabsf(confidence - 0.8f) < 1e-6f;

		// Invalid head: identity orientation, zero position, confidence 0
		TrackingFrameResult::HeadPose invalid;
		OscStreamer::resolveHeadOutput(invalid, position, orientation, confidence);
		headPassed&= confidence == 0.f;
		headPassed&= position == glm::vec3(0.f);
		headPassed&= orientation == glm::quat(1.f, 0.f, 0.f, 0.f);

		if (headPassed)
			MIKAN_LOG_INFO("runOscWriterSelfTest") << "head output passed";
		else
			MIKAN_LOG_ERROR("runOscWriterSelfTest") << "head output FAILED";
		allPassed&= headPassed;
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
