#include "TestCommon.h"

#include "FaceService.h"
#include "SteadyClock.h"
#include "UdpSocket.h"

// Tests for the iFacialMocap receiver: the datagram parser, the receive
// filter, the blendshape name table, and a loopback socket round trip.

namespace
{
bool parseText(const std::string& text, FaceSample& out, uint64_t& unknown)
{
	return FaceService::parseDatagram(text.data(), text.size(), out, unknown);
}

bool nearlyEqual(float a, float b)
{
	return fabsf(a - b) < 1e-5f;
}
} // namespace

static int runFaceTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-face") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-face") << "FAIL " << name;
			failures++;
		}
	};

	// (a) The name table
	{
		bool bNamesValid= true;
		for (int index= 0; index < ARKIT_BLENDSHAPE_COUNT; ++index)
		{
			const char* name= arkitBlendshapeName(index);
			bNamesValid&= name[0] != '\0' && arkitBlendshapeFromName(name) == index;
			for (int other= 0; other < index; ++other)
				bNamesValid&= strcmp(name, arkitBlendshapeName(other)) != 0;
		}
		check(bNamesValid, "52 unique names that round trip through the lookup");
		check(arkitBlendshapeName(-1)[0] == '\0' && arkitBlendshapeName(ARKIT_BLENDSHAPE_COUNT)[0] == '\0',
			  "out-of-range index gives an empty name");
		check(arkitBlendshapeFromName("eyeBlink_L") == arkitBlendshapeFromName("eyeBlinkLeft") &&
				  arkitBlendshapeFromName("eyeBlink_L") >= 0,
			  "_L folds to Left");
		check(arkitBlendshapeFromName("mouthSmile_R") == arkitBlendshapeFromName("mouthSmileRight") &&
				  arkitBlendshapeFromName("mouthSmile_R") >= 0,
			  "_R folds to Right");
		check(arkitBlendshapeFromName("jawOpen") >= 0 && arkitBlendshapeFromName("notABlendshape") == -1 &&
				  arkitBlendshapeFromName("") == -1 && arkitBlendshapeFromName(nullptr) == -1,
			  "unknown, empty, and null names give -1");
	}

	// (b) v1 and v2 blendshape formats
	{
		FaceSample sample;
		uint64_t unknown= 0;
		const bool bParsed= parseText("jawOpen-50|eyeBlink_L-100|mouthSmile_R-25=head#1,2,3,4,5,6", sample, unknown);
		check(bParsed, "v1 datagram parses");
		check(nearlyEqual(sample.blendshapes[arkitBlendshapeFromName("jawOpen")], 0.5f) &&
				  nearlyEqual(sample.blendshapes[arkitBlendshapeFromName("eyeBlinkLeft")], 1.0f) &&
				  nearlyEqual(sample.blendshapes[arkitBlendshapeFromName("mouthSmileRight")], 0.25f),
			  "v1 '-' values are scaled from percent");
		check(unknown == 0, "no unknown names in a clean datagram");

		FaceSample sample2;
		parseText("jawOpen&50|browInnerUp&-12.5|cheekPuff&0", sample2, unknown);
		check(nearlyEqual(sample2.blendshapes[arkitBlendshapeFromName("jawOpen")], 0.5f), "v2 '&' value parses");
		check(nearlyEqual(sample2.blendshapes[arkitBlendshapeFromName("browInnerUp")], -0.125f),
			  "v2 keeps negative values as sent");
	}

	// (c) Unknown names
	{
		FaceSample sample;
		uint64_t unknown= 0;
		const bool bParsed= parseText("jawOpen-10|fooBar-20|bazQux&30", sample, unknown);
		check(bParsed && unknown == 2, "unknown names are counted and the rest still parses");

		FaceSample statusSample;
		uint64_t statusUnknown= 0;
		parseText("jawOpen-10|trackingStatus-0|hapihapi-1", statusSample, statusUnknown);
		check(!statusSample.faceTracked && statusUnknown == 0,
			  "trackingStatus reads as the face-tracked flag and hapihapi is skipped, neither counted unknown");
		parseText("trackingStatus-1", statusSample, statusUnknown);
		check(statusSample.faceTracked, "trackingStatus 1 reports the face tracked again");

		FaceSample onlyUnknown;
		uint64_t unknownOnly= 0;
		check(!parseText("fooBar-20", onlyUnknown, unknownOnly) && unknownOnly == 1,
			  "a datagram of only unknown names reports no parse");
	}

	// (d) Bones
	{
		FaceSample sample;
		uint64_t unknown= 0;
		parseText("jawOpen-1=head#10.5,-20,30,0.1,0.2,-0.3|rightEye#1,2,3|leftEye#4,5,6", sample, unknown);
		check(sample.hasHead && nearlyEqual(sample.headEulerDegrees.x, 10.5f) &&
				  nearlyEqual(sample.headEulerDegrees.y, -20.f) && nearlyEqual(sample.headEulerDegrees.z, 30.f) &&
				  nearlyEqual(sample.headPosition.x, 0.1f) && nearlyEqual(sample.headPosition.z, -0.3f),
			  "head with six values parses rotation and position");
		check(sample.hasEyes && nearlyEqual(sample.rightEyeEulerDegrees.y, 2.f) &&
				  nearlyEqual(sample.leftEyeEulerDegrees.z, 6.f),
			  "eye bones parse into the right and left slots");

		FaceSample fiveValues;
		check(!parseText("jawOpen-0=head#1,2,3,4,5", fiveValues, unknown) || !fiveValues.hasHead,
			  "head with five values is rejected");
		FaceSample badEye;
		parseText("jawOpen-0=rightEye#1,2", badEye, unknown);
		check(!badEye.hasEyes, "an eye bone with two values is rejected");
		FaceSample badNumber;
		parseText("jawOpen-0=head#1,2,x,4,5,6", badNumber, unknown);
		check(!badNumber.hasHead, "a non-numeric head value is rejected");

		// A later datagram without a bone section keeps the earlier bones
		FaceSample held= sample;
		parseText("jawOpen-70", held, unknown);
		check(held.hasHead && nearlyEqual(held.headEulerDegrees.x, 10.5f) &&
				  nearlyEqual(held.blendshapes[arkitBlendshapeFromName("jawOpen")], 0.7f),
			  "values a datagram omits keep their previous state");

		FaceSample padded;
		check(parseText("  jawOpen-20=head#1,2,3,4,5,6\r\n", padded, unknown) && padded.hasHead,
			  "surrounding whitespace is trimmed");
	}

	// (e) Own-handshake filter
	{
		const char* handshake= FaceService::getHandshakeText();
		check(FaceService::isOwnHandshake(handshake, strlen(handshake)), "the handshake text is filtered");
		check(strlen(handshake) == 51 && strncmp(handshake, "iFacialMocap_", 13) == 0, "handshake text is the app's");
		const std::string frame= "jawOpen-50=head#1,2,3,4,5,6";
		check(!FaceService::isOwnHandshake(frame.data(), frame.size()), "a stream datagram is not filtered");
		check(!FaceService::isOwnHandshake("iFacial", 7) && !FaceService::isOwnHandshake(nullptr, 0),
			  "a short or null buffer is not filtered");
	}

	// (f) Empty and garbage input
	{
		FaceSample sample;
		uint64_t unknown= 0;
		check(!FaceService::parseDatagram(nullptr, 0, sample, unknown), "null input is rejected");
		check(!parseText("", sample, unknown), "empty input is rejected");
		check(!parseText("   \r\n", sample, unknown), "whitespace input is rejected");
		check(!parseText("====|||---###&&&,,,", sample, unknown), "punctuation garbage is rejected");
		check(!parseText("jawOpen-|-5|&|head#|#", sample, unknown), "malformed entries are rejected");
		std::string binary;
		for (int i= 0; i < 600; ++i)
			binary.push_back((char)(i * 37));
		parseText(binary, sample, unknown);
		check(true, "binary garbage does not crash");
		check(!sample.hasHead && !sample.hasEyes && !sample.valid, "garbage leaves the sample untouched");
	}

	// (g) Loopback round trip through the service
	{
		FaceService service;
		service.startup();
		FaceServiceConfig config;
		config.enabled= true;
		config.port= 52731;
		config.phoneAddress= "127.0.0.1";
		service.setConfig(config);

		FaceServiceStatus status= service.getStatus();
		check(status.enabled && status.bound, "service binds its port when enabled");
		check(!status.streaming && !status.latest.valid, "service is quiet before any datagram");

		// The handshake burst goes to 127.0.0.1:port, i.e. back to the service
		// itself, and must be dropped as our own broadcast
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		service.update(steadyNowMs());
		status= service.getStatus();
		check(status.latest.valid == false && status.parseFailures == 0,
			  "the looped-back handshake is dropped, not parsed or counted");

		UdpSocket sender;
		check(sender.open(), "sender socket opens");
		const std::string first= "jawOpen&20|fooBar&5=head#1,2,3,4,5,6";
		const std::string second= "jawOpen&80|eyeBlink_L&100=head#7,8,9,0.1,0.2,0.3|leftEye#1,2,3|rightEye#4,5,6";
		check(sender.sendTo("127.0.0.1", config.port, first.data(), (int)first.size()) &&
				  sender.sendTo("127.0.0.1", config.port, second.data(), (int)second.size()),
			  "datagrams are sent");

		const double startMs= steadyNowMs();
		while (steadyNowMs() - startMs < 1000.0 && service.getStatus().latest.sequence < 2)
		{
			service.update(steadyNowMs());
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		FaceSample latest;
		const bool bGot= service.getLatestSample(latest);
		check(bGot && latest.valid && latest.sequence == 2, "both datagrams drained and counted");
		check(bGot && nearlyEqual(latest.blendshapes[arkitBlendshapeFromName("jawOpen")], 0.8f) &&
				  nearlyEqual(latest.headEulerDegrees.x, 7.f),
			  "the last datagram wins");
		check(bGot && latest.hasEyes && nearlyEqual(latest.rightEyeEulerDegrees.x, 4.f), "eyes arrive over the socket");
		check(service.isStreaming(steadyNowMs()), "streaming right after a datagram");
		check(service.getStatus().unknownNameCount == 1, "the unknown name is counted in the status");
		check(!service.isStreaming(steadyNowMs() + 600.0), "not streaming 600 ms after the last datagram");

		// A port change rebinds
		config.port= 52732;
		service.setConfig(config);
		check(service.getStatus().bound, "service rebinds on a port change");
		config.enabled= false;
		service.setConfig(config);
		check(!service.getStatus().bound, "service releases the socket when disabled");

		sender.close();
		service.shutdown();
	}

	// (h) The socket primitives on their own
	{
		UdpSocket receiver;
		check(receiver.openBound(52733, false), "openBound binds a port");
		char buffer[16]= {};
		check(receiver.receiveFrom(buffer, sizeof(buffer)) == 0, "receiveFrom returns 0 with nothing waiting");

		UdpSocket sender;
		sender.open();
		const char payload[]= "0123456789ABCDEFGHIJ";
		sender.sendTo("127.0.0.1", 52733, payload, (int)sizeof(payload) - 1);
		int received= 0;
		std::string source;
		for (int attempt= 0; attempt < 200 && received == 0; ++attempt)
		{
			received= receiver.receiveFrom(buffer, sizeof(buffer), &source);
			if (received == 0)
				std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}
		check(received == (int)sizeof(buffer) && memcmp(buffer, payload, sizeof(buffer)) == 0,
			  "a datagram longer than the buffer is truncated to it");
		check(source == "127.0.0.1", "the source address is reported");
		receiver.close();
		check(receiver.receiveFrom(buffer, sizeof(buffer)) == -1, "receiveFrom on a closed socket is an error");
	}

	if (failures == 0)
		MIKAN_LOG_INFO("test-face") << "All face receiver tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-face", "iFacialMocap receiver: datagram parser, receive filter, loopback socket",
					eTestCategory::SelfTest, runFaceTest);
