#include "TrackingRecording.h"

#include <cstring>

#include "TrackingJson.h"

using json= nlohmann::json;
using namespace TrackingJson;

namespace
{
constexpr uint64_t k_fnvBasis= 0xcbf29ce484222325ull;
constexpr uint64_t k_fnvPrime= 0x100000001b3ull;

void fnvByte(uint64_t& hash, uint8_t byte)
{
	hash^= byte;
	hash*= k_fnvPrime;
}

void fnvFloat(uint64_t& hash, float value)
{
	uint32_t bits= 0;
	std::memcpy(&bits, &value, sizeof(bits));
	fnvByte(hash, (uint8_t)(bits & 0xff));
	fnvByte(hash, (uint8_t)((bits >> 8) & 0xff));
	fnvByte(hash, (uint8_t)((bits >> 16) & 0xff));
	fnvByte(hash, (uint8_t)((bits >> 24) & 0xff));
}

std::string checksumToHex(uint64_t checksum)
{
	char buffer[17];
	snprintf(buffer, sizeof(buffer), "%016llx", (unsigned long long)checksum);
	return buffer;
}

uint64_t checksumFromHex(const std::string& hex)
{
	return strtoull(hex.c_str(), nullptr, 16);
}

json skeletonToJson(const HandSkeleton& skeleton)
{
	json baseInPalm= json::array();
	json phalanx= json::array();
	json neutralDir= json::array();
	for (int fingerIndex= 0; fingerIndex < FINGER_COUNT; ++fingerIndex)
	{
		baseInPalm.push_back(vec3ToJson(skeleton.baseInPalm[fingerIndex]));
		phalanx.push_back(json::array({skeleton.phalanxLengths[fingerIndex][0],
									   skeleton.phalanxLengths[fingerIndex][1],
									   skeleton.phalanxLengths[fingerIndex][2]}));
		neutralDir.push_back(vec3ToJson(skeleton.neutralDirInPalm[fingerIndex]));
	}
	return {{"baseInPalm", baseInPalm}, {"phalanx", phalanx}, {"neutralDir", neutralDir}};
}

void skeletonFromJson(const json& j, HandSkeleton& outSkeleton)
{
	outSkeleton= HandSkeleton();
	if (!j.is_object())
		return;
	const json& baseInPalm= j.value("baseInPalm", json::array());
	const json& phalanx= j.value("phalanx", json::array());
	const json& neutralDir= j.value("neutralDir", json::array());
	for (int fingerIndex= 0; fingerIndex < FINGER_COUNT; ++fingerIndex)
	{
		if (fingerIndex < (int)baseInPalm.size())
			outSkeleton.baseInPalm[fingerIndex]= vec3FromJson(baseInPalm[fingerIndex]);
		if (fingerIndex < (int)phalanx.size() && phalanx[fingerIndex].is_array() &&
			phalanx[fingerIndex].size() == 3)
		{
			for (int bone= 0; bone < 3; ++bone)
				outSkeleton.phalanxLengths[fingerIndex][bone]= floatFromJson(phalanx[fingerIndex][bone]);
		}
		if (fingerIndex < (int)neutralDir.size())
			outSkeleton.neutralDirInPalm[fingerIndex]= vec3FromJson(neutralDir[fingerIndex]);
	}
}

json handInputToJson(const RecordedHandInput& hand)
{
	if (!hand.tracked)
		return {{"tracked", false}};

	json out= {
		{"tracked", true},
		{"side", hand.side},
		{"slotId", hand.slotId},
		{"presence", hand.presence},
		{"handednessScore", hand.handednessScore},
		{"rightProb", hand.rightProb},
		{"imagePoints", landmarksToJson(hand.imagePoints)},
		{"modelPoints", landmarksToJson(hand.modelPoints)},
	};
	if (hand.imageQuality.valid)
		out["imageQuality"]= imageQualityToJson(hand.imageQuality);
	return out;
}

void handInputFromJson(const json& j, RecordedHandInput& outHand)
{
	outHand= RecordedHandInput();
	outHand.tracked= j.value("tracked", false);
	if (!outHand.tracked)
		return;
	outHand.side= j.value("side", 0);
	outHand.slotId= j.value("slotId", -1);
	outHand.presence= j.value("presence", 0.f);
	outHand.handednessScore= j.value("handednessScore", 0.f);
	outHand.rightProb= j.value("rightProb", 0.5f);
	landmarksFromJson(j.value("imagePoints", json::array()), outHand.imagePoints);
	landmarksFromJson(j.value("modelPoints", json::array()), outHand.modelPoints);
	if (j.contains("imageQuality"))
	{
		const json& quality= j["imageQuality"];
		outHand.imageQuality.valid= true;
		outHand.imageQuality.meanLuma= quality.value("meanLuma", 0.f);
		outHand.imageQuality.shadowClipRatio= quality.value("shadowClipRatio", 0.f);
		outHand.imageQuality.highlightClipRatio= quality.value("highlightClipRatio", 0.f);
		outHand.imageQuality.contrast= quality.value("contrast", 0.f);
		outHand.imageQuality.backgroundSeparation= quality.value("backgroundSeparation", 0.f);
		outHand.imageQuality.sharpness= quality.value("sharpness", 0.f);
		outHand.imageQuality.noise= quality.value("noise", 0.f);
	}
}

json depthToJson(const HandDepthMeasurement& depth)
{
	uint32_t validMask= 0;
	for (int landmark= 0; landmark < HAND_LANDMARK_COUNT; ++landmark)
		if (depth.bValid[landmark])
			validMask|= 1u << landmark;
	return {{"validMask", validMask}, {"points", landmarksToJson(depth.cameraPoints)}};
}

void depthFromJson(const json& j, HandDepthMeasurement& outDepth)
{
	outDepth= HandDepthMeasurement();
	const uint32_t validMask= j.value("validMask", 0u);
	outDepth.validCount= 0;
	for (int landmark= 0; landmark < HAND_LANDMARK_COUNT; ++landmark)
	{
		outDepth.bValid[landmark]= (validMask & (1u << landmark)) != 0;
		if (outDepth.bValid[landmark])
			++outDepth.validCount;
	}
	landmarksFromJson(j.value("points", json::array()), outDepth.cameraPoints);
}

json cameraInputToJson(const RecordedCameraInput& camera)
{
	json out= {
		{"cam", camera.cameraIndex},
		{"tMs", camera.timestampMs},
		{"valid", camera.valid},
		{"refLenM", camera.refLengthMeters},
		{"frameIndex", camera.frameIndex},
		{"w", camera.frameWidth},
		{"h", camera.frameHeight},
		{"captureFps", camera.captureFps},
		{"inferenceMs", camera.inferenceMs},
		{"lumaInstability", camera.lumaInstability},
		{"lumaFlickerHz", camera.lumaFlickerHz},
		{"hands", json::array({handInputToJson(camera.hands[0]), handInputToJson(camera.hands[1])})},
	};
	if (camera.bHaveDepth)
		out["depth"]= json::array({depthToJson(camera.depth[0]), depthToJson(camera.depth[1])});
	if (camera.bHaveBodyPose)
		out["body"]= bodyPoseToJson(camera.body);
	return out;
}

bool cameraInputFromJson(const json& j, RecordedCameraInput& outCamera)
{
	outCamera= RecordedCameraInput();
	outCamera.cameraIndex= j.value("cam", -1);
	if (outCamera.cameraIndex < 0)
		return false;
	outCamera.timestampMs= j.value("tMs", 0.0);
	outCamera.valid= j.value("valid", false);
	outCamera.refLengthMeters= j.value("refLenM", 0.f);
	outCamera.frameIndex= j.value("frameIndex", (int64_t)-1);
	outCamera.frameWidth= j.value("w", 0);
	outCamera.frameHeight= j.value("h", 0);
	outCamera.captureFps= j.value("captureFps", 0.f);
	outCamera.inferenceMs= j.value("inferenceMs", 0.f);
	outCamera.lumaInstability= j.value("lumaInstability", 0.f);
	outCamera.lumaFlickerHz= j.value("lumaFlickerHz", 0.f);
	const json& hands= j.value("hands", json::array());
	for (int sideIndex= 0; sideIndex < 2 && sideIndex < (int)hands.size(); ++sideIndex)
		handInputFromJson(hands[sideIndex], outCamera.hands[sideIndex]);
	if (j.contains("depth") && j["depth"].is_array() && j["depth"].size() == 2)
	{
		outCamera.bHaveDepth= true;
		depthFromJson(j["depth"][0], outCamera.depth[0]);
		depthFromJson(j["depth"][1], outCamera.depth[1]);
	}
	if (j.contains("body"))
	{
		outCamera.bHaveBodyPose= true;
		bodyPoseFromJson(j["body"], outCamera.body);
	}
	return true;
}

json poseOutputToJson(const RecordedPoseOutput& pose)
{
	if (!pose.tracked)
		return {{"tracked", false}};

	json out= {
		{"tracked", true},
		{"side", pose.side},
		{"presence", pose.presence},
		{"visibility", pose.visibility},
		{"confidence", pose.confidence},
		{"fkReprojectionPx", pose.fkReprojectionPx},
		{"stereoTriangulated", pose.stereoTriangulated},
		{"fingers", fingersToJson(pose.fingers)},
		{"skeleton", skeletonToJson(pose.skeleton)},
	};
	if (pose.hasCameraPose)
	{
		out["palmPosC"]= vec3ToJson(pose.palmPositionCamera);
		out["palmQuatC"]= quatToJson(pose.palmOrientationCamera);
	}
	if (pose.hasWorldPose)
	{
		out["palmPosW"]= vec3ToJson(pose.palmPositionWorld);
		out["palmQuatW"]= quatToJson(pose.palmOrientationWorld);
	}
	return out;
}

void poseOutputFromJson(const json& j, RecordedPoseOutput& outPose)
{
	outPose= RecordedPoseOutput();
	outPose.tracked= j.value("tracked", false);
	if (!outPose.tracked)
		return;
	outPose.side= j.value("side", 0);
	outPose.presence= j.value("presence", 0.f);
	outPose.visibility= j.value("visibility", 0.f);
	outPose.confidence= j.value("confidence", 0.f);
	outPose.fkReprojectionPx= j.value("fkReprojectionPx", 0.f);
	outPose.stereoTriangulated= j.value("stereoTriangulated", false);
	fingersFromJson(j.value("fingers", json::array()), outPose.fingers);
	skeletonFromJson(j.value("skeleton", json::object()), outPose.skeleton);
	if (j.contains("palmPosC"))
	{
		outPose.hasCameraPose= true;
		outPose.palmPositionCamera= vec3FromJson(j["palmPosC"]);
		outPose.palmOrientationCamera= quatFromJson(j.value("palmQuatC", json::array()));
	}
	if (j.contains("palmPosW"))
	{
		outPose.hasWorldPose= true;
		outPose.palmPositionWorld= vec3FromJson(j["palmPosW"]);
		outPose.palmOrientationWorld= quatFromJson(j.value("palmQuatW", json::array()));
	}
}
} // namespace

namespace TrackingRecording
{
uint64_t computeFusedChecksum(const TrackingFrameResult& preImuFused)
{
	uint64_t hash= k_fnvBasis;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= preImuFused.poses[sideIndex];
		fnvByte(hash, pose.tracked ? 1 : 0);
		if (!pose.tracked)
			continue;

		fnvByte(hash, pose.hasWorldPose ? 1 : 0);
		const glm::vec3& position= pose.hasWorldPose ? pose.palmPositionWorld : pose.palmPositionCamera;
		const glm::quat& orientation=
			pose.hasWorldPose ? pose.palmOrientationWorld : pose.palmOrientationCamera;
		fnvFloat(hash, position.x);
		fnvFloat(hash, position.y);
		fnvFloat(hash, position.z);
		fnvFloat(hash, orientation.x);
		fnvFloat(hash, orientation.y);
		fnvFloat(hash, orientation.z);
		fnvFloat(hash, orientation.w);
		for (int fingerIndex= 0; fingerIndex < FINGER_COUNT; ++fingerIndex)
		{
			const FingerAngles& angles= pose.fingers[fingerIndex];
			fnvFloat(hash, angles.lateral);
			fnvFloat(hash, angles.proximal);
			fnvFloat(hash, angles.intermediate);
			fnvFloat(hash, angles.distal);
		}
		fnvFloat(hash, pose.confidence);
	}
	return hash;
}

void snapshotFusedOutput(const TrackingFrameResult& fused,
						 std::array<RecordedPoseOutput, 2>& outPoses)
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= fused.poses[sideIndex];
		RecordedPoseOutput& out= outPoses[sideIndex];
		out= RecordedPoseOutput();
		out.tracked= pose.tracked;
		if (!pose.tracked)
			continue;
		out.side= (int)pose.side;
		out.presence= pose.presence;
		out.visibility= pose.visibility;
		out.confidence= pose.confidence;
		out.fkReprojectionPx= pose.fkReprojectionPx;
		out.stereoTriangulated= pose.stereoTriangulated;
		out.hasCameraPose= pose.hasCameraPose;
		out.palmPositionCamera= pose.palmPositionCamera;
		out.palmOrientationCamera= pose.palmOrientationCamera;
		out.hasWorldPose= pose.hasWorldPose;
		out.palmPositionWorld= pose.palmPositionWorld;
		out.palmOrientationWorld= pose.palmOrientationWorld;
		out.fingers= pose.fingers;
		out.skeleton= pose.skeleton;
	}
}

std::string makeFrameDirectoryPath(const std::string& recordingFilePath)
{
	const size_t extension= recordingFilePath.find_last_of('.');
	const std::string stem=
		extension == std::string::npos ? recordingFilePath : recordingFilePath.substr(0, extension);
	return stem + "_frames";
}

json headerToJson(const RecordingHeader& header)
{
	json appConfig= json::parse(header.appConfigJsonText, nullptr, false);
	if (appConfig.is_discarded())
		appConfig= json::object();
	return {
		{"type", "header"},
		{"formatVersion", header.formatVersion},
		{"appConfig", appConfig},
		{"cameraCount", header.cameraCount},
	};
}

bool headerFromJson(const json& j, RecordingHeader& outHeader)
{
	outHeader= RecordingHeader();
	if (j.value("type", "") != "header")
		return false;
	outHeader.formatVersion= j.value("formatVersion", 0);
	outHeader.cameraCount= j.value("cameraCount", 0);
	if (j.contains("appConfig"))
		outHeader.appConfigJsonText= j["appConfig"].dump();
	return outHeader.formatVersion == k_formatVersion && outHeader.cameraCount > 0;
}

json frameToJson(const RecordedFrame& frame)
{
	json fresh= json::array();
	for (const RecordedCameraInput& camera : frame.freshCameras)
		fresh.push_back(cameraInputToJson(camera));

	json imu= json::array();
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (frame.imu[sideIndex].hasForearmPose)
			imu.push_back({{"has", true},
						   {"quatW", quatToJson(frame.imu[sideIndex].forearmOrientationWorld)},
						   {"conf", frame.imu[sideIndex].forearmConfidence}});
		else
			imu.push_back({{"has", false}});
	}

	return {
		{"type", "frame"},
		{"seq", frame.seq},
		{"nowMs", frame.nowTimestampMs},
		{"fused", frame.bFused},
		{"fresh", fresh},
		{"out", json::array({poseOutputToJson(frame.outPoses[0]), poseOutputToJson(frame.outPoses[1])})},
		{"imu", imu},
		{"checksum", checksumToHex(frame.checksum)},
	};
}

bool frameFromJson(const json& j, RecordedFrame& outFrame)
{
	outFrame= RecordedFrame();
	if (j.value("type", "") != "frame")
		return false;
	outFrame.seq= j.value("seq", (int64_t)-1);
	outFrame.nowTimestampMs= j.value("nowMs", 0.0);
	outFrame.bFused= j.value("fused", false);
	for (const json& cameraJson : j.value("fresh", json::array()))
	{
		RecordedCameraInput camera;
		if (cameraInputFromJson(cameraJson, camera))
			outFrame.freshCameras.push_back(camera);
	}
	const json& out= j.value("out", json::array());
	for (int sideIndex= 0; sideIndex < 2 && sideIndex < (int)out.size(); ++sideIndex)
		poseOutputFromJson(out[sideIndex], outFrame.outPoses[sideIndex]);
	const json& imu= j.value("imu", json::array());
	for (int sideIndex= 0; sideIndex < 2 && sideIndex < (int)imu.size(); ++sideIndex)
	{
		outFrame.imu[sideIndex].hasForearmPose= imu[sideIndex].value("has", false);
		if (outFrame.imu[sideIndex].hasForearmPose)
		{
			outFrame.imu[sideIndex].forearmOrientationWorld=
				quatFromJson(imu[sideIndex].value("quatW", json::array()));
			outFrame.imu[sideIndex].forearmConfidence= imu[sideIndex].value("conf", 0.f);
		}
	}
	outFrame.checksum= checksumFromHex(j.value("checksum", "0"));
	return true;
}

json footerToJson(const RecordingFooter& footer)
{
	return {
		{"type", "end"},
		{"frameCount", footer.frameCount},
		{"aborted", footer.bAborted},
		{"reason", footer.reason},
	};
}

bool footerFromJson(const json& j, RecordingFooter& outFooter)
{
	outFooter= RecordingFooter();
	if (j.value("type", "") != "end")
		return false;
	outFooter.frameCount= j.value("frameCount", (int64_t)0);
	outFooter.bAborted= j.value("aborted", false);
	outFooter.reason= j.value("reason", "");
	return true;
}
} // namespace TrackingRecording
