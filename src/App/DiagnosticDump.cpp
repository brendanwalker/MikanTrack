#include "DiagnosticDump.h"

#include <filesystem>
#include <fstream>

#include "nlohmann/json.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include "Logger.h"

using json= nlohmann::json;

// -- json helpers ----
static json vec2ToJson(const glm::vec2& v)
{
	return json::array({v.x, v.y});
}

static json vec3ToJson(const glm::vec3& v)
{
	return json::array({v.x, v.y, v.z});
}

static json quatToJson(const glm::quat& q)
{
	return json::array({q.x, q.y, q.z, q.w}); // xyzw, matching the OSC schema
}

static const char* sideName(int side)
{
	return side == 0 ? "left" : (side == 1 ? "right" : "none");
}

static json fingersToJson(const std::array<FingerAngles, FINGER_COUNT>& fingers)
{
	json out= json::array();
	for (const FingerAngles& angles : fingers)
		out.push_back(json::array({angles.lateral, angles.proximal, angles.intermediate, angles.distal}));
	return out;
}

static json landmarksToJson(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points)
{
	json out= json::array();
	for (const glm::vec3& p : points)
		out.push_back(vec3ToJson(p));
	return out;
}

static json diagImuToJson(const DiagImuState& imu)
{
	if (!imu.deviceConnected)
		return {{"deviceConnected", false}};

	return {
		{"deviceConnected", true},
		{"streaming", imu.streaming},
		{"calibrated", imu.calibrated},
		{"orientationValid", imu.orientationValid},
		{"sampleRateHz", imu.sampleRateHz},
		{"msSinceLastSample", imu.millisecondsSinceLastSample},
		{"forearmAxisConsistency", imu.forearmAxisConsistency},
		{"armAxisDominance", imu.armAxisDominance},
		{"twistProgress", imu.twistProgress},
		{"twistReversal", imu.twistReversal},
		{"wristAxialTwistDegrees", imu.wristAxialTwistDegrees},
		{"biasSaturated", imu.biasSaturated},
		{"gyroBiasDegPerSec", json::array({imu.gyroBiasDegreesPerSecond.x,
										   imu.gyroBiasDegreesPerSecond.y,
										   imu.gyroBiasDegreesPerSecond.z})},
		{"yawSigmaRadians", imu.yawSigmaRadians},
		{"filterOrientation", quatToJson(imu.filterOrientation)},
		{"tiltSigmaRadians", imu.tiltSigmaRadians},
		{"gravityAcceptRatio", imu.gravityAcceptRatio},
		{"visionYawCorrectionDegrees", imu.visionYawCorrectionDegrees},
	};
}

static json imageQualityToJson(const HandImageQuality& quality)
{
	return {
		{"meanLuma", quality.meanLuma},
		{"shadowClipRatio", quality.shadowClipRatio},
		{"highlightClipRatio", quality.highlightClipRatio},
		{"contrast", quality.contrast},
		{"backgroundSeparation", quality.backgroundSeparation},
		{"sharpness", quality.sharpness},
		{"noise", quality.noise},
	};
}

static json diagHandToJson(const DiagHandState& hand)
{
	if (!hand.tracked)
		return {{"tracked", false}};

	json out= {
		{"tracked", true},
		{"slotId", hand.slotId},
		{"presence", hand.presence},
		{"handednessScore", hand.handednessScore},
		{"rightProb", hand.rightProb},
		{"labeledSide", sideName(hand.labeledSide)},
		{"visibility", hand.visibility},
		{"confidence", hand.confidence},
		{"fkReprojectionPx", hand.fkReprojectionPx},
		{"stereoTriangulated", hand.stereoTriangulated},
		{"hasWorldPose", hand.hasWorldPose},
		{"palmPositionWorld", vec3ToJson(hand.palmPositionWorld)},
		{"palmOrientationWorld", quatToJson(hand.palmOrientationWorld)},
		{"hasForearmPose", hand.hasForearmPose},
		{"forearmOrientationWorld", quatToJson(hand.forearmOrientationWorld)},
		{"wristPx", vec2ToJson(hand.wristPx)},
		{"fingers", fingersToJson(hand.fingers)},
	};
	if (hand.imageQuality.valid)
		out["imageQuality"]= imageQualityToJson(hand.imageQuality);
	return out;
}

static json fusionDiagnosticsToJson(const FusionDiagnostics& diagnostics)
{
	json clusters= json::array();
	for (const FusionDiagnostics::Cluster& cluster : diagnostics.clusters)
	{
		json observations= json::array();
		for (const FusionDiagnostics::Observation& observation : cluster.observations)
		{
			observations.push_back({
				{"camera", observation.cameraIndex},
				{"labeledSide", sideName(observation.labeledSide)},
				{"weight", observation.weight},
				{"confidence", observation.confidence},
				{"stability", observation.stability},
				{"jitterMm", observation.jitterMm},
				{"sideVoteWeight", observation.sideVoteWeight},
				{"palmWorld", vec3ToJson(observation.palmWorld)},
			});
		}

		json affinity;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			affinity[sideName(sideIndex)]= {
				{"vote", cluster.affinity[sideIndex][0]},
				{"temporal", cluster.affinity[sideIndex][1]},
				{"spatial", cluster.affinity[sideIndex][2]},
				{"total",
				 cluster.affinity[sideIndex][0] + cluster.affinity[sideIndex][1] + cluster.affinity[sideIndex][2]},
			};
		}

		clusters.push_back({
			{"palmWorld", vec3ToJson(cluster.palmWorld)},
			{"bestWeight", cluster.bestWeight},
			{"assignedSide", sideName(cluster.assignedSide)},
			{"triangulated", cluster.triangulated},
			{"triVetoed", cluster.triVetoed},
			{"triResidualRmsPx", cluster.triResidualRmsPx},
			{"triResidualMaxPx", cluster.triResidualMaxPx},
			{"affinity", affinity},
			{"observations", observations},
		});
	}

	return {
		{"totalObservations", diagnostics.totalObservations},
		{"clusters", clusters},
	};
}

static void fillDiagHandFromResult(const TrackingFrameResult& result, int sideIndex, DiagHandState& outHand)
{
	const HandPose& pose= result.poses[sideIndex];
	const TrackedHand& hand= result.hands[sideIndex];

	outHand.tracked= pose.tracked || hand.tracked;
	if (!outHand.tracked)
		return;

	outHand.slotId= hand.slotId;
	outHand.presence= std::max(pose.presence, hand.presence);
	outHand.handednessScore= hand.handednessScore;
	outHand.rightProb= hand.rightProb;
	outHand.labeledSide= (int)(hand.tracked ? hand.side : pose.side);
	outHand.visibility= pose.visibility;
	outHand.confidence= pose.confidence;
	outHand.fkReprojectionPx= pose.fkReprojectionPx;
	outHand.stereoTriangulated= pose.stereoTriangulated;
	outHand.hasWorldPose= pose.hasWorldPose;
	outHand.palmPositionWorld= pose.hasWorldPose ? pose.palmPositionWorld : pose.palmPositionCamera;
	outHand.palmOrientationWorld= pose.hasWorldPose ? pose.palmOrientationWorld : pose.palmOrientationCamera;
	outHand.hasForearmPose= pose.hasForearmPose;
	outHand.forearmOrientationWorld= pose.forearmOrientationWorld;
	outHand.wristPx= hand.tracked ? glm::vec2(hand.imagePoints[0]) : glm::vec2(0.f);
	outHand.fingers= pose.fingers;
	outHand.imageQuality= hand.imageQuality;
}

void DiagnosticDump::record(const std::vector<const CameraFrameResult*>& cameraResults,
							const TrackingFrameResult& fused,
							const FusionDiagnostics& fusionDiagnostics,
							const int dominantCamera[2],
							float autoScaleFactor,
							const DiagImuState imu[2])
{
	DiagFrameRecord record;
	record.fuseTimestampMs= fused.timestampMs;
	record.autoScaleFactor= autoScaleFactor;
	record.fusion= fusionDiagnostics;
	record.imu[0]= imu[0];
	record.imu[1]= imu[1];

	for (const CameraFrameResult* cameraResult : cameraResults)
	{
		DiagCameraState cameraState;
		if (cameraResult != nullptr)
		{
			cameraState.valid= cameraResult->valid;
			cameraState.timestampMs= cameraResult->timestampMs;
			cameraState.captureFps= cameraResult->result.captureFps;
			cameraState.inferenceMs= cameraResult->result.inferenceMs;
			cameraState.lumaInstability= cameraResult->result.lumaInstability;
			cameraState.lumaFlickerHz= cameraResult->result.lumaFlickerHz;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				fillDiagHandFromResult(cameraResult->result, sideIndex, cameraState.sides[sideIndex]);
		}
		record.cameras.push_back(cameraState);
	}

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		fillDiagHandFromResult(fused, sideIndex, record.fused[sideIndex]);
		record.fused[sideIndex].labeledSide= sideIndex; // fused slots ARE the side
		record.dominantCamera[sideIndex]= dominantCamera[sideIndex];
	}

	m_history.push_back(std::move(record));
	while (m_history.size() > kMaxHistory)
		m_history.pop_front();
}

// Draws landmarks/bones, ROI boxes and per-hand labels onto a copy of the
// camera frame (colors match nothing else in the app - chosen for print
// legibility: left amber, right cyan, detections green)
static void annotateFrame(const cv::Mat& frame, const TrackingFrameResult& result, cv::Mat& outAnnotated)
{
	frame.copyTo(outAnnotated);

	for (const DetectionBox& box : result.palmDetections)
	{
		for (int corner= 0; corner < 4; ++corner)
		{
			const glm::vec2& a= box.corners[corner];
			const glm::vec2& b= box.corners[(corner + 1) % 4];
			cv::line(outAnnotated, cv::Point2f(a.x, a.y), cv::Point2f(b.x, b.y), cv::Scalar(0, 200, 0), 1);
		}
	}

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedHand& hand= result.hands[sideIndex];
		if (!hand.tracked)
			continue;

		const cv::Scalar color=
			sideIndex == 0 ? cv::Scalar(0, 180, 255) /*amber*/ : cv::Scalar(255, 220, 0) /*cyan*/;

		for (int connection= 0; connection < HAND_CONNECTION_COUNT; ++connection)
		{
			const glm::vec3& a= hand.imagePoints[HAND_CONNECTIONS[connection][0]];
			const glm::vec3& b= hand.imagePoints[HAND_CONNECTIONS[connection][1]];
			cv::line(outAnnotated, cv::Point2f(a.x, a.y), cv::Point2f(b.x, b.y), color, 2);
		}
		for (int landmark= 0; landmark < HAND_LANDMARK_COUNT; ++landmark)
		{
			const glm::vec3& p= hand.imagePoints[landmark];
			cv::circle(outAnnotated, cv::Point2f(p.x, p.y), 3, color, cv::FILLED);
		}

		char label[64];
		snprintf(label, sizeof(label), "%s p=%.2f h=%.2f", sideIndex == 0 ? "L" : "R", hand.presence,
				 hand.handednessScore);
		const glm::vec3& wrist= hand.imagePoints[0];
		cv::putText(outAnnotated, label, cv::Point2f(wrist.x + 8.f, wrist.y + 18.f), cv::FONT_HERSHEY_SIMPLEX,
					0.6, cv::Scalar(0, 0, 0), 3);
		cv::putText(outAnnotated, label, cv::Point2f(wrist.x + 8.f, wrist.y + 18.f), cv::FONT_HERSHEY_SIMPLEX,
					0.6, color, 1);
	}
}

static json handSnapshotToJson(const TrackedHand& hand)
{
	if (!hand.tracked)
		return {{"tracked", false}};

	json out= {
		{"tracked", true},
		{"side", sideName((int)hand.side)},
		{"slotId", hand.slotId},
		{"presence", hand.presence},
		{"handednessScore", hand.handednessScore},
		{"imagePoints", landmarksToJson(hand.imagePoints)},
	};
	if (hand.imageQuality.valid)
		out["imageQuality"]= imageQualityToJson(hand.imageQuality);
	out["modelPoints"]= landmarksToJson(hand.modelPoints);
	if (hand.hasCameraSpace)
		out["cameraPoints"]= landmarksToJson(hand.cameraPoints);
	if (hand.hasWorldSpace)
		out["worldPoints"]= landmarksToJson(hand.worldPoints);
	return out;
}

static json poseSnapshotToJson(const HandPose& pose)
{
	if (!pose.tracked)
		return {{"tracked", false}};

	json out= {
		{"tracked", true},
		{"side", sideName((int)pose.side)},
		{"presence", pose.presence},
		{"visibility", pose.visibility},
		{"confidence", pose.confidence},
		{"fingers", fingersToJson(pose.fingers)},
		{"fkReprojectionPx", pose.fkReprojectionPx},
		{"stereoTriangulated", pose.stereoTriangulated},
	};
	if (pose.hasCameraPose)
	{
		out["palmPositionCamera"]= vec3ToJson(pose.palmPositionCamera);
		out["palmOrientationCamera"]= quatToJson(pose.palmOrientationCamera);
	}
	if (pose.hasWorldPose)
	{
		out["palmPositionWorld"]= vec3ToJson(pose.palmPositionWorld);
		out["palmOrientationWorld"]= quatToJson(pose.palmOrientationWorld);
	}

	json neutralDirs= json::array();
	for (const glm::vec3& dir : pose.skeleton.neutralDirInPalm)
		neutralDirs.push_back(vec3ToJson(dir));
	out["neutralDirInPalm"]= neutralDirs;

	return out;
}

static json resultSnapshotToJson(const TrackingFrameResult& result)
{
	return {
		{"frameIndex", result.frameIndex},
		{"timestampMs", result.timestampMs},
		{"frameSize", json::array({result.frameWidth, result.frameHeight})},
		{"captureFps", result.captureFps},
		{"inferenceMs", result.inferenceMs},
		{"lumaInstability", result.lumaInstability},
		{"lumaFlickerHz", result.lumaFlickerHz},
		{"hands", json::array({handSnapshotToJson(result.hands[0]), handSnapshotToJson(result.hands[1])})},
		{"poses", json::array({poseSnapshotToJson(result.poses[0]), poseSnapshotToJson(result.poses[1])})},
	};
}

bool DiagnosticDump::write(const std::string& dumpDir,
						   const std::vector<DiagCameraSnapshot>& cameras,
						   const TrackingFrameResult& latestFused,
						   const std::string& configJsonString,
						   const std::vector<DiagImuRawSample> rawImu[2],
						   const DiagImuCapture lastCapture[2]) const
{
	std::error_code ec;
	std::filesystem::create_directories(dumpDir, ec);
	if (ec)
	{
		MIKAN_MT_LOG_ERROR("DiagnosticDump") << "Failed to create " << dumpDir << ": " << ec.message();
		return false;
	}

	bool bSuccess= true;
	json j;

	// The live config (may differ from the saved config.json)
	try
	{
		j["config"]= json::parse(configJsonString);
	}
	catch (const std::exception&)
	{
		j["config"]= configJsonString; // shouldn't happen; keep the raw text
	}

	// Per-camera: current frame images + full landmark snapshot
	json camerasJson= json::array();
	for (size_t cameraIndex= 0; cameraIndex < cameras.size(); ++cameraIndex)
	{
		const DiagCameraSnapshot& camera= cameras[cameraIndex];

		json cameraJson= {
			{"index", (int)cameraIndex},
			{"trackingEnabled", camera.trackingEnabled},
			{"activeEp", camera.activeEp},
			{"deviceFps", camera.deviceFps},
			{"droppedFrames", camera.droppedFrames},
		};

		if (camera.lastResult != nullptr)
		{
			cameraJson["hasExtrinsics"]= camera.lastResult->hasExtrinsics;
			cameraJson["snapshot"]= resultSnapshotToJson(camera.lastResult->result);
		}

		if (camera.frame != nullptr && !camera.frame->empty())
		{
			char rawName[64], annotatedName[64];
			snprintf(rawName, sizeof(rawName), "cam%zu_raw.png", cameraIndex);
			snprintf(annotatedName, sizeof(annotatedName), "cam%zu_annotated.png", cameraIndex);

			const std::filesystem::path dirPath(dumpDir);
			if (!cv::imwrite((dirPath / rawName).string(), *camera.frame))
				bSuccess= false;

			if (camera.lastResult != nullptr)
			{
				cv::Mat annotated;
				annotateFrame(*camera.frame, camera.lastResult->result, annotated);
				if (!cv::imwrite((dirPath / annotatedName).string(), annotated))
					bSuccess= false;
				cameraJson["images"]= {{"raw", rawName}, {"annotated", annotatedName}};
			}
			else
			{
				cameraJson["images"]= {{"raw", rawName}};
			}
		}

		camerasJson.push_back(cameraJson);
	}
	j["cameras"]= camerasJson;

	j["fusedSnapshot"]= resultSnapshotToJson(latestFused);

	// Rolling history, oldest first
	json historyJson= json::array();
	for (const DiagFrameRecord& record : m_history)
	{
		json camerasHistory= json::array();
		for (const DiagCameraState& cameraState : record.cameras)
		{
			camerasHistory.push_back({
				{"valid", cameraState.valid},
				{"timestampMs", cameraState.timestampMs},
				{"captureFps", cameraState.captureFps},
				{"inferenceMs", cameraState.inferenceMs},
				{"lumaInstability", cameraState.lumaInstability},
				{"lumaFlickerHz", cameraState.lumaFlickerHz},
				{"left", diagHandToJson(cameraState.sides[0])},
				{"right", diagHandToJson(cameraState.sides[1])},
			});
		}

		historyJson.push_back({
			{"t", record.fuseTimestampMs},
			{"cameras", camerasHistory},
			{"fused",
			 {
				 {"left", diagHandToJson(record.fused[0])},
				 {"right", diagHandToJson(record.fused[1])},
				 {"dominantCamera", json::array({record.dominantCamera[0], record.dominantCamera[1]})},
			 }},
			{"autoScaleFactor", record.autoScaleFactor},
			{"fusion", fusionDiagnosticsToJson(record.fusion)},
			{"imu",
			 {
				 {"left", diagImuToJson(record.imu[0])},
				 {"right", diagImuToJson(record.imu[1])},
			 }},
		});
	}
	j["history"]= historyJson;
	// Raw IMU samples, oldest first. Deliberately uncorrected: the point is to
	// replay candidate axis mappings against them and score each one on how
	// well it tracks the palm quaternions in the history above.
	json rawImuJson= json::object();
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		json samples= json::array();
		for (const DiagImuRawSample& sample : rawImu[sideIndex])
		{
			samples.push_back({
				{"t", sample.timestampMs},
				{"accel", json::array({sample.acceleration.x, sample.acceleration.y, sample.acceleration.z})},
				{"gyro", json::array({sample.angularVelocity.x, sample.angularVelocity.y,
									  sample.angularVelocity.z})},
			});
		}
		rawImuJson[sideIndex == 0 ? "left" : "right"]= samples;
	}
	j["imuRaw"]= rawImuJson;

	// The calibration currently in force, with the evidence it was built from
	json captureJson= json::object();
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const DiagImuCapture& capture= lastCapture[sideIndex];
		captureJson[sideIndex == 0 ? "left" : "right"]= capture.present
			? json({{"present", true},
					{"forearmToSensor", quatToJson(capture.forearmToSensor)},
					{"motionUsable", capture.motionUsable},
					{"poseSamples", capture.poseSamples},
					{"poseSpreadDegrees", capture.poseSpreadDegrees},
					{"axisDominance", capture.axisDominance},
					{"twistProgress", capture.twistProgress},
					{"twistReversal", capture.twistReversal},
					{"curlDominance", capture.curlDominance},
					{"curlProgress", capture.curlProgress},
					{"curlReversal", capture.curlReversal},
					{"curlStrokes", capture.curlStrokes},
					{"hingeSpreadDegrees", capture.hingeSpreadDegrees},
					{"interAxisAngleDegrees", capture.interAxisAngleDegrees},
					{"lengthMeasured", capture.lengthMeasured},
					{"forearmLengthMeters", capture.forearmLengthMeters},
					{"lengthFitCorrelation", capture.lengthFitCorrelation},
					{"palmarSource", capture.palmarSource}})
			: json({{"present", false}});
	}
	j["imuLastCapture"]= captureJson;


	const std::filesystem::path jsonPath= std::filesystem::path(dumpDir) / "dump.json";
	std::ofstream file(jsonPath);
	if (!file.is_open())
	{
		MIKAN_MT_LOG_ERROR("DiagnosticDump") << "Failed to open " << jsonPath.string() << " for writing";
		return false;
	}
	file << j.dump(1, '\t');

	return bSuccess;
}
