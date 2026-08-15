#include "TestCommon.h"

static int runReplayDumpTool(const TestArgs& args)
{
	if (args.size() < 3)
	{
		MIKAN_LOG_ERROR("replay-dump")
			<< "usage: --replay-dump <recording.jsonl> <firstFrame> <lastFrame> [outPath]";
		return 1;
	}
	const std::string recordingPath= args[0];
	const int firstFrame= atoi(args[1].c_str());
	const int lastFrame= atoi(args[2].c_str());
	const std::string outPath= args.size() > 3
		? args[3]
		: recordingPath + ".frames_" + std::to_string(firstFrame) + "_" +
			std::to_string(lastFrame) + ".json";

	TrackingReplay replay;
	std::string error;
	if (!replay.load(recordingPath, error))
	{
		MIKAN_LOG_ERROR("replay-dump") << "Load failed: " << error;
		return 1;
	}

	// Replay always runs from frame 0 (determinism), capturing the
	// regenerated fusion diagnostics only inside the requested range
	replay.setCaptureDiagnostics(true, firstFrame, lastFrame);
	replay.runAll();

	nlohmann::json framesJson= nlohmann::json::array();
	const int clampedLast= std::min(lastFrame, replay.getFrameCount() - 1);
	for (int frameIndex= std::max(0, firstFrame); frameIndex <= clampedLast; ++frameIndex)
	{
		const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIndex);
		const RecordedFrame& recorded= *frame.recorded;

		char recordedHex[17], replayedHex[17];
		snprintf(recordedHex, sizeof(recordedHex), "%016llx",
				 (unsigned long long)recorded.checksum);
		snprintf(replayedHex, sizeof(replayedHex), "%016llx",
				 (unsigned long long)frame.replayChecksum);

		nlohmann::json frameJson= {
			{"frame", frameIndex},
			{"recordedInputs", TrackingRecording::frameToJson(recorded)},
			{"checksumMatch", frame.bChecksumMatch},
			{"recordedChecksum", recordedHex},
			{"replayChecksum", replayedHex},
		};

		// Replayed fused poses (same schema the F9 dump uses for poses)
		std::array<RecordedPoseOutput, 2> replayedPoses;
		TrackingRecording::snapshotFusedOutput(frame.replayedFused, replayedPoses);
		RecordedFrame replayedAsRecord;
		replayedAsRecord.outPoses= replayedPoses;
		frameJson["replayedOut"]= TrackingRecording::frameToJson(replayedAsRecord)["out"];

		// Body-pose solver output (elbow/shoulder/head). It runs after the
		// fused snapshot, like the IMU overlay, so it is outside the checksum
		// and this is the only place a replay exposes what it produced.
		{
			const float forearmLength= replay.getRecordedConfig().body.forearmLengthMeters;
			nlohmann::json bodyJson= nlohmann::json::array();
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const HandPose& pose= frame.replayedFused.poses[sideIndex];
				nlohmann::json sideJson= {
					{"side", TrackingJson::sideName(sideIndex)},
					{"hasForearmPose", pose.hasForearmPose},
					{"hasShoulder", pose.hasShoulder},
				};
				if (pose.hasForearmPose)
				{
					sideJson["elbowPositionWorld"]=
						TrackingJson::vec3ToJson(pose.getElbowPositionWorld(forearmLength));
					sideJson["forearmConfidence"]= pose.forearmConfidence;
				}
				if (pose.hasShoulder)
				{
					sideJson["shoulderPositionWorld"]= TrackingJson::vec3ToJson(pose.shoulderPositionWorld);
					sideJson["shoulderConfidence"]= pose.shoulderConfidence;
				}
				bodyJson.push_back(sideJson);
			}
			frameJson["replayedBody"]= bodyJson;

			const TrackingFrameResult::HeadPose& head= frame.replayedFused.head;
			frameJson["replayedHead"]= {
				{"valid", head.valid},
				{"positionWorld", TrackingJson::vec3ToJson(head.positionWorld)},
				{"orientationWorld", TrackingJson::quatToJson(head.orientationWorld)},
				{"confidence", head.confidence},
			};
		}

		if (frame.bHasDiagnostics)
			frameJson["fusionDiagnostics"]=
				TrackingJson::fusionDiagnosticsToJson(frame.diagnostics);

		framesJson.push_back(frameJson);
	}

	nlohmann::json out= {
		{"recording", recordingPath},
		{"frameCount", replay.getFrameCount()},
		{"divergentFrames", replay.getDivergentFrames()},
		{"frames", framesJson},
	};

	std::ofstream outFile(outPath, std::ios::trunc);
	if (!outFile.is_open())
	{
		MIKAN_LOG_ERROR("replay-dump") << "Cannot write " << outPath;
		return 1;
	}
	outFile << out.dump(1, '\t');
	outFile.close();

	MIKAN_LOG_INFO("replay-dump")
		<< "Wrote frames " << firstFrame << ".." << clampedLast << " ("
		<< framesJson.size() << " frames, "
		<< replay.getDivergentFrames().size() << " divergent overall) to " << outPath;

	return 0;
}

MIKAN_REGISTER_TEST("--replay-dump", "Emits a recording frame range with regenerated fusion diagnostics", eTestCategory::Tool, runReplayDumpTool);
