#include "TestCommon.h"

static int runReplayVerifyTool(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("replay-verify") << "usage: --replay-verify <recording.jsonl>";
		return 1;
	}

	TrackingReplay replay;
	std::string error;
	if (!replay.load(args[0], error))
	{
		MIKAN_LOG_ERROR("replay-verify") << "Load failed: " << error;
		return 1;
	}

	replay.runAll();

	MIKAN_LOG_INFO("replay-verify")
		<< replay.getFilePath() << ": " << replay.getFrameCount() << " frames, "
		<< replay.getRecordedConfig().cameraCount() << " cameras"
		<< (replay.hasFooter()
			? (replay.getFooter().bAborted ? " (recording ABORTED: " + replay.getFooter().reason + ")"
										   : "")
			: " (no footer - crash-truncated)");

	const std::vector<int>& divergent= replay.getDivergentFrames();
	for (int frameIndex : divergent)
	{
		char recordedHex[17], replayedHex[17];
		snprintf(recordedHex, sizeof(recordedHex), "%016llx",
				 (unsigned long long)replay.getFrame(frameIndex).recorded->checksum);
		snprintf(replayedHex, sizeof(replayedHex), "%016llx",
				 (unsigned long long)replay.getFrame(frameIndex).replayChecksum);
		MIKAN_LOG_ERROR("replay-verify")
			<< "DIVERGENT frame " << frameIndex << ": recorded " << recordedHex
			<< " replayed " << replayedHex;
	}

	MIKAN_LOG_INFO("replay-verify")
		<< (replay.getFrameCount() - (int)divergent.size()) << "/" << replay.getFrameCount()
		<< " frames verified, " << divergent.size()
		<< " divergent (checksums compare against the RECORDED values - same-binary "
		   "guarantee only)";

	return divergent.empty() ? 0 : 1;
}

MIKAN_REGISTER_TEST("--replay-verify", "Re-runs a recording and verifies every frame checksum", eTestCategory::Tool, runReplayVerifyTool);
