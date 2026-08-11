#include "TestCommon.h"

// A/B test of a candidate extrinsics calibration against a recorded session:
// replays the recording once with its own (baseline) extrinsics and once with
// the extrinsics from the given config.json, then compares what the fused
// tracking made of the SAME recorded hands. This measures what actually
// matters downstream - triangulation success and fused confidence - instead
// of a proxy on the calibration board.
static int runReplayExtrinsicsTool(const TestArgs& args)
{
	if (args.size() < 2)
	{
		MIKAN_LOG_ERROR("replay-extrinsics")
			<< "usage: --replay-extrinsics <recording.jsonl> <config.json>";
		return 1;
	}

	TrackingReplay replay;
	std::string error;
	if (!replay.load(args[0], error))
	{
		MIKAN_LOG_ERROR("replay-extrinsics") << "Load failed: " << error;
		return 1;
	}
	replay.runAll();

	// Candidate extrinsics from an ordinary saved config.json
	AppConfig candidateConfig;
	{
		std::ifstream configFile(args[1]);
		if (!configFile.is_open())
		{
			MIKAN_LOG_ERROR("replay-extrinsics") << "Cannot open " << args[1];
			return 1;
		}
		std::string configText((std::istreambuf_iterator<char>(configFile)),
							   std::istreambuf_iterator<char>());
		if (!candidateConfig.loadFromJsonString(configText))
		{
			MIKAN_LOG_ERROR("replay-extrinsics") << "Failed to parse " << args[1];
			return 1;
		}
	}

	TrackingReplay::WhatIfParams params= replay.makeDefaultWhatIfParams();
	params.bOverrideExtrinsics= true;
	for (size_t cameraIndex= 0; cameraIndex < candidateConfig.cameraCount(); ++cameraIndex)
		params.extrinsicsOverride.push_back(candidateConfig.camera(cameraIndex).extrinsics);
	replay.runWhatIf(params);

	// Per side: how often each pass triangulated, how confident it was, and
	// how far apart the two passes' palms landed
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		int trackedBoth= 0;
		int stereoBaseline= 0, stereoCandidate= 0;
		double confidenceBaseline= 0.0, confidenceCandidate= 0.0;
		std::vector<double> palmDeltasMm;

		for (int frameIndex= 0; frameIndex < replay.getFrameCount(); ++frameIndex)
		{
			const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIndex);
			if (!frame.bHasWhatIf)
				continue;
			const HandPose& baseline= frame.replayedFused.poses[sideIndex];
			const HandPose& candidate= frame.whatIfFused.poses[sideIndex];
			if (!baseline.tracked || !candidate.tracked || !baseline.hasWorldPose ||
				!candidate.hasWorldPose)
				continue;

			++trackedBoth;
			stereoBaseline+= baseline.stereoTriangulated ? 1 : 0;
			stereoCandidate+= candidate.stereoTriangulated ? 1 : 0;
			confidenceBaseline+= baseline.confidence;
			confidenceCandidate+= candidate.confidence;
			palmDeltasMm.push_back(
				glm::length(candidate.palmPositionWorld - baseline.palmPositionWorld) * 1000.0);
		}

		if (trackedBoth == 0)
		{
			MIKAN_LOG_INFO("replay-extrinsics")
				<< (sideIndex == 0 ? "LEFT" : "RIGHT") << ": no frames tracked by both passes";
			continue;
		}

		std::sort(palmDeltasMm.begin(), palmDeltasMm.end());
		MIKAN_LOG_INFO("replay-extrinsics")
			<< (sideIndex == 0 ? "LEFT" : "RIGHT") << ": " << trackedBoth << " frames | stereo "
			<< (100 * stereoBaseline / trackedBoth) << "% -> " << (100 * stereoCandidate / trackedBoth)
			<< "% | mean confidence " << (confidenceBaseline / trackedBoth) << " -> "
			<< (confidenceCandidate / trackedBoth) << " | palm delta med "
			<< palmDeltasMm[palmDeltasMm.size() / 2] << " mm, p90 "
			<< palmDeltasMm[(size_t)(0.9 * palmDeltasMm.size())] << " mm, max "
			<< palmDeltasMm.back() << " mm";
	}

	MIKAN_LOG_INFO("replay-extrinsics")
		<< "Higher stereo % and confidence with a similar-or-smaller palm delta "
		   "means the candidate calibration is better; the deltas show how far "
		   "the world itself moved between the two.";
	return 0;
}

MIKAN_REGISTER_TEST("--replay-extrinsics",
					"A/B a candidate extrinsics config against a recording",
					eTestCategory::Tool, runReplayExtrinsicsTool);
