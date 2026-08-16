#include "TestCommon.h"

#include "AnglePriorCalibrator.h"

// Fits the per-side angle prior (mean + precision over the 20 raw finger
// angles) from recorded sessions: replays each recording, collects RAW angles
// from stereo-triangulated, confident fused frames, and fits via
// AnglePriorCalibrator. The estimator uses the result as a weak Mahalanobis
// pull that tells coordinated pose changes from single-DoF landmark snaps.
//
//   --fit-angle-prior <recording.jsonl>... [output-config.json]
//
// Recordings are the .jsonl arguments; the optional .json argument is written
// as the FIRST recording's config plus the fitted prior (same contract as
// --calibrate-bones: nothing mutates the live config silently).

namespace
{
// Streamed angles carry the fused rest offset; the prior lives in RAW space
std::array<FingerAngles, FINGER_COUNT> rawAnglesOf(const HandPose& pose, const AppConfig& config)
{
	std::array<FingerAngles, FINGER_COUNT> rawAngles= pose.fingers;
	const int sideIndex= (int)pose.side;
	if (config.fusedRestAngles.present[sideIndex])
	{
		const std::array<FingerAngles, FINGER_COUNT>& rest= config.fusedRestAngles.angles[sideIndex];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			rawAngles[finger].lateral+= rest[finger].lateral;
			rawAngles[finger].proximal+= rest[finger].proximal;
			rawAngles[finger].intermediate+= rest[finger].intermediate;
			rawAngles[finger].distal+= rest[finger].distal;
		}
	}
	return rawAngles;
}
} // namespace

static int runFitAnglePriorTool(const TestArgs& args)
{
	std::vector<std::string> recordings;
	std::string outputPath;
	for (const std::string& arg : args)
	{
		if (arg.size() > 6 && arg.substr(arg.size() - 6) == ".jsonl")
			recordings.push_back(arg);
		else
			outputPath= arg;
	}
	if (recordings.empty())
	{
		MIKAN_LOG_ERROR("fit-angle-prior")
			<< "usage: --fit-angle-prior <recording.jsonl>... [output-config.json]";
		return 1;
	}

	// Only confident stereo frames teach the prior: monocular articulation
	// carries per-camera bias, and low-confidence frames are the noise the
	// prior exists to reject
	constexpr float kMinConfidence= 0.4f;

	AnglePriorCalibrator calibrator;
	AppConfig baseConfig;
	bool bHaveBaseConfig= false;

	for (const std::string& path : recordings)
	{
		TrackingReplay replay;
		std::string error;
		if (!replay.load(path, error))
		{
			MIKAN_LOG_ERROR("fit-angle-prior") << path << ": load failed: " << error;
			return 1;
		}
		replay.runAll();

		if (!bHaveBaseConfig)
		{
			baseConfig= replay.getRecordedConfig();
			bHaveBaseConfig= true;
		}

		const int before[2]= {calibrator.getSampleCount(eHandSide::Left),
							  calibrator.getSampleCount(eHandSide::Right)};
		for (int frameIndex= 0; frameIndex < replay.getFrameCount(); ++frameIndex)
		{
			const TrackingFrameResult& fused= replay.getFrame(frameIndex).replayedFused;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const HandPose& pose= fused.poses[sideIndex];
				if (!pose.tracked || !pose.hasWorldPose || !pose.stereoTriangulated ||
					pose.confidence < kMinConfidence)
					continue;
				calibrator.addSample((eHandSide)sideIndex,
									 rawAnglesOf(pose, replay.getRecordedConfig()));
			}
		}
		MIKAN_LOG_INFO("fit-angle-prior")
			<< std::filesystem::path(path).filename().string() << ": +"
			<< calibrator.getSampleCount(eHandSide::Left) - before[0] << " left / +"
			<< calibrator.getSampleCount(eHandSide::Right) - before[1] << " right samples";
	}

	bool bAnySide= false;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const eHandSide side= (eHandSide)sideIndex;
		const char* sideName= sideIndex == 0 ? "LEFT" : "RIGHT";

		AnglePriorCalibrator::Prior prior;
		if (!calibrator.finish(side, prior))
		{
			MIKAN_LOG_WARNING("fit-angle-prior")
				<< sideName << ": only " << calibrator.getSampleCount(side) << " samples, need "
				<< AnglePriorCalibrator::k_minSamples;
			continue;
		}

		baseConfig.anglePrior.present[sideIndex]= true;
		baseConfig.anglePrior.mean[sideIndex]= prior.mean;
		baseConfig.anglePrior.precision[sideIndex]= prior.precision;
		bAnySide= true;

		// Sanity: mean squared Mahalanobis distance of the training samples
		// should land near the dimension count (chi-squared expectation);
		// far off means the fit is broken, not just imperfect
		constexpr int kAngles= AnglePriorCalibrator::k_angleCount;
		MIKAN_LOG_INFO("fit-angle-prior")
			<< sideName << ": " << calibrator.getSampleCount(side)
			<< " samples fitted (expect mean Mahalanobis^2 near " << kAngles << ")";
	}
	if (!bAnySide)
	{
		MIKAN_LOG_ERROR("fit-angle-prior") << "Neither hand had enough samples to fit a prior";
		return 1;
	}

	if (outputPath.empty())
	{
		MIKAN_LOG_INFO("fit-angle-prior")
			<< "No output path given - fitted but nothing written (pass output-config.json)";
		return 0;
	}

	std::ofstream outFile(outputPath, std::ios::binary);
	if (!outFile.is_open())
	{
		MIKAN_LOG_ERROR("fit-angle-prior") << "Cannot write " << outputPath;
		return 1;
	}
	outFile << baseConfig.toJsonString();
	outFile.close();
	MIKAN_LOG_INFO("fit-angle-prior") << "Wrote " << outputPath;
	return 0;
}

MIKAN_REGISTER_TEST("--fit-angle-prior",
					"Fit the finger-angle prior from recordings' stereo frames",
					eTestCategory::Tool, runFitAnglePriorTool);
