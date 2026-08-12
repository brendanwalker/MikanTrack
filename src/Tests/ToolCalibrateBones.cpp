#include "TestCommon.h"

// Measures the user's hand bones from a recorded session: replays it, feeds
// every stereo-triangulated frame to the same HandBoneCalibrator the live
// capture uses, and prints what it measured next to the landmark model's own
// proportions.
//
// Replay re-runs fusion, so the triangulated world points are regenerated even
// though the recording format never stored them. That makes this both the
// verification harness for the calibrator and a way to calibrate from a
// capture already on disk.
//
// With an output path it writes a config: the recording's own config plus the
// measured skeleton and the reference bone it implies.

namespace
{
const char* k_boneNames[FINGER_COUNT][3]= {
	{"thumb metacarpal", "thumb proximal", "thumb distal"},
	{"index proximal", "index middle", "index distal"},
	{"middle proximal", "middle middle", "middle distal"},
	{"ring proximal", "ring middle", "ring distal"},
	{"pinky proximal", "pinky middle", "pinky distal"},
};

// The landmark model's own skeleton for the same hand, rescaled on the
// reference bone exactly as LandmarkTo3D does, so the two columns compare
void reportSide(eHandSide side, const HandBoneCalibrator& calibrator)
{
	const char* sideName= side == eHandSide::Left ? "LEFT" : "RIGHT";

	HandSkeleton skeleton;
	HandBoneCalibrator::Quality quality;
	if (!calibrator.finish(side, skeleton, quality))
	{
		MIKAN_LOG_WARNING("calibrate-bones")
			<< sideName << ": only " << calibrator.getSampleCount(side)
			<< " stereo-triangulated samples, need " << HandBoneCalibrator::k_minSamples;
		return;
	}

	const float referenceBone= skeleton.baseInPalm[(int)eFinger::Middle].x * 2.f;
	MIKAN_LOG_INFO("calibrate-bones")
		<< sideName << ": " << quality.sampleCount << " samples | wrist->middle-MCP "
		<< referenceBone * 1000.f << " mm | worst per-bone spread "
		<< quality.worstPhalanxSpread * 1000.f << " mm";

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			MIKAN_LOG_INFO("calibrate-bones")
				<< "   " << k_boneNames[finger][phalanx] << ": "
				<< skeleton.phalanxLengths[finger][phalanx] * 1000.f << " mm (spread "
				<< quality.phalanxSpread[finger][phalanx] * 1000.f << " mm)";
		}
	}

	// Knuckle spacing is not a phalanx, but it is where the landmark model is
	// most obviously wrong, so it is worth printing
	for (int finger= (int)eFinger::Index; finger < FINGER_COUNT - 1; ++finger)
	{
		MIKAN_LOG_INFO("calibrate-bones")
			<< "   base spacing " << finger << "->" << (finger + 1) << ": "
			<< glm::length(skeleton.baseInPalm[finger + 1] - skeleton.baseInPalm[finger]) * 1000.f
			<< " mm";
	}
}
} // namespace

static int runCalibrateBonesTool(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("calibrate-bones")
			<< "usage: --calibrate-bones <recording.jsonl> [output-config.json]";
		return 1;
	}

	TrackingReplay replay;
	std::string error;
	if (!replay.load(args[0], error))
	{
		MIKAN_LOG_ERROR("calibrate-bones") << "Load failed: " << error;
		return 1;
	}
	replay.runAll();

	HandBoneCalibrator calibrator;
	int fusedFrames= 0;
	for (int frameIndex= 0; frameIndex < replay.getFrameCount(); ++frameIndex)
	{
		const TrackingFrameResult& fused= replay.getFrame(frameIndex).replayedFused;
		++fusedFrames;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			// Only the triangulated frames carry measured geometry; a
			// monocular pose is the model's shape wearing a pose
			const HandPose& pose= fused.poses[sideIndex];
			const TrackedHand& hand= fused.hands[sideIndex];
			if (!pose.tracked || !pose.stereoTriangulated || !hand.tracked || !hand.hasWorldSpace)
				continue;
			calibrator.addSample((eHandSide)sideIndex, hand.worldPoints);
		}
	}

	MIKAN_LOG_INFO("calibrate-bones")
		<< replay.getFilePath() << ": " << fusedFrames << " frames, "
		<< calibrator.getSampleCount(eHandSide::Left) << " left / "
		<< calibrator.getSampleCount(eHandSide::Right) << " right stereo samples";

	reportSide(eHandSide::Left, calibrator);
	reportSide(eHandSide::Right, calibrator);

	if (args.size() < 2)
		return 0;

	// Write the recording's config back out with the measured skeleton in it
	AppConfig config= replay.getRecordedConfig();
	bool bAnySide= false;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		HandSkeleton skeleton;
		HandBoneCalibrator::Quality quality;
		if (!calibrator.finish((eHandSide)sideIndex, skeleton, quality))
			continue;

		config.handSkeleton.skeleton[sideIndex]= skeleton;
		config.handSkeleton.present[sideIndex]= true;
		bAnySide= true;
	}
	if (!bAnySide)
	{
		MIKAN_LOG_ERROR("calibrate-bones") << "Neither hand had enough samples to write a config";
		return 1;
	}

	// One scale story: the reference bone the rest of the app reads comes from
	// the measured skeleton (both hands when both were measured)
	double referenceSum= 0.0;
	int referenceCount= 0;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!config.handSkeleton.present[sideIndex])
			continue;
		referenceSum+= config.handSkeleton.skeleton[sideIndex].baseInPalm[(int)eFinger::Middle].x * 2.0;
		++referenceCount;
	}
	config.handScale.refLengthMeters= referenceSum / (double)referenceCount;
	config.handScale.present= true;

	std::ofstream outFile(args[1], std::ios::binary);
	if (!outFile.is_open())
	{
		MIKAN_LOG_ERROR("calibrate-bones") << "Cannot write " << args[1];
		return 1;
	}
	outFile << config.toJsonString();
	outFile.close();

	MIKAN_LOG_INFO("calibrate-bones")
		<< "Wrote " << args[1] << " with refLengthMeters " << config.handScale.refLengthMeters * 1000.0
		<< " mm";
	return 0;
}

MIKAN_REGISTER_TEST("--calibrate-bones",
					"Measure hand bone lengths from a recording's stereo frames",
					eTestCategory::Tool, runCalibrateBonesTool);
