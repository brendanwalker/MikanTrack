#include <limits>

#include "TestCommon.h"

// A/B test of hardware depth against a recorded session: replays the recording
// once as it was captured and once with the recorded depth measurements
// withheld, so every depth-resolved camera falls back to the monocular PnP
// solve. That second pass is what the rig would have produced with an ordinary
// webcam in the depth camera's place - same landmarks, same intrinsics, same
// extrinsics, same triangulation, just no metric depth.
//
// The comparison is only about POSITION and about which path fusion took:
// finger angles never read depth, and triangulation works from 2D landmarks
// alone, so the two passes can only differ where a camera's own metric
// estimate mattered.
//
// Two things this cannot tell you. It cannot predict a replacement camera,
// because the landmarks are fixed recorded inputs - a different lens changes
// them and the replay never sees it. And the per-frame hand scale is a
// recorded input too, so the depth-off pass inherits the scale the depth-on
// session converged to instead of re-converging.

namespace
{

// Sorted-on-demand sample set, so every metric reports the same three numbers
struct Samples
{
	std::vector<double> values;

	void add(double value) { values.push_back(value); }
	bool empty() const { return values.empty(); }
	size_t size() const { return values.size(); }

	double quantile(double fraction)
	{
		if (values.empty())
			return 0.0;
		std::sort(values.begin(), values.end());
		size_t index= (size_t)(fraction * (double)values.size());
		if (index >= values.size())
			index= values.size() - 1;
		return values[index];
	}
};

const char* sideName(int sideIndex)
{
	return sideIndex == 0 ? "LEFT" : "RIGHT";
}

// Constant-velocity residual |p(t) - 2p(t-1) + p(t-2)|, the same jitter
// measure fusion scores cameras with. Normally useless on a free-form capture
// because real acceleration reads as jitter, but both passes replay the SAME
// motion, so the difference between them is attributable.
double accelerationResidualMm(const glm::vec3& previous2,
							  const glm::vec3& previous1,
							  const glm::vec3& current)
{
	return glm::length(current - 2.f * previous1 + previous2) * 1000.0;
}

// Which of a camera's two hand slots corresponds to each fused side. Per
// camera hand labels are slot bookkeeping and get displaced, so the pairing is
// resolved geometrically instead - and resolved ONCE, on the baseline pass, so
// both passes are then scored through the identical mapping and neither is
// favored by its own errors.
void assignCameraHandsToSides(const TrackingFrameResult& cameraResult,
							  const TrackingFrameResult& fused,
							  int outHandForSide[2])
{
	outHandForSide[0]= -1;
	outHandForSide[1]= -1;

	double cost[2][2];
	bool bUsable[2][2];
	for (int handIndex= 0; handIndex < 2; ++handIndex)
	{
		const HandPose& handPose= cameraResult.poses[handIndex];
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const HandPose& fusedPose= fused.poses[sideIndex];
			bUsable[handIndex][sideIndex]= handPose.tracked && handPose.hasWorldPose &&
										   fusedPose.tracked && fusedPose.hasWorldPose;
			cost[handIndex][sideIndex]=
				bUsable[handIndex][sideIndex]
					? glm::length(handPose.palmPositionWorld - fusedPose.palmPositionWorld)
					: std::numeric_limits<double>::max();
		}
	}

	// Two hands, two sides: the cheaper of the two permutations wins
	const bool bStraightUsable= bUsable[0][0] && bUsable[1][1];
	const bool bSwappedUsable= bUsable[0][1] && bUsable[1][0];
	const double straightCost= bStraightUsable ? cost[0][0] + cost[1][1]
											   : std::numeric_limits<double>::max();
	const double swappedCost= bSwappedUsable ? cost[0][1] + cost[1][0]
											 : std::numeric_limits<double>::max();

	if (bStraightUsable && straightCost <= swappedCost)
	{
		outHandForSide[0]= 0;
		outHandForSide[1]= 1;
		return;
	}
	if (bSwappedUsable)
	{
		outHandForSide[0]= 1;
		outHandForSide[1]= 0;
		return;
	}

	// Only one camera hand is usable: give it the side it sits closest to
	for (int handIndex= 0; handIndex < 2; ++handIndex)
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			if (bUsable[handIndex][sideIndex] && outHandForSide[sideIndex] < 0 &&
				(!bUsable[handIndex][1 - sideIndex] ||
				 cost[handIndex][sideIndex] <= cost[handIndex][1 - sideIndex]))
			{
				outHandForSide[sideIndex]= handIndex;
			}
		}
	}
}

// Accuracy of one camera's own monocular palm against the stereo solution,
// split into the component along the camera's view ray and the component
// across it. Monocular depth error is an along-ray error by construction, so
// the split says whether a gap is the depth solve or something else.
struct MonocularError
{
	Samples totalMm;
	Samples alongRayMm;
	Samples signedAlongRayMm; // + = the camera placed the hand too far away
	Samples lateralMm;
	Samples rangeRatio; // monocular range / stereo range, 1 = agrees

	void add(const glm::vec3& monocularPalm,
			 const glm::vec3& stereoPalm,
			 const glm::vec3& cameraOriginWorld)
	{
		const glm::vec3 delta= monocularPalm - stereoPalm;
		const double total= glm::length(delta);
		totalMm.add(total * 1000.0);

		const glm::vec3 rayToStereo= stereoPalm - cameraOriginWorld;
		const float rayLength= glm::length(rayToStereo);
		if (rayLength <= 0.f)
			return;

		const glm::vec3 rayDirection= rayToStereo / rayLength;
		const double alongRay= glm::dot(delta, rayDirection);
		const double lateralSquared= std::max(0.0, total * total - alongRay * alongRay);
		alongRayMm.add(std::abs(alongRay) * 1000.0);
		signedAlongRayMm.add(alongRay * 1000.0);
		lateralMm.add(std::sqrt(lateralSquared) * 1000.0);
		rangeRatio.add((double)glm::length(monocularPalm - cameraOriginWorld) / (double)rayLength);
	}
};

} // namespace

static int runReplayDepthTool(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("replay-depth") << "usage: --replay-depth <recording.jsonl>";
		return 1;
	}

	TrackingReplay replay;
	std::string error;
	if (!replay.load(args[0], error))
	{
		MIKAN_LOG_ERROR("replay-depth") << "Load failed: " << error;
		return 1;
	}
	replay.runAll();

	const AppConfig& recordedConfig= replay.getRecordedConfig();
	const int cameraCount= (int)recordedConfig.cameraCount();
	const int frameCount= replay.getFrameCount();

	// How much depth there was to withhold in the first place
	std::vector<int> depthFrames((size_t)cameraCount, 0);
	std::vector<int> freshFrames((size_t)cameraCount, 0);
	std::vector<double> depthLandmarks((size_t)cameraCount, 0.0);
	for (int frameIndex= 0; frameIndex < frameCount; ++frameIndex)
	{
		const RecordedFrame& recorded= *replay.getFrame(frameIndex).recorded;
		for (const RecordedCameraInput& fresh : recorded.freshCameras)
		{
			if (fresh.cameraIndex < 0 || fresh.cameraIndex >= cameraCount)
				continue;
			++freshFrames[(size_t)fresh.cameraIndex];
			if (!fresh.bHaveDepth)
				continue;
			++depthFrames[(size_t)fresh.cameraIndex];
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				depthLandmarks[(size_t)fresh.cameraIndex]+= fresh.depth[sideIndex].validCount;
		}
	}

	MIKAN_LOG_INFO("replay-depth")
		<< replay.getFilePath() << ": " << frameCount << " frames, " << cameraCount << " cameras";
	int depthCameraCount= 0;
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		const CameraProfile& profile= recordedConfig.camera((size_t)cameraIndex);
		const int fresh= std::max(1, freshFrames[(size_t)cameraIndex]);
		MIKAN_LOG_INFO("replay-depth")
			<< "  cam" << cameraIndex << " " << profile.video.deviceName << ": "
			<< freshFrames[(size_t)cameraIndex] << " fresh frames, depth on "
			<< (100 * depthFrames[(size_t)cameraIndex] / fresh) << "% of them ("
			<< (depthLandmarks[(size_t)cameraIndex] /
				(double)std::max(1, depthFrames[(size_t)cameraIndex]))
			<< " resolved landmarks per frame, both hands)";
		if (depthFrames[(size_t)cameraIndex] > 0)
			++depthCameraCount;
	}
	if (depthCameraCount == 0)
	{
		MIKAN_LOG_ERROR("replay-depth")
			<< "No camera in this recording carried depth - nothing to withhold";
		return 1;
	}

	TrackingReplay::WhatIfParams params= replay.makeDefaultWhatIfParams();
	params.bUseRecordedDepth= false;
	replay.runWhatIf(params);

	// -- Fused output, per side -----
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		int trackedDepthOn= 0, trackedDepthOff= 0, trackedBoth= 0;
		int stereoDepthOn= 0, stereoDepthOff= 0;
		int pathLostStereo= 0, pathGainedStereo= 0;
		double confidenceDepthOn= 0.0, confidenceDepthOff= 0.0;
		Samples palmDeltaStereoMm, palmDeltaMonoMm;
		Samples jitterDepthOn, jitterDepthOff;

		for (int frameIndex= 0; frameIndex < frameCount; ++frameIndex)
		{
			const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIndex);
			if (!frame.bHasWhatIf)
				continue;

			const HandPose& depthOn= frame.replayedFused.poses[sideIndex];
			const HandPose& depthOff= frame.whatIfFused.poses[sideIndex];
			if (depthOn.tracked)
				++trackedDepthOn;
			if (depthOff.tracked)
				++trackedDepthOff;
			if (!depthOn.tracked || !depthOff.tracked || !depthOn.hasWorldPose ||
				!depthOff.hasWorldPose)
				continue;

			++trackedBoth;
			stereoDepthOn+= depthOn.stereoTriangulated ? 1 : 0;
			stereoDepthOff+= depthOff.stereoTriangulated ? 1 : 0;
			if (depthOn.stereoTriangulated && !depthOff.stereoTriangulated)
				++pathLostStereo;
			if (!depthOn.stereoTriangulated && depthOff.stereoTriangulated)
				++pathGainedStereo;
			confidenceDepthOn+= depthOn.confidence;
			confidenceDepthOff+= depthOff.confidence;

			// Split by the path the depth-on pass took: on triangulated frames
			// position comes from 2D geometry and depth should barely matter,
			// so a delta there means depth changed a fusion DECISION, while a
			// delta on monocular frames is depth doing its actual job
			const double deltaMm=
				glm::length(depthOff.palmPositionWorld - depthOn.palmPositionWorld) * 1000.0;
			if (depthOn.stereoTriangulated)
				palmDeltaStereoMm.add(deltaMm);
			else
				palmDeltaMonoMm.add(deltaMm);

			// Jitter needs three consecutive frames valid in BOTH passes, so
			// the two numbers are measured over the identical frame set
			if (frameIndex < 2)
				continue;
			const TrackingReplay::ReplayFrame& previous1= replay.getFrame(frameIndex - 1);
			const TrackingReplay::ReplayFrame& previous2= replay.getFrame(frameIndex - 2);
			if (!previous1.bHasWhatIf || !previous2.bHasWhatIf)
				continue;
			const HandPose& on1= previous1.replayedFused.poses[sideIndex];
			const HandPose& on2= previous2.replayedFused.poses[sideIndex];
			const HandPose& off1= previous1.whatIfFused.poses[sideIndex];
			const HandPose& off2= previous2.whatIfFused.poses[sideIndex];
			if (!on1.tracked || !on2.tracked || !off1.tracked || !off2.tracked ||
				!on1.hasWorldPose || !on2.hasWorldPose || !off1.hasWorldPose || !off2.hasWorldPose)
				continue;
			jitterDepthOn.add(accelerationResidualMm(
				on2.palmPositionWorld, on1.palmPositionWorld, depthOn.palmPositionWorld));
			jitterDepthOff.add(accelerationResidualMm(
				off2.palmPositionWorld, off1.palmPositionWorld, depthOff.palmPositionWorld));
		}

		if (trackedBoth == 0)
		{
			MIKAN_LOG_INFO("replay-depth") << sideName(sideIndex) << ": no frames tracked by both passes";
			continue;
		}

		MIKAN_LOG_INFO("replay-depth")
			<< sideName(sideIndex) << ": tracked " << trackedDepthOn << " -> " << trackedDepthOff
			<< " frames (" << trackedBoth << " comparable) | stereo "
			<< (100 * stereoDepthOn / trackedBoth) << "% -> " << (100 * stereoDepthOff / trackedBoth)
			<< "% (" << pathLostStereo << " lost, " << pathGainedStereo << " gained) | mean confidence "
			<< (confidenceDepthOn / trackedBoth) << " -> " << (confidenceDepthOff / trackedBoth);
		MIKAN_LOG_INFO("replay-depth")
			<< sideName(sideIndex) << ": palm moved med " << palmDeltaStereoMm.quantile(0.5)
			<< " mm, p90 " << palmDeltaStereoMm.quantile(0.9) << " mm on the "
			<< palmDeltaStereoMm.size() << " stereo frames | med " << palmDeltaMonoMm.quantile(0.5)
			<< " mm, p90 " << palmDeltaMonoMm.quantile(0.9) << " mm, max "
			<< palmDeltaMonoMm.quantile(1.0) << " mm on the " << palmDeltaMonoMm.size()
			<< " monocular frames";
		MIKAN_LOG_INFO("replay-depth")
			<< sideName(sideIndex) << ": fused palm jitter med " << jitterDepthOn.quantile(0.5)
			<< " -> " << jitterDepthOff.quantile(0.5) << " mm, p90 " << jitterDepthOn.quantile(0.9)
			<< " -> " << jitterDepthOff.quantile(0.9) << " mm (" << jitterDepthOn.size()
			<< " frames, identical motion in both passes)";
	}

	// -- Each camera's own monocular estimate, refereed by stereo -----
	// On frames the depth-on pass triangulated, the fused palm comes from 2D
	// geometry across both cameras and owes nothing to depth, so it is a fair
	// referee for both passes' single-camera estimates.
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		const CameraProfile& profile= recordedConfig.camera((size_t)cameraIndex);
		if (!profile.extrinsics.present)
			continue;
		const glm::vec3 cameraOriginWorld= glm::vec3(profile.extrinsics.markerFromCamera[3]);

		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			MonocularError depthOnError, depthOffError;

			for (int frameIndex= 0; frameIndex < frameCount; ++frameIndex)
			{
				const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIndex);
				if (!frame.bHasWhatIf || (int)frame.perCamera.size() <= cameraIndex ||
					(int)frame.whatIfPerCamera.size() <= cameraIndex)
					continue;

				const HandPose& stereoPose= frame.replayedFused.poses[sideIndex];
				if (!stereoPose.tracked || !stereoPose.hasWorldPose || !stereoPose.stereoTriangulated)
					continue;

				const TrackingFrameResult& depthOnCamera= frame.perCamera[(size_t)cameraIndex];
				const TrackingFrameResult& depthOffCamera= frame.whatIfPerCamera[(size_t)cameraIndex];

				int handForSide[2];
				assignCameraHandsToSides(depthOnCamera, frame.replayedFused, handForSide);
				const int handIndex= handForSide[sideIndex];
				if (handIndex < 0)
					continue;

				const HandPose& depthOnHand= depthOnCamera.poses[handIndex];
				const HandPose& depthOffHand= depthOffCamera.poses[handIndex];
				if (!depthOnHand.tracked || !depthOnHand.hasWorldPose || !depthOffHand.tracked ||
					!depthOffHand.hasWorldPose)
					continue;

				depthOnError.add(depthOnHand.palmPositionWorld, stereoPose.palmPositionWorld,
								 cameraOriginWorld);
				depthOffError.add(depthOffHand.palmPositionWorld, stereoPose.palmPositionWorld,
								  cameraOriginWorld);
			}

			if (depthOnError.totalMm.empty())
				continue;

			MIKAN_LOG_INFO("replay-depth")
				<< "cam" << cameraIndex << " " << sideName(sideIndex) << " vs stereo ("
				<< depthOnError.totalMm.size() << " frames): depth-on med "
				<< depthOnError.totalMm.quantile(0.5) << " mm (along-ray "
				<< depthOnError.alongRayMm.quantile(0.5) << ", lateral "
				<< depthOnError.lateralMm.quantile(0.5) << ") | depth-off med "
				<< depthOffError.totalMm.quantile(0.5) << " mm (along-ray "
				<< depthOffError.alongRayMm.quantile(0.5) << ", lateral "
				<< depthOffError.lateralMm.quantile(0.5) << ")";
			// A near-constant signed offset with a range ratio away from 1 is a
			// scale disagreement with the stereo solution, not per-frame noise
			MIKAN_LOG_INFO("replay-depth")
				<< "cam" << cameraIndex << " " << sideName(sideIndex)
				<< " range bias: depth-on signed along-ray med "
				<< depthOnError.signedAlongRayMm.quantile(0.5) << " mm, range ratio med "
				<< depthOnError.rangeRatio.quantile(0.5) << " | depth-off signed "
				<< depthOffError.signedAlongRayMm.quantile(0.5) << " mm, ratio "
				<< depthOffError.rangeRatio.quantile(0.5);
		}
	}

	MIKAN_LOG_INFO("replay-depth")
		<< "Read the monocular-frame palm delta and the per-camera vs-stereo rows first: "
		   "they are where depth can change an answer. A large stereo-frame delta instead "
		   "means depth was steering fusion decisions rather than supplying position.";
	return 0;
}

MIKAN_REGISTER_TEST("--replay-depth",
					"A/B hardware depth against a recording by withholding it",
					eTestCategory::Tool, runReplayDepthTool);
