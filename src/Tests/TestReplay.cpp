#include "TestCommon.h"

static int runReplayTest(const TestArgs& args)
{
	int result= 0;
	const std::string recordingPath=
		(std::filesystem::temp_directory_path() / "mikan-test-replay.jsonl").string();

	// -- Synthetic 2-camera rig ------------------------------------
	const float fx= 800.f, fy= 800.f, cx= 640.f, cy= 360.f;

	AppConfig config;
	config.cameras.clear();
	// Camera 0: overhead at (0,0,0.8) looking straight down
	// Camera 1: side at (0,0.7,0.15) looking along -Y
	const glm::dmat4 camFromWorld[2]= {
		glm::dmat4(glm::dvec4(1, 0, 0, 0), glm::dvec4(0, -1, 0, 0), glm::dvec4(0, 0, -1, 0),
				   glm::dvec4(0, 0, 0.8, 1)),
		glm::dmat4(glm::dvec4(-1, 0, 0, 0), glm::dvec4(0, 0, -1, 0), glm::dvec4(0, -1, 0, 0),
				   glm::dvec4(0, 0.15, 0.7, 1)),
	};
	for (int cameraIndex= 0; cameraIndex < 2; ++cameraIndex)
	{
		CameraProfile profile;
		profile.intrinsics.present= true;
		profile.intrinsics.intrinsics.pixel_width= 1280;
		profile.intrinsics.intrinsics.pixel_height= 720;
		profile.intrinsics.intrinsics.undistorted_camera_matrix.x0= fx;
		profile.intrinsics.intrinsics.undistorted_camera_matrix.y1= fy;
		profile.intrinsics.intrinsics.undistorted_camera_matrix.z0= cx;
		profile.intrinsics.intrinsics.undistorted_camera_matrix.z1= cy;
		profile.extrinsics.present= true;
		profile.extrinsics.markerFromCamera= glm::inverse(camFromWorld[cameraIndex]);
		config.cameras.push_back(profile);
	}

	// Canonical hand model, wrist at origin, slightly non-planar (the
	// same shape the PnP self-test uses)
	std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints;
	for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
	{
		const float along= 0.02f + 0.15f * (float)(lm % 4) / 4.f * (lm >= 1 ? 1.f : 0.f);
		const float spread= ((float)(lm / 4) - 2.f) * 0.02f;
		const float curl= 0.012f * (float)(lm % 4);
		modelPoints[lm]= glm::vec3(spread, along, curl);
	}
	modelPoints[0]= glm::vec3(0.f);
	const float boneLength= glm::length(modelPoints[(int)eHandLandmark::MIDDLE_MCP]);
	config.handScale.present= true;
	config.handScale.refLengthMeters= boneLength;

	auto handTransform= [](int frameIdx) {
		const float t= (float)frameIdx / 30.f;
		glm::mat4 xf= glm::translate(glm::mat4(1.f),
									 glm::vec3(0.05f * sinf(t), 0.03f * cosf(t), 0.12f));
		xf= glm::rotate(xf, 0.4f * sinf(0.7f * t), glm::vec3(0.f, 0.f, 1.f));
		xf= glm::rotate(xf, 0.3f * cosf(0.9f * t), glm::vec3(1.f, 0.f, 0.f));
		return xf;
	};

	// Builds one camera's recorded input for a frame (the source of
	// truth for both the live pass and, via the file, replay)
	auto buildCameraInput= [&](int cameraIndex, int frameIdx, bool bValid, bool bWithDepth) {
		RecordedCameraInput input;
		input.cameraIndex= cameraIndex;
		input.timestampMs= 1000.0 + frameIdx * 33.0;
		input.valid= bValid;
		input.frameIndex= frameIdx;
		input.frameWidth= 1280;
		input.frameHeight= 720;
		input.captureFps= 30.f;
		input.inferenceMs= 5.f;
		if (!bValid)
			return input;

		input.refLengthMeters= boneLength * (1.f + 0.0005f * (float)frameIdx);

		const glm::mat4 worldFromModel= handTransform(frameIdx);
		RecordedHandInput& hand= input.hands[(int)eHandSide::Right];
		hand.tracked= true;
		hand.side= (int)eHandSide::Right;
		hand.slotId= 0;
		hand.presence= 0.95f;
		hand.handednessScore= 0.9f;
		hand.rightProb= 0.9f;
		hand.modelPoints= modelPoints;
		for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
		{
			const glm::vec3 worldPoint=
				glm::vec3(worldFromModel * glm::vec4(modelPoints[lm], 1.f));
			const glm::vec3 cameraPoint=
				glm::vec3(camFromWorld[cameraIndex] * glm::dvec4(glm::dvec3(worldPoint), 1.0));
			hand.imagePoints[lm]= glm::vec3(fx * cameraPoint.x / cameraPoint.z + cx,
											fy * cameraPoint.y / cameraPoint.z + cy, 0.f);
			if (bWithDepth)
			{
				input.depth[(int)eHandSide::Right].bValid[lm]= true;
				input.depth[(int)eHandSide::Right].cameraPoints[lm]= cameraPoint;
			}
		}
		if (bWithDepth)
		{
			input.bHaveDepth= true;
			input.depth[(int)eHandSide::Right].validCount= HAND_LANDMARK_COUNT;
		}
		return input;
	};

	// -- Live pass: the exact VisionThread call sequence, recorded
	// through a real TrackingRecorder ------------------------------
	std::vector<TrackingFrameResult> liveOutputs;
	{
		HandFusion fusion;
		// Build the fusion config exactly as the vision thread does
		HandFusionConfig fusionConfig;
		fusionConfig.stalenessWindowMs= config.fusion.stalenessWindowMs;
		fusionConfig.wristMatchMaxDistM= config.fusion.wristMatchMaxDistM;
		fusionConfig.minCameraConfidence= config.fusion.minCameraConfidence;
		fusionConfig.jitterReferenceM= config.fusion.jitterReferenceMm * 0.001f;
		fusionConfig.smoothingEnabled= config.tracking.smoothingEnabled;
		fusionConfig.palmMinCutoff= config.tracking.palmMinCutoff;
		fusionConfig.palmBeta= config.tracking.palmBeta;
		fusionConfig.angleMinCutoff= config.tracking.angleMinCutoff;
		fusionConfig.angleBeta= config.tracking.angleBeta;
		fusionConfig.triangulationEnabled= config.fusion.triangulationEnabled;
		fusionConfig.triangulationMaxResidualPx= config.fusion.triangulationMaxResidualPx;
		fusionConfig.residualReferencePx= config.fusion.residualReferencePx;
		fusion.configure(fusionConfig);
		fusion.resetTransientState();

		std::array<LandmarkTo3D, 2> landmarkTo3D;
		std::array<CameraFrameResult, 2> mirrors;
		for (int cameraIndex= 0; cameraIndex < 2; ++cameraIndex)
		{
			landmarkTo3D[cameraIndex].configure(config.cameras[cameraIndex].intrinsics.intrinsics,
												config.handScale.refLengthMeters);
			landmarkTo3D[cameraIndex].resetTransientState();
			mirrors[cameraIndex].cameraIndex= cameraIndex;
		}

		RecordingHeader header;
		header.formatVersion= TrackingRecording::k_formatVersion;
		header.appConfigJsonText= config.toJsonString();
		header.cameraCount= 2;

		TrackingRecorder recorder;
		if (!recorder.start(recordingPath, TrackingRecording::headerToJson(header)))
		{
			MIKAN_LOG_ERROR("test-replay") << "Recorder failed to open " << recordingPath;
			return 1;
		}

		for (int frameIdx= 0; frameIdx < 60; ++frameIdx)
		{
			RecordedFrame frame;
			frame.seq= frameIdx;

			// Freshness plan: frame 0 = cam0 only, invalid (the
			// passthrough case); i%5==3 = cam0 only (stale-candidate
			// case); frame 20 = cam1 fresh-but-invalid; 30..32 = cam0
			// carries synthetic depth
			std::vector<RecordedCameraInput> fresh;
			if (frameIdx == 0)
			{
				fresh.push_back(buildCameraInput(0, frameIdx, false, false));
			}
			else
			{
				const bool bDepth= frameIdx >= 30 && frameIdx <= 32;
				fresh.push_back(buildCameraInput(0, frameIdx, true, bDepth));
				if (frameIdx == 20)
					fresh.push_back(buildCameraInput(1, frameIdx, false, false));
				else if (frameIdx % 5 != 3)
					fresh.push_back(buildCameraInput(1, frameIdx, true, false));
			}

			// Per-camera stage (mirrors VisionThread::processCameraFrame)
			for (const RecordedCameraInput& input : fresh)
			{
				const int cameraIndex= input.cameraIndex;
				TrackingFrameResult camResult;
				camResult.frameIndex= input.frameIndex;
				camResult.timestampMs= input.timestampMs;
				camResult.frameWidth= input.frameWidth;
				camResult.frameHeight= input.frameHeight;
				camResult.captureFps= input.captureFps;
				camResult.inferenceMs= input.inferenceMs;
				if (input.valid)
				{
					for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
					{
						const RecordedHandInput& recordedHand= input.hands[sideIndex];
						TrackedHand& hand= camResult.hands[sideIndex];
						hand.tracked= recordedHand.tracked;
						if (!recordedHand.tracked)
							continue;
						hand.side= (eHandSide)recordedHand.side;
						hand.slotId= recordedHand.slotId;
						hand.presence= recordedHand.presence;
						hand.handednessScore= recordedHand.handednessScore;
						hand.rightProb= recordedHand.rightProb;
						hand.imagePoints= recordedHand.imagePoints;
						hand.modelPoints= recordedHand.modelPoints;
					}
					landmarkTo3D[cameraIndex].setRefLengthMeters(input.refLengthMeters);
					landmarkTo3D[cameraIndex].process(
						camResult, input.bHaveDepth ? &input.depth : nullptr);
					applyWorldTransform(camResult,
										config.cameras[cameraIndex].extrinsics.markerFromCamera);
				}

				CameraFrameResult& mirror= mirrors[cameraIndex];
				mirror.valid= input.valid;
				mirror.timestampMs= input.timestampMs;
				mirror.hasExtrinsics= true;
				mirror.markerFromCamera= config.cameras[cameraIndex].extrinsics.markerFromCamera;
				mirror.hasIntrinsics= true;
				mirror.fx= fx;
				mirror.fy= fy;
				mirror.cx= cx;
				mirror.cy= cy;
				mirror.result= camResult;

				frame.freshCameras.push_back(input);
			}

			// Fusion stage over all mirrors (fresh or stale)
			double nowMs= 0.0;
			for (const CameraFrameResult& mirror : mirrors)
				nowMs= std::max(nowMs, mirror.timestampMs);
			frame.nowTimestampMs= nowMs;

			bool bAnyWorld= false;
			for (const CameraFrameResult& mirror : mirrors)
				bAnyWorld|= mirror.valid && mirror.hasExtrinsics;

			TrackingFrameResult fused;
			if (bAnyWorld)
			{
				std::vector<const CameraFrameResult*> candidates;
				for (const CameraFrameResult& mirror : mirrors)
					candidates.push_back(&mirror);
				fusion.fuse(candidates, nowMs, fused);
				frame.bFused= true;
			}
			else
			{
				fused= mirrors[0].result;
				frame.bFused= false;
			}

			TrackingRecording::snapshotFusedOutput(fused, frame.outPoses);
			frame.checksum= TrackingRecording::computeFusedChecksum(fused);
			liveOutputs.push_back(fused);
			recorder.enqueueFrame(std::move(frame));
		}
		recorder.stop();
	}

	// -- (a) Replay pass: load + runAll must match bit-exactly -----
	TrackingReplay replay;
	std::string error;
	if (!replay.load(recordingPath, error))
	{
		MIKAN_LOG_ERROR("test-replay") << "(a) FAILED: load: " << error;
		return 1;
	}
	replay.runAll();

	MIKAN_LOG_INFO("test-replay")
		<< "(a) frames=" << replay.getFrameCount()
		<< " divergent=" << replay.getDivergentFrames().size();
	if (replay.getFrameCount() != 60 || !replay.getDivergentFrames().empty())
	{
		MIKAN_LOG_ERROR("test-replay") << "(a) FAILED: replay must match every checksum";
		result= 1;
	}
	if (replay.getFrame(0).recorded->bFused)
	{
		MIKAN_LOG_ERROR("test-replay") << "(a) FAILED: frame 0 must be a passthrough frame";
		result= 1;
	}

	// Float-exact comparison of the replayed poses vs the live pass
	{
		int mismatchCount= 0;
		for (int frameIdx= 0; frameIdx < replay.getFrameCount(); ++frameIdx)
		{
			const TrackingFrameResult& live= liveOutputs[frameIdx];
			const TrackingFrameResult& replayed= replay.getFrame(frameIdx).replayedFused;
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				const HandPose& livePose= live.poses[sideIndex];
				const HandPose& replayedPose= replayed.poses[sideIndex];
				if (livePose.tracked != replayedPose.tracked)
				{
					++mismatchCount;
					continue;
				}
				if (!livePose.tracked)
					continue;
				if (memcmp(&livePose.palmPositionWorld, &replayedPose.palmPositionWorld,
						   sizeof(glm::vec3)) != 0 ||
					memcmp(&livePose.palmOrientationWorld, &replayedPose.palmOrientationWorld,
						   sizeof(glm::quat)) != 0 ||
					memcmp(livePose.fingers.data(), replayedPose.fingers.data(),
						   sizeof(FingerAngles) * FINGER_COUNT) != 0)
					++mismatchCount;
			}
		}
		MIKAN_LOG_INFO("test-replay") << "(b) float-exact pose mismatches=" << mismatchCount;
		if (mismatchCount != 0)
		{
			MIKAN_LOG_ERROR("test-replay")
				<< "(b) FAILED: JSONL round-trip must be byte-lossless";
			result= 1;
		}
	}

	// -- (c) What-if must diverge while plain replay matches -------
	{
		TrackingReplay::WhatIfParams params= replay.makeDefaultWhatIfParams();
		params.fusionConfig.triangulationEnabled= false;
		replay.runWhatIf(params);

		int whatIfDifferent= 0;
		for (int frameIdx= 0; frameIdx < replay.getFrameCount(); ++frameIdx)
		{
			const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIdx);
			if (frame.bHasWhatIf &&
				TrackingRecording::computeFusedChecksum(frame.whatIfFused) !=
					frame.recorded->checksum)
				++whatIfDifferent;
		}
		MIKAN_LOG_INFO("test-replay") << "(c) what-if divergent frames=" << whatIfDifferent;
		if (whatIfDifferent == 0)
		{
			MIKAN_LOG_ERROR("test-replay")
				<< "(c) FAILED: disabling triangulation must change the output";
			result= 1;
		}
	}

	// -- (d) Corrupted checksum must be reported -------------------
	{
		const std::string corruptPath=
			(std::filesystem::temp_directory_path() / "mikan-test-replay-corrupt.jsonl").string();
		std::ifstream inFile(recordingPath);
		std::ofstream outFile(corruptPath, std::ios::trunc);
		std::string line;
		int lineIndex= 0;
		while (std::getline(inFile, line))
		{
			if (lineIndex == 41) // an arbitrary mid-recording frame line
			{
				nlohmann::json j= nlohmann::json::parse(line);
				j["checksum"]= "0000000000000000";
				line= j.dump();
			}
			outFile << line << '\n';
			++lineIndex;
		}
		inFile.close();
		outFile.close();

		TrackingReplay corruptReplay;
		if (!corruptReplay.load(corruptPath, error))
		{
			MIKAN_LOG_ERROR("test-replay") << "(d) FAILED: corrupt-copy load: " << error;
			result= 1;
		}
		else
		{
			corruptReplay.runAll();
			MIKAN_LOG_INFO("test-replay")
				<< "(d) corrupt-copy divergent=" << corruptReplay.getDivergentFrames().size();
			if (corruptReplay.getDivergentFrames().size() != 1)
			{
				MIKAN_LOG_ERROR("test-replay")
					<< "(d) FAILED: exactly the corrupted frame must diverge";
				result= 1;
			}
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-replay") << "All replay checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-replay", "Record/replay determinism, checksums, what-if", eTestCategory::SelfTest, runReplayTest);
