#include "TestCommon.h"

#include "HandStateEstimator.h"

// Pop metrics: replays recordings twice - as recorded (baseline) and with the
// hand state estimator forced on - and reports the discontinuity statistics of
// the fused output side by side. This is the objective gate for the estimator:
// pops (palm steps, orientation snaps, palmar flips, angle jumps) must go
// down while the FK reprojection residual - the anti-cheat metric, since a
// frozen hand has zero pops - must not regress.
//
//   --replay-popmetrics <recording.jsonl>... [prior-config.json]
//
// The optional .json argument is a config carrying a fitted angle prior
// (--fit-angle-prior output); it is injected into the estimator pass, so a
// prior can be A/B'd against recordings whose headers predate it.

namespace
{
struct StepStats
{
	std::vector<float> values;
	// Frame index of the largest value (so the worst offender is directly
	// diggable with --replay-dump)
	int worstFrame= -1;
	float worstValue= -1.f;

	void add(float value, int frameIndex= -1)
	{
		values.push_back(value);
		if (value > worstValue)
		{
			worstValue= value;
			worstFrame= frameIndex;
		}
	}

	float percentile(float fraction) const
	{
		if (values.empty())
			return 0.f;
		std::vector<float> sorted= values;
		std::sort(sorted.begin(), sorted.end());
		const size_t index= std::min(sorted.size() - 1, (size_t)((float)sorted.size() * fraction));
		return sorted[index];
	}
	float median() const { return percentile(0.5f); }
	float p90() const { return percentile(0.9f); }
	float maxValue() const { return values.empty() ? 0.f : *std::max_element(values.begin(), values.end()); }
	int countAbove(float threshold) const
	{
		return (int)std::count_if(values.begin(), values.end(),
								  [threshold](float v) { return v > threshold; });
	}
};

// Per-side metrics for one pass over one recording
struct SideMetrics
{
	int fusedRecords= 0;
	int trackedRecords= 0;
	int dropouts= 0;        // tracked -> untracked transitions
	int pathTransitions= 0; // stereoTriangulated toggles between tracked records
	// Records since the last (re)acquisition: the first few steps after one
	// are settling (both passes converge from their seed), not tracking pops,
	// so they stay out of the step statistics
	int recordsSinceAcquire= 0;
	StepStats posStepMm;
	StepStats rotStepDeg;
	int palmarFlips= 0; // >60 deg steps whose rotation axis is near palm +X
	StepStats angleStepDeg; // pooled across the 20 DoF
	int angleJumps= 0;      // single-DoF steps > 0.3 rad
	StepStats fkResidualPx; // fused pose FK vs each fresh camera's pixels

	// Estimator-pass attribution (from the what-if diagnostics)
	int estimatorFallbacks= 0; // fuses where the classic pose streamed instead
	int estimatorReseeds= 0;
	int estimatorHolds= 0;     // bad-fit holds (previous state streamed)

	// Anatomical plausibility: tracked records whose RAW angles land outside
	// the joint limits, and the worst excursion seen. On clean data this
	// should be near zero - a high baseline rate is exactly the implausible
	// output the limit prior exists to remove.
	int limitViolationRecords= 0;
	float worstLimitViolationDeg= 0.f;
};

struct CameraGeometry
{
	bool valid= false;
	glm::dmat4 markerFromCamera{1.0};
	glm::dmat4 cameraFromWorld{1.0};
	float fx= 0.f, fy= 0.f, cx= 0.f, cy= 0.f;
};

float angleDof(const FingerAngles& angles, int dof)
{
	switch (dof)
	{
	case 0: return angles.lateral;
	case 1: return angles.proximal;
	case 2: return angles.intermediate;
	default: return angles.distal;
	}
}

// FK reprojection of a fused pose against one fresh camera's recorded pixels.
// The pose's streamed angles carry the rest offset, so it is added back before
// FK (the state/geometry convention); the camera's hand is matched by wrist
// projection distance because per-camera side labels are not evidence.
bool fusedFkResidualPx(const HandPose& pose, const AppConfig& config,
					   const CameraGeometry& geometry, const RecordedCameraInput& fresh,
					   float& outMeanPx)
{
	const int sideIndex= (int)pose.side;

	std::array<FingerAngles, FINGER_COUNT> rawAngles= pose.fingers;
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

	HandStateEstimator::Pose fkPose;
	fkPose.palmPositionWorld= pose.palmPositionWorld;
	fkPose.palmOrientationWorld= pose.palmOrientationWorld;
	fkPose.rawAngles= rawAngles;
	std::array<glm::vec3, HAND_LANDMARK_COUNT> worldPoints;
	HandStateEstimator::predictWorldLandmarks(fkPose, pose.skeleton, worldPoints);

	auto projectPx= [&](const glm::vec3& world, glm::vec2& outPx) {
		const glm::dvec4 camPoint= geometry.cameraFromWorld * glm::dvec4(glm::dvec3(world), 1.0);
		if (camPoint.z < 1e-3)
			return false;
		outPx= glm::vec2((float)(geometry.fx * camPoint.x / camPoint.z + geometry.cx),
						 (float)(geometry.fy * camPoint.y / camPoint.z + geometry.cy));
		return true;
	};

	// Match the camera's recorded hand by projected wrist distance
	glm::vec2 wristPx;
	if (!projectPx(worldPoints[(int)eHandLandmark::WRIST], wristPx))
		return false;
	const RecordedHandInput* matched= nullptr;
	float bestWristDistPx= 150.f; // beyond this the camera saw a different hand
	for (const RecordedHandInput& hand : fresh.hands)
	{
		if (!hand.tracked)
			continue;
		const float dist= glm::length(glm::vec2(hand.imagePoints[(int)eHandLandmark::WRIST]) - wristPx);
		if (dist < bestWristDistPx)
		{
			bestWristDistPx= dist;
			matched= &hand;
		}
	}
	if (matched == nullptr)
		return false;

	float sum= 0.f;
	int count= 0;
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		glm::vec2 projected;
		if (!projectPx(worldPoints[i], projected))
			return false;
		sum+= glm::length(projected - glm::vec2(matched->imagePoints[i]));
		++count;
	}
	outMeanPx= sum / (float)count;
	return true;
}

void accumulateFrame(const TrackingFrameResult& fused, const RecordedFrame& recorded, int frameIndex,
					 const AppConfig& config, const std::vector<CameraGeometry>& geometry,
					 std::array<SideMetrics, 2>& metrics,
					 std::array<HandPose, 2>& previousPoses, std::array<bool, 2>& bHavePrevious)
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		SideMetrics& side= metrics[sideIndex];
		const HandPose& pose= fused.poses[sideIndex];
		++side.fusedRecords;

		const bool bTracked= pose.tracked && pose.hasWorldPose;
		if (!bTracked)
		{
			if (bHavePrevious[sideIndex])
				++side.dropouts;
			bHavePrevious[sideIndex]= false;
			side.recordsSinceAcquire= 0;
			continue;
		}
		++side.trackedRecords;
		++side.recordsSinceAcquire;

		constexpr int kSettleRecords= 5;
		if (bHavePrevious[sideIndex] && side.recordsSinceAcquire > kSettleRecords)
		{
			const HandPose& previous= previousPoses[sideIndex];

			side.posStepMm.add(glm::length(pose.palmPositionWorld - previous.palmPositionWorld) * 1000.f,
							   frameIndex);

			// Body-frame rotation step (the palm's own axes): axis near local
			// +X on a large step is the palmar-flip signature
			glm::quat dq= glm::inverse(previous.palmOrientationWorld) * pose.palmOrientationWorld;
			if (dq.w < 0.f)
				dq= -dq;
			const float stepRad= 2.f * acosf(std::min(dq.w, 1.f));
			side.rotStepDeg.add(glm::degrees(stepRad), frameIndex);
			if (glm::degrees(stepRad) > 60.f)
			{
				const glm::vec3 axis(dq.x, dq.y, dq.z);
				const float axisLength= glm::length(axis);
				if (axisLength > 1e-6f &&
					fabsf(axis.x / axisLength) > cosf(glm::radians(30.f)))
					++side.palmarFlips;
			}

			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				for (int dof= 0; dof < 4; ++dof)
				{
					const float stepAngleRad= fabsf(angleDof(pose.fingers[finger], dof) -
													angleDof(previous.fingers[finger], dof));
					side.angleStepDeg.add(glm::degrees(stepAngleRad));
					if (stepAngleRad > 0.3f)
						++side.angleJumps;
				}
			}

			if (pose.stereoTriangulated != previous.stereoTriangulated)
				++side.pathTransitions;
		}

		// Anatomical plausibility of the RAW angles (streamed + rest offset).
		// Stereo records only: mono fallback angles carry PER-CAMERA rest
		// offsets, so adding the fused offset back would manufacture
		// violations out of a convention mismatch.
		if (pose.stereoTriangulated)
		{
			std::array<FingerAngles, FINGER_COUNT> rawAngles= pose.fingers;
			if (config.fusedRestAngles.present[sideIndex])
			{
				const std::array<FingerAngles, FINGER_COUNT>& rest=
					config.fusedRestAngles.angles[sideIndex];
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					rawAngles[finger].lateral+= rest[finger].lateral;
					rawAngles[finger].proximal+= rest[finger].proximal;
					rawAngles[finger].intermediate+= rest[finger].intermediate;
					rawAngles[finger].distal+= rest[finger].distal;
				}
			}
			float worstViolationRad= 0.f;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				for (int dof= 0; dof < 4; ++dof)
					worstViolationRad= std::max(
						worstViolationRad,
						fabsf(HandStateEstimator::angleLimitViolation(
							(eFinger)finger, dof, angleDof(rawAngles[finger], dof))));
			if (worstViolationRad > 0.f)
				++side.limitViolationRecords;
			side.worstLimitViolationDeg=
				std::max(side.worstLimitViolationDeg, glm::degrees(worstViolationRad));
		}

		// FK reprojection against every FRESH camera that saw this hand
		for (const RecordedCameraInput& fresh : recorded.freshCameras)
		{
			if (!fresh.valid || fresh.cameraIndex < 0 ||
				fresh.cameraIndex >= (int)geometry.size() || !geometry[fresh.cameraIndex].valid)
				continue;
			float residualPx= 0.f;
			if (fusedFkResidualPx(pose, config, geometry[fresh.cameraIndex], fresh, residualPx))
				side.fkResidualPx.add(residualPx);
		}

		previousPoses[sideIndex]= pose;
		bHavePrevious[sideIndex]= true;
	}
}

void logSideMetrics(const char* passName, int sideIndex, const SideMetrics& side)
{
	const char* sideName= sideIndex == 0 ? "L" : "R";
	const float trackedPct= side.fusedRecords > 0
		? 100.f * (float)side.trackedRecords / (float)side.fusedRecords
		: 0.f;
	MIKAN_LOG_INFO("replay-popmetrics")
		<< "  " << passName << " " << sideName
		<< ": tracked " << trackedPct << "% (" << side.trackedRecords << "/" << side.fusedRecords
		<< "), dropouts " << side.dropouts
		<< ", path transitions " << side.pathTransitions;
	MIKAN_LOG_INFO("replay-popmetrics")
		<< "    palm step mm med/p90/max " << side.posStepMm.median() << "/" << side.posStepMm.p90()
		<< "/" << side.posStepMm.maxValue() << " (frame " << side.posStepMm.worstFrame << ")"
		<< ", >50mm " << side.posStepMm.countAbove(50.f)
		<< ", >100mm " << side.posStepMm.countAbove(100.f);
	MIKAN_LOG_INFO("replay-popmetrics")
		<< "    rot step deg med/p90/max " << side.rotStepDeg.median() << "/" << side.rotStepDeg.p90()
		<< "/" << side.rotStepDeg.maxValue() << " (frame " << side.rotStepDeg.worstFrame << ")"
		<< ", >30deg " << side.rotStepDeg.countAbove(30.f)
		<< ", >60deg " << side.rotStepDeg.countAbove(60.f)
		<< ", palmar flips " << side.palmarFlips;
	MIKAN_LOG_INFO("replay-popmetrics")
		<< "    angle step deg med/p90 " << side.angleStepDeg.median() << "/" << side.angleStepDeg.p90()
		<< ", jumps >0.3rad " << side.angleJumps
		<< ", limit violations " << side.limitViolationRecords << " records (worst "
		<< side.worstLimitViolationDeg << " deg)";
	MIKAN_LOG_INFO("replay-popmetrics")
		<< "    FK reprojection px med/p90 " << side.fkResidualPx.median() << "/"
		<< side.fkResidualPx.p90() << " (" << (int)side.fkResidualPx.values.size() << " samples)";
	if (side.estimatorFallbacks > 0 || side.estimatorReseeds > 0 || side.estimatorHolds > 0)
	{
		MIKAN_LOG_INFO("replay-popmetrics")
			<< "    estimator fallbacks " << side.estimatorFallbacks << ", reseeds "
			<< side.estimatorReseeds << ", bad-fit holds " << side.estimatorHolds;
	}
}
} // namespace

static int runReplayPopMetricsTool(const TestArgs& args)
{
	std::vector<std::string> recordings;
	std::string priorConfigPath;
	for (const std::string& arg : args)
	{
		if (arg.size() > 6 && arg.substr(arg.size() - 6) == ".jsonl")
			recordings.push_back(arg);
		else
			priorConfigPath= arg;
	}
	if (recordings.empty())
	{
		MIKAN_LOG_ERROR("replay-popmetrics")
			<< "Usage: --replay-popmetrics <recording.jsonl>... [prior-config.json]";
		return 1;
	}

	AppConfig priorConfig;
	bool bHavePriorConfig= false;
	if (!priorConfigPath.empty())
	{
		std::ifstream priorFile(priorConfigPath, std::ios::binary);
		if (!priorFile.is_open())
		{
			MIKAN_LOG_ERROR("replay-popmetrics") << "Cannot read " << priorConfigPath;
			return 1;
		}
		std::string text((std::istreambuf_iterator<char>(priorFile)), std::istreambuf_iterator<char>());
		if (!priorConfig.loadFromJsonString(text))
		{
			MIKAN_LOG_ERROR("replay-popmetrics") << "Cannot parse " << priorConfigPath;
			return 1;
		}
		bHavePriorConfig= true;
		MIKAN_LOG_INFO("replay-popmetrics")
			<< "Injecting angle prior from " << priorConfigPath << " (left="
			<< priorConfig.anglePrior.present[0] << " right=" << priorConfig.anglePrior.present[1]
			<< ") into the estimator pass";
	}

	int result= 0;
	for (const std::string& path : recordings)
	{
		TrackingReplay replay;
		std::string error;
		if (!replay.load(path, error))
		{
			MIKAN_LOG_ERROR("replay-popmetrics") << path << ": load failed: " << error;
			result= 1;
			continue;
		}

		replay.setCaptureDiagnostics(true);
		replay.runAll();

		TrackingReplay::WhatIfParams params= replay.makeDefaultWhatIfParams();
		params.fusionConfig.estimatorEnabled= true;
		if (bHavePriorConfig)
		{
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				HandStateEstimatorConfig::AnglePrior& prior=
					params.fusionConfig.estimator.anglePrior[sideIndex];
				prior.present= priorConfig.anglePrior.present[sideIndex];
				if (prior.present)
				{
					prior.mean= priorConfig.anglePrior.mean[sideIndex];
					prior.precision= priorConfig.anglePrior.precision[sideIndex];
					prior.weight= priorConfig.fusion.estimatorAnglePriorWeight;
				}
			}
		}
		replay.runWhatIf(params);

		const AppConfig& config= replay.getRecordedConfig();

		std::vector<CameraGeometry> geometry(config.cameraCount());
		for (int cameraIndex= 0; cameraIndex < (int)config.cameraCount(); ++cameraIndex)
		{
			const CameraProfile& profile= config.camera(cameraIndex);
			if (!profile.intrinsics.present || !profile.extrinsics.present)
				continue;
			CameraGeometry& cam= geometry[cameraIndex];
			cam.valid= true;
			cam.markerFromCamera= profile.extrinsics.markerFromCamera;
			cam.cameraFromWorld= glm::inverse(cam.markerFromCamera);
			const MikanMatrix3d& cameraMatrix= profile.intrinsics.intrinsics.undistorted_camera_matrix;
			cam.fx= (float)cameraMatrix.x0;
			cam.fy= (float)cameraMatrix.y1;
			cam.cx= (float)cameraMatrix.z0;
			cam.cy= (float)cameraMatrix.z1;
		}

		std::array<SideMetrics, 2> baseline{};
		std::array<SideMetrics, 2> estimator{};
		std::array<HandPose, 2> previousBaseline{};
		std::array<HandPose, 2> previousEstimator{};
		std::array<bool, 2> bHaveBaseline{};
		std::array<bool, 2> bHaveEstimator{};

		for (int frameIndex= 0; frameIndex < replay.getFrameCount(); ++frameIndex)
		{
			const TrackingReplay::ReplayFrame& frame= replay.getFrame(frameIndex);
			if (frame.recorded == nullptr || !frame.recorded->bFused)
				continue;

			accumulateFrame(frame.replayedFused, *frame.recorded, frameIndex, config, geometry,
							baseline, previousBaseline, bHaveBaseline);
			if (frame.bHasWhatIf)
				accumulateFrame(frame.whatIfFused, *frame.recorded, frameIndex, config, geometry,
								estimator, previousEstimator, bHaveEstimator);

			if (frame.bHasWhatIfDiagnostics)
			{
				for (const FusionDiagnostics::Cluster& cluster : frame.whatIfDiagnostics.clusters)
				{
					if (cluster.assignedSide < 0 || cluster.assignedSide > 1)
						continue;
					SideMetrics& side= estimator[cluster.assignedSide];
					if (!cluster.estimatorUsed)
						++side.estimatorFallbacks;
					if (cluster.estimatorReseeded)
						++side.estimatorReseeds;
					if (cluster.estimatorHeldBadFit)
						++side.estimatorHolds;
				}
			}
		}

		MIKAN_LOG_INFO("replay-popmetrics")
			<< std::filesystem::path(path).filename().string() << " (" << replay.getFrameCount()
			<< " frames, " << config.cameraCount() << " cameras):";
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			logSideMetrics("baseline ", sideIndex, baseline[sideIndex]);
			logSideMetrics("estimator", sideIndex, estimator[sideIndex]);
		}
	}

	return result;
}

MIKAN_REGISTER_TEST("--replay-popmetrics",
					"Pop statistics of a recording: baseline vs hand estimator",
					eTestCategory::Tool, runReplayPopMetricsTool);
