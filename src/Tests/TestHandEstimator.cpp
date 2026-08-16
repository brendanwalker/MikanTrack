#include "TestCommon.h"

#include "HandStateEstimator.h"

// Self test for the angle-space multi-view hand state estimator: synthetic
// cameras project a known FK hand, and the estimator has to recover and TRACK
// the truth from the 2D landmarks alone - including across the seams that pop
// the classic per-frame pipeline (camera dropout, mono stretches).

namespace
{
constexpr double kFrameMs= 33.0;

HandSkeleton makeTruthSkeleton()
{
	HandSkeleton skeleton;
	const float baseY[FINGER_COUNT]= {0.045f, 0.03f, 0.f, -0.01f, -0.03f};
	const float baseX[FINGER_COUNT]= {-0.01f, 0.035f, 0.0425f, 0.035f, 0.03f};
	const float proximal[FINGER_COUNT]= {0.038f, 0.0415f, 0.0441f, 0.0401f, 0.032f};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		skeleton.baseInPalm[finger]= glm::vec3(baseX[finger], baseY[finger], 0.f);
		skeleton.phalanxLengths[finger]= {proximal[finger], 0.025f, 0.020f};
	}
	skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
	return skeleton;
}

// A synthetic camera: OpenCV-convention axes (x right, y down, z forward
// toward the scene) placed in world and aimed at a target, plus pinhole
// intrinsics. markerFromCamera's columns are the camera axes in world.
struct TestCamera
{
	glm::dmat4 markerFromCamera{1.0};
	glm::dmat4 cameraFromWorld{1.0};
	float fx= 600.f;
	float fy= 600.f;
	float cx= 640.f;
	float cy= 360.f;
};

TestCamera makeCamera(const glm::vec3& position, const glm::vec3& target)
{
	const glm::vec3 forward= glm::normalize(target - position);
	const glm::vec3 right= glm::normalize(glm::cross(forward, glm::vec3(0.f, 0.f, 1.f)));
	const glm::vec3 down= glm::cross(forward, right);

	TestCamera camera;
	camera.markerFromCamera= glm::dmat4(
		glm::dvec4(glm::dvec3(right), 0.0),
		glm::dvec4(glm::dvec3(down), 0.0),
		glm::dvec4(glm::dvec3(forward), 0.0),
		glm::dvec4(glm::dvec3(position), 1.0));
	camera.cameraFromWorld= glm::inverse(camera.markerFromCamera);
	return camera;
}

// Truth pose at a time step: smooth palm motion + smooth articulation, all
// well inside anatomical range so the truth itself is a plausible hand
HandStateEstimator::Pose truthPose(int frame)
{
	const float t= 0.033f * (float)frame;

	HandStateEstimator::Pose pose;
	pose.palmPositionWorld= glm::vec3(0.06f * sinf(0.8f * t), 0.04f * cosf(0.6f * t), 0.02f * sinf(0.5f * t));
	pose.palmOrientationWorld=
		glm::angleAxis(0.5f * sinf(0.7f * t), glm::normalize(glm::vec3(0.2f, 1.f, 0.3f))) *
		glm::angleAxis(0.3f * cosf(0.9f * t), glm::normalize(glm::vec3(1.f, 0.f, 0.2f)));
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const float phase= 0.9f * t + 0.6f * (float)finger;
		pose.rawAngles[finger].lateral= 0.1f * sinf(phase);
		pose.rawAngles[finger].proximal= 0.5f + 0.35f * sinf(phase * 0.7f);
		pose.rawAngles[finger].intermediate= 0.35f + 0.25f * sinf(phase * 1.3f);
		pose.rawAngles[finger].distal= 0.2f + 0.15f * sinf(phase * 1.7f);
	}
	return pose;
}

float pseudoNoise(int index, int axis)
{
	const float seed= sinf((float)(index * 7919 + axis * 104729)) * 43758.5453f;
	return (seed - floorf(seed)) * 2.f - 1.f;
}

// Projects the truth pose into a camera as MediaPipe-style imagePoints
// (z carries nothing the estimator reads)
std::array<glm::vec3, HAND_LANDMARK_COUNT> projectTruth(
	const HandStateEstimator::Pose& pose, const HandSkeleton& skeleton, const TestCamera& camera,
	float noisePx, int noiseSeed)
{
	std::array<glm::vec3, HAND_LANDMARK_COUNT> worldPoints;
	HandStateEstimator::predictWorldLandmarks(pose, skeleton, worldPoints);

	std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints{};
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		const glm::dvec4 camPoint= camera.cameraFromWorld * glm::dvec4(glm::dvec3(worldPoints[i]), 1.0);
		imagePoints[i]= glm::vec3(
			(float)(camera.fx * camPoint.x / camPoint.z + camera.cx) +
				noisePx * pseudoNoise(noiseSeed * HAND_LANDMARK_COUNT + i, 0),
			(float)(camera.fy * camPoint.y / camPoint.z + camera.cy) +
				noisePx * pseudoNoise(noiseSeed * HAND_LANDMARK_COUNT + i, 1),
			0.f);
	}
	return imagePoints;
}

HandStateEstimator::Observation makeObservation(
	int cameraIndex, const TestCamera& camera, double timestampMs,
	const std::array<glm::vec3, HAND_LANDMARK_COUNT>& imagePoints)
{
	HandStateEstimator::Observation observation;
	observation.cameraIndex= cameraIndex;
	observation.timestampMs= timestampMs;
	observation.presence= 0.95f;
	observation.markerFromCamera= camera.markerFromCamera;
	observation.fx= camera.fx;
	observation.fy= camera.fy;
	observation.cx= camera.cx;
	observation.cy= camera.cy;
	observation.imagePoints= &imagePoints;
	return observation;
}

float poseAngleErrorRad(const HandStateEstimator::Pose& a, const HandStateEstimator::Pose& b)
{
	float worst= 0.f;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		worst= std::max(worst, fabsf(a.rawAngles[finger].lateral - b.rawAngles[finger].lateral));
		worst= std::max(worst, fabsf(a.rawAngles[finger].proximal - b.rawAngles[finger].proximal));
		worst= std::max(worst, fabsf(a.rawAngles[finger].intermediate - b.rawAngles[finger].intermediate));
		worst= std::max(worst, fabsf(a.rawAngles[finger].distal - b.rawAngles[finger].distal));
	}
	return worst;
}

// Mean |error| across all 20 angle DoF (the tight tracking metric, where the
// per-frame worst above is a max-statistic)
float poseAngleMeanErrorRad(const HandStateEstimator::Pose& a, const HandStateEstimator::Pose& b)
{
	float sum= 0.f;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		sum+= fabsf(a.rawAngles[finger].lateral - b.rawAngles[finger].lateral);
		sum+= fabsf(a.rawAngles[finger].proximal - b.rawAngles[finger].proximal);
		sum+= fabsf(a.rawAngles[finger].intermediate - b.rawAngles[finger].intermediate);
		sum+= fabsf(a.rawAngles[finger].distal - b.rawAngles[finger].distal);
	}
	return sum / (float)(FINGER_COUNT * 4);
}

float quatErrorRad(const glm::quat& a, const glm::quat& b)
{
	const float dot= std::min(fabsf(glm::dot(a, b)), 1.f);
	return 2.f * acosf(dot);
}
} // namespace

static int runHandEstimatorTest(const TestArgs& args)
{
	int result= 0;
	const HandSkeleton skeleton= makeTruthSkeleton();

	// Two cameras with a healthy stereo baseline (~70 deg parallax at the hand)
	const TestCamera cam0= makeCamera(glm::vec3(0.1f, 0.5f, 0.6f), glm::vec3(0.f));
	const TestCamera cam1= makeCamera(glm::vec3(0.6f, -0.4f, 0.5f), glm::vec3(0.f));

	HandStateEstimatorConfig config;

	// -- (a) Clean two-camera recovery from a perturbed seed -----
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		const HandStateEstimator::Pose truth= truthPose(0);
		HandStateEstimator::Pose seed= truth;
		seed.palmPositionWorld+= glm::vec3(0.03f, -0.02f, 0.04f);
		seed.palmOrientationWorld=
			glm::angleAxis(0.15f, glm::normalize(glm::vec3(1.f, 1.f, 0.f))) * seed.palmOrientationWorld;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			seed.rawAngles[finger].proximal+= 0.2f;

		HandStateEstimator::Pose output;
		HandStateEstimator::UpdateResult update;
		for (int frame= 0; frame < 6; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			const auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
			const auto points1= projectTruth(truth, skeleton, cam1, 0.f, 0);
			const std::vector<HandStateEstimator::Observation> observations= {
				makeObservation(0, cam0, timestampMs, points0),
				makeObservation(1, cam1, timestampMs, points1)};

			update= estimator.update(eHandSide::Right, observations, timestampMs,
									 frame == 0 ? &seed : nullptr, skeleton, output);
			if (!update.bUpdated)
				break;
		}

		const float posErrorMm= glm::length(output.palmPositionWorld - truth.palmPositionWorld) * 1000.f;
		const float rotErrorDeg= glm::degrees(quatErrorRad(output.palmOrientationWorld, truth.palmOrientationWorld));
		const float angleErrorDeg= glm::degrees(poseAngleErrorRad(output, truth));
		MIKAN_LOG_INFO("test-handestimator")
			<< "(a) clean stereo recovery: pos " << posErrorMm << " mm, rot " << rotErrorDeg
			<< " deg, worst angle " << angleErrorDeg << " deg, residual " << update.residualAfterPx << " px";
		if (!update.bUpdated || posErrorMm > 1.f || rotErrorDeg > 0.5f || angleErrorDeg > 1.f ||
			update.residualAfterPx > 0.5f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (a) clean projections of a known hand must be recovered near-exactly";
			result= 1;
		}
	}

	// -- (b) Tracking a moving hand under pixel noise -----
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		constexpr float kNoisePx= 1.5f;
		constexpr int kFrames= 120;
		float worstPosErrorMm= 0.f;
		float worstAngleErrorDeg= 0.f;
		float angleErrorSumDeg= 0.f;
		int scoredFrames= 0;
		int updatedFrames= 0;

		for (int frame= 0; frame < kFrames; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			const HandStateEstimator::Pose truth= truthPose(frame);
			const HandStateEstimator::Pose seed= truthPose(frame);
			const auto points0= projectTruth(truth, skeleton, cam0, kNoisePx, frame * 2);
			const auto points1= projectTruth(truth, skeleton, cam1, kNoisePx, frame * 2 + 1);
			const std::vector<HandStateEstimator::Observation> observations= {
				makeObservation(0, cam0, timestampMs, points0),
				makeObservation(1, cam1, timestampMs, points1)};

			HandStateEstimator::Pose output;
			const HandStateEstimator::UpdateResult update= estimator.update(
				eHandSide::Right, observations, timestampMs, frame == 0 ? &seed : nullptr, skeleton, output);
			if (!update.bUpdated)
				continue;
			++updatedFrames;

			// Let the first frames converge before scoring
			if (frame < 5)
				continue;
			worstPosErrorMm=
				std::max(worstPosErrorMm, glm::length(output.palmPositionWorld - truth.palmPositionWorld) * 1000.f);
			worstAngleErrorDeg= std::max(worstAngleErrorDeg, glm::degrees(poseAngleErrorRad(output, truth)));
			angleErrorSumDeg+= glm::degrees(poseAngleMeanErrorRad(output, truth));
			++scoredFrames;
		}
		const float meanAngleErrorDeg= scoredFrames > 0 ? angleErrorSumDeg / (float)scoredFrames : 0.f;

		// The worst bound is loose on purpose: it is a max-statistic over 20
		// DoF x 100+ frames, and a distal joint's angle is genuinely several
		// degrees uncertain per frame at this pixel noise (1.5 px on a 20 mm
		// bone). The mean is what says tracking is tight.
		MIKAN_LOG_INFO("test-handestimator")
			<< "(b) " << kNoisePx << " px noise, moving hand: worst pos error " << worstPosErrorMm
			<< " mm, worst finger angle error " << worstAngleErrorDeg << " deg (mean "
			<< meanAngleErrorDeg << " deg) over " << updatedFrames << " frames";
		if (updatedFrames < kFrames || worstPosErrorMm > 10.f || worstAngleErrorDeg > 12.f ||
			meanAngleErrorDeg > 3.f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (b) noisy stereo tracking must stay within tolerance of truth";
			result= 1;
		}
	}

	// -- (c) Camera dropout must not step the output -----
	// The classic pipeline pops here: losing the second camera switches tri ->
	// mono and the mono depth error lands as a position step. The estimator
	// just loses measurement rows, so a static hand must stay put.
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		const HandStateEstimator::Pose truth= truthPose(10);
		constexpr float kNoisePx= 1.f;
		glm::vec3 previousPos(0.f);
		bool bHavePrevious= false;
		float worstStepMm= 0.f;
		float worstAngleStepDeg= 0.f;
		HandStateEstimator::Pose previousOutput;

		for (int frame= 0; frame < 45; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			const auto points0= projectTruth(truth, skeleton, cam0, kNoisePx, 500 + frame * 2);
			const auto points1= projectTruth(truth, skeleton, cam1, kNoisePx, 500 + frame * 2 + 1);

			// Frames 15-29: camera 1 stops delivering (stale timestamps get
			// deduped away live; here it simply is not offered)
			std::vector<HandStateEstimator::Observation> observations;
			observations.push_back(makeObservation(0, cam0, timestampMs, points0));
			const bool bCam1Fresh= frame < 15 || frame >= 30;
			if (bCam1Fresh)
				observations.push_back(makeObservation(1, cam1, timestampMs, points1));

			HandStateEstimator::Pose output;
			const HandStateEstimator::UpdateResult update= estimator.update(
				eHandSide::Right, observations, timestampMs, frame == 0 ? &truth : nullptr, skeleton, output);
			if (!update.bUpdated)
			{
				MIKAN_LOG_ERROR("test-handestimator") << "FAILED: (c) update dropped out at frame " << frame;
				result= 1;
				break;
			}

			if (bHavePrevious && frame >= 5)
			{
				worstStepMm= std::max(worstStepMm, glm::length(output.palmPositionWorld - previousPos) * 1000.f);
				worstAngleStepDeg=
					std::max(worstAngleStepDeg, glm::degrees(poseAngleErrorRad(output, previousOutput)));
			}
			previousPos= output.palmPositionWorld;
			previousOutput= output;
			bHavePrevious= true;
		}

		MIKAN_LOG_INFO("test-handestimator")
			<< "(c) camera dropout on a static hand: worst step " << worstStepMm << " mm, worst angle step "
			<< worstAngleStepDeg << " deg";
		if (worstStepMm > 3.f || worstAngleStepDeg > 2.f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (c) losing a camera must not step the output (this is the pop being fixed)";
			result= 1;
		}
	}

	// -- (d) Mono stretch: lateral motion tracks, depth stays leashed -----
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		// Establish stereo state first
		HandStateEstimator::Pose truth= truthPose(20);
		for (int frame= 0; frame < 5; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			const auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
			const auto points1= projectTruth(truth, skeleton, cam1, 0.f, 0);
			const std::vector<HandStateEstimator::Observation> observations= {
				makeObservation(0, cam0, timestampMs, points0),
				makeObservation(1, cam1, timestampMs, points1)};
			HandStateEstimator::Pose output;
			estimator.update(eHandSide::Right, observations, timestampMs,
							 frame == 0 ? &truth : nullptr, skeleton, output);
		}

		// Mono: truth slides laterally (perpendicular to cam0's view ray)
		const glm::vec3 cam0Pos= glm::vec3(glm::dvec3(cam0.markerFromCamera[3]));
		const glm::vec3 viewRay= glm::normalize(truth.palmPositionWorld - cam0Pos);
		const glm::vec3 lateralDir= glm::normalize(glm::cross(viewRay, glm::vec3(0.f, 0.f, 1.f)));

		HandStateEstimator::Pose output;
		for (int frame= 5; frame < 35; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			truth.palmPositionWorld+= lateralDir * 0.002f;
			const auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
			const std::vector<HandStateEstimator::Observation> observations= {
				makeObservation(0, cam0, timestampMs, points0)};
			estimator.update(eHandSide::Right, observations, timestampMs, nullptr, skeleton, output);
		}

		const glm::vec3 error= output.palmPositionWorld - truth.palmPositionWorld;
		const float lateralErrorMm= fabsf(glm::dot(error, lateralDir)) * 1000.f;
		const float depthErrorMm= fabsf(glm::dot(error, viewRay)) * 1000.f;
		MIKAN_LOG_INFO("test-handestimator")
			<< "(d) 60 mm lateral mono slide: lateral error " << lateralErrorMm << " mm, depth error "
			<< depthErrorMm << " mm";
		if (lateralErrorMm > 5.f || depthErrorMm > 30.f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (d) mono tracking must follow lateral motion without losing depth";
			result= 1;
		}
	}

	// -- (e) Determinism: two identical runs are bit-identical -----
	{
		auto runSequence= [&](HandStateEstimator& estimator, HandStateEstimator::Pose& outFinal) {
			estimator.configure(config);
			for (int frame= 0; frame < 30; ++frame)
			{
				const double timestampMs= 1000.0 + kFrameMs * frame;
				const HandStateEstimator::Pose truth= truthPose(frame);
				const HandStateEstimator::Pose seed= truthPose(frame);
				const auto points0= projectTruth(truth, skeleton, cam0, 1.5f, frame * 2);
				const auto points1= projectTruth(truth, skeleton, cam1, 1.5f, frame * 2 + 1);
				const std::vector<HandStateEstimator::Observation> observations= {
					makeObservation(0, cam0, timestampMs, points0),
					makeObservation(1, cam1, timestampMs, points1)};
				estimator.update(eHandSide::Right, observations, timestampMs,
								 frame == 0 ? &seed : nullptr, skeleton, outFinal);
			}
		};

		HandStateEstimator estimatorA;
		HandStateEstimator estimatorB;
		HandStateEstimator::Pose finalA;
		HandStateEstimator::Pose finalB;
		runSequence(estimatorA, finalA);
		runSequence(estimatorB, finalB);

		const bool bIdentical=
			memcmp(&finalA.palmPositionWorld, &finalB.palmPositionWorld, sizeof(glm::vec3)) == 0 &&
			memcmp(&finalA.palmOrientationWorld, &finalB.palmOrientationWorld, sizeof(glm::quat)) == 0 &&
			memcmp(finalA.rawAngles.data(), finalB.rawAngles.data(),
				   sizeof(FingerAngles) * FINGER_COUNT) == 0;
		MIKAN_LOG_INFO("test-handestimator") << "(e) determinism: bit-identical=" << bIdentical;
		if (!bIdentical)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (e) identical inputs must produce bit-identical output (replay contract)";
			result= 1;
		}
	}

	// -- (f) Divergence guard: garbage correspondence drops the state -----
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		const HandStateEstimator::Pose truth= truthPose(0);
		for (int frame= 0; frame < 5; ++frame)
		{
			const double timestampMs= 1000.0 + kFrameMs * frame;
			const auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
			const auto points1= projectTruth(truth, skeleton, cam1, 0.f, 0);
			const std::vector<HandStateEstimator::Observation> observations= {
				makeObservation(0, cam0, timestampMs, points0),
				makeObservation(1, cam1, timestampMs, points1)};
			HandStateEstimator::Pose output;
			estimator.update(eHandSide::Right, observations, timestampMs,
							 frame == 0 ? &truth : nullptr, skeleton, output);
		}

		// Both cameras suddenly report a hand hundreds of px away (a wrong
		// cluster pairing). The guard must HOLD the previous state through the
		// bad fits (streaming the compromise would be the pop) and only drop
		// the state once the streak persists.
		auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
		auto points1= projectTruth(truth, skeleton, cam1, 0.f, 0);
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			points0[i]+= glm::vec3(400.f, 200.f, 0.f);
			points1[i]+= glm::vec3(-350.f, 250.f, 0.f);
		}

		HandStateEstimator::Pose output;
		HandStateEstimator::UpdateResult update;
		bool bHeldThroughStreak= true;
		int garbageFuse= 0;
		for (; garbageFuse < 5; ++garbageFuse)
		{
			const double garbageTimestampMs= 1000.0 + kFrameMs * (5 + garbageFuse);
			const std::vector<HandStateEstimator::Observation> garbage= {
				makeObservation(0, cam0, garbageTimestampMs, points0),
				makeObservation(1, cam1, garbageTimestampMs, points1)};
			update= estimator.update(eHandSide::Right, garbage, garbageTimestampMs, nullptr, skeleton,
									 output);
			if (!update.bUpdated)
				break;
			// While holding, the streamed pose must stay the pre-garbage state
			bHeldThroughStreak&= update.bHeldBadFit &&
				glm::length(output.palmPositionWorld - truth.palmPositionWorld) < 0.02f;
		}

		const bool bDropped= !update.bUpdated && !estimator.hasState(eHandSide::Right);
		MIKAN_LOG_INFO("test-handestimator")
			<< "(f) garbage correspondence: held " << garbageFuse << " fuses then dropped="
			<< bDropped << " (residual " << update.residualAfterPx << " px), held cleanly="
			<< bHeldThroughStreak;
		if (!bDropped || !bHeldThroughStreak || garbageFuse < 2)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (f) bad fits must hold the previous state, then a sustained streak "
				   "must drop it - never stream the compromise";
			result= 1;
		}

		// ...and a good seed on the next fuse recovers
		const double recoveryTimestampMs= 1000.0 + kFrameMs * 11;
		const auto goodPoints0= projectTruth(truth, skeleton, cam0, 0.f, 0);
		const auto goodPoints1= projectTruth(truth, skeleton, cam1, 0.f, 0);
		const std::vector<HandStateEstimator::Observation> good= {
			makeObservation(0, cam0, recoveryTimestampMs, goodPoints0),
			makeObservation(1, cam1, recoveryTimestampMs, goodPoints1)};
		const HandStateEstimator::UpdateResult recovery=
			estimator.update(eHandSide::Right, good, recoveryTimestampMs, &truth, skeleton, output);
		MIKAN_LOG_INFO("test-handestimator")
			<< "(f) reseed after drop: reseeded=" << recovery.bReseeded << " updated=" << recovery.bUpdated;
		if (!recovery.bReseeded || !recovery.bUpdated)
		{
			MIKAN_LOG_ERROR("test-handestimator") << "FAILED: (f) a good seed must recover the state";
			result= 1;
		}
	}

	// -- (g) Re-presenting an unchanged observation must not move the state,
	// and one aged past the measurement window must be excluded (hold) -----
	{
		HandStateEstimator estimator;
		estimator.configure(config);

		const HandStateEstimator::Pose truth= truthPose(0);
		const auto points0= projectTruth(truth, skeleton, cam0, 0.f, 0);
		const std::vector<HandStateEstimator::Observation> observations= {
			makeObservation(0, cam0, 1000.0, points0)};
		HandStateEstimator::Pose first;
		estimator.update(eHandSide::Right, observations, 1000.0, &truth, skeleton, first);

		// Same observation re-fit (fusion re-presents mirrors whenever any
		// camera delivers): the converged state must stay converged
		HandStateEstimator::Pose refit;
		const HandStateEstimator::UpdateResult refitUpdate=
			estimator.update(eHandSide::Right, observations, 1033.0, nullptr, skeleton, refit);
		const float refitStepMm= glm::length(refit.palmPositionWorld - first.palmPositionWorld) * 1000.f;

		// Aged past the window: no rows, pose held bit-exactly
		HandStateEstimator::Pose held;
		const HandStateEstimator::UpdateResult hold=
			estimator.update(eHandSide::Right, observations, 1000.0 + 80.0, nullptr, skeleton, held);
		const bool bHeld= hold.bUpdated && hold.cameraCount == 0 &&
			memcmp(&held.palmPositionWorld, &refit.palmPositionWorld, sizeof(glm::vec3)) == 0;

		MIKAN_LOG_INFO("test-handestimator")
			<< "(g) unchanged re-fit step " << refitStepMm << " mm (cameras "
			<< refitUpdate.cameraCount << "), aged-out held=" << bHeld;
		if (!refitUpdate.bUpdated || refitUpdate.cameraCount != 1 || refitStepMm > 0.01f || !bHeld)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (g) an unchanged observation must not move a converged state, and an "
				   "aged-out one must hold the pose";
			result= 1;
		}
	}

	// -- (h) Mono depth-bias drift must be resisted, not followed -----
	// During a mono stretch the only depth signal is apparent scale, which
	// carries per-camera bias. Simulated here as the observed hand slowly
	// shrinking (truth sliding away along the view ray): the anisotropic
	// prior must hold the stereo-established depth so the second camera's
	// return does not snap the palm back (the measured 106 mm pop). The
	// isotropic run is the negative control proving the scenario drifts.
	{
		auto runDriftSequence= [&](float monoDepthPriorFactor) {
			float driftDepthMm= 0.f;
			HandStateEstimatorConfig driftConfig= config;
			driftConfig.monoDepthPriorFactor= monoDepthPriorFactor;
			HandStateEstimator estimator;
			estimator.configure(driftConfig);

			const HandStateEstimator::Pose truth= truthPose(30);
			const glm::vec3 cam0Pos= glm::vec3(glm::dvec3(cam0.markerFromCamera[3]));
			const glm::vec3 viewRay= glm::normalize(truth.palmPositionWorld - cam0Pos);

			HandStateEstimator::Pose output;
			for (int frame= 0; frame < 35; ++frame)
			{
				const double timestampMs= 1000.0 + kFrameMs * frame;
				HandStateEstimator::Pose observed= truth;
				// Frames 5-29: cam0 only, its depth signal drifting away
				// along the ray by 2 mm per frame (40 mm total)
				const bool bMono= frame >= 5 && frame < 30;
				if (bMono)
					observed.palmPositionWorld+=
						viewRay * (0.002f * (float)(std::min(frame, 29) - 4));

				const auto points0= projectTruth(observed, skeleton, cam0, 0.f, 0);
				const auto points1= projectTruth(truth, skeleton, cam1, 0.f, 0);
				std::vector<HandStateEstimator::Observation> observations;
				observations.push_back(makeObservation(0, cam0, timestampMs, points0));
				if (!bMono)
					observations.push_back(makeObservation(1, cam1, timestampMs, points1));

				estimator.update(eHandSide::Right, observations, timestampMs,
								 frame == 0 ? &truth : nullptr, skeleton, output);
				if (frame == 29)
					driftDepthMm= fabsf(glm::dot(output.palmPositionWorld - truth.palmPositionWorld,
												 viewRay)) * 1000.f;
			}
			// The depth error at the end of the mono stretch IS the pending
			// snap size when the second camera returns
			return driftDepthMm;
		};

		const float driftAnisoMm= runDriftSequence(0.2f);
		const float driftIsoMm= runDriftSequence(1.f);
		MIKAN_LOG_INFO("test-handestimator")
			<< "(h) 40 mm mono scale-bias drift: depth error " << driftAnisoMm
			<< " mm anisotropic vs " << driftIsoMm << " mm isotropic";
		if (driftAnisoMm > 10.f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (h) the anisotropic prior must hold the stereo depth through mono bias";
			result= 1;
		}
		if (driftIsoMm < 20.f)
		{
			MIKAN_LOG_ERROR("test-handestimator")
				<< "FAILED: (h) negative control - the isotropic prior was expected to drift, so "
				   "this scenario no longer exercises the anisotropic hold";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-handestimator") << "All hand-estimator checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-handestimator",
					"Angle-space multi-view hand state estimator",
					eTestCategory::SelfTest, runHandEstimatorTest);
