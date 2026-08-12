#include "TestCommon.h"

namespace
{
// The hand the synthetic samples are generated from. Proportions are the ones
// measured from stereo on a real hand, not the landmark model's.
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

// One 21-landmark set: forward kinematics through the skeleton, plus the wrist
// reconstruction the PnP object model also relies on (palm +X is wrist ->
// middle MCP about the palm center, so the wrist mirrors that base)
std::array<glm::vec3, HAND_LANDMARK_COUNT> buildLandmarks(
	const glm::mat4& palmTransform,
	const HandSkeleton& skeleton,
	const std::array<FingerAngles, FINGER_COUNT>& angles)
{
	std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
	HandPoseModel::buildFingerJoints(palmTransform, skeleton, angles, joints);

	std::array<glm::vec3, HAND_LANDMARK_COUNT> points{};
	const glm::vec3 wristInPalm(-skeleton.baseInPalm[(int)eFinger::Middle].x, 0.f, 0.f);
	points[(int)eHandLandmark::WRIST]= glm::vec3(palmTransform * glm::vec4(wristInPalm, 1.f));
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int joint= 0; joint < 4; ++joint)
			points[FINGER_JOINTS[finger][joint]]= joints[finger][joint];
	return points;
}

// A spread of poses and palm placements, so no bone stays aligned with one
// direction - the same thing the live capture asks the user to do
std::array<FingerAngles, FINGER_COUNT> poseForSample(int sampleIndex)
{
	std::array<FingerAngles, FINGER_COUNT> angles{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const float phase= 0.37f * (float)sampleIndex + 0.6f * (float)finger;
		angles[finger].lateral= 0.12f * sinf(phase);
		angles[finger].proximal= 0.45f + 0.35f * sinf(phase * 0.7f);
		angles[finger].intermediate= 0.35f + 0.3f * sinf(phase * 1.3f);
		angles[finger].distal= 0.2f + 0.15f * sinf(phase * 1.7f);
	}
	return angles;
}

glm::mat4 palmForSample(int sampleIndex)
{
	const float t= 0.21f * (float)sampleIndex;
	const glm::quat rotation= glm::angleAxis(0.9f * sinf(t), glm::normalize(glm::vec3(0.3f, 1.f, 0.2f))) *
		glm::angleAxis(0.6f * cosf(t * 0.8f), glm::normalize(glm::vec3(1.f, 0.1f, 0.4f)));
	glm::mat4 transform= glm::mat4_cast(rotation);
	transform[3]= glm::vec4(0.1f * sinf(t), 0.05f * cosf(t), 0.7f + 0.1f * sinf(t * 0.5f), 1.f);
	return transform;
}

// Deterministic pseudo-noise: Date/random are unavailable in this codebase's
// offline paths and a fixed sequence keeps the test reproducible
float pseudoNoise(int index, int axis)
{
	const float seed= sinf((float)(index * 7919 + axis * 104729)) * 43758.5453f;
	return (seed - floorf(seed)) * 2.f - 1.f;
}

float maxPhalanxError(const HandSkeleton& recovered, const HandSkeleton& truth)
{
	float worst= 0.f;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			worst= std::max(worst, fabsf(recovered.phalanxLengths[finger][phalanx] -
										 truth.phalanxLengths[finger][phalanx]));
	return worst;
}
} // namespace

static int runBoneCalibrationTest(const TestArgs& args)
{
	int result= 0;
	const HandSkeleton truth= makeTruthSkeleton();
	constexpr int kSampleCount= 200;

	// -- (a) Noise-free recovery -----
	{
		HandBoneCalibrator calibrator;
		for (int sample= 0; sample < kSampleCount; ++sample)
		{
			calibrator.addSample(eHandSide::Right,
								 buildLandmarks(palmForSample(sample), truth, poseForSample(sample)));
		}

		HandSkeleton recovered;
		HandBoneCalibrator::Quality quality;
		if (!calibrator.finish(eHandSide::Right, recovered, quality))
		{
			MIKAN_LOG_ERROR("test-bonecalib") << "FAILED: (a) finish rejected a full sample set";
			return 1;
		}

		const float boneError= maxPhalanxError(recovered, truth);
		float baseError= 0.f;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			baseError= std::max(baseError, glm::length(recovered.baseInPalm[finger] - truth.baseInPalm[finger]));

		MIKAN_LOG_INFO("test-bonecalib")
			<< "(a) noise-free: max phalanx error " << boneError * 1000.f << " mm, max base error "
			<< baseError * 1000.f << " mm over " << quality.sampleCount << " samples";
		if (boneError > 1e-5f || baseError > 1e-5f)
		{
			MIKAN_LOG_ERROR("test-bonecalib")
				<< "FAILED: (a) a noise-free hand must recover its own skeleton exactly";
			result= 1;
		}
	}

	// -- (b) Recovery under landmark noise -----
	// Documents the bias as much as it bounds it: independent endpoint error
	// can only lengthen a segment, so measured bones read slightly long.
	{
		constexpr float kNoiseMeters= 0.004f;
		HandBoneCalibrator calibrator;
		for (int sample= 0; sample < kSampleCount; ++sample)
		{
			std::array<glm::vec3, HAND_LANDMARK_COUNT> points=
				buildLandmarks(palmForSample(sample), truth, poseForSample(sample));
			for (int landmark= 0; landmark < HAND_LANDMARK_COUNT; ++landmark)
			{
				const int index= sample * HAND_LANDMARK_COUNT + landmark;
				points[landmark]+= kNoiseMeters * glm::vec3(pseudoNoise(index, 0), pseudoNoise(index, 1),
															pseudoNoise(index, 2));
			}
			calibrator.addSample(eHandSide::Right, points);
		}

		HandSkeleton recovered;
		HandBoneCalibrator::Quality quality;
		if (!calibrator.finish(eHandSide::Right, recovered, quality))
		{
			MIKAN_LOG_ERROR("test-bonecalib") << "FAILED: (b) finish rejected a full sample set";
			return 1;
		}

		float lengthBias= 0.f;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				lengthBias+= recovered.phalanxLengths[finger][phalanx] - truth.phalanxLengths[finger][phalanx];
		lengthBias/= (float)(FINGER_COUNT * 3);

		const float boneError= maxPhalanxError(recovered, truth);
		MIKAN_LOG_INFO("test-bonecalib")
			<< "(b) " << kNoiseMeters * 1000.f << " mm landmark noise: max phalanx error "
			<< boneError * 1000.f << " mm, mean length bias " << lengthBias * 1000.f
			<< " mm, worst spread " << quality.worstPhalanxSpread * 1000.f << " mm";
		if (boneError > 0.004f)
		{
			MIKAN_LOG_ERROR("test-bonecalib")
				<< "FAILED: (b) medians must stay within a few mm of truth under noise";
			result= 1;
		}
		if (lengthBias < 0.f)
		{
			MIKAN_LOG_ERROR("test-bonecalib")
				<< "FAILED: (b) endpoint noise can only lengthen bones - a negative bias means "
				   "the sample generator, not the estimator, is being measured";
			result= 1;
		}
	}

	// -- (c) The object model PnP builds from a calibrated skeleton -----
	// LandmarkTo3D rebuilds its 21 object points exactly this way, so the
	// reconstruction has to reproduce the geometry it was built from.
	{
		const std::array<FingerAngles, FINGER_COUNT> angles= poseForSample(3);
		const std::array<glm::vec3, HAND_LANDMARK_COUNT> points=
			buildLandmarks(glm::mat4(1.f), truth, angles);

		HandSkeleton rebuilt;
		HandPoseModel::computeSkeleton(points, eHandSide::Right, rebuilt);

		const float boneError= maxPhalanxError(rebuilt, truth);
		const float referenceBone= glm::length(points[(int)eHandLandmark::MIDDLE_MCP] -
											   points[(int)eHandLandmark::WRIST]);
		const float expectedBone= truth.baseInPalm[(int)eFinger::Middle].x * 2.f;

		MIKAN_LOG_INFO("test-bonecalib")
			<< "(c) FK object model: max phalanx error " << boneError * 1000.f
			<< " mm, wrist->middle-MCP " << referenceBone * 1000.f << " mm (expected "
			<< expectedBone * 1000.f << " mm)";
		if (boneError > 1e-5f || fabsf(referenceBone - expectedBone) > 1e-5f)
		{
			MIKAN_LOG_ERROR("test-bonecalib")
				<< "FAILED: (c) the FK-built landmark set must carry the calibrated geometry";
			result= 1;
		}
	}

	// -- (d) The minimum-sample gate -----
	{
		HandBoneCalibrator calibrator;
		for (int sample= 0; sample < HandBoneCalibrator::k_minSamples - 1; ++sample)
		{
			calibrator.addSample(eHandSide::Left,
								 buildLandmarks(palmForSample(sample), truth, poseForSample(sample)));
		}

		HandSkeleton recovered;
		HandBoneCalibrator::Quality quality;
		const bool bAcceptedShort= calibrator.finish(eHandSide::Left, recovered, quality);

		calibrator.addSample(eHandSide::Left, buildLandmarks(palmForSample(99), truth, poseForSample(99)));
		const bool bAcceptedFull= calibrator.finish(eHandSide::Left, recovered, quality);

		MIKAN_LOG_INFO("test-bonecalib")
			<< "(d) sample gate: " << (HandBoneCalibrator::k_minSamples - 1) << " rejected="
			<< !bAcceptedShort << ", " << HandBoneCalibrator::k_minSamples << " accepted=" << bAcceptedFull;
		if (bAcceptedShort || !bAcceptedFull)
		{
			MIKAN_LOG_ERROR("test-bonecalib") << "FAILED: (d) the minimum-sample gate is not holding";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-bonecalib") << "All bone-calibration checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-bonecalib",
					"Hand bone calibration from triangulated landmarks",
					eTestCategory::SelfTest, runBoneCalibrationTest);
