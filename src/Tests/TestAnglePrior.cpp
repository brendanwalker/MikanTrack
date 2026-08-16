#include "TestCommon.h"

// Self test for the fitted finger-angle prior: the calibrator has to recover
// the mean and the CORRELATION STRUCTURE of a known distribution, because the
// structure is the entire value - it is what makes a coordinated pose change
// cheap and an uncorrelated single-DoF snap expensive under the Mahalanobis
// metric the estimator applies.

namespace
{
constexpr int kAngles= AnglePriorCalibrator::k_angleCount;

float pseudoNoise(int index, int axis)
{
	const float seed= sinf((float)(index * 7919 + axis * 104729)) * 43758.5453f;
	return (seed - floorf(seed)) * 2.f - 1.f;
}

// Ground-truth generative model: two latent "synergy" factors (a whole-hand
// curl and a splay) plus small independent noise - the low-dimensional
// structure real hand poses actually have
std::array<FingerAngles, FINGER_COUNT> sampleFromModel(int sampleIndex,
													   const std::array<float, kAngles>& mean)
{
	const float curl= 0.5f * pseudoNoise(sampleIndex, 0);
	const float splay= 0.2f * pseudoNoise(sampleIndex, 1);

	std::array<FingerAngles, FINGER_COUNT> angles{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const float jitter= 0.03f;
		angles[finger].lateral= mean[finger * 4 + 0] + splay * (float)(finger - 2) * 0.5f +
			jitter * pseudoNoise(sampleIndex, 10 + finger * 4);
		angles[finger].proximal= mean[finger * 4 + 1] + curl +
			jitter * pseudoNoise(sampleIndex, 11 + finger * 4);
		angles[finger].intermediate= mean[finger * 4 + 2] + curl * 0.9f +
			jitter * pseudoNoise(sampleIndex, 12 + finger * 4);
		angles[finger].distal= mean[finger * 4 + 3] + curl * 0.6f +
			jitter * pseudoNoise(sampleIndex, 13 + finger * 4);
	}
	return angles;
}

float mahalanobis(const AnglePriorCalibrator::Prior& prior, const std::array<float, kAngles>& x)
{
	std::array<float, kAngles> d;
	for (int i= 0; i < kAngles; ++i)
		d[i]= x[i] - prior.mean[i];
	float sum= 0.f;
	for (int i= 0; i < kAngles; ++i)
	{
		float row= 0.f;
		for (int j= 0; j < kAngles; ++j)
			row+= prior.precision[i * kAngles + j] * d[j];
		sum+= d[i] * row;
	}
	return sum;
}

std::array<float, kAngles> flatten(const std::array<FingerAngles, FINGER_COUNT>& angles)
{
	std::array<float, kAngles> flat{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		flat[finger * 4 + 0]= angles[finger].lateral;
		flat[finger * 4 + 1]= angles[finger].proximal;
		flat[finger * 4 + 2]= angles[finger].intermediate;
		flat[finger * 4 + 3]= angles[finger].distal;
	}
	return flat;
}
} // namespace

static int runAnglePriorTest(const TestArgs& args)
{
	int result= 0;

	std::array<float, kAngles> trueMean{};
	for (int i= 0; i < kAngles; ++i)
		trueMean[i]= 0.3f + 0.05f * (float)(i % 4);

	// -- (a) Mean recovery + correlation structure -----
	{
		AnglePriorCalibrator calibrator;
		constexpr int kSamples= 1500;
		for (int sample= 0; sample < kSamples; ++sample)
			calibrator.addSample(eHandSide::Right, sampleFromModel(sample, trueMean));

		AnglePriorCalibrator::Prior prior;
		if (!calibrator.finish(eHandSide::Right, prior))
		{
			MIKAN_LOG_ERROR("test-angleprior") << "FAILED: (a) finish rejected a full sample set";
			return 1;
		}

		float meanErrorRad= 0.f;
		for (int i= 0; i < kAngles; ++i)
			meanErrorRad= std::max(meanErrorRad, fabsf(prior.mean[i] - trueMean[i]));

		// Typical training sample lands near the chi-squared expectation
		float mahalanobisSum= 0.f;
		for (int sample= 0; sample < kSamples; ++sample)
			mahalanobisSum+= mahalanobis(prior, flatten(sampleFromModel(sample, trueMean)));
		const float mahalanobisMean= mahalanobisSum / (float)kSamples;

		// The discriminating property: a coordinated whole-hand curl of some
		// size is CHEAP (it is how the training data moves), an uncorrelated
		// single-DoF excursion of the same size is EXPENSIVE
		std::array<FingerAngles, FINGER_COUNT> coordinated{};
		std::array<FingerAngles, FINGER_COUNT> snap{};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			coordinated[finger].lateral= trueMean[finger * 4 + 0];
			coordinated[finger].proximal= trueMean[finger * 4 + 1] + 0.3f;
			coordinated[finger].intermediate= trueMean[finger * 4 + 2] + 0.27f;
			coordinated[finger].distal= trueMean[finger * 4 + 3] + 0.18f;
			snap[finger].lateral= trueMean[finger * 4 + 0];
			snap[finger].proximal= trueMean[finger * 4 + 1];
			snap[finger].intermediate= trueMean[finger * 4 + 2];
			snap[finger].distal= trueMean[finger * 4 + 3];
		}
		// Same total excursion magnitude, concentrated in one DoF
		snap[(int)eFinger::Index].proximal+= 0.3f;

		const float coordinatedCost= mahalanobis(prior, flatten(coordinated));
		const float snapCost= mahalanobis(prior, flatten(snap));

		MIKAN_LOG_INFO("test-angleprior")
			<< "(a) mean error " << glm::degrees(meanErrorRad) << " deg, sample Mahalanobis^2 mean "
			<< mahalanobisMean << " (dim " << kAngles << "), coordinated-curl cost " << coordinatedCost
			<< " vs single-DoF-snap cost " << snapCost;
		if (meanErrorRad > 0.02f)
		{
			MIKAN_LOG_ERROR("test-angleprior") << "FAILED: (a) mean must be recovered";
			result= 1;
		}
		// Unshrunk, the training mean would sit at exactly dim x (n-1)/n (the
		// trace identity). Shrinkage deliberately inflates the low-variance
		// directions of this strongly-correlated data, so the mean lands well
		// BELOW the dimension - the bound only catches a broken fit (near
		// zero = degenerate precision, far above = variance collapse).
		if (mahalanobisMean < (float)kAngles * 0.05f || mahalanobisMean > (float)kAngles * 2.f)
		{
			MIKAN_LOG_ERROR("test-angleprior")
				<< "FAILED: (a) training-sample Mahalanobis is outside any sane fit's range";
			result= 1;
		}
		if (snapCost < 3.f * coordinatedCost)
		{
			MIKAN_LOG_ERROR("test-angleprior")
				<< "FAILED: (a) an uncorrelated snap must cost several times a coordinated "
				   "move of the same size - without that the prior discriminates nothing";
			result= 1;
		}
	}

	// -- (b) Minimum-sample gate -----
	{
		AnglePriorCalibrator calibrator;
		for (int sample= 0; sample < AnglePriorCalibrator::k_minSamples - 1; ++sample)
			calibrator.addSample(eHandSide::Left, sampleFromModel(sample, trueMean));
		AnglePriorCalibrator::Prior prior;
		const bool bAcceptedShort= calibrator.finish(eHandSide::Left, prior);
		calibrator.addSample(eHandSide::Left, sampleFromModel(9999, trueMean));
		const bool bAcceptedFull= calibrator.finish(eHandSide::Left, prior);

		MIKAN_LOG_INFO("test-angleprior")
			<< "(b) sample gate: short rejected=" << !bAcceptedShort << ", full accepted=" << bAcceptedFull;
		if (bAcceptedShort || !bAcceptedFull)
		{
			MIKAN_LOG_ERROR("test-angleprior") << "FAILED: (b) the minimum-sample gate is not holding";
			result= 1;
		}
	}

	// -- (c) Variance floor: an unexercised DoF must stay soft -----
	{
		AnglePriorCalibrator calibrator;
		for (int sample= 0; sample < 800; ++sample)
		{
			std::array<FingerAngles, FINGER_COUNT> angles= sampleFromModel(sample, trueMean);
			// The pinky distal never moves in this training set
			angles[(int)eFinger::Pinky].distal= trueMean[(int)eFinger::Pinky * 4 + 3];
			calibrator.addSample(eHandSide::Right, angles);
		}
		AnglePriorCalibrator::Prior prior;
		if (!calibrator.finish(eHandSide::Right, prior))
		{
			MIKAN_LOG_ERROR("test-angleprior") << "FAILED: (c) finish rejected the sample set";
			return 1;
		}

		// Marginal cost of moving ONLY that DoF by 0.1 rad; the floor bounds
		// the DIAGONAL precision, and the conditional term can only add a
		// bounded amount on top, so a generous ceiling is the honest assert
		std::array<FingerAngles, FINGER_COUNT> probe{};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			probe[finger].lateral= trueMean[finger * 4 + 0];
			probe[finger].proximal= trueMean[finger * 4 + 1];
			probe[finger].intermediate= trueMean[finger * 4 + 2];
			probe[finger].distal= trueMean[finger * 4 + 3];
		}
		probe[(int)eFinger::Pinky].distal+= 0.1f;
		const float floorCost= mahalanobis(prior, flatten(probe));
		const float unflooredCost= 0.1f * 0.1f / 1e-8f; // what a collapsed variance would charge

		MIKAN_LOG_INFO("test-angleprior")
			<< "(c) unexercised-DoF probe cost " << floorCost << " (collapsed variance would be ~"
			<< unflooredCost << ")";
		if (floorCost > 30.f)
		{
			MIKAN_LOG_ERROR("test-angleprior")
				<< "FAILED: (c) the variance floor must keep an unexercised DoF soft";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-angleprior") << "All angle-prior checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-angleprior",
					"Finger-angle prior fitting (mean, correlations, floors)",
					eTestCategory::SelfTest, runAnglePriorTest);
