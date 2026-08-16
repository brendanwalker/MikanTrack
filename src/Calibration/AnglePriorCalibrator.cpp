#include "AnglePriorCalibrator.h"

#include <cmath>

namespace
{
constexpr int kN= AnglePriorCalibrator::k_angleCount;

// In-place inverse of a symmetric positive-definite matrix via Cholesky
// (A = L L^T, invert L, A^-1 = L^-T L^-1). False when not positive definite.
bool invertSpd(std::array<double, kN * kN>& a)
{
	// Cholesky: lower triangle of a becomes L
	for (int i= 0; i < kN; ++i)
	{
		for (int j= 0; j <= i; ++j)
		{
			double sum= a[i * kN + j];
			for (int k= 0; k < j; ++k)
				sum-= a[i * kN + k] * a[j * kN + k];
			if (i == j)
			{
				if (sum <= 0.0)
					return false;
				a[i * kN + i]= sqrt(sum);
			}
			else
			{
				a[i * kN + j]= sum / a[j * kN + j];
			}
		}
	}

	// Invert L in place (lower triangular)
	for (int i= 0; i < kN; ++i)
	{
		a[i * kN + i]= 1.0 / a[i * kN + i];
		for (int j= i + 1; j < kN; ++j)
		{
			double sum= 0.0;
			for (int k= i; k < j; ++k)
				sum-= a[j * kN + k] * a[k * kN + i];
			a[j * kN + i]= sum / a[j * kN + j];
		}
	}

	// A^-1 = L^-T L^-1 (symmetric; fill both triangles)
	std::array<double, kN * kN> inverse{};
	for (int i= 0; i < kN; ++i)
	{
		for (int j= 0; j <= i; ++j)
		{
			double sum= 0.0;
			for (int k= i; k < kN; ++k)
				sum+= a[k * kN + i] * a[k * kN + j];
			inverse[i * kN + j]= sum;
			inverse[j * kN + i]= sum;
		}
	}
	a= inverse;
	return true;
}
} // namespace

void AnglePriorCalibrator::reset()
{
	m_samples[0].clear();
	m_samples[1].clear();
}

void AnglePriorCalibrator::addSample(eHandSide side,
									 const std::array<FingerAngles, FINGER_COUNT>& rawAngles)
{
	std::array<float, k_angleCount> flat{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		flat[finger * 4 + 0]= rawAngles[finger].lateral;
		flat[finger * 4 + 1]= rawAngles[finger].proximal;
		flat[finger * 4 + 2]= rawAngles[finger].intermediate;
		flat[finger * 4 + 3]= rawAngles[finger].distal;
	}
	for (float value : flat)
	{
		if (!std::isfinite(value))
			return;
	}
	m_samples[(int)side].push_back(flat);
}

bool AnglePriorCalibrator::finish(eHandSide side, Prior& outPrior) const
{
	const std::vector<std::array<float, k_angleCount>>& samples= m_samples[(int)side];
	const int count= (int)samples.size();
	if (count < k_minSamples)
		return false;

	std::array<double, kN> mean{};
	for (const std::array<float, k_angleCount>& sample : samples)
		for (int i= 0; i < kN; ++i)
			mean[i]+= sample[i];
	for (int i= 0; i < kN; ++i)
		mean[i]/= (double)count;

	std::array<double, kN * kN> covariance{};
	for (const std::array<float, k_angleCount>& sample : samples)
	{
		for (int i= 0; i < kN; ++i)
		{
			const double di= sample[i] - mean[i];
			for (int j= 0; j <= i; ++j)
				covariance[i * kN + j]+= di * (sample[j] - mean[j]);
		}
	}
	for (int i= 0; i < kN; ++i)
		for (int j= 0; j <= i; ++j)
		{
			covariance[i * kN + j]/= (double)(count - 1);
			covariance[j * kN + i]= covariance[i * kN + j];
		}

	// Shrink correlations toward the diagonal, keep measured variances, and
	// floor the variances so an unexercised DoF stays soft
	for (int i= 0; i < kN; ++i)
	{
		for (int j= 0; j < kN; ++j)
		{
			if (i != j)
				covariance[i * kN + j]*= 1.0 - (double)k_shrinkage;
		}
		covariance[i * kN + i]= std::max(covariance[i * kN + i], (double)k_varianceFloorRadSq);
	}

	if (!invertSpd(covariance))
		return false;

	for (int i= 0; i < kN; ++i)
	{
		outPrior.mean[i]= (float)mean[i];
		for (int j= 0; j < kN; ++j)
			outPrior.precision[i * kN + j]= (float)covariance[i * kN + j];
	}
	return true;
}
