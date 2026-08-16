#pragma once

#include <array>
#include <vector>

#include "TrackingTypes.h"

// Fits a Gaussian prior over the 20-dim RAW finger-angle vector from a user's
// own tracked data: hand poses live in a much lower-dimensional subspace than
// 20 independent DoF (finger joints are tendon-coupled and habitually
// coordinated), and a covariance fitted from real sessions captures exactly
// the correlations - DIP follows PIP, neighboring fingers move together -
// that let the estimator tell a coordinated pose change from a single-DoF
// landmark snap.
//
// Samples must be RAW angles (flat-hand zero, pre rest-offset), from
// stereo-quality frames only: monocular articulation carries per-camera bias
// that would smear the fitted correlations. Per side - lateral's sign
// convention is chirality-geometric, so the sides are not mirror images.
//
// The output is the PRECISION matrix (inverse covariance), ready for a
// Mahalanobis residual. Two regularizers keep it sane:
//   - shrinkage toward the diagonal (correlations from finite noisy data are
//     overconfident; per-DoF variances are kept as measured)
//   - a per-DoF variance floor, so a DoF the capture sessions never exercised
//     cannot become rigidly pinned to its mean
//
// THREAD AFFINITY: none of its own (offline tool / test use).
class AnglePriorCalibrator
{
public:
	static constexpr int k_angleCount= FINGER_COUNT * 4;
	// Below this a fit is refused: a covariance over 20 DoF needs real data
	// or its correlations are noise
	static constexpr int k_minSamples= 300;
	// Shrinkage toward the diagonal
	static constexpr float k_shrinkage= 0.1f;
	// Variance floor per DoF (radians squared): ~2 degrees of standard
	// deviation, so no DoF is ever pinned harder than that
	static constexpr float k_varianceFloorRadSq= 0.0012f;

	struct Prior
	{
		std::array<float, k_angleCount> mean{};
		// Row-major inverse covariance (symmetric positive definite)
		std::array<float, k_angleCount * k_angleCount> precision{};
	};

	void reset();
	void addSample(eHandSide side, const std::array<FingerAngles, FINGER_COUNT>& rawAngles);
	int getSampleCount(eHandSide side) const { return (int)m_samples[(int)side].size(); }

	// Mean + shrinkage-regularized precision over the samples. False below
	// k_minSamples or if the regularized covariance fails to invert.
	bool finish(eHandSide side, Prior& outPrior) const;

private:
	std::array<std::vector<std::array<float, k_angleCount>>, 2> m_samples;
};
