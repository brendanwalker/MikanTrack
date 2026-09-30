#include "ImuBiasCalibrator.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

static constexpr double k_calibrationSeconds= 4.0;
// Rest gates: a controller reading faster than this, or whose specific force
// strays from 1g by more than this, is being handled
static constexpr float k_restRateRadiansPerSecond= 0.15f; // ~8.6 deg/s
static constexpr float k_restAccelTolerance= 0.6f;        // m/s^2 around 1g
static constexpr float k_gravityMetersPerSecond2= 9.80665f;

void ImuBiasCalibrator::begin()
{
	m_bRunning= true;
	m_bDisturbed= false;
	m_sum= glm::dvec3(0.0);
	m_sampleCount= 0;
	m_seconds= 0.0;
}

void ImuBiasCalibrator::cancel()
{
	m_bRunning= false;
}

bool ImuBiasCalibrator::addSample(const ImuSample& sample, float dtSeconds, glm::vec3& outBias)
{
	if (!m_bRunning)
		return false;

	const float rateMagnitude= glm::length(sample.angularVelocity);
	const float accelMagnitude= glm::length(sample.acceleration);
	const bool bAtRest= rateMagnitude < k_restRateRadiansPerSecond &&
						fabsf(accelMagnitude - k_gravityMetersPerSecond2) < k_restAccelTolerance;
	if (!bAtRest)
	{
		// Start over rather than average in motion - a single nudge would
		// otherwise be baked into the bias permanently
		m_bDisturbed= true;
		m_sum= glm::dvec3(0.0);
		m_sampleCount= 0;
		m_seconds= 0.0;
		return false;
	}

	m_sum+= glm::dvec3(sample.angularVelocity);
	m_sampleCount++;
	m_seconds+= dtSeconds;

	if (m_seconds < k_calibrationSeconds || m_sampleCount <= 0)
		return false;

	outBias= glm::vec3(m_sum / (double)m_sampleCount);
	m_bRunning= false;
	m_bDisturbed= false;
	return true;
}

float ImuBiasCalibrator::getProgress() const
{
	return m_bRunning ? (float)std::clamp(m_seconds / k_calibrationSeconds, 0.0, 1.0) : -1.f;
}
