#include "OneEuroFilter.h"

#include <cmath>

static constexpr float kTwoPi= 6.28318530717959f;

void OneEuroFilter::configure(float minCutoff, float beta, float dCutoff)
{
	m_minCutoff= minCutoff;
	m_beta= beta;
	m_dCutoff= dCutoff;
}

float OneEuroFilter::smoothingAlpha(float cutoff, float dtSeconds)
{
	const float tau= 1.f / (kTwoPi * cutoff);
	return 1.f / (1.f + tau / dtSeconds);
}

float OneEuroFilter::filter(float value, float dtSeconds)
{
	if (!m_bInitialized || dtSeconds <= 0.f)
	{
		m_bInitialized= true;
		m_lastValue= value;
		m_lastDerivative= 0.f;
		return value;
	}

	// filtered derivative of the signal
	const float rawDerivative= (value - m_lastValue) / dtSeconds;
	const float derivativeAlpha= smoothingAlpha(m_dCutoff, dtSeconds);
	m_lastDerivative= derivativeAlpha * rawDerivative + (1.f - derivativeAlpha) * m_lastDerivative;

	// speed-adaptive cutoff
	const float cutoff= m_minCutoff + m_beta * std::abs(m_lastDerivative);
	const float alpha= smoothingAlpha(cutoff, dtSeconds);
	m_lastValue= alpha * value + (1.f - alpha) * m_lastValue;

	return m_lastValue;
}

void OneEuroFilterVec3::configure(float minCutoff, float beta, float dCutoff)
{
	for (OneEuroFilter& axisFilter : m_axisFilters)
		axisFilter.configure(minCutoff, beta, dCutoff);
}

glm::vec3 OneEuroFilterVec3::filter(const glm::vec3& value, float dtSeconds)
{
	return glm::vec3(
		m_axisFilters[0].filter(value.x, dtSeconds),
		m_axisFilters[1].filter(value.y, dtSeconds),
		m_axisFilters[2].filter(value.z, dtSeconds));
}

void OneEuroFilterVec3::reset()
{
	for (OneEuroFilter& axisFilter : m_axisFilters)
		axisFilter.reset();
}
