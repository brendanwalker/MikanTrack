#pragma once

#include <array>

#include "glm/ext/vector_float3.hpp"

#include "TrackingTypes.h"

// Standard one-euro filter (Casiez et al. 2012) for a scalar series.
// cutoff= minCutoff + beta * |dx|; alpha(fc)= 1 / (1 + tau/dt), tau= 1/(2*pi*fc)
class OneEuroFilter
{
public:
	void configure(float minCutoff, float beta, float dCutoff);
	float filter(float value, float dtSeconds);
	void reset() { m_bInitialized= false; }

private:
	static float smoothingAlpha(float cutoff, float dtSeconds);

	float m_minCutoff= 1.f;
	float m_beta= 0.f;
	float m_dCutoff= 1.f;

	bool m_bInitialized= false;
	float m_lastValue= 0.f;
	float m_lastDerivative= 0.f;
};

// One-euro filter applied per component of a glm::vec3 series
class OneEuroFilterVec3
{
public:
	void configure(float minCutoff, float beta, float dCutoff);
	glm::vec3 filter(const glm::vec3& value, float dtSeconds);
	void reset();

private:
	std::array<OneEuroFilter, 3> m_axisFilters;
};

// Per-hand filter bank: one vec3 filter per hand landmark plus one for the
// elbow, per side. Reset a side when its hand loses tracking so stale filter
// state doesn't bleed into a re-acquired hand.
class HandOneEuroBank
{
public:
	void configure(float minCutoff, float beta, float dCutoff);

	OneEuroFilterVec3& landmarkFilter(eHandSide side, int landmarkIndex);
	OneEuroFilterVec3& elbowFilter(eHandSide side);

	void resetSide(eHandSide side);
	void resetAll();

private:
	std::array<std::array<OneEuroFilterVec3, HAND_LANDMARK_COUNT>, 2> m_landmarkFilters;
	std::array<OneEuroFilterVec3, 2> m_elbowFilters;
};
