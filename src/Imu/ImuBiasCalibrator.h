#pragma once

#include "glm/ext/vector_double3.hpp"
#include "glm/vec3.hpp"

#include "ImuTypes.h"

// STATIC GYRO BIAS CALIBRATION. With the controller resting untouched, true
// angular velocity is zero, so the raw gyro reading IS the bias - measured
// directly on all three axes.
//
// This is not redundant with the filter's online estimate: gravity only
// makes the bias observable about the TILT axes. The component about the
// gravity axis is not inertially observable at all, and it is exactly the
// one that shows up later as yaw drift.
class ImuBiasCalibrator
{
public:
	void begin();
	void cancel();
	bool isRunning() const { return m_bRunning; }

	// Feeds one raw sample while running. A disturbed controller restarts the
	// measurement rather than averaging motion in. Returns true once the rest
	// window is complete; outBias then holds the measured bias (rad/s) and
	// the calibrator has stopped.
	bool addSample(const ImuSample& sample, float dtSeconds, glm::vec3& outBias);

	// 0..1 while running, -1 otherwise
	float getProgress() const;
	// The controller was moved and the measurement restarted
	bool wasDisturbed() const { return m_bDisturbed; }
	int getSampleCount() const { return m_sampleCount; }

private:
	bool m_bRunning= false;
	bool m_bDisturbed= false;
	glm::dvec3 m_sum{0.0};
	int m_sampleCount= 0;
	double m_seconds= 0.0;
};
