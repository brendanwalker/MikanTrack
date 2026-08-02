#pragma once

#include "glm/ext/vector_float3.hpp"

// VENDOR-NEUTRAL inertial sample. This is the seam between IMU backends and
// the fusion layer: Joy-Cons today, SlimeVR (ESP32 + 9-axis) tomorrow. A
// backend only has to produce these; nothing downstream changes.
//
// Frame convention: the SENSOR's own right-handed axes, as mounted. The
// rotation from sensor frame to the tracked body frame (forearm) is a
// separate calibrated quantity - see ImuMountCalibration - so a backend is
// NOT responsible for reorienting anything.
struct ImuSample
{
	// Shared steady_clock milliseconds, same base as camera frame timestamps
	// (this is what makes vision/IMU fusion alignable at all)
	double timestampMs= 0.0;

	// Angular velocity, radians/second, sensor frame
	glm::vec3 angularVelocity{0.f};
	// Proper acceleration, meters/second^2, sensor frame. At rest this reads
	// +9.81 along whichever axis points UP (specific force, not gravity).
	glm::vec3 acceleration{0.f};
};

enum class eImuSide : int
{
	Left= 0,
	Right= 1,
	Count= 2,
	Unassigned= -1,
};
