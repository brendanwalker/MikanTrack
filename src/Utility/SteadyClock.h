#pragma once

#include <chrono>

// Milliseconds on the steady clock: the one time base shared by camera frame
// timestamps, IMU samples, and the vision thread's own timing
inline double steadyNowMs()
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
