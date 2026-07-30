#pragma once

#include <chrono>

class FrameTimer
{
public:
	explicit FrameTimer(int tickIntervalMs= 30);

	void waitForNextFrame();

private:
	const std::chrono::milliseconds m_tickInterval;
	std::chrono::steady_clock::time_point m_nextTime;

	std::chrono::milliseconds getTimeToNextFrame() const;
};
