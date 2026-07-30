#include "FrameTimer.h"

#include <thread>

FrameTimer::FrameTimer(int tickIntervalMs)
	: m_tickInterval(tickIntervalMs)
	, m_nextTime(std::chrono::steady_clock::now() + std::chrono::milliseconds(tickIntervalMs))
{
}

void FrameTimer::waitForNextFrame()
{
	std::this_thread::sleep_for(getTimeToNextFrame());

	m_nextTime+= m_tickInterval;
}

std::chrono::milliseconds FrameTimer::getTimeToNextFrame() const
{
	const auto now= std::chrono::steady_clock::now();
	const auto remaining= m_nextTime - now;

	return remaining > std::chrono::milliseconds::zero()
			   ? std::chrono::duration_cast<std::chrono::milliseconds>(remaining)
			   : std::chrono::milliseconds::zero();
}
