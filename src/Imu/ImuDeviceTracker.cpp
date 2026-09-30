#include "ImuDeviceTracker.h"

#include "glm/gtc/quaternion.hpp"

#include "Logger.h"

// A streaming Joy-Con delivers ~200 samples/second, so a second of total
// silence already means it is gone (asleep, or the link dropped)
static constexpr double k_deviceSilentTimeoutMs= 1500.0;
// Frames to wait between reopen attempts (~2s at camera rate)
static constexpr int k_reopenCooldownFrames= 60;
// ~30 seconds of raw samples kept for the axis-convention diagnostic
static constexpr size_t k_rawHistoryMaxSamples= 6000;

ImuDeviceTracker::ImuDeviceTracker(const std::shared_ptr<IImuDevice>& device,
								   const ImuOrientationFilterConfig& filterConfig)
	: m_device(device)
{
	m_filter.configure(filterConfig);
}

bool ImuDeviceTracker::hasRecentSamples() const
{
	const double silentMs= m_device->getMillisecondsSinceLastSample();
	return m_device->isStreaming() && silentMs >= 0.0 && silentMs < k_deviceSilentTimeoutMs;
}

bool ImuDeviceTracker::pollReopenRequest()
{
	// Joy-Cons sleep on their own schedule and Bluetooth links drop; either
	// way the fix is the same close/open cycle, so heal automatically instead
	// of making the user notice and press a button
	const double silentMs= m_device->getMillisecondsSinceLastSample();
	if (silentMs > k_deviceSilentTimeoutMs)
	{
		if (!m_bAwaitingReopen && m_reopenCooldownFrames <= 0)
		{
			m_bAwaitingReopen= true;
			m_reopenCooldownFrames= k_reopenCooldownFrames;
			return true;
		}
		if (m_reopenCooldownFrames > 0)
			m_reopenCooldownFrames--;
	}
	else if (m_reopenCooldownFrames > 0)
	{
		m_reopenCooldownFrames= 0;
	}
	return false;
}

void ImuDeviceTracker::onReopenCompleted()
{
	m_bAwaitingReopen= false;
	m_lastSampleTimestampMs= -1.0;
}

void ImuDeviceTracker::update(std::vector<ImuSample>& sampleScratch, eMountingMotion recording)
{
	sampleScratch.clear();
	m_device->fetchSamples(sampleScratch);

	// Keep the untouched samples first: the axis-convention diagnostic has
	// to replay candidate mappings against what the device actually sent,
	// not against whatever correction is currently enabled
	for (const ImuSample& raw : sampleScratch)
	{
		m_rawHistory.push_back(raw);
		if (m_rawHistory.size() > k_rawHistoryMaxSamples)
			m_rawHistory.pop_front();
	}

	for (const ImuSample& sample : sampleScratch)
	{
		constexpr double k_nominalSpacingMs= 1000.0 / 200.0; // Joy-Con rate
		constexpr double k_maxPlausibleSpacingMs= 100.0;

		// Sample times are back-dated from HID arrival, so a Bluetooth stall
		// followed by a burst yields timestamps that go BACKWARDS or bunch
		// microseconds apart. Taking those at face value collapses the filter
		// covariance (see the dt floor in predict). Fall back to the nominal
		// spacing and keep the clock monotonic instead - the samples
		// themselves are still good, only their arrival times are not.
		double sampleTimeMs= sample.timestampMs;
		float dtSeconds= (float)(k_nominalSpacingMs / 1000.0);
		if (m_lastSampleTimestampMs >= 0.0)
		{
			const double deltaMs= sampleTimeMs - m_lastSampleTimestampMs;
			if (deltaMs <= 0.0 || deltaMs > k_maxPlausibleSpacingMs)
				sampleTimeMs= m_lastSampleTimestampMs + k_nominalSpacingMs;
			else
				dtSeconds= (float)(deltaMs / 1000.0);
		}
		m_lastSampleTimestampMs= sampleTimeMs;

		if (m_bias.isRunning())
		{
			// Deliberately BEFORE the filter: a resting controller carries no
			// orientation information, and folding its samples in while the
			// user is told not to touch it just wastes them
			glm::vec3 measuredBias;
			if (m_bias.addSample(sample, dtSeconds, measuredBias))
			{
				m_filter.setGyroBias(measuredBias);
				MIKAN_LOG_INFO("ImuDeviceTracker")
					<< m_device->getFriendlyName() << " gyro bias calibrated: " << glm::degrees(measuredBias).x
					<< ", " << glm::degrees(measuredBias).y << ", " << glm::degrees(measuredBias).z
					<< " deg/s over " << m_bias.getSampleCount() << " samples";
			}
			continue;
		}

		m_filter.processSample(sample, dtSeconds);

		const glm::vec3 rate= sample.angularVelocity - m_filter.getGyroBias();
		m_mounting.addMotionSample(rate, sample.acceleration, dtSeconds, recording);
	}
}

void ImuDeviceTracker::getRawSampleHistory(std::vector<ImuSample>& outSamples) const
{
	outSamples.assign(m_rawHistory.begin(), m_rawHistory.end());
}
