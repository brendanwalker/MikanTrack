#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "IImuDevice.h"
#include "ImuBiasCalibrator.h"
#include "ImuMountingCalibrator.h"
#include "ImuOrientationFilter.h"

// One IMU device as the service tracks it: the device, its orientation
// filter, the mounting and bias calibrators fed from its samples, the sample
// clock, and the silence/reopen bookkeeping. Vision-thread state. The device
// itself is shared with the discovery worker, which may be reopening it.
class ImuDeviceTracker
{
public:
	ImuDeviceTracker(const std::shared_ptr<IImuDevice>& device, const ImuOrientationFilterConfig& filterConfig);

	const std::shared_ptr<IImuDevice>& getDevice() const { return m_device; }
	ImuOrientationFilter& getFilter() { return m_filter; }
	const ImuOrientationFilter& getFilter() const { return m_filter; }
	ImuMountingCalibrator& getMounting() { return m_mounting; }
	const ImuMountingCalibrator& getMounting() const { return m_mounting; }
	ImuBiasCalibrator& getBias() { return m_bias; }
	const ImuBiasCalibrator& getBias() const { return m_bias; }

	// Samples are arriving now. The device's own isStreaming() only says
	// "samples arrived at some point"; a controller that fell asleep still
	// reports true, so this gates on recent traffic.
	bool hasRecentSamples() const;
	// The device has streamed and the filter's tilt has settled: the
	// precondition for every side-level read of this tracker
	bool isReady() const { return m_device->isStreaming() && m_filter.isTiltConverged(); }

	// -- Silence recovery -----
	// Advances the reopen throttle. Returns true when the device has gone
	// silent and a close/open cycle should be requested now; the tracker then
	// leaves the device alone until onReopenCompleted(). Throttled so a
	// genuinely absent controller does not spin.
	bool pollReopenRequest();
	bool isAwaitingReopen() const { return m_bAwaitingReopen; }
	// A completed reopen restarts the sample clock: the orientation survives,
	// but integrating across the gap must not happen
	void onReopenCompleted();

	// Drains the device's buffered samples and advances the filter and the
	// calibrators. Samples arrive in chronological order and carry their own
	// timestamps, so integrating a whole backlog at once is exact.
	// sampleScratch is the caller's reusable buffer.
	void update(std::vector<ImuSample>& sampleScratch, eMountingMotion recording);

	// Recent RAW samples, oldest first, exactly as the device reported them -
	// before the bias and the filter. Kept so the sensor's axis convention
	// can be MEASURED against vision by replaying candidate mappings offline.
	void getRawSampleHistory(std::vector<ImuSample>& outSamples) const;

private:
	std::shared_ptr<IImuDevice> m_device;
	ImuOrientationFilter m_filter;
	ImuMountingCalibrator m_mounting;
	ImuBiasCalibrator m_bias;

	double m_lastSampleTimestampMs= -1.0;
	int m_reopenCooldownFrames= 0;
	// The discovery worker is close()/open()ing the device right now, so
	// nothing here touches it until the worker reports back. The filter
	// state stays put, so a reconnect keeps its converged orientation.
	bool m_bAwaitingReopen= false;

	// Rolling window of raw samples for the axis-convention diagnostic.
	// ~30 seconds at the Joy-Con's 200 Hz.
	std::deque<ImuSample> m_rawHistory;
};
