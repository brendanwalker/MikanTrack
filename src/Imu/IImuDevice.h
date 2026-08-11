#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ImuTypes.h"

// One inertial tracker (a Joy-Con strapped to a wrist today; a SlimeVR node
// later). Devices stream on their own thread and buffer samples until the
// fusion thread drains them.
class IImuDevice
{
public:
	virtual ~IImuDevice() {}

	// -- Identity
	virtual const char* getDevicePath() const= 0;
	virtual const char* getFriendlyName() const= 0;
	// Which wrist this device is strapped to. Backends may guess (a Joy-Con L
	// defaults to the left wrist); the user can override in config.
	virtual eImuSide getSide() const= 0;
	virtual void setSide(eImuSide side)= 0;

	// -- Streaming
	virtual bool open()= 0;
	virtual void close()= 0;
	virtual bool isOpen() const= 0;
	// True once samples have actually arrived (a device can be open but
	// silent if the IMU-enable handshake failed)
	virtual bool isStreaming() const= 0;

	// Drains buffered samples in chronological order (fusion thread).
	// Returns the number appended to outSamples.
	virtual size_t fetchSamples(std::vector<ImuSample>& outSamples)= 0;

	// -- Diagnostics
	// Milliseconds since the last delivered sample; -1 if never. A streaming
	// device should keep this in the low tens - a large value means it slept
	// or the link dropped, which the service uses to trigger recovery.
	virtual double getMillisecondsSinceLastSample() const= 0;
	virtual float getSampleRateHz() const= 0;
	// 0..1, or -1 when the backend doesn't report it
	virtual float getBatteryLevel() const= 0;
};

// Enumerates and owns the IMU devices for one backend.
//
// DISCOVERY BLOCKS. Enumerating a HID/Bluetooth bus and opening a device take
// hundreds of milliseconds on Windows (measured ~200 ms), so refreshing must
// never happen on a thread with a frame deadline - ImuService owns a discovery
// worker for exactly this reason. Any future backend (hidraw/BlueZ, IOKit)
// inherits that contract, and may replace the polling with a native
// device-arrival notification behind this same interface.
class IImuDeviceManager
{
public:
	virtual ~IImuDeviceManager() {}

	virtual bool startup()= 0;
	virtual void shutdown()= 0;
	// Re-scans for connected devices (safe to call while streaming; already
	// known devices keep their state). BLOCKS - see the note above.
	virtual void refreshConnectedDevices()= 0;

	virtual size_t getDeviceCount() const= 0;
	// Shared ownership so discovery can never free a device another thread is
	// still draining: a consumer that keeps its reference across a refresh
	// which dropped that device keeps a valid object, and the device closes
	// when the last reference goes.
	virtual std::shared_ptr<IImuDevice> getDeviceByIndex(size_t index)= 0;
};
