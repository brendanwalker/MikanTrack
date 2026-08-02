#pragma once

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
	virtual float getSampleRateHz() const= 0;
	// 0..1, or -1 when the backend doesn't report it
	virtual float getBatteryLevel() const= 0;
};

// Enumerates and owns the IMU devices for one backend.
class IImuDeviceManager
{
public:
	virtual ~IImuDeviceManager() {}

	virtual bool startup()= 0;
	virtual void shutdown()= 0;
	// Re-scans for connected devices (safe to call while streaming; already
	// known devices keep their state)
	virtual void refreshConnectedDevices()= 0;

	virtual size_t getDeviceCount() const= 0;
	virtual IImuDevice* getDeviceByIndex(size_t index)= 0;
};
