#pragma once

#include <memory>
#include <string>
#include <vector>

#include "IUsbVideoDevice.h"
#include "IUsbVideoDeviceManager.h"
#include "RealSenseDepthView.h"

// RealSense backend: enumerates connected RealSense devices through the
// dynamically-loaded librealsense2 C API and exposes each as an
// IUsbVideoDevice streaming synchronized COLOR (RGB24, section 0) + DEPTH
// (Z16, section 1) frames through the same listener callback the WMF path
// uses. Device paths are namespaced "rs://<serial>" so they can share the
// config/UI plumbing with WMF device paths.
//
// When realsense2.dll is absent this manager simply reports zero devices.
class RealSenseVideoDevice;

class RealSenseVideoDeviceManager : public IUsbVideoDeviceManager
{
public:
	RealSenseVideoDeviceManager()= default;
	virtual ~RealSenseVideoDeviceManager();

	// IUsbVideoDeviceManager
	virtual void addListener(IUsbVideoDeviceManagerListener* listener) override;
	virtual void removeListener(IUsbVideoDeviceManagerListener* listener) override;
	virtual bool startup() override;
	virtual void update(float deltaTime) override;
	virtual void shutdown() override;
	virtual size_t getDeviceCount() const override;
	virtual IUsbVideoDevice* getDeviceByIndex(size_t index) override;
	virtual IUsbVideoDevice* getDeviceByPath(const char* devicePath) override;

	// Re-enumerates connected RealSense devices (not part of the base
	// interface; VideoCaptureSystem calls it via the concrete type, matching
	// the WMF manager's refreshConnectedDevices pattern)
	void refreshConnectedDevices();

	static bool isRealSensePath(const std::string& devicePath);

	// Latest depth frame + calibration for an open device ("" path = any).
	// Safe to call from the inference thread: the view points into a
	// double-buffered depth copy owned by the device that stays valid until
	// the next fetch for the same device.
	bool fetchDepthView(const std::string& devicePath, double colorTimestampMs, RealSenseDepthView& outView);

private:
	std::vector<IUsbVideoDeviceManagerListener*> m_listeners;
	std::vector<std::shared_ptr<RealSenseVideoDevice>> m_devices;
	struct rs2_context* m_context= nullptr;
};
