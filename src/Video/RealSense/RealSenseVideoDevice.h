#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "IUsbVideoDevice.h"
#include "RealSenseDepthView.h"

// One RealSense camera exposed through the app's USB-video-device interface.
// Streams synchronized COLOR (RGB24) + DEPTH (Z16) via an rs2 pipeline on a
// dedicated capture thread; each frameset is delivered to the listeners as a
// single UsbVideoFrameBuffer with format USBVideo_RGB24_DEPTH16:
//   section[0] = RGB24 color plane (what the existing pipeline consumes)
//   section[1] = Z16 depth plane (own dimensions, NOT aligned to color)
// Depth calibration (intrinsics/extrinsics/units) is captured at stream
// start and exposed through fetchDepthView() for per-keypoint metric lookup.
class RealSenseVideoDevice : public IUsbVideoDevice
{
public:
	RealSenseVideoDevice(struct rs2_context* context, const std::string& serial, const std::string& name);
	virtual ~RealSenseVideoDevice();

	// IUsbVideoDevice
	virtual void addListener(IUsbVideoDeviceListener* listener) override;
	virtual void removeListener(IUsbVideoDeviceListener* listener) override;
	virtual const char* getDevicePath() const override { return m_devicePath.c_str(); }
	virtual const char* getFriendlyName() const override { return m_friendlyName.c_str(); }
	virtual bool getIsOpen() const override { return m_bOpen; }
	virtual bool open() override;
	virtual void close() override;
	virtual size_t getAvailableVideoModesCount() const override;
	virtual bool getVideoModeProperties(size_t index, UsbVideoModeProperties& outProperties) const override;
	virtual int getVideoModeIndex() const override { return m_videoModeIndex; }
	virtual const char* getVideoModeName() const override;
	virtual bool setVideoModeByName(const char* szVideoModeName) override;
	virtual bool setVideoModeByIndex(size_t index) override;
	virtual eVideoStreamingStatus startVideoStream() override;
	virtual eVideoStreamingStatus getVideoStreamingStatus() const override { return m_streamingStatus; }
	virtual void stopVideoStream() override;

	// -- IVideoDevice video settings: not exposed for RealSense devices
	// (exposure/gain etc. are tuned in RealSense Viewer; the color stream's
	// auto-exposure default behaves well for tracking)
	virtual bool isVideoSettingSupported(const eVideoSettingType propertyType) const override { return false; }
	virtual bool getVideoSettingConstraint(const eVideoSettingType propertyType,
										   VideoSettingConstraint& outConstraint) const override
	{
		return false;
	}
	virtual void setVideoSetting(const eVideoSettingType propertyType, int desiredValue) override {}
	virtual int getVideoSetting(const eVideoSettingType propertyType) const override { return 0; }

	const std::string& getSerial() const { return m_serial; }

	// Copies the newest depth frame + calibration into outView's backing
	// store (double-buffered inside the device; the returned pointers stay
	// valid until the next fetch). Inference-thread safe.
	bool fetchDepthView(RealSenseDepthView& outView, std::vector<uint16_t>& ioDepthStorage);

private:
	void captureThreadLoop();
	void notifyFrame(const UsbVideoFrameBuffer& buffer);

	struct rs2_context* m_context= nullptr;
	std::string m_serial;
	std::string m_friendlyName;
	std::string m_devicePath;

	std::vector<IUsbVideoDeviceListener*> m_listeners;
	std::mutex m_listenerMutex;

	// Color mode table (depth is always 848x480@30 - the D400 sweet spot,
	// and lower x-resolution lowers min-Z, which matters at desk range)
	struct ColorMode
	{
		const char* name;
		int width, height, fps;
	};
	static const std::array<ColorMode, 3> k_colorModes;
	int m_videoModeIndex= 0;

	bool m_bOpen= false;
	std::atomic<eVideoStreamingStatus> m_streamingStatus{eVideoStreamingStatus::stopped};
	std::atomic_bool m_bStopRequested{false};
	std::thread m_captureThread;

	// Depth handoff: capture thread writes, inference thread reads
	std::mutex m_depthMutex;
	std::vector<uint16_t> m_depthBuffer;
	RealSenseDepthView m_depthCalib; // valid= stream running; depthData unused here
	double m_depthTimestampMs= 0.0;
	bool m_bDepthFresh= false;

	// Scratch color buffer for the listener callback (capture thread only)
	std::vector<uint8_t> m_frameScratch;
};
