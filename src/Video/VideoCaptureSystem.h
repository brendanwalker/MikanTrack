#pragma once

#include "IUsbVideoDevice.h"
#include "IUsbVideoDeviceManager.h"
#include "VideoFrame.h"

#include "readerwriterqueue.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cv
{
class Mat;
}

// App-facing webcam capture facade for one or more cameras.
//
// Owns Media Foundation startup/shutdown, the WMF device manager (hotplug),
// and one CameraSlot per configured camera. Each slot is its own
// IUsbVideoDeviceListener (the WMF frame callback doesn't identify the source
// device, so the listener object's identity is the camera identity) and owns
// its own lock-free frame handoff to the inference thread.
//
// Threading model:
// - startup()/update()/shutdown() and all device management calls must happen
//   on the same (main) thread: the device manager owns the hotplug HWND and
//   pumps its messages from update().
// - Each slot's notifyVideoFrameReceived() runs on that device's Media
//   Foundation worker thread and only touches the slot's lock-free queues
//   (per-device MF callback = sole producer per slot, preserving SPSC).
// - tryPopFrame()/releaseFrame() are for a single consumer (inference) thread.
//   Stop that thread before calling shutdown() or shrinking the slot count.
class VideoCaptureSystem : public IUsbVideoDeviceManagerListener
{
public:
	VideoCaptureSystem();
	virtual ~VideoCaptureSystem();

	// -- Main thread API -----
	bool startup();
	void update(float deltaTime);
	void shutdown();

	// Keeps the slot list in sync with the configured camera count.
	// Growing is always safe; shrinking closes the removed slots' devices
	// (pause the inference thread first).
	void setCameraSlotCount(size_t count);
	size_t getCameraSlotCount() const { return m_slots.size(); }

	// Device list (enumeration is global, not per-slot)
	void refreshDeviceList();
	size_t getDeviceCount() const;
	bool getDevicePath(size_t index, std::string& outDevicePath) const;
	bool getDeviceFriendlyName(size_t index, std::string& outFriendlyName) const;

	// Device open/close (per camera slot)
	bool openDeviceByPath(int cameraIndex, const std::string& devicePath);
	void closeDevice(int cameraIndex);
	bool getIsDeviceOpen(int cameraIndex) const;
	IUsbVideoDevice* getCurrentDevice(int cameraIndex) const;
	std::string getCurrentDevicePath(int cameraIndex) const;
	std::string getCurrentDeviceFriendlyName(int cameraIndex) const;

	// True if the path is already open in a different slot (duplicate guard)
	bool isDevicePathOpenElsewhere(const std::string& devicePath, int excludeCameraIndex) const;

	// Video modes (per camera slot)
	std::string getCurrentVideoModeName(int cameraIndex) const;
	bool setVideoModeByName(int cameraIndex, const std::string& videoModeName);

	// Streaming (per camera slot)
	bool startStream(int cameraIndex);
	void stopStream(int cameraIndex);
	bool isStreaming(int cameraIndex) const;

	// Event callbacks (always invoked on the main thread, with the camera index)
	void setDeviceListChangedCallback(std::function<void()> callback) { m_onDeviceListChanged= callback; }
	void setVideoModeChangedCallback(std::function<void(int)> callback) { m_onVideoModeChanged= callback; }
	void setDeviceDisconnectedCallback(std::function<void(int)> callback) { m_onDeviceDisconnected= callback; }

	// Diagnostics
	uint64_t getDroppedFrameCount(int cameraIndex) const;
	// Frame rate the DEVICE is delivering (measured at the MF callback,
	// before any queueing/drops)
	float getDeviceFrameRate(int cameraIndex) const;

	// -- Inference thread API -----
	// Returns the newest available frame for the camera, recycling any stale
	// frames back to the freelist (latest-wins). Returns nullptr if no frame
	// is pending. The returned block must be given back via releaseFrame().
	VideoFrameBlock* tryPopFrame(int cameraIndex);
	void releaseFrame(int cameraIndex, VideoFrameBlock* block);

	// Converts a raw captured frame (NV12/YUY2/RGB24) to a BGR cv::Mat,
	// respecting section start offsets and strides (drivers pad planes)
	static void convertFrameToBGR(const VideoFrameBlock& block, cv::Mat& outBgr);

	// -- IUsbVideoDeviceManagerListener -----
	virtual void onConnectedDeviceListChanged() override;

private:
	static constexpr size_t k_frameBlockCount= 6;

	// One per configured camera. Being its own listener makes 'this' the
	// camera identity in the WMF callbacks.
	struct CameraSlot final : public IUsbVideoDeviceListener
	{
		VideoCaptureSystem* owner= nullptr;
		int cameraIndex= -1;
		IUsbVideoDevice* device= nullptr;

		// Frame handoff:
		// - frameQueue: MF worker thread (producer) -> inference thread (consumer)
		// - frameFreelist: inference thread (producer) -> MF worker thread (consumer)
		std::vector<std::unique_ptr<VideoFrameBlock>> frameBlockStorage;
		moodycamel::ReaderWriterQueue<VideoFrameBlock*> frameFreelist;
		moodycamel::ReaderWriterQueue<VideoFrameBlock*> frameQueue;
		int64_t nextFrameIndex= 0; // only touched on the MF worker thread
		std::atomic<uint64_t> droppedFrameCount{0};

		// Device-side delivery rate, measured at the MF callback BEFORE any
		// queueing/drops - distinguishes "the camera only delivers N fps"
		// (e.g. auto-exposure halving the rate in low light) from "the
		// pipeline drops frames"
		double lastArrivalMs= 0.0; // MF worker thread only
		std::atomic<float> deviceFps{0.f};

		// Set on device removal, consumed on the main thread in update()
		std::atomic<bool> bDeviceDisconnected{false};

		CameraSlot();
		virtual ~CameraSlot();

		// -- IUsbVideoDeviceListener -----
		virtual void notifyVideoDeviceDisconnected(const IUsbVideoDevice* device) override;
		virtual void notifyVideoModePropertiesChanged(const IUsbVideoDevice* device) override;
		virtual void notifyVideoFrameReceived(const UsbVideoFrameBuffer& bufferInfo) override;
	};

	CameraSlot* getSlot(int cameraIndex);
	const CameraSlot* getSlot(int cameraIndex) const;

	IUsbVideoDevice* findDeviceByPath(const std::string& devicePath) const;
	IUsbVideoDevice* getMergedDeviceByIndex(size_t index) const;

	class CaptureUsbVideoDeviceManager* m_deviceManager= nullptr;
	bool m_bMediaFoundationInitialized= false;

	std::vector<std::unique_ptr<CameraSlot>> m_slots;

	std::function<void()> m_onDeviceListChanged;
	std::function<void(int)> m_onVideoModeChanged;
	std::function<void(int)> m_onDeviceDisconnected;
};
