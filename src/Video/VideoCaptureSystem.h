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

// App-facing webcam capture facade.
//
// Owns Media Foundation startup/shutdown, the WMF device manager (hotplug),
// the currently open device and the frame handoff to the inference thread.
//
// Threading model:
// - startup()/update()/shutdown() and all device management calls must happen
//   on the same (main) thread: the device manager owns the hotplug HWND and
//   pumps its messages from update().
// - notifyVideoFrameReceived() runs on the Media Foundation worker thread and
//   only touches the lock-free frame queues.
// - tryPopFrame()/releaseFrame() are for a single consumer (inference) thread.
//   Stop that thread before calling shutdown().
class VideoCaptureSystem : public IUsbVideoDeviceManagerListener, public IUsbVideoDeviceListener
{
public:
	VideoCaptureSystem();
	virtual ~VideoCaptureSystem();

	// -- Main thread API -----
	bool startup();
	void update(float deltaTime);
	void shutdown();

	// Device list
	void refreshDeviceList();
	size_t getDeviceCount() const;
	bool getDevicePath(size_t index, std::string& outDevicePath) const;
	bool getDeviceFriendlyName(size_t index, std::string& outFriendlyName) const;

	// Device open/close
	bool openDeviceByPath(const std::string& devicePath);
	void closeDevice();
	bool getIsDeviceOpen() const { return m_currentDevice != nullptr; }
	IUsbVideoDevice* getCurrentDevice() const { return m_currentDevice; }
	std::string getCurrentDevicePath() const;
	std::string getCurrentDeviceFriendlyName() const;

	// Video modes
	std::string getCurrentVideoModeName() const;
	bool setVideoModeByName(const std::string& videoModeName);

	// Streaming
	bool startStream();
	void stopStream();
	bool isStreaming() const;

	// Event callbacks (always invoked on the main thread)
	void setDeviceListChangedCallback(std::function<void()> callback) { m_onDeviceListChanged= callback; }
	void setVideoModeChangedCallback(std::function<void()> callback) { m_onVideoModeChanged= callback; }
	void setDeviceDisconnectedCallback(std::function<void()> callback) { m_onDeviceDisconnected= callback; }

	// Diagnostics
	uint64_t getDroppedFrameCount() const { return m_droppedFrameCount.load(std::memory_order_relaxed); }

	// -- Inference thread API -----
	// Returns the newest available frame, recycling any stale frames back to
	// the freelist (latest-wins). Returns nullptr if no frame is pending.
	// The returned block must be given back via releaseFrame().
	VideoFrameBlock* tryPopFrame();
	void releaseFrame(VideoFrameBlock* block);

	// Converts a raw captured frame (NV12/YUY2/RGB24) to a BGR cv::Mat,
	// respecting section start offsets and strides (drivers pad planes)
	static void convertFrameToBGR(const VideoFrameBlock& block, cv::Mat& outBgr);

	// -- IUsbVideoDeviceManagerListener -----
	virtual void onConnectedDeviceListChanged() override;

	// -- IUsbVideoDeviceListener -----
	virtual void notifyVideoDeviceDisconnected(const IUsbVideoDevice* device) override;
	virtual void notifyVideoModePropertiesChanged(const IUsbVideoDevice* device) override;
	virtual void notifyVideoFrameReceived(const UsbVideoFrameBuffer& bufferInfo) override;

private:
	static constexpr size_t k_frameBlockCount= 6;

	class CaptureUsbVideoDeviceManager* m_deviceManager= nullptr;
	IUsbVideoDevice* m_currentDevice= nullptr;
	bool m_bMediaFoundationInitialized= false;

	// Frame handoff:
	// - m_frameQueue: MF worker thread (producer) -> inference thread (consumer)
	// - m_frameFreelist: inference thread (producer) -> MF worker thread (consumer)
	// Blocks are owned by m_frameBlockStorage and only recirculate through the queues.
	std::vector<std::unique_ptr<VideoFrameBlock>> m_frameBlockStorage;
	moodycamel::ReaderWriterQueue<VideoFrameBlock*> m_frameFreelist;
	moodycamel::ReaderWriterQueue<VideoFrameBlock*> m_frameQueue;
	int64_t m_nextFrameIndex= 0; // only touched on the MF worker thread
	std::atomic<uint64_t> m_droppedFrameCount{0};

	// Set when the open device gets removed, consumed on the main thread in update()
	std::atomic<bool> m_bDeviceDisconnected{false};

	std::function<void()> m_onDeviceListChanged;
	std::function<void()> m_onVideoModeChanged;
	std::function<void()> m_onDeviceDisconnected;
};
