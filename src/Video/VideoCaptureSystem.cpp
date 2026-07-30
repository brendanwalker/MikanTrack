#include "VideoCaptureSystem.h"
#include "Logger.h"
#include "MikanWMFVideoDeviceManager.h"
#include "VideoModeUtils.h"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include <Mfidl.h>
#include <Mfapi.h>

#include <chrono>

// Default video mode used when a freshly opened device has no mode selected yet
// (same defaults as MikanXR's USBVideoSourceComponent)
#define DEFAULT_DESIRED_VIDEO_WIDTH 1280
#define DEFAULT_DESIRED_VIDEO_HEIGHT 720
#define DEFAULT_DESIRED_FRAME_RATE 30

// Small subclass that exposes the protected rebuildDeviceList() so the app
// can force a manual device list refresh (normally it only runs on hotplug events)
class CaptureUsbVideoDeviceManager : public MikanWMFVideoDeviceManager
{
public:
	void refreshConnectedDevices() { rebuildDeviceList(); }
};

// -- VideoCaptureSystem -----
VideoCaptureSystem::VideoCaptureSystem()
	: m_frameFreelist(k_frameBlockCount * 2)
	, m_frameQueue(k_frameBlockCount * 2)
{
}

VideoCaptureSystem::~VideoCaptureSystem() { shutdown(); }

// -- Main thread API -----
bool VideoCaptureSystem::startup()
{
	if (!m_bMediaFoundationInitialized)
	{
		HRESULT hr= MFStartup(MF_VERSION);
		if (SUCCEEDED(hr))
		{
			MIKAN_LOG_INFO("VideoCaptureSystem::startup") << "Successfully initialized Media Foundation";
			m_bMediaFoundationInitialized= true;
		}
		else
		{
			MIKAN_LOG_ERROR("VideoCaptureSystem::startup") << "Failed to initialize Media Foundation.";
			return false;
		}
	}

	// Allocate the frame blocks and prime the freelist (once)
	if (m_frameBlockStorage.empty())
	{
		m_frameBlockStorage.reserve(k_frameBlockCount);
		for (size_t i= 0; i < k_frameBlockCount; ++i)
		{
			m_frameBlockStorage.emplace_back(std::make_unique<VideoFrameBlock>());
			m_frameFreelist.enqueue(m_frameBlockStorage.back().get());
		}
	}

	if (m_deviceManager == nullptr)
	{
		m_deviceManager= new CaptureUsbVideoDeviceManager();
		m_deviceManager->addListener(this);

		if (!m_deviceManager->startup())
		{
			MIKAN_LOG_ERROR("VideoCaptureSystem::startup") << "Failed to start USB video device manager";
			shutdown();
			return false;
		}
	}

	return true;
}

void VideoCaptureSystem::update(float deltaTime)
{
	// Pump hotplug notifications.
	// Must be called on the thread that created the manager (it owns the hotplug HWND).
	if (m_deviceManager != nullptr)
	{
		m_deviceManager->update(deltaTime);
	}

	// Forward device disconnects flagged by notifyVideoDeviceDisconnected()
	if (m_bDeviceDisconnected.exchange(false))
	{
		MIKAN_LOG_INFO("VideoCaptureSystem::update") << "Current video device was disconnected";

		if (m_onDeviceDisconnected)
		{
			m_onDeviceDisconnected();
		}
	}
}

void VideoCaptureSystem::shutdown()
{
	closeDevice();

	if (m_deviceManager != nullptr)
	{
		m_deviceManager->removeListener(this);
		m_deviceManager->shutdown();
		delete m_deviceManager;
		m_deviceManager= nullptr;
	}

	if (m_bMediaFoundationInitialized)
	{
		MFShutdown();
		m_bMediaFoundationInitialized= false;
	}
}

// -- Device list -----
void VideoCaptureSystem::refreshDeviceList()
{
	if (m_deviceManager != nullptr)
	{
		m_deviceManager->refreshConnectedDevices();
	}
}

size_t VideoCaptureSystem::getDeviceCount() const
{
	return m_deviceManager != nullptr ? m_deviceManager->getDeviceCount() : 0;
}

bool VideoCaptureSystem::getDevicePath(size_t index, std::string& outDevicePath) const
{
	if (m_deviceManager != nullptr)
	{
		IUsbVideoDevice* device= m_deviceManager->getDeviceByIndex(index);
		if (device != nullptr)
		{
			outDevicePath= device->getDevicePath();
			return true;
		}
	}

	return false;
}

bool VideoCaptureSystem::getDeviceFriendlyName(size_t index, std::string& outFriendlyName) const
{
	if (m_deviceManager != nullptr)
	{
		IUsbVideoDevice* device= m_deviceManager->getDeviceByIndex(index);
		if (device != nullptr)
		{
			outFriendlyName= device->getFriendlyName();
			return true;
		}
	}

	return false;
}

// -- Device open/close -----
bool VideoCaptureSystem::openDeviceByPath(const std::string& devicePath)
{
	if (m_deviceManager == nullptr || devicePath.empty())
		return false;

	// Close any previously open device first
	closeDevice();

	IUsbVideoDevice* device= m_deviceManager->getDeviceByPath(devicePath.c_str());
	if (device == nullptr)
	{
		MIKAN_LOG_WARNING("VideoCaptureSystem::openDeviceByPath") << "No device found at path: " << devicePath;
		return false;
	}

	// If the device doesn't have a video mode selected yet, pick a sensible default
	if (device->getVideoModeName() == nullptr)
	{
		const int videoModeIndex= VideoModeUtils::findBestVideoModeIndex(
			device, DEFAULT_DESIRED_VIDEO_WIDTH, DEFAULT_DESIRED_VIDEO_HEIGHT, DEFAULT_DESIRED_FRAME_RATE);
		if (videoModeIndex == -1 || !device->setVideoModeByIndex(videoModeIndex))
		{
			MIKAN_LOG_ERROR("VideoCaptureSystem::openDeviceByPath")
				<< "Failed to set a default video mode on device: " << devicePath;
			return false;
		}
	}

	// Attempt to open the video device
	if (!device->open())
	{
		MIKAN_LOG_ERROR("VideoCaptureSystem::openDeviceByPath") << "Failed to open device: " << devicePath;
		return false;
	}

	// Listen for events from the video device
	device->addListener(this);

	m_currentDevice= device;
	m_bDeviceDisconnected= false;

	MIKAN_LOG_INFO("VideoCaptureSystem::openDeviceByPath")
		<< "Opened device: " << device->getFriendlyName() << " (" << device->getVideoModeName() << ")";

	return true;
}

void VideoCaptureSystem::closeDevice()
{
	if (m_currentDevice != nullptr)
	{
		m_currentDevice->stopVideoStream();
		m_currentDevice->removeListener(this);
		m_currentDevice->close();
		m_currentDevice= nullptr;
	}
}

std::string VideoCaptureSystem::getCurrentDevicePath() const
{
	return m_currentDevice != nullptr ? m_currentDevice->getDevicePath() : "";
}

std::string VideoCaptureSystem::getCurrentDeviceFriendlyName() const
{
	return m_currentDevice != nullptr ? m_currentDevice->getFriendlyName() : "";
}

// -- Video modes -----
std::string VideoCaptureSystem::getCurrentVideoModeName() const
{
	if (m_currentDevice != nullptr)
	{
		const char* szVideoModeName= m_currentDevice->getVideoModeName();
		if (szVideoModeName != nullptr)
		{
			return szVideoModeName;
		}
	}

	return "";
}

bool VideoCaptureSystem::setVideoModeByName(const std::string& videoModeName)
{
	if (m_currentDevice == nullptr)
		return false;

	// Early out if this is already the current mode
	const char* szCurrentModeName= m_currentDevice->getVideoModeName();
	if (szCurrentModeName != nullptr && videoModeName == szCurrentModeName)
		return true;

	const bool bWasStreaming= isStreaming();

	// This closes the device if the mode actually changed
	// (and fires notifyVideoModePropertiesChanged)
	if (!m_currentDevice->setVideoModeByName(videoModeName.c_str()))
	{
		MIKAN_LOG_WARNING("VideoCaptureSystem::setVideoModeByName") << "Invalid video mode name: " << videoModeName;
		return false;
	}

	// Re-open the device with the new mode
	if (!m_currentDevice->open())
	{
		MIKAN_LOG_ERROR("VideoCaptureSystem::setVideoModeByName")
			<< "Failed to re-open device after video mode change: " << videoModeName;
		closeDevice();
		return false;
	}

	// Resume streaming if we were streaming before the mode change
	if (bWasStreaming)
	{
		startStream();
	}

	return true;
}

// -- Streaming -----
bool VideoCaptureSystem::startStream()
{
	if (m_currentDevice == nullptr)
		return false;

	const eVideoStreamingStatus status= m_currentDevice->startVideoStream();

	return status == eVideoStreamingStatus::started || status == eVideoStreamingStatus::pendingStart;
}

void VideoCaptureSystem::stopStream()
{
	if (m_currentDevice != nullptr)
	{
		m_currentDevice->stopVideoStream();
	}
}

bool VideoCaptureSystem::isStreaming() const
{
	return m_currentDevice != nullptr
		   && m_currentDevice->getVideoStreamingStatus() == eVideoStreamingStatus::started;
}

// -- Inference thread API -----
VideoFrameBlock* VideoCaptureSystem::tryPopFrame()
{
	// Drain to the newest available frame (latest-wins),
	// recycling any stale frames back to the freelist
	VideoFrameBlock* newestBlock= nullptr;
	VideoFrameBlock* candidateBlock= nullptr;
	while (m_frameQueue.try_dequeue(candidateBlock))
	{
		if (newestBlock != nullptr)
		{
			m_frameFreelist.enqueue(newestBlock);
		}

		newestBlock= candidateBlock;
	}

	return newestBlock;
}

void VideoCaptureSystem::releaseFrame(VideoFrameBlock* block)
{
	if (block != nullptr)
	{
		m_frameFreelist.enqueue(block);
	}
}

void VideoCaptureSystem::convertFrameToBGR(const VideoFrameBlock& block, cv::Mat& outBgr)
{
	// Ported from MikanXR USBVideoSourceComponent::notifyVideoFrameReceived
	const uint8_t* frameData= block.data.data();

	if (block.format == eUSBVideoFrameBufferFormat::USBVideo_NV12)
	{
		// NV12 layout: [Y plane: width × height] [UV plane: width × height/2]
		// Always use section-based approach (handles inter-plane padding correctly)
		if (block.sectionCount == 2)
		{
			const UsbVideoFrameSection& ySection= block.sections[0];
			const UsbVideoFrameSection& uvSection= block.sections[1];

			// Create separate Mat views for Y and UV planes using section offsets
			cv::Mat yPlane(ySection.pixel_height, ySection.pixel_width, CV_8UC1,
						   (void*)(frameData + ySection.start_offset), ySection.stride);

			// For NV12, UV plane is interleaved U and V bytes
			// When using CV_8UC2, width is half of Y plane width (each pixel = 2 bytes)
			cv::Mat uvPlane(uvSection.pixel_height,
							uvSection.pixel_width / 2, // Half width since each pixel is 2 bytes (U+V)
							CV_8UC2, (void*)(frameData + uvSection.start_offset), uvSection.stride);

			// Use cvtColorTwoPlane for proper NV12 conversion with separate planes
			cv::cvtColorTwoPlane(yPlane, uvPlane, outBgr, cv::COLOR_YUV2BGR_NV12);
		}
		else
		{
			// Should not happen, but provide fallback
			const UsbVideoFrameSection& section= block.sections[0];
			outBgr= cv::Mat(section.pixel_height, section.pixel_width, CV_8UC3, cv::Scalar(0, 0, 0));
		}
	}
	else if (block.format == eUSBVideoFrameBufferFormat::USBVideo_YUY2)
	{
		// YUY2 (YUYV) format: 4:2:2 packed format (2 bytes per pixel)
		const UsbVideoFrameSection& section= block.sections[0];
		cv::Mat yuy2Mat(section.pixel_height, section.pixel_width, CV_8UC2,
						(void*)(frameData + section.start_offset),
						section.stride); // stride is already in bytes per row
		cv::cvtColor(yuy2Mat, outBgr, cv::COLOR_YUV2BGR_YUY2);
	}
	else if (block.format == eUSBVideoFrameBufferFormat::USBVideo_RGB24)
	{
		// RGB24 format: convert to BGR
		const UsbVideoFrameSection& section= block.sections[0];
		cv::Mat rgb24Mat(section.pixel_height, section.pixel_width, CV_8UC3,
						 (void*)(frameData + section.start_offset),
						 section.stride); // stride is already in bytes per row
		cv::cvtColor(rgb24Mat, outBgr, cv::COLOR_RGB2BGR);
	}
	else
	{
		// Unknown format - create empty mat as fallback
		if (block.sectionCount > 0)
		{
			const UsbVideoFrameSection& section= block.sections[0];
			outBgr= cv::Mat(section.pixel_height, section.pixel_width, CV_8UC3, cv::Scalar(0, 0, 0));
		}
		else
		{
			outBgr= cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 0)); // Default fallback size
		}
	}
}

// -- IUsbVideoDeviceManagerListener -----
void VideoCaptureSystem::onConnectedDeviceListChanged()
{
	// Called on the main thread (from the manager's hotplug message pump)
	if (m_onDeviceListChanged)
	{
		m_onDeviceListChanged();
	}
}

// -- IUsbVideoDeviceListener -----
void VideoCaptureSystem::notifyVideoDeviceDisconnected(const IUsbVideoDevice* device)
{
	if (device == m_currentDevice)
	{
		// The manager is about to destroy this device (hotplug removal) and has
		// already closed it. Drop our reference immediately (it's about to dangle)
		// and flag the main thread to fire the disconnect callback from update().
		// NOTE: don't call removeListener() here - the device is iterating its
		// listener set and will be deleted right after this notification anyway.
		m_currentDevice= nullptr;
		m_bDeviceDisconnected= true;
	}
}

void VideoCaptureSystem::notifyVideoModePropertiesChanged(const IUsbVideoDevice* device)
{
	// Called on the main thread (video modes are only changed from the main thread)
	if (device == m_currentDevice && m_onVideoModeChanged)
	{
		m_onVideoModeChanged();
	}
}

void VideoCaptureSystem::notifyVideoFrameReceived(const UsbVideoFrameBuffer& bufferInfo)
{
	// Runs on the Media Foundation worker thread - keep this lock-free and allocation-light
	VideoFrameBlock* block= nullptr;
	if (m_frameFreelist.try_dequeue(block))
	{
		const double timestampMs=
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();

		block->copyFrom(bufferInfo, m_nextFrameIndex++, timestampMs);
		m_frameQueue.enqueue(block);
	}
	else
	{
		// No free blocks - the consumer is behind, drop the frame (no logging on the hot path)
		m_droppedFrameCount.fetch_add(1, std::memory_order_relaxed);
	}
}
