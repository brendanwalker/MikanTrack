#include "RealSenseVideoDevice.h"

#include <algorithm>
#include <cstring>

#include "Logger.h"
#include "RealSenseApi.h"
#include "ThreadUtils.h"

#include "librealsense2/rs.h"

const std::array<RealSenseVideoDevice::ColorMode, 3> RealSenseVideoDevice::k_colorModes= {{
	{"1280x720 @30 (RGB+depth)", 1280, 720, 30},
	{"848x480 @30 (RGB+depth)", 848, 480, 30},
	{"640x480 @30 (RGB+depth)", 640, 480, 30},
}};

// Depth stream: fixed at the D400 sweet-spot resolution. Lower x-resolution
// also lowers min-Z (~35cm at 848 wide on a D455), which matters at desk range.
static constexpr int kDepthWidth= 848;
static constexpr int kDepthHeight= 480;
static constexpr int kDepthFps= 30;

static void copyIntrinsics(const rs2_intrinsics& in, DepthFrameView::Intrinsics& out)
{
	out.width= in.width;
	out.height= in.height;
	out.fx= in.fx;
	out.fy= in.fy;
	out.ppx= in.ppx;
	out.ppy= in.ppy;
	out.model= (int)in.model;
	for (int i= 0; i < 5; ++i)
		out.coeffs[i]= in.coeffs[i];
}

RealSenseVideoDevice::RealSenseVideoDevice(rs2_context* context, const std::string& serial,
										   const std::string& name)
	: m_context(context)
	, m_serial(serial)
	, m_friendlyName("RealSense " + name + " [" + serial + "]")
	, m_devicePath("rs://" + serial)
{
}

RealSenseVideoDevice::~RealSenseVideoDevice()
{
	stopVideoStream();
}

void RealSenseVideoDevice::addListener(IUsbVideoDeviceListener* listener)
{
	std::lock_guard<std::mutex> lock(m_listenerMutex);
	m_listeners.push_back(listener);
}

void RealSenseVideoDevice::removeListener(IUsbVideoDeviceListener* listener)
{
	std::lock_guard<std::mutex> lock(m_listenerMutex);
	m_listeners.erase(std::remove(m_listeners.begin(), m_listeners.end(), listener), m_listeners.end());
}

bool RealSenseVideoDevice::open()
{
	m_bOpen= true;
	return true;
}

void RealSenseVideoDevice::close()
{
	stopVideoStream();
	m_bOpen= false;
}

size_t RealSenseVideoDevice::getAvailableVideoModesCount() const
{
	return k_colorModes.size();
}

bool RealSenseVideoDevice::getVideoModeProperties(size_t index, UsbVideoModeProperties& outProperties) const
{
	if (index >= k_colorModes.size())
		return false;

	const ColorMode& mode= k_colorModes[index];
	outProperties.index= index;
	outProperties.name= mode.name;
	outProperties.width= mode.width;
	outProperties.height= mode.height;
	outProperties.frame_rate_numerator= mode.fps;
	outProperties.frame_rate_demonenator= 1;
	outProperties.format= "RGB24+Z16";
	outProperties.colorimetry= VideoColorimetry();
	return true;
}

const char* RealSenseVideoDevice::getVideoModeName() const
{
	return k_colorModes[m_videoModeIndex].name;
}

bool RealSenseVideoDevice::setVideoModeByName(const char* szVideoModeName)
{
	for (size_t i= 0; i < k_colorModes.size(); ++i)
	{
		if (std::strcmp(k_colorModes[i].name, szVideoModeName) == 0)
			return setVideoModeByIndex(i);
	}
	return false;
}

bool RealSenseVideoDevice::setVideoModeByIndex(size_t index)
{
	if (index >= k_colorModes.size())
		return false;

	const bool bWasStreaming= m_streamingStatus == eVideoStreamingStatus::started;
	if (bWasStreaming)
		stopVideoStream();
	m_videoModeIndex= (int)index;
	if (bWasStreaming)
		startVideoStream();
	return true;
}

eVideoStreamingStatus RealSenseVideoDevice::startVideoStream()
{
	if (m_streamingStatus == eVideoStreamingStatus::started)
		return eVideoStreamingStatus::started;

	m_bStopRequested= false;
	m_streamingStatus= eVideoStreamingStatus::pendingStart;
	m_captureThread= std::thread([this]() { captureThreadLoop(); });

	return eVideoStreamingStatus::pendingStart;
}

void RealSenseVideoDevice::stopVideoStream()
{
	m_bStopRequested= true;
	if (m_captureThread.joinable())
		m_captureThread.join();
	m_streamingStatus= eVideoStreamingStatus::stopped;
	{
		std::lock_guard<std::mutex> lock(m_depthMutex);
		m_depthCalib.valid= false;
		m_bDepthFresh= false;
	}
}

void RealSenseVideoDevice::notifyFrame(const UsbVideoFrameBuffer& buffer)
{
	std::lock_guard<std::mutex> lock(m_listenerMutex);
	for (IUsbVideoDeviceListener* listener : m_listeners)
		listener->notifyVideoFrameReceived(buffer);
}

void RealSenseVideoDevice::captureThreadLoop()
{
	ThreadUtils::setCurrentThreadName("RealSenseCapture");

	const RealSenseApi* api= RealSenseApi::get();
	const ColorMode& mode= k_colorModes[m_videoModeIndex];
	rs2_error* error= nullptr;

	rs2_pipeline* pipeline= api->create_pipeline(m_context, &error);
	if (api->checkError(error, "create_pipeline") || pipeline == nullptr)
	{
		m_streamingStatus= eVideoStreamingStatus::failed;
		return;
	}

	rs2_config* config= api->create_config(&error);
	api->checkError(error, "create_config");
	api->config_enable_device(config, m_serial.c_str(), &error);
	api->checkError(error, "config_enable_device");
	api->config_enable_stream(config, RS2_STREAM_COLOR, -1, mode.width, mode.height, RS2_FORMAT_RGB8, mode.fps,
							  &error);
	api->checkError(error, "enable color stream");
	api->config_enable_stream(config, RS2_STREAM_DEPTH, -1, kDepthWidth, kDepthHeight, RS2_FORMAT_Z16, kDepthFps,
							  &error);
	api->checkError(error, "enable depth stream");

	rs2_pipeline_profile* profile= api->pipeline_start_with_config(pipeline, config, &error);
	if (api->checkError(error, "pipeline_start") || profile == nullptr)
	{
		api->delete_config(config);
		api->delete_pipeline(pipeline);
		m_streamingStatus= eVideoStreamingStatus::failed;
		return;
	}

	MIKAN_MT_LOG_INFO("RealSenseVideoDevice")
		<< m_friendlyName << " streaming " << mode.width << "x" << mode.height << " color + " << kDepthWidth
		<< "x" << kDepthHeight << " depth";
	m_streamingStatus= eVideoStreamingStatus::started;

	bool bCalibCaptured= false;
	while (!m_bStopRequested)
	{
		rs2_frame* frameset= api->pipeline_wait_for_frames(pipeline, 1000, &error);
		if (error != nullptr)
		{
			api->checkError(error, "wait_for_frames");
			error= nullptr;
			continue;
		}
		if (frameset == nullptr)
			continue;

		rs2_frame* colorFrame= nullptr;
		rs2_frame* depthFrame= nullptr;
		const int embedded= api->embedded_frames_count(frameset, &error);
		api->checkError(error, "embedded_frames_count");
		for (int i= 0; i < embedded; ++i)
		{
			rs2_frame* frame= api->extract_frame(frameset, i, &error);
			if (api->checkError(error, "extract_frame") || frame == nullptr)
				continue;

			const rs2_stream_profile* streamProfile= api->get_frame_stream_profile(frame, &error);
			api->checkError(error, "get_frame_stream_profile");
			rs2_stream stream= RS2_STREAM_ANY;
			rs2_format format= RS2_FORMAT_ANY;
			int index= 0, uniqueId= 0, framerate= 0;
			api->get_stream_profile_data(streamProfile, &stream, &format, &index, &uniqueId, &framerate, &error);
			api->checkError(error, "get_stream_profile_data");

			if (stream == RS2_STREAM_COLOR && colorFrame == nullptr)
				colorFrame= frame;
			else if (stream == RS2_STREAM_DEPTH && depthFrame == nullptr)
				depthFrame= frame;
			else
				api->release_frame(frame);
		}

		// Capture static calibration once both stream profiles are live
		if (!bCalibCaptured && colorFrame != nullptr && depthFrame != nullptr)
		{
			const rs2_stream_profile* colorProfile= api->get_frame_stream_profile(colorFrame, &error);
			api->checkError(error, "color profile");
			const rs2_stream_profile* depthProfile= api->get_frame_stream_profile(depthFrame, &error);
			api->checkError(error, "depth profile");

			rs2_intrinsics colorIntrinsics{}, depthIntrinsics{};
			rs2_extrinsics depthToColor{};
			api->get_video_stream_intrinsics(colorProfile, &colorIntrinsics, &error);
			api->checkError(error, "color intrinsics");
			api->get_video_stream_intrinsics(depthProfile, &depthIntrinsics, &error);
			api->checkError(error, "depth intrinsics");
			api->get_extrinsics(depthProfile, colorProfile, &depthToColor, &error);
			api->checkError(error, "depth->color extrinsics");
			const float depthUnits= api->depth_frame_get_units(depthFrame, &error);
			api->checkError(error, "depth units");

			std::lock_guard<std::mutex> lock(m_depthMutex);
			copyIntrinsics(colorIntrinsics, m_depthCalib.colorIntrinsics);
			copyIntrinsics(depthIntrinsics, m_depthCalib.depthIntrinsics);
			for (int i= 0; i < 9; ++i)
				m_depthCalib.depthToColorRotation[i]= depthToColor.rotation[i];
			for (int i= 0; i < 3; ++i)
				m_depthCalib.depthToColorTranslation[i]= depthToColor.translation[i];
			m_depthCalib.depthWidth= depthIntrinsics.width;
			m_depthCalib.depthHeight= depthIntrinsics.height;
			m_depthCalib.depthUnitsMeters= depthUnits > 0.f ? depthUnits : 0.001f;
			m_depthCalib.valid= true;
			bCalibCaptured= true;
		}

		if (colorFrame != nullptr)
		{
			const uint8_t* colorData= (const uint8_t*)api->get_frame_data(colorFrame, &error);
			api->checkError(error, "color data");
			const double timestampMs= api->get_frame_timestamp(colorFrame, &error);
			api->checkError(error, "color timestamp");
			const int width= api->get_frame_width(colorFrame, &error);
			const int height= api->get_frame_height(colorFrame, &error);
			api->checkError(error, "color dims");

			// Stash the depth plane (own thread-safe copy for fetchDepthView)
			const uint16_t* depthData= nullptr;
			int depthW= 0, depthH= 0;
			if (depthFrame != nullptr)
			{
				depthData= (const uint16_t*)api->get_frame_data(depthFrame, &error);
				api->checkError(error, "depth data");
				depthW= api->get_frame_width(depthFrame, &error);
				depthH= api->get_frame_height(depthFrame, &error);
				api->checkError(error, "depth dims");
			}

			if (colorData != nullptr && width > 0 && height > 0)
			{
				const size_t colorBytes= (size_t)width * height * 3;
				const size_t depthBytes=
					depthData != nullptr ? (size_t)depthW * depthH * sizeof(uint16_t) : 0;

				m_frameScratch.resize(colorBytes + depthBytes);
				std::memcpy(m_frameScratch.data(), colorData, colorBytes);
				if (depthBytes > 0)
					std::memcpy(m_frameScratch.data() + colorBytes, depthData, depthBytes);

				UsbVideoFrameBuffer buffer{};
				buffer.data= m_frameScratch.data();
				buffer.byte_count= colorBytes + depthBytes;
				buffer.data_format= depthBytes > 0 ? eUSBVideoFrameBufferFormat::USBVideo_RGB24_DEPTH16
												   : eUSBVideoFrameBufferFormat::USBVideo_RGB24;
				buffer.section_count= depthBytes > 0 ? 2 : 1;
				buffer.sections[0].pixel_width= width;
				buffer.sections[0].pixel_height= height;
				buffer.sections[0].stride= (size_t)width * 3;
				buffer.sections[0].start_offset= 0;
				buffer.sections[0].byte_count= colorBytes;
				if (depthBytes > 0)
				{
					buffer.sections[1].pixel_width= depthW;
					buffer.sections[1].pixel_height= depthH;
					buffer.sections[1].stride= (size_t)depthW * sizeof(uint16_t);
					buffer.sections[1].start_offset= colorBytes;
					buffer.sections[1].byte_count= depthBytes;
				}

				notifyFrame(buffer);

				if (depthBytes > 0)
				{
					std::lock_guard<std::mutex> lock(m_depthMutex);
					m_depthBuffer.assign(depthData, depthData + (size_t)depthW * depthH);
					m_depthTimestampMs= timestampMs;
					m_bDepthFresh= true;
				}
			}
		}

		if (colorFrame != nullptr)
			api->release_frame(colorFrame);
		if (depthFrame != nullptr)
			api->release_frame(depthFrame);
		api->release_frame(frameset);
	}

	api->pipeline_stop(pipeline, &error);
	api->checkError(error, "pipeline_stop");
	api->delete_pipeline_profile(profile);
	api->delete_config(config);
	api->delete_pipeline(pipeline);
}

bool RealSenseVideoDevice::fetchDepthView(DepthFrameView& outView, std::vector<uint16_t>& ioDepthStorage)
{
	std::lock_guard<std::mutex> lock(m_depthMutex);
	if (!m_depthCalib.valid || m_depthBuffer.empty())
		return false;

	ioDepthStorage= m_depthBuffer; // caller-owned copy: stays valid across frames
	outView= m_depthCalib;
	outView.depthData= ioDepthStorage.data();
	outView.valid= true;
	return true;
}
