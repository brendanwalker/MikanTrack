#include "RealSenseVideoDeviceManager.h"

#include <cstring>

#include "Logger.h"
#include "RealSenseApi.h"
#include "RealSenseVideoDevice.h"

#include "librealsense2/rs.h"

RealSenseVideoDeviceManager::~RealSenseVideoDeviceManager()
{
	shutdown();
}

void RealSenseVideoDeviceManager::addListener(IUsbVideoDeviceManagerListener* listener)
{
	m_listeners.push_back(listener);
}

void RealSenseVideoDeviceManager::removeListener(IUsbVideoDeviceManagerListener* listener)
{
	m_listeners.erase(std::remove(m_listeners.begin(), m_listeners.end(), listener), m_listeners.end());
}

bool RealSenseVideoDeviceManager::startup()
{
	const RealSenseApi* api= RealSenseApi::get();
	if (!api->isAvailable())
		return true; // no dll: backend stays empty, app runs normally

	rs2_error* error= nullptr;
	m_context= api->create_context(RS2_API_VERSION, &error);
	if (api->checkError(error, "create_context") || m_context == nullptr)
	{
		m_context= nullptr;
		return true;
	}

	refreshConnectedDevices();
	return true;
}

void RealSenseVideoDeviceManager::update(float deltaTime)
{
	// Hotplug is handled by the explicit refresh path (the app's device list
	// UI calls refreshConnectedDevices), matching the WMF manager's behavior
}

void RealSenseVideoDeviceManager::shutdown()
{
	m_devices.clear();
	if (m_context != nullptr)
	{
		RealSenseApi::get()->delete_context(m_context);
		m_context= nullptr;
	}
}

void RealSenseVideoDeviceManager::refreshConnectedDevices()
{
	if (m_context == nullptr)
		return;

	const RealSenseApi* api= RealSenseApi::get();
	rs2_error* error= nullptr;

	rs2_device_list* list= api->query_devices(m_context, &error);
	if (api->checkError(error, "query_devices") || list == nullptr)
		return;

	const int count= api->get_device_count(list, &error);
	api->checkError(error, "get_device_count");

	std::vector<std::shared_ptr<RealSenseVideoDevice>> devices;
	for (int i= 0; i < count; ++i)
	{
		rs2_device* device= api->create_device(list, i, &error);
		if (api->checkError(error, "create_device") || device == nullptr)
			continue;

		const char* serial= api->get_device_info(device, RS2_CAMERA_INFO_SERIAL_NUMBER, &error);
		api->checkError(error, "serial");
		const char* name= api->get_device_info(device, RS2_CAMERA_INFO_NAME, &error);
		api->checkError(error, "name");

		if (serial != nullptr)
		{
			// Keep an existing device object (with its stream state) when the
			// same camera is still connected
			std::shared_ptr<RealSenseVideoDevice> existing;
			for (const auto& knownDevice : m_devices)
			{
				if (knownDevice->getSerial() == serial)
					existing= knownDevice;
			}
			if (existing != nullptr)
				devices.push_back(existing);
			else
				devices.push_back(std::make_shared<RealSenseVideoDevice>(
					m_context, serial, name != nullptr ? name : "device"));
		}

		api->delete_device(device);
	}
	api->delete_device_list(list);

	const bool bChanged= devices.size() != m_devices.size();
	m_devices= std::move(devices);
	if (bChanged)
	{
		for (IUsbVideoDeviceManagerListener* listener : m_listeners)
			listener->onConnectedDeviceListChanged();
	}

	if (!m_devices.empty())
		MIKAN_LOG_INFO("RealSenseVideoDeviceManager") << m_devices.size() << " RealSense device(s) found";
}

size_t RealSenseVideoDeviceManager::getDeviceCount() const
{
	return m_devices.size();
}

IUsbVideoDevice* RealSenseVideoDeviceManager::getDeviceByIndex(size_t index)
{
	return index < m_devices.size() ? m_devices[index].get() : nullptr;
}

IUsbVideoDevice* RealSenseVideoDeviceManager::getDeviceByPath(const char* devicePath)
{
	for (const auto& device : m_devices)
	{
		if (std::strcmp(device->getDevicePath(), devicePath) == 0)
			return device.get();
	}
	return nullptr;
}

bool RealSenseVideoDeviceManager::isRealSensePath(const std::string& devicePath)
{
	return devicePath.rfind("rs://", 0) == 0;
}

bool RealSenseVideoDeviceManager::fetchDepthView(const std::string& devicePath, double colorTimestampMs,
												 RealSenseDepthView& outView)
{
	// The caller owns depth storage via a thread-local; simplest correct
	// implementation routes through the device's double-buffered copy.
	// (colorTimestampMs reserved for future frame matching; the newest depth
	// frame is within one frame interval of the newest color frame since the
	// pipeline delivers synchronized framesets.)
	static thread_local std::vector<uint16_t> s_depthStorage;
	for (const auto& device : m_devices)
	{
		if (devicePath.empty() || devicePath == device->getDevicePath())
			return device->fetchDepthView(outView, s_depthStorage);
	}
	return false;
}
