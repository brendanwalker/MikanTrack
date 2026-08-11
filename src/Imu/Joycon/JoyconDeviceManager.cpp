#include "JoyconDeviceManager.h"

#include <string>

#include <windows.h>

#include <setupapi.h>

extern "C"
{
#include <hidsdi.h>
}

#include "Logger.h"

JoyconDeviceManager::~JoyconDeviceManager()
{
	shutdown();
}

bool JoyconDeviceManager::startup()
{
	refreshConnectedDevices();
	return true;
}

void JoyconDeviceManager::shutdown()
{
	for (std::shared_ptr<JoyconDevice>& device : m_devices)
		device->close();
	m_devices.clear();
}

void JoyconDeviceManager::refreshConnectedDevices()
{
	GUID hidGuid;
	HidD_GetHidGuid(&hidGuid);

	HDEVINFO deviceInfoSet= SetupDiGetClassDevsA(&hidGuid, nullptr, nullptr,
												 DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
	if (deviceInfoSet == INVALID_HANDLE_VALUE)
	{
		MIKAN_LOG_ERROR("JoyconDeviceManager") << "SetupDiGetClassDevs failed";
		return;
	}

	std::vector<std::shared_ptr<JoyconDevice>> devices;

	SP_DEVICE_INTERFACE_DATA interfaceData= {};
	interfaceData.cbSize= sizeof(interfaceData);
	for (DWORD index= 0; SetupDiEnumDeviceInterfaces(deviceInfoSet, nullptr, &hidGuid, index, &interfaceData);
		 ++index)
	{
		DWORD requiredSize= 0;
		SetupDiGetDeviceInterfaceDetailA(deviceInfoSet, &interfaceData, nullptr, 0, &requiredSize, nullptr);
		if (requiredSize == 0)
			continue;

		std::vector<unsigned char> detailBuffer(requiredSize, 0);
		auto* detail= reinterpret_cast<PSP_DEVICE_INTERFACE_DETAIL_DATA_A>(detailBuffer.data());
		detail->cbSize= sizeof(SP_DEVICE_INTERFACE_DETAIL_DATA_A);
		if (!SetupDiGetDeviceInterfaceDetailA(deviceInfoSet, &interfaceData, detail, requiredSize, nullptr,
											  nullptr))
		{
			continue;
		}

		// Probe VID/PID without disturbing a device we don't want: open
		// shared, query attributes, close.
		HANDLE probe= ::CreateFileA(detail->DevicePath, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
									OPEN_EXISTING, 0, nullptr);
		if (probe == INVALID_HANDLE_VALUE)
			continue;

		HIDD_ATTRIBUTES attributes= {};
		attributes.Size= sizeof(attributes);
		const bool bHaveAttributes= HidD_GetAttributes(probe, &attributes) != FALSE;
		::CloseHandle(probe);

		if (!bHaveAttributes || attributes.VendorID != JoyconDevice::k_vendorId ||
			!JoyconDevice::isSupportedProductId(attributes.ProductID))
		{
			continue;
		}

		const std::string devicePath= detail->DevicePath;

		// Keep an already-open device object (and its stream) across refreshes
		bool bReused= false;
		for (std::shared_ptr<JoyconDevice>& existing : m_devices)
		{
			if (existing != nullptr && devicePath == existing->getDevicePath())
			{
				devices.push_back(std::move(existing));
				bReused= true;
				break;
			}
		}
		if (!bReused)
			devices.push_back(std::make_shared<JoyconDevice>(devicePath, attributes.ProductID));
	}

	SetupDiDestroyDeviceInfoList(deviceInfoSet);

	// Anything not carried over is gone. Dropping the reference closes it
	// through the destructor unless a consumer still holds one - deliberately
	// NOT an explicit close here: this runs on the discovery worker, and
	// closing a device out from under the thread draining its samples is what
	// shared ownership exists to prevent. The last holder closes it.
	m_devices= std::move(devices);

	MIKAN_LOG_INFO("JoyconDeviceManager") << m_devices.size() << " Joy-Con device(s) found";
}
