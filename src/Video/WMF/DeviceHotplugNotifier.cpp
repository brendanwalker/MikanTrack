// -- include -----
#include "DeviceHotplugNotifier.h"
#include "Logger.h"

#define ANSI
#define WIN32_LEAN_AND_MEAN

#include <windows.h> // Required for data types
#include <winuser.h>
#include <Dbt.h>
#include <guiddef.h>
#include <setupapi.h> // Device setup APIs
#include <assert.h>
#include <strsafe.h>
#include <winreg.h>
#include <Shellapi.h>

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

//-- constants -----
GUID GUID_DEVCLASS_IMAGE= {0x6bdd1fc6, 0x810f, 0x11d0, 0xbe, 0xc7, 0x08, 0x00, 0x2b, 0xe2, 0x09, 0x2f};

#define CLS_NAME "DEVICE_LISTENER_CLASS"
#define HWND_MESSAGE_ONLY ((HWND) - 3)

// -- globals ----
struct DeviceHotplugNotifierImpl
{
	IDeviceHotplugListener* listener= nullptr;
	HDEVNOTIFY hImageDeviceNotify= nullptr;
	HWND hWnd= nullptr;
	WNDCLASSEX wx;
};

//-- private prototypes -----
static HDEVNOTIFY register_device_class_notification(HWND__* hwnd, const GUID& guid);
static LRESULT message_handler(HWND__* hwnd, UINT uint, WPARAM wparam, LPARAM lparam);

// -- definitions -----
DeviceHotplugNotifier::DeviceHotplugNotifier()
	: m_impl(new DeviceHotplugNotifierImpl)
{
}

DeviceHotplugNotifier::~DeviceHotplugNotifier() { delete m_impl; }

bool DeviceHotplugNotifier::startup(IDeviceHotplugListener* listener)
{
	bool bSuccess= true;

	if (m_impl->hWnd == nullptr)
	{
		ZeroMemory(&m_impl->wx, sizeof(m_impl->wx));

		m_impl->wx.cbSize= sizeof(WNDCLASSEX);
		m_impl->wx.lpfnWndProc= reinterpret_cast<WNDPROC>(message_handler);
		m_impl->wx.style= CS_HREDRAW | CS_VREDRAW;
		m_impl->wx.hInstance= GetModuleHandle(0);
		m_impl->wx.hbrBackground= (HBRUSH)(COLOR_WINDOW);
		m_impl->wx.lpszClassName= CLS_NAME;

		if (RegisterClassEx(&m_impl->wx))
		{
			m_impl->hWnd= CreateWindow(CLS_NAME, "DevNotifWnd", WS_ICONIC, 0, 0, CW_USEDEFAULT, 0, HWND_MESSAGE_ONLY,
									   NULL, GetModuleHandle(0),
									   this); // Pass 'this' as the window lpParam
		}

		if (m_impl->hWnd != nullptr)
		{
			m_impl->listener= listener;
		}
		else
		{
			MIKAN_LOG_ERROR("DeviceHotplugAPIWin32::startup") << "Could not create message window!";
			bSuccess= false;
		}
	}
	else
	{
		MIKAN_LOG_ERROR("DeviceHotplugAPIWin32::startup") << "Message handler window already created";
	}

	return bSuccess;
}

void DeviceHotplugNotifier::update()
{
	MSG msg;
	while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE) > 0)
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
}

void DeviceHotplugNotifier::shutdown()
{
	if (m_impl->hImageDeviceNotify != nullptr)
	{
		UnregisterDeviceNotification(m_impl->hImageDeviceNotify);
		m_impl->hImageDeviceNotify= nullptr;
	}

	if (m_impl->hWnd != nullptr)
	{
		DestroyWindow(m_impl->hWnd);
		m_impl->hWnd= nullptr;
	}

	if (m_impl->wx.hInstance != nullptr)
	{
		UnregisterClassA(m_impl->wx.lpszClassName, m_impl->wx.hInstance);
		ZeroMemory(&m_impl->wx, sizeof(m_impl->wx));
	}
}

//-- private helper methods -----
LRESULT message_handler(HWND__* hwnd, UINT msg_type, WPARAM wparam, LPARAM lparam)
{
	auto pThis= reinterpret_cast<DeviceHotplugNotifier*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

	switch (msg_type)
	{
	case WM_NCCREATE:
	{
		auto cs= reinterpret_cast<CREATESTRUCT*>(lparam);
		auto pThis= static_cast<DeviceHotplugNotifier*>(cs->lpCreateParams);

		// Store pointer for later messages
		SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pThis));
	}
	break;

	case WM_DESTROY:
	{
		SetWindowLongPtr(hwnd, GWLP_USERDATA, 0);
		PostQuitMessage(0);
	}
		return 0;

	case WM_CREATE:
	{
		if (pThis != nullptr)
		{
			pThis->getPrivateImpl()->hImageDeviceNotify= register_device_class_notification(hwnd, GUID_DEVCLASS_IMAGE);
		}
		break;
	}

	case WM_DEVICECHANGE:
	{
		PDEV_BROADCAST_HDR lpdb= (PDEV_BROADCAST_HDR)lparam;

		if (lpdb->dbch_devicetype == DBT_DEVTYP_DEVICEINTERFACE)
		{
			PDEV_BROADCAST_DEVICEINTERFACE lpdbv= (PDEV_BROADCAST_DEVICEINTERFACE)lpdb;
			std::string path= std::string(lpdbv->dbcc_name);

			if (IsEqualCLSID(lpdbv->dbcc_classguid, GUID_DEVCLASS_IMAGE))
			{
				switch (wparam)
				{
				case DBT_DEVICEARRIVAL:
					pThis->getPrivateImpl()->listener->onDeviceConnected(path);
					break;

				case DBT_DEVICEREMOVECOMPLETE:
					pThis->getPrivateImpl()->listener->onDeviceDisconnected(path);
					break;
				}
			}
		}
		break;
	}
	}

	return DefWindowProc(hwnd, msg_type, wparam, lparam);
}

static HDEVNOTIFY register_device_class_notification(HWND__* hwnd, const GUID& guid)
{
	DEV_BROADCAST_DEVICEINTERFACE NotificationFilter;
	ZeroMemory(&NotificationFilter, sizeof(NotificationFilter));
	NotificationFilter.dbcc_size= sizeof(DEV_BROADCAST_DEVICEINTERFACE);
	NotificationFilter.dbcc_devicetype= DBT_DEVTYP_DEVICEINTERFACE;
	NotificationFilter.dbcc_classguid= guid;
	HDEVNOTIFY dev_notify= RegisterDeviceNotification(hwnd, &NotificationFilter, DEVICE_NOTIFY_WINDOW_HANDLE);

	if (dev_notify == nullptr)
	{
		MIKAN_LOG_ERROR("RegisterDeviceClassNotification") << "Could not register for device notifications!";
	}

	return dev_notify;
}