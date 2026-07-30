#pragma once

#include <memory>
#include <stdint.h>

class IUsbVideoDeviceManagerListener
{
public:
	virtual ~IUsbVideoDeviceManagerListener() {};

	virtual void onConnectedDeviceListChanged()= 0;
};

class IUsbVideoDeviceManager
{
public:
	IUsbVideoDeviceManager()= default;
	virtual ~IUsbVideoDeviceManager() {}

	virtual void addListener(IUsbVideoDeviceManagerListener* eventListener)= 0;
	virtual void removeListener(IUsbVideoDeviceManagerListener* eventListener)= 0;

	virtual bool startup()= 0;
	virtual void update(float deltaTime)= 0;
	virtual void shutdown()= 0;

	virtual size_t getDeviceCount() const= 0;
	virtual class IUsbVideoDevice* getDeviceByIndex(size_t index)= 0;
	virtual class IUsbVideoDevice* getDeviceByPath(const char* devicePath)= 0;
};
using IUsbVideoDeviceManagerPtr= std::shared_ptr<IUsbVideoDeviceManager>;