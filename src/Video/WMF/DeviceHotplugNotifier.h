#pragma once

#include <string>

class IDeviceHotplugListener
{
public:
	virtual void onDeviceConnected(const std::string& path)= 0;
	virtual void onDeviceDisconnected(const std::string& path)= 0;
};

class DeviceHotplugNotifier
{
public:
	DeviceHotplugNotifier();
	virtual ~DeviceHotplugNotifier();

	bool startup(IDeviceHotplugListener* listener);
	void update();
	void shutdown();

	struct DeviceHotplugNotifierImpl* getPrivateImpl() { return m_impl; }

private:
	struct DeviceHotplugNotifierImpl* m_impl;
};