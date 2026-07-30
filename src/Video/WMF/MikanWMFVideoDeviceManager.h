#pragma once

#include "IUsbVideoDevice.h"
#include "IUsbVideoDeviceManager.h"
#include "DeviceHotplugNotifier.h"

#include <map>
#include <memory>
#include <vector>

using MikanWMFVideoDevicePtr= std::shared_ptr<class MikanWMFVideoDevice>;

class MikanWMFVideoDeviceManager : public IUsbVideoDeviceManager, public IDeviceHotplugListener
{
public:
	MikanWMFVideoDeviceManager();
	virtual ~MikanWMFVideoDeviceManager();

	// IUsbVideoDeviceManager
	virtual void addListener(IUsbVideoDeviceManagerListener* eventListener) override;
	virtual void removeListener(IUsbVideoDeviceManagerListener* eventListener) override;

	virtual bool startup() override;
	virtual void update(float deltaTime) override;
	virtual void shutdown() override;

	virtual size_t getDeviceCount() const override;
	virtual class IUsbVideoDevice* getDeviceByIndex(size_t index) override;
	virtual class IUsbVideoDevice* getDeviceByPath(const char* devicePath) override;

	// IDeviceHotplugListener
	virtual void onDeviceConnected(const std::string& path) override;
	virtual void onDeviceDisconnected(const std::string& path) override;

protected:
	void rebuildDeviceList();

private:
	std::vector<IUsbVideoDeviceManagerListener*> m_eventListeners;
	class WMFDeviceList* m_wmfDeviceList;
	class DeviceHotplugNotifier* m_deviceHotplugNotifier;
	std::map<std::string, MikanWMFVideoDevicePtr> m_deviceMap;
};