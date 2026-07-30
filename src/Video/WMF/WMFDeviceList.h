#pragma once

#include <string>
#include <vector>

class WMFDeviceList
{
public:
	WMFDeviceList()= default;

	const std::vector<struct WMFDeviceInfo>& getDevices() const { return m_deviceList; }
	bool isDevicePresent(const std::string& devicePath) const;

	size_t getDeviceCount() const { return m_deviceList.size(); }
	const struct WMFDeviceInfo* getDeviceByIndex(size_t index) const;

	bool rebuild();

protected:
	std::vector<struct WMFDeviceInfo> m_deviceList;
};
