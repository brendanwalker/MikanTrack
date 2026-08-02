#pragma once

#include <memory>
#include <vector>

#include "IImuDevice.h"
#include "JoyconDevice.h"

// Enumerates paired Joy-Cons through the Windows HID interface class.
// A Joy-Con appears here once Windows has it paired as a Bluetooth HID
// device - pairing itself is done in Windows Bluetooth settings (hold the
// small sync button on the rail until the lights run).
class JoyconDeviceManager : public IImuDeviceManager
{
public:
	virtual ~JoyconDeviceManager();

	virtual bool startup() override;
	virtual void shutdown() override;
	virtual void refreshConnectedDevices() override;
	virtual size_t getDeviceCount() const override { return m_devices.size(); }
	virtual IImuDevice* getDeviceByIndex(size_t index) override
	{
		return index < m_devices.size() ? m_devices[index].get() : nullptr;
	}

private:
	std::vector<std::unique_ptr<JoyconDevice>> m_devices;
};
