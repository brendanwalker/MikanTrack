#pragma once

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include "readerwriterqueue.h"

#include "IImuDevice.h"

// Nintendo Switch Joy-Con over Bluetooth HID.
//
// Protocol (per dekuNukem/Nintendo_Switch_Reverse_Engineering):
//   OUTPUT report 0x01: [0]=0x01 (report id) [1]=packet counter (wraps 0-F)
//     [2..9]=rumble (neutral) [10]=subcommand [11..]=args
//   Handshake: subcommand 0x40 arg 0x01 (enable IMU), then subcommand 0x03
//     arg 0x30 (input report mode = standard full w/ IMU)
//   INPUT report 0x30: [0]=0x30 [1]=timer [2]=battery/connection
//     [3..5]=buttons [6..8]=L stick [9..11]=R stick [12]=vibrator
//     [13..24],[25..36],[37..48]= THREE 6-axis samples, 5 ms apart
//     each sample: accel XYZ then gyro XYZ, int16 little-endian
//   Scale: accel raw * 0.000244 g, gyro raw * 0.070 deg/s
//
// Deliberately NOT implemented: factory calibration readback from SPI flash
// (subcommand 0x10). Gyro bias is estimated online by the fusion filter,
// which is both more robust and drift-tracking; accel scale error is
// second-order because only the gravity DIRECTION is used for tilt.
class JoyconDevice : public IImuDevice
{
public:
	// Nintendo vendor + Joy-Con/Pro product ids
	static constexpr unsigned short k_vendorId= 0x057E;
	static constexpr unsigned short k_productIdJoyconLeft= 0x2006;
	static constexpr unsigned short k_productIdJoyconRight= 0x2007;
	static constexpr unsigned short k_productIdProController= 0x2009;

	static bool isSupportedProductId(unsigned short productId);

	JoyconDevice(const std::string& devicePath, unsigned short productId);
	virtual ~JoyconDevice();

	// IImuDevice
	virtual const char* getDevicePath() const override { return m_devicePath.c_str(); }
	virtual const char* getFriendlyName() const override { return m_friendlyName.c_str(); }
	virtual eImuSide getSide() const override { return m_side; }
	virtual void setSide(eImuSide side) override { m_side= side; }
	virtual bool open() override;
	virtual void close() override;
	virtual bool isOpen() const override { return m_bOpen; }
	virtual bool isStreaming() const override { return m_bStreaming; }
	virtual size_t fetchSamples(std::vector<ImuSample>& outSamples) override;
	virtual float getSampleRateHz() const override { return m_sampleRateHz.load(); }
	virtual float getBatteryLevel() const override { return m_batteryLevel.load(); }

	// Decodes one 6-axis sample out of an input report buffer at byteOffset.
	// Static + pure so the self test can exercise it without hardware.
	static ImuSample decodeSample(const unsigned char* report, int byteOffset, double timestampMs);

	virtual double getMillisecondsSinceLastSample() const override;

private:
	void readThreadLoop();
	// Sends output report 0x01 with a subcommand; returns false on write failure
	bool sendSubcommand(unsigned char subcommand, const unsigned char* args, int argCount);
	// Output report 0x10: neutral rumble, no subcommand. Sent periodically as
	// a KEEPALIVE - a Joy-Con that hears nothing from the host eventually
	// sleeps, even while it is happily streaming input reports at us.
	bool sendRumbleKeepalive();
	// Shared output-report construction (reportId 0x01 = rumble+subcommand,
	// 0x10 = rumble only)
	bool sendOutputReport(unsigned char reportId, int subcommand, const unsigned char* args, int argCount);

	std::string m_devicePath;
	std::string m_friendlyName;
	unsigned short m_productId= 0;
	eImuSide m_side= eImuSide::Unassigned;

	void* m_handle= nullptr; // HANDLE (avoid dragging windows.h into the header)
	unsigned long m_inputReportLength= 0;
	unsigned long m_outputReportLength= 0;
	unsigned char m_packetCounter= 0;

	std::atomic_bool m_bOpen{false};
	std::atomic_bool m_bStreaming{false};
	std::atomic_bool m_bStopRequested{false};
	std::thread m_readThread;

	// Read thread (producer) -> fusion thread (consumer)
	moodycamel::ReaderWriterQueue<ImuSample> m_sampleQueue;

	std::atomic<float> m_sampleRateHz{0.f};
	std::atomic<float> m_batteryLevel{-1.f};
	// steady_clock ms of the last delivered sample (-1 = never)
	std::atomic<double> m_lastSampleArrivalMs{-1.0};
};
