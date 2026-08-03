#include "JoyconDevice.h"

#include <chrono>
#include <cstring>

#include <windows.h>

extern "C"
{
#include <hidsdi.h>
}

#include "Logger.h"
#include "ThreadUtils.h"

// Raw -> physical scales (dekuNukem reverse-engineering notes)
static constexpr float k_accelScaleG= 0.000244f;      // g per LSB (+/-8g range)
static constexpr float k_gyroScaleDps= 0.070f;        // deg/s per LSB (+/-2000dps)
static constexpr float k_gravityMetersPerSecond2= 9.80665f;
static constexpr float k_degreesToRadians= 0.01745329252f;

// Report layout
static constexpr int k_imuFirstSampleOffset= 13;
static constexpr int k_imuSampleStride= 12;
static constexpr int k_imuSampleCount= 3;
static constexpr double k_imuSampleSpacingMs= 5.0;

// Neutral rumble payload every output report has to carry
static const unsigned char k_neutralRumble[8]= {0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40};

static double nowSteadyMs()
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool JoyconDevice::isSupportedProductId(unsigned short productId)
{
	return productId == k_productIdJoyconLeft || productId == k_productIdJoyconRight ||
		productId == k_productIdProController;
}

JoyconDevice::JoyconDevice(const std::string& devicePath, unsigned short productId)
	: m_devicePath(devicePath)
	, m_productId(productId)
	, m_sampleQueue(256)
{
	switch (productId)
	{
		case k_productIdJoyconLeft:
			m_friendlyName= "Joy-Con (L)";
			// A Joy-Con L defaults to the left wrist; overridable
			m_side= eImuSide::Left;
			break;
		case k_productIdJoyconRight:
			m_friendlyName= "Joy-Con (R)";
			m_side= eImuSide::Right;
			break;
		default:
			m_friendlyName= "Switch Pro Controller";
			m_side= eImuSide::Unassigned;
			break;
	}
}

JoyconDevice::~JoyconDevice()
{
	close();
}

ImuSample JoyconDevice::decodeSample(const unsigned char* report, int byteOffset, double timestampMs)
{
	auto readInt16= [report](int offset) -> int {
		return (int)(short)((unsigned short)report[offset] | ((unsigned short)report[offset + 1] << 8));
	};

	ImuSample sample;
	sample.timestampMs= timestampMs;

	// accel XYZ then gyro XYZ, int16 little-endian
	sample.acceleration= glm::vec3((float)readInt16(byteOffset + 0), (float)readInt16(byteOffset + 2),
								   (float)readInt16(byteOffset + 4)) *
		(k_accelScaleG * k_gravityMetersPerSecond2);
	sample.angularVelocity= glm::vec3((float)readInt16(byteOffset + 6), (float)readInt16(byteOffset + 8),
									  (float)readInt16(byteOffset + 10)) *
		(k_gyroScaleDps * k_degreesToRadians);

	return sample;
}

bool JoyconDevice::sendSubcommand(unsigned char subcommand, const unsigned char* args, int argCount)
{
	return sendOutputReport(0x01, (int)subcommand, args, argCount);
}

bool JoyconDevice::sendRumbleKeepalive()
{
	return sendOutputReport(0x10, -1, nullptr, 0);
}

bool JoyconDevice::sendOutputReport(unsigned char reportId, int subcommand, const unsigned char* args,
									int argCount)
{
	if (m_handle == nullptr || m_outputReportLength == 0)
		return false;

	std::vector<unsigned char> buffer(m_outputReportLength, 0);
	buffer[0]= reportId; // 0x01 = rumble + subcommand, 0x10 = rumble only
	buffer[1]= (unsigned char)(m_packetCounter & 0x0F);
	m_packetCounter++;
	std::memcpy(&buffer[2], k_neutralRumble, sizeof(k_neutralRumble));
	if (subcommand >= 0)
	{
		buffer[10]= (unsigned char)subcommand;
		for (int i= 0; i < argCount && (11 + i) < (int)buffer.size(); ++i)
			buffer[11 + i]= args[i];
	}

	// Overlapped write (the handle is opened FILE_FLAG_OVERLAPPED)
	OVERLAPPED overlapped= {};
	overlapped.hEvent= ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
	if (overlapped.hEvent == nullptr)
		return false;

	DWORD written= 0;
	bool bSuccess= ::WriteFile((HANDLE)m_handle, buffer.data(), (DWORD)buffer.size(), &written, &overlapped) != FALSE;
	if (!bSuccess && ::GetLastError() == ERROR_IO_PENDING)
		bSuccess= ::WaitForSingleObject(overlapped.hEvent, 200) == WAIT_OBJECT_0 &&
			::GetOverlappedResult((HANDLE)m_handle, &overlapped, &written, FALSE) != FALSE;
	if (!bSuccess)
		::CancelIo((HANDLE)m_handle);

	::CloseHandle(overlapped.hEvent);
	return bSuccess;
}

bool JoyconDevice::open()
{
	if (m_bOpen)
		return true;

	HANDLE handle= ::CreateFileA(m_devicePath.c_str(), GENERIC_READ | GENERIC_WRITE,
								 FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
								 FILE_FLAG_OVERLAPPED, nullptr);
	if (handle == INVALID_HANDLE_VALUE)
	{
		MIKAN_LOG_ERROR("JoyconDevice::open")
			<< m_friendlyName << ": CreateFile failed (" << ::GetLastError()
			<< ") - is the controller paired and not claimed by another app?";
		return false;
	}

	// Report lengths come from the HID caps (Windows requires writes/reads of
	// exactly these sizes, zero padded)
	PHIDP_PREPARSED_DATA preparsed= nullptr;
	if (!HidD_GetPreparsedData(handle, &preparsed))
	{
		MIKAN_LOG_ERROR("JoyconDevice::open") << m_friendlyName << ": HidD_GetPreparsedData failed";
		::CloseHandle(handle);
		return false;
	}
	HIDP_CAPS caps= {};
	const bool bHaveCaps= HidP_GetCaps(preparsed, &caps) == HIDP_STATUS_SUCCESS;
	HidD_FreePreparsedData(preparsed);
	if (!bHaveCaps)
	{
		MIKAN_LOG_ERROR("JoyconDevice::open") << m_friendlyName << ": HidP_GetCaps failed";
		::CloseHandle(handle);
		return false;
	}

	m_handle= handle;
	m_inputReportLength= caps.InputReportByteLength;
	m_outputReportLength= caps.OutputReportByteLength;
	m_packetCounter= 0;
	m_bOpen= true;
	m_bStopRequested= false;

	// Handshake. Order matters: leave shipment low-power mode FIRST (a
	// controller still in it sleeps aggressively no matter what else we do),
	// then enable the IMU, then switch to the report mode that carries it.
	const unsigned char disableShipmentMode= 0x00;
	const unsigned char enableImu= 0x01;
	const unsigned char reportMode= 0x30;
	sendSubcommand(0x08, &disableShipmentMode, 1);
	::Sleep(30); // the controller needs a beat between subcommands
	const bool bImuEnabled= sendSubcommand(0x40, &enableImu, 1);
	::Sleep(30);
	const bool bModeSet= sendSubcommand(0x03, &reportMode, 1);
	if (!bImuEnabled || !bModeSet)
	{
		MIKAN_LOG_ERROR("JoyconDevice::open") << m_friendlyName << ": IMU handshake failed";
		close();
		return false;
	}

	// Player LED: 1 for the left wrist, 2 for the right. Purely so a glance
	// at the controllers tells you which one the app thinks is which.
	::Sleep(30);
	const unsigned char playerLight= m_side == eImuSide::Right ? 0x02 : 0x01;
	sendSubcommand(0x30, &playerLight, 1);

	m_readThread= std::thread([this]() { readThreadLoop(); });

	MIKAN_LOG_INFO("JoyconDevice::open")
		<< m_friendlyName << " opened (input report " << m_inputReportLength << " bytes, output "
		<< m_outputReportLength << ")";
	return true;
}

void JoyconDevice::close()
{
	m_bStopRequested= true;
	if (m_handle != nullptr)
		::CancelIoEx((HANDLE)m_handle, nullptr); // unblock a pending read

	if (m_readThread.joinable())
		m_readThread.join();

	if (m_handle != nullptr)
	{
		::CloseHandle((HANDLE)m_handle);
		m_handle= nullptr;
	}
	m_bOpen= false;
	m_bStreaming= false;
	m_sampleRateHz= 0.f;
}

void JoyconDevice::readThreadLoop()
{
	ThreadUtils::setCurrentThreadName("JoyconRead");

	std::vector<unsigned char> report(m_inputReportLength, 0);
	HANDLE readEvent= ::CreateEvent(nullptr, TRUE, FALSE, nullptr);
	if (readEvent == nullptr)
		return;

	int samplesThisSecond= 0;
	double rateWindowStartMs= nowSteadyMs();
	// Keepalive cadence. The controller streams input reports at us
	// regardless, but hears nothing back unless we send something - and a
	// Joy-Con that hears nothing from the host eventually sleeps. Sending
	// from THIS thread (rather than a timer thread) keeps all I/O on the
	// handle single-threaded.
	constexpr double kKeepaliveIntervalMs= 1000.0;
	double lastKeepaliveMs= nowSteadyMs();

	while (!m_bStopRequested)
	{
		if (nowSteadyMs() - lastKeepaliveMs >= kKeepaliveIntervalMs)
		{
			sendRumbleKeepalive();
			lastKeepaliveMs= nowSteadyMs();
		}

		OVERLAPPED overlapped= {};
		overlapped.hEvent= readEvent;
		::ResetEvent(readEvent);

		DWORD bytesRead= 0;
		bool bSuccess= ::ReadFile((HANDLE)m_handle, report.data(), (DWORD)report.size(), &bytesRead,
								  &overlapped) != FALSE;
		if (!bSuccess)
		{
			if (::GetLastError() != ERROR_IO_PENDING)
				break;

			// 200ms timeout so a silent controller doesn't wedge shutdown
			const DWORD waitResult= ::WaitForSingleObject(readEvent, 200);
			if (waitResult == WAIT_TIMEOUT)
			{
				::CancelIo((HANDLE)m_handle);
				continue;
			}
			if (waitResult != WAIT_OBJECT_0 ||
				!::GetOverlappedResult((HANDLE)m_handle, &overlapped, &bytesRead, FALSE))
			{
				break;
			}
		}

		// Only the full-mode report carries IMU data (the controller also
		// emits 0x3F button-only reports before the mode switch takes)
		if (bytesRead < (DWORD)(k_imuFirstSampleOffset + k_imuSampleStride * k_imuSampleCount) ||
			report[0] != 0x30)
		{
			continue;
		}

		const double arrivalMs= nowSteadyMs();

		// Battery: high nibble of byte 2, 8=full .. 0=empty
		m_batteryLevel= (float)((report[2] & 0xF0) >> 4) / 8.f;

		// Three samples 5ms apart. Sample 0 is treated as the OLDEST, so the
		// newest lands at the arrival timestamp. If fusion ever shows a
		// systematic ~10ms orientation lead/lag, this ordering is the first
		// thing to flip.
		for (int i= 0; i < k_imuSampleCount; ++i)
		{
			const double sampleTimeMs=
				arrivalMs - (double)(k_imuSampleCount - 1 - i) * k_imuSampleSpacingMs;
			const ImuSample sample=
				decodeSample(report.data(), k_imuFirstSampleOffset + i * k_imuSampleStride, sampleTimeMs);

			// Drop oldest on overflow rather than block the read thread
			if (!m_sampleQueue.try_enqueue(sample))
			{
				ImuSample discarded;
				m_sampleQueue.try_dequeue(discarded);
				m_sampleQueue.try_enqueue(sample);
			}
			samplesThisSecond++;
		}

		m_bStreaming= true;
		m_lastSampleArrivalMs= arrivalMs;

		if (arrivalMs - rateWindowStartMs >= 1000.0)
		{
			m_sampleRateHz= (float)(samplesThisSecond * 1000.0 / (arrivalMs - rateWindowStartMs));
			samplesThisSecond= 0;
			rateWindowStartMs= arrivalMs;
		}
	}

	::CloseHandle(readEvent);
	m_bStreaming= false;
}

double JoyconDevice::getMillisecondsSinceLastSample() const
{
	const double lastArrival= m_lastSampleArrivalMs.load();
	return lastArrival < 0.0 ? -1.0 : nowSteadyMs() - lastArrival;
}

size_t JoyconDevice::fetchSamples(std::vector<ImuSample>& outSamples)
{
	size_t count= 0;
	ImuSample sample;
	while (m_sampleQueue.try_dequeue(sample))
	{
		outSamples.push_back(sample);
		count++;
	}
	return count;
}
