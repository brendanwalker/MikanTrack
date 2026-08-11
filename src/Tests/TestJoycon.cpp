#include "TestCommon.h"

static int runJoyconTest(const TestArgs& args)
{
	int result= 0;

	// (a) Decode unit test: hand-built report bytes with known values
	{
		unsigned char report[64]= {};
		report[0]= 0x30;
		// sample 0 at offset 13: accel (1000, -2000, 4096), gyro (100, -100, 0)
		auto writeInt16= [&report](int offset, short value) {
			report[offset]= (unsigned char)(value & 0xFF);
			report[offset + 1]= (unsigned char)((value >> 8) & 0xFF);
		};
		writeInt16(13, 1000);
		writeInt16(15, -2000);
		writeInt16(17, 4096);
		writeInt16(19, 100);
		writeInt16(21, -100);
		writeInt16(23, 0);

		const ImuSample sample= JoyconDevice::decodeSample(report, 13, 1234.0);
		// 4096 raw * 0.000244 g * 9.80665 = ~9.80 m/s^2 (i.e. ~1g)
		const float expectedZ= 4096.f * 0.000244f * 9.80665f;
		const float expectedGyroX= 100.f * 0.070f * 0.01745329252f;
		MIKAN_LOG_INFO("test-joycon")
			<< "(a) decode: accel=(" << sample.acceleration.x << "," << sample.acceleration.y << ","
			<< sample.acceleration.z << ") gyro=(" << sample.angularVelocity.x << ","
			<< sample.angularVelocity.y << "," << sample.angularVelocity.z << ")";
		if (fabsf(sample.acceleration.z - expectedZ) > 1e-4f ||
			fabsf(sample.angularVelocity.x - expectedGyroX) > 1e-6f ||
			sample.acceleration.y >= 0.f || sample.angularVelocity.y >= 0.f ||
			sample.timestampMs != 1234.0)
		{
			MIKAN_LOG_ERROR("test-joycon") << "(a) FAILED: sample decode mismatch";
			result= 1;
		}
	}

	// (b) Live readout: enumerate, open, stream for a few seconds.
	// This is the hardware gate for the whole IMU feature - if a
	// Joy-Con won't stream here, nothing downstream can work.
	{
		JoyconDeviceManager manager;
		manager.startup();

		const size_t deviceCount= manager.getDeviceCount();
		MIKAN_LOG_INFO("test-joycon") << "(b) devices found: " << deviceCount;
		if (deviceCount == 0)
		{
			MIKAN_LOG_WARNING("test-joycon")
				<< "No Joy-Cons paired. Pair them in Windows Bluetooth settings first "
				   "(hold the small sync button on the rail until the lights run), then rerun.";
		}

		for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
		{
			std::shared_ptr<IImuDevice> device= manager.getDeviceByIndex(deviceIndex);
			MIKAN_LOG_INFO("test-joycon")
				<< "  [" << deviceIndex << "] " << device->getFriendlyName() << " side="
				<< (device->getSide() == eImuSide::Left ? "Left"
													   : (device->getSide() == eImuSide::Right ? "Right"
																						      : "Unassigned"));
			if (!device->open())
			{
				MIKAN_LOG_ERROR("test-joycon") << "  open FAILED";
				result= 1;
			}
		}

		// Stream for 5 seconds, reporting once a second
		std::vector<ImuSample> samples;
		for (int second= 0; second < 5 && deviceCount > 0; ++second)
		{
			std::this_thread::sleep_for(std::chrono::seconds(1));
			for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
			{
				std::shared_ptr<IImuDevice> device= manager.getDeviceByIndex(deviceIndex);
				samples.clear();
				device->fetchSamples(samples);
				if (samples.empty())
				{
					MIKAN_LOG_WARNING("test-joycon")
						<< "  " << device->getFriendlyName() << ": no samples this second";
					continue;
				}

				const ImuSample& newest= samples.back();
				const float accelMagnitude= glm::length(newest.acceleration);
				MIKAN_LOG_INFO("test-joycon")
					<< "  " << device->getFriendlyName() << ": " << samples.size() << " samples, "
					<< device->getSampleRateHz() << " Hz, battery "
					<< (int)(device->getBatteryLevel() * 100.f) << "%"
					<< " | accel (" << newest.acceleration.x << ", " << newest.acceleration.y << ", "
					<< newest.acceleration.z << ") |a|=" << accelMagnitude
					<< " | gyro (" << newest.angularVelocity.x << ", " << newest.angularVelocity.y
					<< ", " << newest.angularVelocity.z << ")";

				// Held still, |accel| must read ~9.81 - the single best
				// sanity check that scaling and decode are right
				if (second == 4 && fabsf(accelMagnitude - 9.80665f) > 2.5f)
				{
					MIKAN_LOG_WARNING("test-joycon")
						<< "  |accel| is " << accelMagnitude
						<< " m/s^2, expected ~9.81 when held still - check the accel scale";
				}
			}
		}

		bool bAnyStreamed= false;
		for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
			bAnyStreamed|= manager.getDeviceByIndex(deviceIndex)->isStreaming();
		if (deviceCount > 0 && !bAnyStreamed)
		{
			MIKAN_LOG_ERROR("test-joycon")
				<< "(b) FAILED: devices opened but never streamed IMU samples";
			result= 1;
		}

		manager.shutdown();
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-joycon") << "Joy-Con checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-joycon", "Joy-Con sample decode + live HID streaming", eTestCategory::Hardware, runJoyconTest);
