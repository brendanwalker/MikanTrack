#include "TestCommon.h"

static int runImuDiscoveryTest(const TestArgs& args)
{
	int result= 0;

	// Device discovery (HID enumeration + the Bluetooth open handshake)
	// must never run on the caller's thread: ImuService::update() is
	// called from the vision thread's frame loop, and doing it inline
	// stalled that loop ~200 ms every 150 frames, starving EVERY camera
	// at once (diagnosed from recording 2026-08-10_00-57-02).
	//
	// Hardware-independent: enumeration runs whether or not a controller
	// is paired, so what is asserted is the invariant - startup and
	// update return promptly no matter what discovery is doing.
	auto elapsedMs= [](const std::chrono::steady_clock::time_point& start) {
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
			.count();
	};

	ImuService service;
	ImuServiceConfig serviceConfig;
	serviceConfig.enabled= true;
	service.setConfig(serviceConfig);

	const auto startupBegin= std::chrono::steady_clock::now();
	service.startup();
	const double startupMs= elapsedMs(startupBegin);

	// Well past the 150-frame rescan cooldown, so the periodic scan
	// request fires several times during this loop
	double worstUpdateMs= 0.0;
	const auto updatesBegin= std::chrono::steady_clock::now();
	for (int frameIndex= 0; frameIndex < 400; ++frameIndex)
	{
		const auto updateBegin= std::chrono::steady_clock::now();
		service.update();
		worstUpdateMs= std::max(worstUpdateMs, elapsedMs(updateBegin));
	}
	const double updatesMs= elapsedMs(updatesBegin);

	const auto shutdownBegin= std::chrono::steady_clock::now();
	service.shutdown();
	const double shutdownMs= elapsedMs(shutdownBegin);

	MIKAN_LOG_INFO("test-imudiscovery")
		<< "(a) startup=" << startupMs << " ms, 400 updates=" << updatesMs
		<< " ms (worst single " << worstUpdateMs << " ms), shutdown=" << shutdownMs << " ms";

	// startup() spawns the worker and returns; update() only swaps
	// already-built results. Thresholds are generous - the point is that
	// neither absorbs a multi-hundred-ms enumeration.
	if (startupMs > 50.0 || worstUpdateMs > 20.0)
	{
		MIKAN_LOG_ERROR("test-imudiscovery")
			<< "(a) FAILED: discovery must not block the caller";
		result= 1;
	}
	// shutdown() joins a worker that may be mid-enumeration, so it can
	// legitimately wait out one scan - but it must not hang
	if (shutdownMs > 3000.0)
	{
		MIKAN_LOG_ERROR("test-imudiscovery") << "(a) FAILED: shutdown must join promptly";
		result= 1;
	}

	// (b) Restartable: the worker's exit flag has to reset, or a second
	// session would spawn a thread already told to quit
	{
		const auto restartBegin= std::chrono::steady_clock::now();
		service.startup();
		for (int frameIndex= 0; frameIndex < 10; ++frameIndex)
			service.update();
		service.shutdown();
		const double restartMs= elapsedMs(restartBegin);

		MIKAN_LOG_INFO("test-imudiscovery") << "(b) restart cycle=" << restartMs << " ms";
		if (restartMs > 3000.0)
		{
			MIKAN_LOG_ERROR("test-imudiscovery") << "(b) FAILED: restart must work and not hang";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-imudiscovery") << "All IMU discovery checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-imudiscovery", "IMU device discovery never blocks the caller", eTestCategory::SelfTest, runImuDiscoveryTest);
