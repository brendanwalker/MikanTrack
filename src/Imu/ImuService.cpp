#include "ImuService.h"

#include <algorithm>

#include "glm/gtc/quaternion.hpp"

#include "JoyconDeviceManager.h"
#include "Logger.h"

// A streaming Joy-Con delivers ~200 samples/second, so a second of total
// silence already means it is gone (asleep, or the link dropped)
static constexpr double k_deviceSilentTimeoutMs= 1500.0;
// Angular-velocity scatter (mounting axis estimation)
static constexpr float k_scatterMinRateRadiansPerSecond= 0.35f; // ~20 deg/s: deliberate motion, not jiggle
static constexpr float k_scatterHalfLifeSeconds= 8.f;
// Frames to wait between reopen attempts (~2s at camera rate)
static constexpr int k_reopenCooldownFrames= 60;
// Frames between scans for newly-paired controllers (~5s at camera rate)
static constexpr int k_rescanCooldownFrames= 150;

// Recording cap, ~60 s per motion at 200 Hz. A window this long is already
// far more than any gate needs; the bound only stops a wizard left open
// overnight from growing without limit.
static constexpr size_t k_maxMotionRecordingSamples= 12000;
// The pose average needs a decent run of tracked frames behind it
static constexpr int k_poseMountingMinSamples= 60;
static constexpr float k_poseMountingMinConfidence= 0.4f;
// ~30 seconds of raw samples kept for the axis-convention diagnostic
static constexpr size_t k_rawHistoryMaxSamples= 6000;

// -- Wrist axial residual (mounting-roll health check) -----
// The palm estimate has to be worth trusting before it says anything
static constexpr float k_axialResidualMinConfidence= 0.5f;
// Enough samples that the average means something before reporting it
static constexpr int k_axialResidualMinSamples= 60;

// -- Static gyro bias calibration -----
static constexpr double k_biasCalibrationSeconds= 4.0;
// A resting controller reads well under this; anything more means it was
// touched, and the average would be poisoned
static constexpr float k_biasRestRateRadiansPerSecond= 0.15f; // ~8.6 deg/s
static constexpr float k_biasRestAccelTolerance= 0.6f;        // m/s^2 around 1g
static constexpr float k_gravityMetersPerSecond2= 9.80665f;

ImuService::ImuService()= default;

ImuService::~ImuService()
{
	shutdown();
}

bool ImuService::startup()
{
	if (m_bStarted)
		return true;

	m_deviceManager= std::make_unique<JoyconDeviceManager>();
	m_bStarted= true;

	// The manager's own startup() enumerates, so it runs on the worker too -
	// this keeps the vision thread's first iteration free of HID work
	m_bDiscoveryExit= false;
	m_discoveryThread= std::thread([this]() { discoveryLoop(); });
	requestDiscovery();
	return true;
}

void ImuService::shutdown()
{
	// Stop the worker before touching anything it owns
	if (m_discoveryThread.joinable())
	{
		{
			std::lock_guard<std::mutex> lock(m_discoveryMutex);
			m_bDiscoveryExit= true;
		}
		m_discoveryCondition.notify_all();
		m_discoveryThread.join();
	}

	{
		std::lock_guard<std::mutex> lock(m_discoveryMutex);
		m_bDiscoveryExit= false;
		m_bDiscoveryRequested= false;
		m_bDiscoveryResultReady= false;
		m_discoveredDevices.clear();
		m_reopenRequests.clear();
		m_reopenCompletedPaths.clear();
	}

	m_devices.clear();
	if (m_deviceManager != nullptr)
	{
		m_deviceManager->shutdown();
		m_deviceManager= nullptr;
	}
	m_bStarted= false;
}

void ImuService::setConfig(const ImuServiceConfig& config)
{
	const bool bFilterChanged=
		config.filter.gyroNoiseDensity != m_config.filter.gyroNoiseDensity ||
		config.filter.gyroBiasRandomWalk != m_config.filter.gyroBiasRandomWalk ||
		config.filter.accelNoise != m_config.filter.accelNoise ||
		config.filter.accelGate != m_config.filter.accelGate;

	m_config= config;

	// Only rebuild filter state when the filter itself was retuned - a
	// mounting recapture or a side swap must not throw away a converged
	// bias estimate that took 30 seconds to earn
	if (bFilterChanged)
	{
		for (std::unique_ptr<DeviceEntry>& entry : m_devices)
			entry->filter.configure(m_config.filter);
	}
}

void ImuService::requestDiscovery()
{
	{
		std::lock_guard<std::mutex> lock(m_discoveryMutex);
		m_bDiscoveryRequested= true;
	}
	m_discoveryCondition.notify_one();
}

void ImuService::requestReopen(const std::shared_ptr<IImuDevice>& device)
{
	{
		std::lock_guard<std::mutex> lock(m_discoveryMutex);
		m_reopenRequests.push_back(device);
	}
	m_discoveryCondition.notify_one();
}

void ImuService::discoveryLoop()
{
	// Sole owner of m_deviceManager while this thread runs (created before the
	// thread starts, destroyed after it joins), so the blocking HID calls here
	// never contend with the frame loop
	if (m_deviceManager != nullptr)
		m_deviceManager->startup();

	while (true)
	{
		bool bDiscover= false;
		std::vector<std::shared_ptr<IImuDevice>> reopenBatch;
		{
			std::unique_lock<std::mutex> lock(m_discoveryMutex);
			m_discoveryCondition.wait(lock, [this]() {
				return m_bDiscoveryExit || m_bDiscoveryRequested || !m_reopenRequests.empty();
			});
			if (m_bDiscoveryExit)
				return;

			bDiscover= m_bDiscoveryRequested;
			m_bDiscoveryRequested= false;
			reopenBatch.swap(m_reopenRequests);
		}

		// Reopens first: a controller that went silent matters more than
		// finding a new one, and the service has already stopped draining it
		for (const std::shared_ptr<IImuDevice>& device : reopenBatch)
		{
			MIKAN_MT_LOG_WARNING("ImuService") << device->getFriendlyName() << " silent - reopening";
			device->close();
			device->open();

			std::lock_guard<std::mutex> lock(m_discoveryMutex);
			m_reopenCompletedPaths.push_back(device->getDevicePath());
		}

		if (bDiscover && m_deviceManager != nullptr)
		{
			m_deviceManager->refreshConnectedDevices();

			std::vector<std::shared_ptr<IImuDevice>> devices;
			for (size_t deviceIndex= 0; deviceIndex < m_deviceManager->getDeviceCount(); ++deviceIndex)
			{
				std::shared_ptr<IImuDevice> device= m_deviceManager->getDeviceByIndex(deviceIndex);
				if (device == nullptr)
					continue;

				if (!device->isOpen())
					device->open();

				devices.push_back(std::move(device));
			}

			std::lock_guard<std::mutex> lock(m_discoveryMutex);
			m_discoveredDevices= std::move(devices);
			m_bDiscoveryResultReady= true;
		}
	}
}

void ImuService::adoptDiscoveryResults()
{
	std::vector<std::shared_ptr<IImuDevice>> discovered;
	std::vector<std::string> reopenedPaths;
	bool bHaveResult= false;
	{
		std::lock_guard<std::mutex> lock(m_discoveryMutex);
		if (m_bDiscoveryResultReady)
		{
			discovered= std::move(m_discoveredDevices);
			m_discoveredDevices.clear();
			m_bDiscoveryResultReady= false;
			bHaveResult= true;
		}
		reopenedPaths.swap(m_reopenCompletedPaths);
	}

	// A completed reopen restarts that device's sample clock: the nominal
	// orientation survives, but integrating across the gap must not happen
	for (const std::string& path : reopenedPaths)
	{
		for (std::unique_ptr<DeviceEntry>& entry : m_devices)
		{
			if (entry != nullptr && entry->device != nullptr && path == entry->device->getDevicePath())
			{
				entry->bAwaitingReopen= false;
				entry->lastSampleTimestampMs= -1.0;
			}
		}
	}

	if (!bHaveResult)
		return;

	// Rebuild the list, carrying over each known device's entry (and its
	// converged filter). A device that vanished falls out here, and the last
	// reference dropping closes it on THIS thread, where nothing is draining
	// it.
	std::vector<std::unique_ptr<DeviceEntry>> devices;
	for (const std::shared_ptr<IImuDevice>& device : discovered)
	{
		std::unique_ptr<DeviceEntry> entry;
		for (std::unique_ptr<DeviceEntry>& existing : m_devices)
		{
			if (existing != nullptr && existing->device == device)
			{
				entry= std::move(existing);
				break;
			}
		}
		if (entry == nullptr)
		{
			entry= std::make_unique<DeviceEntry>();
			entry->device= device;
			entry->filter.configure(m_config.filter);
		}

		devices.push_back(std::move(entry));
	}

	m_devices= std::move(devices);
}

int ImuService::findDeviceIndexForSide(eHandSide side) const
{
	const eImuSide wanted= m_config.swapSides
		? (side == eHandSide::Left ? eImuSide::Right : eImuSide::Left)
		: (side == eHandSide::Left ? eImuSide::Left : eImuSide::Right);

	for (size_t deviceIndex= 0; deviceIndex < m_devices.size(); ++deviceIndex)
	{
		if (m_devices[deviceIndex]->device != nullptr && m_devices[deviceIndex]->device->getSide() == wanted)
			return (int)deviceIndex;
	}
	return -1;
}

void ImuService::update()
{
	if (!m_config.enabled)
		return;

	// Pick up whatever the discovery worker finished (new devices, completed
	// reopens). Cheap: a mutexed swap of already-built results.
	adoptDiscoveryResults();

	// Look for controllers we do not have yet, so pairing one mid-session just
	// starts working instead of needing the user to know to press a button.
	// Throttled, and skipped entirely once both wrists are covered. Only the
	// REQUEST happens here - the enumeration runs on the discovery worker,
	// because doing it inline stalled this thread for ~200 ms every 150 frames
	// and starved both cameras (measured: recording 2026-08-10_00-57-02).
	if (m_devices.size() < 2)
	{
		if (m_rescanCooldownFrames <= 0)
		{
			requestDiscovery();
			m_rescanCooldownFrames= k_rescanCooldownFrames;
		}
		else
		{
			m_rescanCooldownFrames--;
		}
	}

	for (std::unique_ptr<DeviceEntry>& entry : m_devices)
	{
		if (entry->device == nullptr)
			continue;

		// Recover a controller that went quiet. Joy-Cons sleep on their own
		// schedule and Bluetooth links drop; either way the fix is the same,
		// so heal automatically instead of making the user notice and press
		// a button. Throttled so a genuinely absent controller doesn't spin.
		const double silentMs= entry->device->getMillisecondsSinceLastSample();
		if (silentMs > k_deviceSilentTimeoutMs)
		{
			if (!entry->bAwaitingReopen && entry->reopenCooldownFrames <= 0)
			{
				// Hand it to the discovery worker: close+open runs a Bluetooth
				// handshake, far too slow for this thread. The entry (and its
				// converged filter) stays here; only the device is off-limits
				// until the worker reports the reopen complete.
				entry->bAwaitingReopen= true;
				entry->reopenCooldownFrames= k_reopenCooldownFrames;
				requestReopen(entry->device);
			}
			else if (entry->reopenCooldownFrames > 0)
			{
				entry->reopenCooldownFrames--;
			}
		}
		else if (entry->reopenCooldownFrames > 0)
		{
			entry->reopenCooldownFrames= 0;
		}

		// The worker owns this device right now - do not touch it
		if (entry->bAwaitingReopen)
			continue;

		m_sampleScratch.clear();
		entry->device->fetchSamples(m_sampleScratch);

		// Keep the untouched samples first: the axis-convention diagnostic has
		// to replay candidate mappings against what the device actually sent,
		// not against whatever correction is currently enabled
		for (const ImuSample& raw : m_sampleScratch)
		{
			entry->rawHistory.push_back(raw);
			if (entry->rawHistory.size() > k_rawHistoryMaxSamples)
				entry->rawHistory.pop_front();
		}

		// Samples arrive in chronological order and carry their own
		// timestamps, so integrating a whole backlog at once is exact - a
		// caller running at camera rate loses nothing but output freshness
		for (const ImuSample& sample : m_sampleScratch)
		{
			constexpr double k_nominalSpacingMs= 1000.0 / 200.0; // Joy-Con rate
			constexpr double k_maxPlausibleSpacingMs= 100.0;

			// Sample times are back-dated from HID arrival, so a Bluetooth
			// stall followed by a burst yields timestamps that go BACKWARDS or
			// bunch microseconds apart. Taking those at face value collapses
			// the filter covariance (see the dt floor in predict). Fall back to
			// the nominal spacing and keep the clock monotonic instead - the
			// samples themselves are still good, only their arrival times are
			// not.
			double sampleTimeMs= sample.timestampMs;
			float dtSeconds= (float)(k_nominalSpacingMs / 1000.0);
			if (entry->lastSampleTimestampMs >= 0.0)
			{
				const double deltaMs= sampleTimeMs - entry->lastSampleTimestampMs;
				if (deltaMs <= 0.0 || deltaMs > k_maxPlausibleSpacingMs)
					sampleTimeMs= entry->lastSampleTimestampMs + k_nominalSpacingMs;
				else
					dtSeconds= (float)(deltaMs / 1000.0);
			}
			entry->lastSampleTimestampMs= sampleTimeMs;

			if (entry->bCalibratingBias)
			{
				// Deliberately BEFORE the filter: a resting controller carries
				// no orientation information, and folding its samples in while
				// the user is told not to touch it just wastes them
				accumulateBiasCalibration(*entry, sample, dtSeconds);
				continue;
			}

			entry->filter.processSample(sample, dtSeconds);

			// Accumulate the angular-velocity scatter used by mounting
			// calibration. Only real rotation carries axis information, and
			// weighting by |w|^2 lets deliberate twisting dominate incidental
			// jiggle. Decays so a calibration reflects RECENT motion.
			const glm::vec3 rate= sample.angularVelocity - entry->filter.getGyroBias();
			const float rateMagnitude= glm::length(rate);
			if (rateMagnitude > k_scatterMinRateRadiansPerSecond)
			{
				const float decay= expf(-dtSeconds / k_scatterHalfLifeSeconds);
				entry->rotationScatter*= decay;
				entry->rotationScatterWeight*= decay;
				entry->rotationPathRadians*= decay;
				entry->rotationNet*= decay;

				entry->rotationScatter+= glm::outerProduct(rate, rate) * dtSeconds;
				entry->rotationScatterWeight+= rateMagnitude * rateMagnitude * dtSeconds;
				entry->rotationPathRadians+= rateMagnitude * dtSeconds;
				entry->rotationNet+= rate * dtSeconds;
			}

			// The calibration recording keeps EVERY sample, including the slow
			// ones the scatter gate above skips: the centripetal fit needs the
			// near-stationary samples to anchor gravity, and the stroke split
			// needs the turnarounds to know where one stroke ends.
			if (m_motionRecording != eMountingMotion::None)
			{
				std::vector<MotionSample>& recording= m_motionRecording == eMountingMotion::Twist
					? entry->twistRecording
					: entry->curlRecording;
				if (recording.size() < k_maxMotionRecordingSamples)
				{
					MotionSample recorded;
					recorded.rate= rate;
					recorded.acceleration= sample.acceleration;
					recorded.dtSeconds= dtSeconds;
					recording.push_back(recorded);
				}
			}
		}
	}
}

void ImuService::accumulateBiasCalibration(DeviceEntry& entry, const ImuSample& sample, float dtSeconds)
{
	const float rateMagnitude= glm::length(sample.angularVelocity);
	const float accelMagnitude= glm::length(sample.acceleration);
	const bool bAtRest= rateMagnitude < k_biasRestRateRadiansPerSecond &&
						fabsf(accelMagnitude - k_gravityMetersPerSecond2) < k_biasRestAccelTolerance;
	if (!bAtRest)
	{
		// Start over rather than average in motion - a single nudge would
		// otherwise be baked into the bias permanently
		entry.bBiasDisturbed= true;
		entry.biasSum= glm::dvec3(0.0);
		entry.biasSampleCount= 0;
		entry.biasSeconds= 0.0;
		return;
	}

	entry.biasSum+= glm::dvec3(sample.angularVelocity);
	entry.biasSampleCount++;
	entry.biasSeconds+= dtSeconds;

	if (entry.biasSeconds < k_biasCalibrationSeconds || entry.biasSampleCount <= 0)
		return;

	const glm::vec3 measuredBias= glm::vec3(entry.biasSum / (double)entry.biasSampleCount);
	entry.filter.setGyroBias(measuredBias);
	entry.bCalibratingBias= false;
	entry.bBiasDisturbed= false;
	MIKAN_LOG_INFO("ImuService") << (entry.device != nullptr ? entry.device->getFriendlyName() : "device")
								 << " gyro bias calibrated: " << glm::degrees(measuredBias).x << ", "
								 << glm::degrees(measuredBias).y << ", " << glm::degrees(measuredBias).z
								 << " deg/s over " << entry.biasSampleCount << " samples";
}

void ImuService::beginBiasCalibration()
{
	for (std::unique_ptr<DeviceEntry>& entry : m_devices)
	{
		entry->bCalibratingBias= true;
		entry->bBiasDisturbed= false;
		entry->biasSum= glm::dvec3(0.0);
		entry->biasSampleCount= 0;
		entry->biasSeconds= 0.0;
	}
}

void ImuService::cancelBiasCalibration()
{
	for (std::unique_ptr<DeviceEntry>& entry : m_devices)
		entry->bCalibratingBias= false;
}

bool ImuService::isBiasCalibrationRunning() const
{
	for (const std::unique_ptr<DeviceEntry>& entry : m_devices)
	{
		if (entry->bCalibratingBias)
			return true;
	}
	return false;
}

void ImuService::updateWristAxialResidual(eHandSide side, const glm::quat& palmOrientationWorld,
										  float confidence)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return;

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr || !entry.device->isStreaming() || !entry.filter.isTiltConverged())
		return;

	// A shaky palm estimate says nothing about a calibration
	if (confidence < k_axialResidualMinConfidence)
		return;

	const glm::quat forearm=
		glm::normalize(entry.filter.getOrientation() * m_config.forearmToSensor[(int)side]);
	const glm::quat wrist= glm::normalize(glm::inverse(forearm) * glm::normalize(palmOrientationWorld));

	// THE constraint: the wrist has no axial degree of freedom, so whatever
	// this reads is mounting roll error rather than anatomy
	const float axialDegrees= imuSignedComponentDegrees(wrist, glm::vec3(1.f, 0.f, 0.f));

	constexpr float kEmaAlpha= 0.02f;
	entry.twistResidualDegreesEma= entry.twistSamples == 0
		? axialDegrees
		: entry.twistResidualDegreesEma * (1.f - kEmaAlpha) + axialDegrees * kEmaAlpha;
	entry.twistSamples++;
}

void ImuService::applyVisionPalmOrientation(eHandSide side, const glm::quat& palmOrientationWorld)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return;

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (!entry.filter.isInitialized())
		return;

	// Convert the vision PALM orientation into the equivalent SENSOR
	// orientation using the mounting rotation, then let the filter take only
	// its yaw (see updateWithYawReference for why not the whole thing).
	const glm::quat referenceSensorToWorld=
		palmOrientationWorld * glm::inverse(m_config.forearmToSensor[(int)side]);
	entry.filter.updateWithYawReference(referenceSensorToWorld, m_config.visionYawSigma);
}

bool ImuService::getForearmOrientation(eHandSide side, glm::quat& outForearmToWorld)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return false;

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return false;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr || !entry.device->isStreaming() || !entry.filter.isTiltConverged())
		return false;

	// q_fw = q_sw * q_fs
	outForearmToWorld=
		glm::normalize(entry.filter.getOrientation() * m_config.forearmToSensor[(int)side]);

	// Score the mounting against real motion. Twisting a forearm about its
	// own long axis must leave the forearm frame's +X fixed - so when the
	// incremental rotation is a twist, its axis should BE +X. A mounting
	// captured at a bent wrist fails this and shows up as an elbow that
	// sweeps a cone instead of staying put.
	if (entry.bHasLastPublishedForearm)
	{
		const glm::quat delta= glm::inverse(entry.lastPublishedForearm) * outForearmToWorld;
		const glm::vec3 axisPart(delta.x, delta.y, delta.z);
		const float axisLength= glm::length(axisPart);
		// Only meaningful rotation carries information about the axis
		if (axisLength > 0.004f) // ~0.5 deg
		{
			const glm::vec3 axis= axisPart / axisLength;
			const float alignment= fabsf(axis.x); // +X in the forearm's own frame
			constexpr float kEmaAlpha= 0.05f;
			entry.axisConsistencyEma= entry.axisConsistencyEma < 0.f
				? alignment
				: entry.axisConsistencyEma * (1.f - kEmaAlpha) + alignment * kEmaAlpha;
			entry.axisConsistencySamples++;
		}
	}
	entry.lastPublishedForearm= outForearmToWorld;
	entry.bHasLastPublishedForearm= true;

	return true;
}

void ImuService::accumulatePoseMounting(eHandSide side, const glm::quat& palmOrientationWorld,
										float confidence)
{
	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr || !entry.device->isStreaming() || !entry.filter.isTiltConverged())
		return;
	if (confidence < k_poseMountingMinConfidence)
		return;

	const glm::quat sample= glm::normalize(glm::inverse(entry.filter.getOrientation()) *
										   glm::normalize(palmOrientationWorld));

	// Quaternions double-cover rotations, so align each sample to the running
	// mean's hemisphere before summing - otherwise q and -q cancel and the
	// average collapses toward zero
	glm::quat aligned= sample;
	if (entry.poseMountingSamples > 0)
	{
		const glm::quat mean= glm::normalize(glm::quat(entry.poseMountingSum.w, entry.poseMountingSum.x,
													   entry.poseMountingSum.y, entry.poseMountingSum.z));
		if (glm::dot(aligned, mean) < 0.f)
			aligned= -aligned;

		const glm::quat delta= glm::inverse(mean) * aligned;
		entry.poseSpreadSumDegrees+= glm::degrees(2.f * asinf(std::min(
			1.f, glm::length(glm::vec3(delta.x, delta.y, delta.z)))));
	}

	entry.poseMountingSum+= glm::vec4(aligned.x, aligned.y, aligned.z, aligned.w);
	entry.poseMountingSamples++;
}

bool ImuService::captureMounting(eHandSide side, MountingCaptureResult& outResult)
{
	outResult= MountingCaptureResult();

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return false;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr || !entry.device->isStreaming() || !entry.filter.isTiltConverged())
		return false;

	// The pose average is no longer the geometry - a wrist that would not hold
	// still is exactly what broke that - but it is still the reference that
	// says which side of the hinge axis the palm is on, a decision with a 180
	// degree margin that it clears easily.
	glm::quat palmarHint(1.f, 0.f, 0.f, 0.f);
	ePalmarSource hintSource= ePalmarSource::None;
	if (entry.poseMountingSamples >= k_poseMountingMinSamples)
	{
		palmarHint= glm::normalize(glm::quat(entry.poseMountingSum.w, entry.poseMountingSum.x,
											 entry.poseMountingSum.y, entry.poseMountingSum.z));
		hintSource= ePalmarSource::Vision;
	}
	else if (m_config.mountingPresent[(int)side])
	{
		// Recalibrating without the cameras seeing the hand: the side already
		// known to be on is a valid reference for a two-way choice, even
		// though it is useless as a mounting
		palmarHint= m_config.forearmToSensor[(int)side];
		hintSource= ePalmarSource::PreviousMounting;
	}

	imuSolveMountingFromMotions(entry.twistRecording, entry.curlRecording, &palmarHint, hintSource,
								outResult);

	outResult.poseSamples= entry.poseMountingSamples;
	outResult.poseSpreadDegrees= entry.poseMountingSamples > 0
		? entry.poseSpreadSumDegrees / (float)entry.poseMountingSamples
		: 0.f;
	outResult.bCaptured= true;
	m_lastCapture[(int)side]= outResult;

	// A recapture invalidates the old mounting's quality score, and the axial
	// residual measured against the OLD mounting says nothing about this one
	entry.axisConsistencyEma= -1.f;
	entry.axisConsistencySamples= 0;
	entry.bHasLastPublishedForearm= false;
	entry.twistResidualDegreesEma= 0.f;
	entry.twistSamples= 0;
	return true;
}

void ImuService::resetMountingMotion()
{
	for (std::unique_ptr<DeviceEntry>& entry : m_devices)
	{
		entry->rotationScatter= glm::mat3(0.f);
		entry->rotationScatterWeight= 0.f;
		entry->rotationPathRadians= 0.f;
		entry->rotationNet= glm::vec3(0.f);
		entry->poseMountingSum= glm::vec4(0.f);
		entry->poseMountingSamples= 0;
		entry->poseSpreadSumDegrees= 0.f;
	}
	m_motionEpoch++;
}

void ImuService::beginMotionRecording(eMountingMotion motion)
{
	m_motionRecording= motion;
	for (std::unique_ptr<DeviceEntry>& entry : m_devices)
	{
		if (motion == eMountingMotion::Twist)
			entry->twistRecording.clear();
		else if (motion == eMountingMotion::Curl)
			entry->curlRecording.clear();
	}

	// The live scatter drives the progress bars, so it has to describe the
	// stage being asked for rather than the one before it
	if (motion != eMountingMotion::None)
		resetMountingMotion();
}

void ImuService::endMotionRecording()
{
	m_motionRecording= eMountingMotion::None;
}

void ImuService::getRawSampleHistory(eHandSide side, std::vector<ImuSample>& outSamples) const
{
	outSamples.clear();

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return;

	const DeviceEntry& entry= *m_devices[deviceIndex];
	outSamples.assign(entry.rawHistory.begin(), entry.rawHistory.end());
}

ImuSideStatus ImuService::getSideStatus(eHandSide side) const
{
	ImuSideStatus status;
	status.calibrated= m_config.mountingPresent[(int)side];

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return status;

	const DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr)
		return status;

	status.deviceConnected= entry.device->isOpen();
	// isStreaming() only says "samples arrived at some point"; a controller
	// that fell asleep still reports true, so gate on recent traffic
	const double silentMs= entry.device->getMillisecondsSinceLastSample();
	status.streaming= entry.device->isStreaming() && silentMs >= 0.0 && silentMs < k_deviceSilentTimeoutMs;
	status.millisecondsSinceLastSample= silentMs;
	status.sampleRateHz= entry.device->getSampleRateHz();
	status.batteryLevel= entry.device->getBatteryLevel();
	status.deviceName= entry.device->getFriendlyName();
	status.gyroBiasDegreesPerSecond= glm::degrees(entry.filter.getGyroBias());
	status.biasSaturated= entry.filter.isBiasSaturated();
	status.yawSigmaRadians= entry.filter.getOrientationSigma().z;
	status.motionEpoch= m_motionEpoch;
	// Needs a bit of motion before it means anything
	status.forearmAxisConsistency= entry.axisConsistencySamples >= 30 ? entry.axisConsistencyEma : -1.f;
	imuEvaluateTwist(entry.rotationScatter, entry.rotationPathRadians, entry.rotationNet,
					 status.armAxisDominance, status.twistProgress, status.twistReversal);
	if (entry.bCalibratingBias)
	{
		status.biasCalibrationProgress=
			(float)std::clamp(entry.biasSeconds / k_biasCalibrationSeconds, 0.0, 1.0);
		status.biasCalibrationDisturbed= entry.bBiasDisturbed;
	}
	status.wristAxialTwistDegrees=
		entry.twistSamples >= k_axialResidualMinSamples ? entry.twistResidualDegreesEma : -999.f;
	status.filterOrientation= entry.filter.getOrientation();
	status.tiltSigmaRadians= entry.filter.getTiltSigma();
	status.gravityAcceptRatio= entry.filter.getGravityAcceptRatio();
	status.visionYawCorrectionDegrees= entry.filter.getVisionYawCorrectionDegrees();
	status.filterConverged= entry.filter.isTiltConverged();
	status.orientationValid= status.calibrated && status.streaming && status.filterConverged;
	return status;
}
