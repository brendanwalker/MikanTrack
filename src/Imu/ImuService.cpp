#include "ImuService.h"

#include <algorithm>

#include "glm/gtc/quaternion.hpp"

#include "JoyconDeviceManager.h"
#include "Logger.h"

// Frames between scans for newly-paired controllers (~5s at camera rate)
static constexpr int k_rescanCooldownFrames= 150;

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

	m_trackers.clear();
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
		for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
			tracker->getFilter().configure(m_config.filter);
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

	for (const std::string& path : reopenedPaths)
	{
		for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
		{
			if (path == tracker->getDevice()->getDevicePath())
				tracker->onReopenCompleted();
		}
	}

	if (!bHaveResult)
		return;

	// Rebuild the list, carrying over each known device's tracker (and its
	// converged filter). A device that vanished falls out here, and the last
	// reference dropping closes it on THIS thread, where nothing is draining
	// it.
	std::vector<std::unique_ptr<ImuDeviceTracker>> trackers;
	for (const std::shared_ptr<IImuDevice>& device : discovered)
	{
		std::unique_ptr<ImuDeviceTracker> tracker;
		for (std::unique_ptr<ImuDeviceTracker>& existing : m_trackers)
		{
			if (existing != nullptr && existing->getDevice() == device)
			{
				tracker= std::move(existing);
				break;
			}
		}
		if (tracker == nullptr)
			tracker= std::make_unique<ImuDeviceTracker>(device, m_config.filter);

		trackers.push_back(std::move(tracker));
	}

	m_trackers= std::move(trackers);
}

ImuDeviceTracker* ImuService::findTrackerForSide(eHandSide side)
{
	return const_cast<ImuDeviceTracker*>(static_cast<const ImuService*>(this)->findTrackerForSide(side));
}

const ImuDeviceTracker* ImuService::findTrackerForSide(eHandSide side) const
{
	const eImuSide wanted= m_config.swapSides
		? (side == eHandSide::Left ? eImuSide::Right : eImuSide::Left)
		: (side == eHandSide::Left ? eImuSide::Left : eImuSide::Right);

	for (const std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
	{
		if (tracker->getDevice()->getSide() == wanted)
			return tracker.get();
	}
	return nullptr;
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
	if (m_trackers.size() < 2)
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

	for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
	{
		// A silent controller goes to the discovery worker for a close/open
		// cycle: that runs a Bluetooth handshake, far too slow for this
		// thread. The tracker (and its converged filter) stays here; only the
		// device is off-limits until the worker reports the reopen complete.
		if (tracker->pollReopenRequest())
			requestReopen(tracker->getDevice());
		if (tracker->isAwaitingReopen())
			continue;

		tracker->update(m_sampleScratch, m_motionRecording);
	}
}

void ImuService::beginBiasCalibration()
{
	for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
		tracker->getBias().begin();
}

void ImuService::cancelBiasCalibration()
{
	for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
		tracker->getBias().cancel();
}

bool ImuService::isBiasCalibrationRunning() const
{
	for (const std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
	{
		if (tracker->getBias().isRunning())
			return true;
	}
	return false;
}

void ImuService::updateWristAxialResidual(eHandSide side, const glm::quat& palmOrientationWorld,
										  float confidence)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return;

	ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr || !tracker->isReady())
		return;

	const glm::quat forearm=
		glm::normalize(tracker->getFilter().getOrientation() * m_config.forearmToSensor[(int)side]);
	tracker->getMounting().updateWristAxialResidual(forearm, palmOrientationWorld, confidence);
}

void ImuService::applyVisionPalmOrientation(eHandSide side, const glm::quat& palmOrientationWorld)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return;

	ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr || !tracker->getFilter().isInitialized())
		return;

	// Convert the vision PALM orientation into the equivalent SENSOR
	// orientation using the mounting rotation, then let the filter take only
	// its yaw (see updateWithYawReference for why not the whole thing).
	const glm::quat referenceSensorToWorld=
		palmOrientationWorld * glm::inverse(m_config.forearmToSensor[(int)side]);
	tracker->getFilter().updateWithYawReference(referenceSensorToWorld, m_config.visionYawSigma);
}

bool ImuService::getForearmOrientation(eHandSide side, glm::quat& outForearmToWorld)
{
	if (!m_config.enabled || !m_config.mountingPresent[(int)side])
		return false;

	ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr || !tracker->isReady())
		return false;

	// q_fw = q_sw * q_fs
	outForearmToWorld=
		glm::normalize(tracker->getFilter().getOrientation() * m_config.forearmToSensor[(int)side]);
	tracker->getMounting().notePublishedForearm(outForearmToWorld);
	return true;
}

void ImuService::accumulatePoseMounting(eHandSide side, const glm::quat& palmOrientationWorld,
										float confidence)
{
	ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr || !tracker->isReady())
		return;

	tracker->getMounting().accumulatePoseMounting(tracker->getFilter().getOrientation(), palmOrientationWorld,
												  confidence);
}

bool ImuService::captureMounting(eHandSide side, MountingCaptureResult& outResult)
{
	outResult= MountingCaptureResult();

	ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr || !tracker->isReady())
		return false;

	const glm::quat* previousMounting=
		m_config.mountingPresent[(int)side] ? &m_config.forearmToSensor[(int)side] : nullptr;
	tracker->getMounting().capture(previousMounting, outResult);
	outResult.bCaptured= true;
	m_lastCapture[(int)side]= outResult;
	return true;
}

void ImuService::resetMountingMotion()
{
	for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
		tracker->getMounting().resetLiveMotion();
	m_motionEpoch++;
}

void ImuService::beginMotionRecording(eMountingMotion motion)
{
	m_motionRecording= motion;
	for (std::unique_ptr<ImuDeviceTracker>& tracker : m_trackers)
		tracker->getMounting().clearRecording(motion);

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

	const ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker != nullptr)
		tracker->getRawSampleHistory(outSamples);
}

ImuSideStatus ImuService::getSideStatus(eHandSide side) const
{
	ImuSideStatus status;
	status.calibrated= m_config.mountingPresent[(int)side];

	const ImuDeviceTracker* tracker= findTrackerForSide(side);
	if (tracker == nullptr)
		return status;

	const IImuDevice& device= *tracker->getDevice();
	const ImuOrientationFilter& filter= tracker->getFilter();
	const ImuMountingCalibrator& mounting= tracker->getMounting();
	const ImuBiasCalibrator& bias= tracker->getBias();

	status.deviceConnected= device.isOpen();
	status.streaming= tracker->hasRecentSamples();
	status.millisecondsSinceLastSample= device.getMillisecondsSinceLastSample();
	status.sampleRateHz= device.getSampleRateHz();
	status.batteryLevel= device.getBatteryLevel();
	status.deviceName= device.getFriendlyName();
	status.gyroBiasDegreesPerSecond= glm::degrees(filter.getGyroBias());
	status.biasSaturated= filter.isBiasSaturated();
	status.yawSigmaRadians= filter.getOrientationSigma().z;
	status.motionEpoch= m_motionEpoch;
	status.forearmAxisConsistency= mounting.getForearmAxisConsistency();
	mounting.getLiveTwist(status.armAxisDominance, status.twistProgress, status.twistReversal);
	status.biasCalibrationProgress= bias.getProgress();
	if (bias.isRunning())
		status.biasCalibrationDisturbed= bias.wasDisturbed();
	status.wristAxialTwistDegrees= mounting.getWristAxialTwistDegrees();
	status.filterOrientation= filter.getOrientation();
	status.tiltSigmaRadians= filter.getTiltSigma();
	status.gravityAcceptRatio= filter.getGravityAcceptRatio();
	status.visionYawCorrectionDegrees= filter.getVisionYawCorrectionDegrees();
	status.filterConverged= filter.isTiltConverged();
	status.orientationValid= status.calibrated && status.streaming && status.filterConverged;
	return status;
}
