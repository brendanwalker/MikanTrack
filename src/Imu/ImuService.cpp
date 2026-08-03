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

// -- Twist quality gates (mounting calibration) -----
// Total rotation the forearm must travel before the measured axis is
// trusted. ~5 rad is roughly two full pronation/supination sweeps.
static constexpr float k_minTwistPathRadians= 5.f;
// The motion must substantially REVERSE. A steady turn (or an uncorrected
// rate offset) also produces a rank-1 scatter, but pointing somewhere that
// has nothing to do with the arm - a Joy-Con whose bias had run away scored
// 0.9999 dominance while sitting still.
static constexpr float k_minTwistReversal= 0.5f;
static constexpr float k_minTwistDominance= 0.7f;

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
	m_deviceManager->startup();
	m_bStarted= true;

	refreshDevices();
	return true;
}

void ImuService::shutdown()
{
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

void ImuService::refreshDevices()
{
	if (!m_bStarted || m_deviceManager == nullptr)
		return;

	m_deviceManager->refreshConnectedDevices();

	std::vector<std::unique_ptr<DeviceEntry>> devices;
	for (size_t deviceIndex= 0; deviceIndex < m_deviceManager->getDeviceCount(); ++deviceIndex)
	{
		IImuDevice* device= m_deviceManager->getDeviceByIndex(deviceIndex);
		if (device == nullptr)
			continue;

		// Carry over the existing entry (and its converged filter) when this
		// device was already known
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

		if (!device->isOpen())
			device->open();

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

	// Look for controllers we do not have yet, so pairing one mid-session just
	// starts working instead of needing the user to know to press a button.
	// Throttled, and skipped entirely once both wrists are covered.
	if (m_devices.size() < 2)
	{
		if (m_rescanCooldownFrames <= 0)
		{
			refreshDevices();
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
			if (entry->reopenCooldownFrames <= 0)
			{
				MIKAN_LOG_WARNING("ImuService")
					<< entry->device->getFriendlyName() << " silent for " << (int)silentMs
					<< " ms - reopening";
				entry->device->close();
				if (entry->device->open())
				{
					// The nominal orientation survives, but the sample clock
					// restarts: don't integrate across the gap
					entry->lastSampleTimestampMs= -1.0;
				}
				entry->reopenCooldownFrames= k_reopenCooldownFrames;
			}
			else
			{
				entry->reopenCooldownFrames--;
			}
		}
		else if (entry->reopenCooldownFrames > 0)
		{
			entry->reopenCooldownFrames= 0;
		}

		m_sampleScratch.clear();
		entry->device->fetchSamples(m_sampleScratch);

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

glm::vec3 imuDominantRotationAxis(const glm::mat3& scatter, float& outDominance)
{
	glm::vec3 axis(1.f, 0.f, 0.f);
	for (int iteration= 0; iteration < 64; ++iteration)
	{
		const glm::vec3 next= scatter * axis;
		const float length= glm::length(next);
		if (length < 1e-20f)
		{
			outDominance= 0.f;
			return glm::vec3(1.f, 0.f, 0.f);
		}
		axis= next / length;
	}

	const float eigenvalue= glm::dot(axis, scatter * axis);
	const float trace= scatter[0][0] + scatter[1][1] + scatter[2][2];
	outDominance= trace > 1e-20f ? eigenvalue / trace : 0.f;
	return axis;
}

glm::quat imuAlignMountingToArmAxis(const glm::quat& poseMounting, glm::vec3 sensorAxis)
{
	const float sensorAxisLength= glm::length(sensorAxis);
	if (sensorAxisLength < 1e-6f)
		return poseMounting;
	sensorAxis/= sensorAxisLength;

	// Where the held pose thinks the arm axis is, expressed in sensor frame
	const glm::vec3 poseAxis= glm::normalize(poseMounting * glm::vec3(1.f, 0.f, 0.f));

	// The eigenvector's sign is arbitrary; +X must point toward the hand, so
	// resolve it by agreeing with the pose
	if (glm::dot(sensorAxis, poseAxis) < 0.f)
		sensorAxis= -sensorAxis;

	const float alignment= glm::clamp(glm::dot(poseAxis, sensorAxis), -1.f, 1.f);
	if (alignment > 0.999999f)
		return poseMounting;

	const glm::vec3 rotationAxis= glm::cross(poseAxis, sensorAxis);
	const float rotationAxisLength= glm::length(rotationAxis);
	if (rotationAxisLength < 1e-6f) // exactly antiparallel: the sign flip above rules this out
		return poseMounting;

	// Left-multiply: the correction acts in SENSOR frame, which is where both
	// axes live
	return glm::normalize(
		glm::angleAxis(acosf(alignment), rotationAxis / rotationAxisLength) * poseMounting);
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

void imuEvaluateTwist(const glm::mat3& scatter, float pathRadians, const glm::vec3& net,
					  float& outDominance, float& outProgress, float& outReversal)
{
	outProgress= std::min(1.f, pathRadians / k_minTwistPathRadians);
	outReversal= pathRadians > 1e-6f ? std::clamp(1.f - glm::length(net) / pathRadians, 0.f, 1.f) : 0.f;
	outDominance= -1.f;
	if (pathRadians > 1e-6f)
		imuDominantRotationAxis(scatter, outDominance);
}

bool imuIsTwistUsable(float dominance, float progress, float reversal)
{
	return progress >= 1.f && reversal >= k_minTwistReversal && dominance >= k_minTwistDominance;
}

bool ImuService::captureMounting(eHandSide side, const glm::quat& palmOrientationWorld,
								 MountingCaptureResult& outResult)
{
	outResult= MountingCaptureResult();

	const int deviceIndex= findDeviceIndexForSide(side);
	if (deviceIndex < 0)
		return false;

	DeviceEntry& entry= *m_devices[deviceIndex];
	if (entry.device == nullptr || !entry.device->isStreaming() || !entry.filter.isTiltConverged())
		return false;

	// Start from the held pose: with the wrist straight the forearm frame IS
	// the palm frame, so q_fs = inverse(q_sw) * q_palm. This gets the roll
	// about the arm right, but its ARM AXIS is only as good as the pose was.
	glm::quat mounting=
		glm::normalize(glm::inverse(entry.filter.getOrientation()) * glm::normalize(palmOrientationWorld));

	float dominance= -1.f;
	imuEvaluateTwist(entry.rotationScatter, entry.rotationPathRadians, entry.rotationNet, dominance,
					 outResult.twistProgress, outResult.twistReversal);
	outResult.axisDominance= std::max(0.f, dominance);
	outResult.bMotionUsable=
		imuIsTwistUsable(outResult.axisDominance, outResult.twistProgress, outResult.twistReversal);

	// Then let measured motion fix the axis. The dominant rotation axis in
	// the SENSOR frame is the forearm's long axis; the mounting must map
	// forearm +X onto it. Rotating the pose-derived mounting by the minimal
	// rotation that does so corrects the two degrees of freedom motion can
	// see, and leaves the third (roll) exactly as the pose set it.
	if (outResult.bMotionUsable)
	{
		const glm::vec3 sensorAxis= imuDominantRotationAxis(entry.rotationScatter, dominance);
		mounting= imuAlignMountingToArmAxis(mounting, sensorAxis);
	}

	outResult.forearmToSensor= mounting;
	outResult.bCaptured= true;

	// A recapture invalidates the old mounting's quality score
	entry.axisConsistencyEma= -1.f;
	entry.axisConsistencySamples= 0;
	entry.bHasLastPublishedForearm= false;
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
	}
	m_motionEpoch++;
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
	status.filterConverged= entry.filter.isTiltConverged();
	status.orientationValid= status.calibrated && status.streaming && status.filterConverged;
	return status;
}
