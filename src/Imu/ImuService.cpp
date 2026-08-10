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

// -- Curl quality gates (the second calibration motion) -----
// A curl sweeps the whole forearm rather than spinning it in place, so it
// carries more soft-tissue wobble than a twist and cannot hold a twist's
// dominance. Measured at 0.94 on a clean capture against the twist's 0.99.
static constexpr float k_minCurlDominance= 0.85f;
// The two motions must be independent for the frame to close. Anatomically
// this is near 90 degrees (measured 89.3); far below it means the shoulder
// was turning during the curl and roll is being extrapolated.
static constexpr float k_minInterAxisAngleDegrees= 60.f;
// Half-strokes whose individually measured hinge axes disagree by more than
// this were not made at one settled pronation, so roll is not repeatable.
// A clean capture holds ~8 degrees once the opening stroke is dropped.
static constexpr float k_maxHingeSpreadDegrees= 15.f;
// A half-stroke shorter than this is a direction change, not a stroke
static constexpr int k_minStrokeSamples= 40;
// Rate below which a sample is between strokes rather than in one
static constexpr float k_strokeDeadbandRadiansPerSecond= 0.5f;
// The centripetal fit's radius is an elbow-to-sensor distance, so it has to
// come out a forearm length. Outside this the model did not apply and the
// measurement is discarded rather than believed.
static constexpr float k_minForearmLengthMeters= 0.10f;
static constexpr float k_maxForearmLengthMeters= 0.40f;
static constexpr float k_minLengthFitCorrelation= 0.5f;
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

namespace
{
struct ImuMotionStats
{
	glm::mat3 scatter{0.f};
	float pathRadians= 0.f;
	glm::vec3 net{0.f};
};

ImuMotionStats imuAccumulateMotionStats(const MotionSample* samples, size_t count)
{
	ImuMotionStats stats;
	for (size_t index= 0; index < count; ++index)
	{
		const glm::vec3& rate= samples[index].rate;
		const float dtSeconds= samples[index].dtSeconds;
		stats.scatter+= glm::outerProduct(rate, rate) * dtSeconds;
		stats.pathRadians+= glm::length(rate) * dtSeconds;
		stats.net+= rate * dtSeconds;
	}
	return stats;
}
} // namespace

bool imuFitCentripetalRadius(const std::vector<MotionSample>& curl, const glm::vec3& longAxis,
							 const glm::vec3& hingeAxis, float& outSignedRadius, float& outCorrelation)
{
	outSignedRadius= 0.f;
	outCorrelation= 0.f;

	constexpr size_t k_minFitSamples= 100;
	constexpr float k_gravity= 9.81f;
	if (curl.size() < k_minFitSamples)
		return false;

	// Seed the gravity direction from the quietest sample in the opening
	// stretch, where the accelerometer reads little but gravity
	size_t seedIndex= 0;
	float seedRate= glm::length(curl[0].rate);
	const size_t seedSearch= std::min<size_t>(curl.size(), k_minFitSamples);
	for (size_t index= 1; index < seedSearch; ++index)
	{
		const float rateMagnitude= glm::length(curl[index].rate);
		if (rateMagnitude < seedRate)
		{
			seedRate= rateMagnitude;
			seedIndex= index;
		}
	}

	glm::vec3 up= curl[seedIndex].acceleration;
	if (glm::length(up) < 1e-3f)
		return false;
	up= glm::normalize(up);

	double sumX= 0.0, sumY= 0.0, sumXX= 0.0, sumYY= 0.0, sumXY= 0.0;
	int count= 0;
	for (size_t index= seedIndex; index < curl.size(); ++index)
	{
		const MotionSample& sample= curl[index];

		// World up carried through the sensor's own rotation. Integrated here
		// rather than read from the filter so the fit stays a pure function of
		// the recording, testable with no filter and no device.
		up-= glm::cross(sample.rate, up) * sample.dtSeconds;
		const float upLength= glm::length(up);
		if (upLength < 1e-3f)
			return false;
		up/= upLength;

		// Re-anchor whenever a sample looks like near-pure gravity. Integration
		// alone drifts, and the whole fit rests on the residual being
		// centripetal rather than accumulated tilt error.
		const float accelerationMagnitude= glm::length(sample.acceleration);
		if (glm::length(sample.rate) < 0.5f && fabsf(accelerationMagnitude - k_gravity) < 0.5f)
			up= glm::normalize(glm::mix(up, sample.acceleration / accelerationMagnitude, 0.02f));

		const float hingeRate= glm::dot(sample.rate, hingeAxis);
		const double x= (double)(hingeRate * hingeRate);
		const double y= (double)glm::dot(sample.acceleration - up * k_gravity, longAxis);
		sumX+= x;
		sumY+= y;
		sumXX+= x * x;
		sumYY+= y * y;
		sumXY+= x * y;
		count++;
	}
	if ((size_t)count < k_minFitSamples)
		return false;

	const double n= (double)count;
	const double varianceX= sumXX - sumX * sumX / n;
	const double varianceY= sumYY - sumY * sumY / n;
	const double covariance= sumXY - sumX * sumY / n;
	if (varianceX < 1e-6 || varianceY < 1e-9)
		return false;

	// residual . longAxis = -omega^2 * radius when longAxis points at the hand
	outSignedRadius= (float)(-covariance / varianceX);
	outCorrelation= (float)(covariance / sqrt(varianceX * varianceY));
	return true;
}

void imuSolveMountingFromMotions(const std::vector<MotionSample>& twist,
								 const std::vector<MotionSample>& curl, const glm::quat* palmarHint,
								 ePalmarSource hintSource, MountingCaptureResult& outResult)
{
	outResult= MountingCaptureResult();

	// -- The twist fixes the forearm's long axis -----
	const ImuMotionStats twistStats= imuAccumulateMotionStats(twist.data(), twist.size());
	float twistDominance= -1.f;
	imuEvaluateTwist(twistStats.scatter, twistStats.pathRadians, twistStats.net, twistDominance,
					 outResult.twistProgress, outResult.twistReversal);
	outResult.axisDominance= std::max(0.f, twistDominance);
	glm::vec3 longAxis= imuDominantRotationAxis(twistStats.scatter, twistDominance);

	// -- The curl fixes the roll about it -----
	const ImuMotionStats curlStats= imuAccumulateMotionStats(curl.data(), curl.size());
	float curlDominance= -1.f;
	imuEvaluateTwist(curlStats.scatter, curlStats.pathRadians, curlStats.net, curlDominance,
					 outResult.curlProgress, outResult.curlReversal);
	const glm::vec3 provisionalHinge= imuDominantRotationAxis(curlStats.scatter, curlDominance);

	// Split into half-strokes at reversals about that hinge and DROP THE
	// FIRST. The elbow hinge belongs to the ulna while the sensor rides the
	// pronating distal forearm, so the axis it measures only means something
	// once the pronation has settled - and on the opening stroke it has not,
	// landing tens of degrees off the strokes that follow it.
	std::vector<glm::vec3> strokeAxes;
	glm::mat3 acceptedScatter(0.f);
	{
		auto flushStroke= [&](size_t begin, size_t end) {
			if (end <= begin || end - begin < (size_t)k_minStrokeSamples)
				return;
			const ImuMotionStats strokeStats= imuAccumulateMotionStats(curl.data() + begin, end - begin);
			float strokeDominance= 0.f;
			glm::vec3 axis= imuDominantRotationAxis(strokeStats.scatter, strokeDominance);
			if (glm::dot(axis, provisionalHinge) < 0.f)
				axis= -axis;
			strokeAxes.push_back(axis);
			if (strokeAxes.size() > 1)
				acceptedScatter+= strokeStats.scatter;
		};

		size_t strokeStart= 0;
		int strokeSign= 0;
		for (size_t index= 0; index < curl.size(); ++index)
		{
			const float hingeRate= glm::dot(curl[index].rate, provisionalHinge);
			const int sign= hingeRate > k_strokeDeadbandRadiansPerSecond
				? 1
				: (hingeRate < -k_strokeDeadbandRadiansPerSecond ? -1 : 0);
			if (sign == 0)
				continue;

			if (strokeSign == 0)
			{
				strokeSign= sign;
				strokeStart= index;
			}
			else if (sign != strokeSign)
			{
				flushStroke(strokeStart, index);
				strokeSign= sign;
				strokeStart= index;
			}
		}
		if (strokeSign != 0)
			flushStroke(strokeStart, curl.size());
	}
	outResult.curlStrokes= (int)strokeAxes.size();

	glm::vec3 hingeAxis= provisionalHinge;
	if (strokeAxes.size() >= 2)
	{
		float acceptedDominance= 0.f;
		hingeAxis= imuDominantRotationAxis(acceptedScatter, acceptedDominance);
		if (glm::dot(hingeAxis, provisionalHinge) < 0.f)
			hingeAxis= -hingeAxis;
		curlDominance= acceptedDominance;

		for (size_t index= 1; index < strokeAxes.size(); ++index)
		{
			const float alignment= glm::clamp(glm::dot(strokeAxes[index], hingeAxis), -1.f, 1.f);
			outResult.hingeSpreadDegrees=
				std::max(outResult.hingeSpreadDegrees, glm::degrees(acosf(alignment)));
		}
	}
	outResult.curlDominance= std::max(0.f, curlDominance);

	const float axisAlignment= glm::clamp(fabsf(glm::dot(longAxis, hingeAxis)), 0.f, 1.f);
	outResult.interAxisAngleDegrees= glm::degrees(acosf(axisAlignment));

	// +X is trusted over the hinge, so the carrying angle lands entirely in +Y
	glm::vec3 yAxis= hingeAxis - longAxis * glm::dot(longAxis, hingeAxis);
	const float yLength= glm::length(yAxis);
	if (yLength < 1e-4f)
		return; // parallel axes: nothing to orthogonalize, and the gate below refuses it
	yAxis/= yLength;

	// -- Which end of the long axis is the hand -----
	bool bLongAxisSignResolved= false;
	float signedRadius= 0.f;
	float fitCorrelation= 0.f;
	if (imuFitCentripetalRadius(curl, longAxis, hingeAxis, signedRadius, fitCorrelation))
	{
		if (signedRadius < 0.f)
		{
			// The eigenvector came out pointing at the elbow
			longAxis= -longAxis;
			yAxis= -yAxis; // keep the triad right-handed
			signedRadius= -signedRadius;
			fitCorrelation= -fitCorrelation;
		}
		outResult.forearmLengthMeters= signedRadius;
		outResult.lengthFitCorrelation= fabsf(fitCorrelation);
		bLongAxisSignResolved= outResult.lengthFitCorrelation >= k_minLengthFitCorrelation;
		outResult.bLengthMeasured= bLongAxisSignResolved &&
			signedRadius >= k_minForearmLengthMeters && signedRadius <= k_maxForearmLengthMeters;
	}

	// -- Which side of the hinge the palm is on -----
	//
	// The two candidates differ by a half turn about the long axis, so any
	// reference that is even roughly right picks correctly. That is the whole
	// reason a pose average far too noisy to BE the mounting is still a fine
	// discriminator here.
	const glm::vec3 zAxis= glm::cross(longAxis, yAxis);
	const glm::quat candidate= glm::normalize(glm::quat_cast(glm::mat3(longAxis, yAxis, zAxis)));
	const glm::quat flipped= glm::normalize(glm::quat_cast(glm::mat3(longAxis, -yAxis, -zAxis)));

	outResult.forearmToSensor= candidate;
	if (palmarHint != nullptr && hintSource != ePalmarSource::None)
	{
		const float alignment= fabsf(glm::dot(candidate, *palmarHint));
		const float flippedAlignment= fabsf(glm::dot(flipped, *palmarHint));
		outResult.forearmToSensor= alignment >= flippedAlignment ? candidate : flipped;
		outResult.palmarSource= hintSource;
	}

	outResult.bMotionUsable=
		imuIsTwistUsable(outResult.axisDominance, outResult.twistProgress, outResult.twistReversal) &&
		outResult.curlProgress >= 1.f && outResult.curlReversal >= k_minTwistReversal &&
		outResult.curlDominance >= k_minCurlDominance && outResult.curlStrokes >= 3 &&
		outResult.hingeSpreadDegrees <= k_maxHingeSpreadDegrees &&
		outResult.interAxisAngleDegrees >= k_minInterAxisAngleDegrees && bLongAxisSignResolved &&
		outResult.palmarSource != ePalmarSource::None;
}


// Signed rotation of `rotation` about the given unit axis, degrees. Used to
// pull the axial component out of a measured wrist joint.
static float signedComponentDegrees(const glm::quat& rotation, const glm::vec3& axis)
{
	glm::quat q= rotation;
	if (q.w < 0.f) // shortest arc, or the sign of the whole thing flips
		q= -q;

	const glm::vec3 axisPart(q.x, q.y, q.z);
	const float axisLength= glm::length(axisPart);
	if (axisLength < 1e-6f)
		return 0.f;

	const float angle= 2.f * asinf(std::min(1.f, axisLength));
	return glm::degrees(angle * glm::dot(axisPart / axisLength, axis));
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
	const float axialDegrees= signedComponentDegrees(wrist, glm::vec3(1.f, 0.f, 0.f));

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
