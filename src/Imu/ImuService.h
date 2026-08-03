#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "glm/ext/quaternion_float.hpp"

#include "IImuDevice.h"
#include "ImuOrientationFilter.h"
#include "TrackingTypes.h" // eHandSide

// Owns the IMU devices, one orientation filter per device, and the mounting
// calibration that turns a sensor orientation into a FOREARM orientation.
//
// FRAME CONVENTION
// The forearm frame is defined so that it EQUALS the palm frame when the
// wrist is held straight. That makes the wrist joint rotation
// (inverse(forearm) * palm) identity at neutral, which is the semantics the
// OSC schema promises, and it makes calibration a single natural pose:
// hold the hand in line with the forearm and capture.
//
// Calibration math: with q_sw = sensor->world (from the filter) and
// q_fs = forearm->sensor (the constant mounting rotation we want),
//   q_fw = q_sw * q_fs
// At capture time q_fw must equal the vision-measured palm orientation, so
//   q_fs = inverse(q_sw) * q_palm
// This absorbs everything physical - the L/R sensor mounting difference,
// how the strap sits, which way the controller faces - so nothing about
// sensor axes is hardcoded anywhere.
struct ImuServiceConfig
{
	bool enabled= true;
	// Vision yaw-anchor strength (radians). Loose on purpose: vision sees
	// the palm, not the forearm, so its yaw is only an approximate reference
	// - it should correct slow drift without fighting real wrist motion.
	float visionYawSigma= 0.35f;
	// Swap which physical controller drives which wrist (a Joy-Con L worn on
	// the right wrist, say)
	bool swapSides= false;

	// Persisted mounting calibration, indexed by eHandSide
	bool mountingPresent[2]= {false, false};
	std::array<glm::quat, 2> forearmToSensor{glm::quat(1.f, 0.f, 0.f, 0.f), glm::quat(1.f, 0.f, 0.f, 0.f)};

	ImuOrientationFilterConfig filter;
};

// Per-side snapshot for UI/diagnostics
struct ImuSideStatus
{
	bool deviceConnected= false;
	bool streaming= false;
	bool calibrated= false;
	bool orientationValid= false; // calibrated AND the filter has converged
	float sampleRateHz= 0.f;
	float batteryLevel= -1.f;
	// -1 = never delivered a sample; large = asleep / link dropped
	double millisecondsSinceLastSample= -1.0;

	// MOUNTING QUALITY, 0..1, -1 until enough motion has been seen.
	// Rotating a forearm about its own long axis (pronation/supination) must
	// leave the forearm frame's +X fixed, because +X IS that axis when the
	// mounting is right. So |dot(rotation axis, +X)| over real motion scores
	// the calibration: near 1 = good, near 0 = the captured pose was not a
	// straight wrist and +X points somewhere other than along the arm - which
	// makes the elbow sweep a cone as you twist.
	float forearmAxisConsistency= -1.f;
	glm::vec3 gyroBiasDegreesPerSecond{0.f};
	float yawSigmaRadians= 0.f;
	std::string deviceName;
};

class ImuService
{
public:
	ImuService();
	~ImuService();

	bool startup();
	void shutdown();
	void setConfig(const ImuServiceConfig& config);
	const ImuServiceConfig& getConfig() const { return m_config; }

	// Re-scans for controllers and opens any that aren't streaming yet
	void refreshDevices();

	// Drains every device's buffered samples and advances its filter.
	// Call once per pipeline tick (the samples carry their own timestamps,
	// so a slow caller loses no information - only output freshness).
	void update();

	// Anchors yaw for one side from a vision-measured palm orientation.
	// Ignored until that side is calibrated (the mounting rotation is what
	// relates the palm to the sensor at all).
	void applyVisionPalmOrientation(eHandSide side, const glm::quat& palmOrientationWorld);

	// Forearm orientation in world space. False when that side has no
	// calibrated, streaming, converged device.
	// NOT const: also feeds the mounting-quality metric below, which needs to
	// watch the orientation actually being published.
	bool getForearmOrientation(eHandSide side, glm::quat& outForearmToWorld);

	// Captures the mounting rotation for one side from the CURRENT sensor
	// orientation plus a vision palm orientation, with the wrist held
	// straight. Returns false when the side has no converged device.
	bool captureMounting(eHandSide side, const glm::quat& palmOrientationWorld,
						 glm::quat& outForearmToSensor);

	ImuSideStatus getSideStatus(eHandSide side) const;

private:
	struct DeviceEntry
	{
		IImuDevice* device= nullptr;
		ImuOrientationFilter filter;
		double lastSampleTimestampMs= -1.0;
		int reopenCooldownFrames= 0;

		// Mounting-quality tracking (see ImuSideStatus::forearmAxisConsistency)
		glm::quat lastPublishedForearm{1.f, 0.f, 0.f, 0.f};
		bool bHasLastPublishedForearm= false;
		float axisConsistencyEma= -1.f;
		int axisConsistencySamples= 0;
	};

	// Index into m_devices for a wrist, honoring swapSides; -1 when none
	int findDeviceIndexForSide(eHandSide side) const;

	ImuServiceConfig m_config;
	std::unique_ptr<class JoyconDeviceManager> m_deviceManager;
	std::vector<std::unique_ptr<DeviceEntry>> m_devices;
	std::vector<ImuSample> m_sampleScratch;
	bool m_bStarted= false;
};
