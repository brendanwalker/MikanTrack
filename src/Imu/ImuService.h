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
	// The filter's tilt has settled. Independent of `calibrated` on purpose:
	// this is the precondition for CAPTURING a mounting, so it has to be
	// knowable before one exists.
	bool filterConverged= false;
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

	// LIVE twist conditioning, 0..1, -1 until any motion has been seen. How
	// dominant a single axis is in the recent angular-velocity scatter - i.e.
	// how much of what this sensor has been doing lately is pure forearm
	// twist. This is what makes the arm axis measurable, so the mounting
	// wizard watches it to decide when enough twisting has happened.
	float armAxisDominance= -1.f;
	glm::vec3 gyroBiasDegreesPerSecond{0.f};
	float yawSigmaRadians= 0.f;
	std::string deviceName;
};

// Dominant eigenvector of a symmetric angular-velocity scatter sum(w w^T),
// plus how dominant it is (lambda1 / trace): 1 = all rotation about a single
// axis, 1/3 = isotropic and therefore uninformative. Free function so the
// mounting math can be tested without a physical device attached.
glm::vec3 imuDominantRotationAxis(const glm::mat3& scatter, float& outDominance);

// Rotates a pose-derived mounting so that forearm +X lands on the measured
// sensor-frame arm axis, by the MINIMAL rotation that does so - which leaves
// roll about that axis exactly as the held pose set it. sensorAxis need not be
// signed correctly; it is flipped to agree with the pose.
glm::quat imuAlignMountingToArmAxis(const glm::quat& poseMounting, glm::vec3 sensorAxis);

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

	// Captures the mounting rotation for one side.
	//
	// Two sources, because they are good at different things:
	// - MOTION decides the forearm's long axis. Pronating/supinating rotates
	//   the arm about that axis and nothing else, so the dominant axis of the
	//   recent angular-velocity scatter IS the arm axis, measured in the
	//   sensor's own frame over thousands of samples. This is the part the
	//   elbow depends on, and it needs no pose to be held correctly.
	// - The held POSE decides the remaining degree of freedom (roll about
	//   that axis), which motion cannot observe and which a straight-wrist
	//   pose gives for free.
	// A single held pose alone had to get all three right at one instant and
	// kept getting the axis wrong by ~60 deg.
	//
	// outAxisDominance (0..1) reports how well-conditioned the motion was;
	// below ~0.7 the axis is unreliable and the caller should say so rather
	// than bake in a bad mounting.
	bool captureMounting(eHandSide side, const glm::quat& palmOrientationWorld,
						 glm::quat& outForearmToSensor, float& outAxisDominance);

	// Discards the accumulated angular-velocity scatter on every device, so a
	// fresh calibration measures only the twisting the user does from here on
	// (the scatter decays on its own, but a wizard should not start with a
	// half-full history of whatever the arms happened to be doing before).
	void resetMountingMotion();

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

		// Decaying scatter of sensor-frame angular velocity, sum(w w^T). Its
		// dominant eigenvector is the axis the arm has been rotating about -
		// i.e. the forearm's long axis, if the user has been twisting.
		glm::mat3 rotationScatter{0.f};
		float rotationScatterWeight= 0.f;
	};

	// Index into m_devices for a wrist, honoring swapSides; -1 when none
	int findDeviceIndexForSide(eHandSide side) const;

	ImuServiceConfig m_config;
	std::unique_ptr<class JoyconDeviceManager> m_deviceManager;
	std::vector<std::unique_ptr<DeviceEntry>> m_devices;
	std::vector<ImuSample> m_sampleScratch;
	bool m_bStarted= false;
};
