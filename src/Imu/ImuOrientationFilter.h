#pragma once

#include <array>

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

#include "ImuTypes.h"

// Error-state Kalman filter (ESKF / multiplicative EKF) estimating the
// SENSOR's orientation in the Z-up world frame, plus the gyro bias.
//
// Why error-state: a quaternion has 4 components but only 3 degrees of
// freedom, so a direct EKF over quaternion components has a singular
// covariance and needs constant renormalization hacks. The ESKF keeps the
// orientation in a nominal quaternion (outside the covariance) and runs the
// filter over a minimal 3-parameter ERROR rotation that is always near zero,
// which is both correct and numerically well behaved.
//
// State (6): [ orientation error (3, body frame) | gyro bias (3, rad/s) ]
// Nominal:   quaternion (sensor -> world) + bias vector
//
// Measurements:
// - Gravity (accelerometer): corrects roll/pitch only. At rest the
//   accelerometer measures specific force = +1g along whatever axis points
//   UP, so its direction is an absolute tilt reference. Gated: during real
//   acceleration the reading is NOT gravity, so samples whose magnitude
//   strays from 1g are rejected rather than believed.
// - Orientation (vision): corrects all 3 axes, and is the ONLY source of
//   absolute yaw for a 6-axis IMU (no magnetometer -> yaw is unobservable
//   from inertial data alone and would otherwise drift forever; a Joy-Con
//   was measured at ~114 deg/minute of yaw bias drift).
//
// The filter is per-device (each sensor has its own bias) and knows nothing
// about hands - it estimates where the SENSOR is pointing. Mapping that to a
// forearm/palm frame is a separate calibrated rotation.
struct ImuOrientationFilterConfig
{
	// Gyro white noise density, rad/s/sqrt(Hz). Default derived from measured
	// Joy-Con jitter (~0.004 rad/s per sample at 200 Hz).
	float gyroNoiseDensity= 3e-4f;
	// Gyro bias random walk, rad/s/sqrt(s) - how fast the bias is allowed to
	// wander. Too low and the filter stops tracking thermal drift; too high
	// and bias soaks up real rotation.
	float gyroBiasRandomWalk= 1e-4f;
	// Accelerometer noise, m/s^2 (converted to a normalized-direction sigma)
	float accelNoise= 0.35f;
	// Reject the gravity update when | |a| - g | exceeds this (m/s^2): the
	// sensor is accelerating, so the reading isn't gravity
	float accelGate= 1.0f;
	// Vision orientation measurement noise, radians
	float visionNoise= 0.05f;
	// Hard bound on the estimated gyro bias, rad/s. A real MEMS gyro bias is a
	// few deg/s; anything approaching this is the filter diverging, not a
	// sensor property. Bounding it matters because the bias feeds straight
	// back into predict(), so a runaway bias spins the nominal orientation,
	// which produces more apparent error, which grows the bias further - a
	// Joy-Con was observed at -5371 deg/s on one axis.
	float maxGyroBias= 1.0f; // rad/s (~57 deg/s)

	// Initial uncertainty
	float initialTiltSigma= 1.0f;    // rad; gravity fixes this fast
	float initialYawSigma= 3.14159f; // rad; yaw is unknown until vision says
	float initialBiasSigma= 0.05f;   // rad/s
};

class ImuOrientationFilter
{
public:
	void configure(const ImuOrientationFilterConfig& config);
	void reset();

	// Propagates with a gyro sample. dtSeconds is clamped internally against
	// timestamp glitches.
	void predict(const glm::vec3& angularVelocity, float dtSeconds);

	// Gravity/tilt correction from an accelerometer sample (m/s^2, sensor
	// frame). Returns false when the sample was gated out as non-gravity.
	bool updateWithGravity(const glm::vec3& acceleration);

	// Absolute orientation correction (sensor -> world) from vision. This is
	// what pins yaw.
	void updateWithOrientation(const glm::quat& sensorToWorld);

	// Yaw-ONLY correction from a reference orientation. Used when the visual
	// reference is only loosely related to the sensor: vision measures the
	// PALM, but the sensor rides the FOREARM, and the (unknown, time-varying)
	// wrist joint sits between them. Roll/pitch from such a reference would
	// fight gravity, which measures them properly - but its yaw is still the
	// only absolute yaw information in the system, and yaw drifts otherwise.
	// So this believes the reference about rotation around world-up and
	// almost nothing else.
	void updateWithYawReference(const glm::quat& sensorToWorldReference, float yawSigma);

	// Convenience: feed a whole sample (predict + gravity update)
	void processSample(const ImuSample& sample, float dtSeconds);

	// -- Accessors -----
	const glm::quat& getOrientation() const { return m_orientation; }
	const glm::vec3& getGyroBias() const { return m_gyroBias; }
	bool isInitialized() const { return m_bInitialized; }
	// Bias-corrected angular velocity from the last predict (rad/s) - useful
	// for forward-predicting orientation to a render/output timestamp
	const glm::vec3& getAngularVelocity() const { return m_lastAngularVelocity; }
	// 1-sigma uncertainty (radians) about each body axis
	glm::vec3 getOrientationSigma() const;
	// True once tilt uncertainty has converged (gravity has been seen enough)
	bool isTiltConverged() const;
	// True when the bias estimate is pinned at its bound, i.e. the filter has
	// been diverging. Surfaced rather than hidden: it invalidates everything
	// downstream and is otherwise invisible.
	bool isBiasSaturated() const;

	// Seeds the nominal orientation directly, e.g. from the first gravity
	// sample, without waiting for the filter to converge from an arbitrary
	// start. Leaves yaw at identity (unobservable).
	void initializeFromGravity(const glm::vec3& acceleration);

	// Overwrites the bias with a directly MEASURED one (sensor at rest, where
	// the raw gyro reading is the bias) and tightens its covariance to match.
	// Worth doing because the gravity update cannot observe the bias component
	// about the gravity axis at all, and that is the one that becomes yaw
	// drift. Clamped to the configured bound like any other bias update.
	void setGyroBias(const glm::vec3& gyroBias, float sigma= 0.002f);

private:
	// 6x6 / 6x3 / 3x3 row-major fixed-size matrices (no Eigen dependency)
	using Mat6= std::array<float, 36>;
	using Mat63= std::array<float, 18>;
	using Mat3= std::array<float, 9>;

	// Shared Kalman correction for a 3-vector measurement with Jacobian
	// H= [Htheta | Hbias] (each 3x3); applies Joseph-form covariance update
	// and injects the error into the nominal state.
	void applyCorrection(const glm::vec3& residual, const Mat3& hTheta, const Mat3& hBias,
						 const Mat3& measurementNoise);

	ImuOrientationFilterConfig m_config;

	glm::quat m_orientation{1.f, 0.f, 0.f, 0.f}; // sensor -> world
	glm::vec3 m_gyroBias{0.f};
	glm::vec3 m_lastAngularVelocity{0.f};
	Mat6 m_covariance{};

	bool m_bInitialized= false;
	bool m_bBiasSaturated= false;
};
