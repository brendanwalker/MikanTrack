#pragma once

#include <vector>

#include "glm/ext/quaternion_float.hpp"
#include "glm/mat3x3.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"

#include "ImuMountingMath.h"

// Which calibration motion is being recorded, if any
enum class eMountingMotion
{
	None,
	Twist,
	Curl,
};

// One device's mounting calibration state: the live motion statistics the
// wizard's progress bars watch, the two recorded calibration motions the
// solve consumes, the vision pose average that resolves the palmar side, and
// the two health checks on whatever mounting is currently in use.
class ImuMountingCalibrator
{
public:
	// -- Motion input, every filtered sample -----

	// rate is the bias-corrected angular velocity in the sensor frame. Feeds
	// the live decaying scatter, and appends to the recording named by
	// `recording` (None records nothing).
	void addMotionSample(const glm::vec3& rate, const glm::vec3& acceleration, float dtSeconds,
						 eMountingMotion recording);

	// Discards one motion's recording (a stage being started over)
	void clearRecording(eMountingMotion motion);

	// Discards the live scatter and the pose average, so a fresh calibration
	// measures only the motion the user does from here on
	void resetLiveMotion();

	// The live twist readout (see imuEvaluateTwist): how single-axis the
	// recent motion is, how much of the needed rotation has accumulated, and
	// how much it reverses
	void getLiveTwist(float& outDominance, float& outProgress, float& outReversal) const;

	// -- Vision input -----

	// Accumulates inverse(sensorToWorld) * palm, one sample per tracked frame.
	//
	// Under the correct model that quantity is CONSTANT - both terms rotate
	// together with the arm - so the only thing that perturbs it is wrist
	// bend, which is bounded and averages out. A mean over hundreds of
	// samples spanning many arm orientations cannot flip the way one instant
	// can, which is what makes it a safe palmar-side reference for capture().
	void accumulatePoseMounting(const glm::quat& sensorToWorld, const glm::quat& palmOrientationWorld,
								float confidence);

	// -- The capture -----

	// Solves forearm -> sensor from the recorded twist and curl (see
	// imuSolveMountingFromMotions). The pose average, when it has enough
	// samples behind it, picks which side of the hinge axis the palm is on;
	// otherwise previousMountingHint does (the side already known to be on is
	// a valid reference for a two-way choice); null means no hint. The result
	// reports every gate it checked and leaves bCaptured to the caller. Also
	// resets the two health checks below, which measured the OLD mounting.
	void capture(const glm::quat* previousMountingHint, MountingCaptureResult& outResult);

	// -- Health checks on the mounting in use -----

	// Scores the mounting against real motion. Twisting a forearm about its
	// own long axis must leave the forearm frame's +X fixed - so when the
	// incremental rotation between two published forearms is a twist, its
	// axis should BE +X. A mounting captured at a bent wrist fails this and
	// shows up as an elbow that sweeps a cone instead of staying put.
	void notePublishedForearm(const glm::quat& forearmToWorld);
	// 0..1, -1 until enough motion has been seen
	float getForearmAxisConsistency() const;

	// Measures the anatomically impossible axial component of the wrist joint
	// inverse(forearm) * palm. The wrist has no axial degree of freedom, so
	// whatever this reads is mounting roll error rather than anatomy.
	// Confidence gates it: a shaky palm estimate says nothing about a
	// calibration.
	void updateWristAxialResidual(const glm::quat& forearmToWorld, const glm::quat& palmOrientationWorld,
								  float confidence);
	// Signed degrees, -999 until measured
	float getWristAxialTwistDegrees() const;

private:
	// Decaying scatter of sensor-frame angular velocity, sum(w w^T). Its
	// dominant eigenvector is the axis the arm has been rotating about -
	// i.e. the forearm's long axis, if the user has been twisting.
	glm::mat3 m_rotationScatter{0.f};
	// Total rotation travelled, sum(|w| dt) - "how much twisting happened"
	float m_rotationPathRadians= 0.f;
	// Net rotation, sum(w dt). Back-and-forth twisting cancels out here
	// while the path keeps growing; a one-way turn or a constant rate
	// offset makes the two equal.
	glm::vec3 m_rotationNet{0.f};

	// The two calibration motions, recorded in full while the wizard asks
	// for them. Separate from the decaying scatter above, which stays the
	// LIVE readout the progress bars watch: a recording must not fade out
	// from under a user who is still performing it.
	std::vector<MotionSample> m_twistRecording;
	std::vector<MotionSample> m_curlRecording;

	// Running mean of inverse(q_sensor) * q_palm, hemisphere-aligned
	glm::vec4 m_poseMountingSum{0.f};
	int m_poseMountingSamples= 0;
	float m_poseSpreadSumDegrees= 0.f;

	// Forearm-axis consistency (see notePublishedForearm)
	glm::quat m_lastPublishedForearm{1.f, 0.f, 0.f, 0.f};
	bool m_bHasLastPublishedForearm= false;
	float m_axisConsistencyEma= -1.f;
	int m_axisConsistencySamples= 0;

	// Rolling mean of the wrist's axial residual (see updateWristAxialResidual)
	float m_twistResidualDegreesEma= 0.f;
	int m_twistSamples= 0;
};
