#pragma once

#include <vector>

#include "glm/ext/quaternion_float.hpp"
#include "glm/mat3x3.hpp"
#include "glm/vec3.hpp"

// The wrist IMU mounting solve: pure functions of recorded motion, with no
// device, filter, or world frame behind them, so the whole calibration can be
// tested from synthetic samples.

// One recorded sample of a calibration motion: the bias-corrected rotation
// rate and the specific force measured alongside it.
//
// Recorded rather than folded into a running sum because two things the
// solve needs cannot be recovered from a scatter matrix: which half-stroke a
// sample belongs to, and how the accelerometer's centripetal term grows with
// the square of the rotation rate.
struct MotionSample
{
	glm::vec3 rate{0.f};         // rad/s, sensor frame, gyro bias removed
	glm::vec3 acceleration{0.f}; // m/s^2, sensor frame, specific force
	float dtSeconds= 0.f;
};

// Where the palmar side of the frame came from. Geometry fixes two axes and
// the centripetal term fixes which end is the hand, but which SIDE of the
// hinge axis the palm sits on is not in the motion at all.
enum class ePalmarSource
{
	None, // unresolved - the capture is refused rather than guessed
	Vision,
	PreviousMounting,
};

// Outcome of one mounting capture. A struct rather than out-params because
// the caller has to tell several different failures apart to say anything
// useful about them.
struct MountingCaptureResult
{
	// A mounting was computed at all (device streaming + filter converged)
	bool bCaptured= false;
	glm::quat forearmToSensor{1.f, 0.f, 0.f, 0.f};

	// -- Twist window: fixes the forearm's long axis -----
	float axisDominance= 0.f;
	float twistProgress= 0.f;
	float twistReversal= 0.f;

	// -- Curl window: fixes the remaining roll about that axis -----
	float curlDominance= 0.f;
	float curlProgress= 0.f;
	float curlReversal= 0.f;
	// Half-strokes the curl was split into, and how much their individually
	// measured hinge axes disagreed. The spread is the honest error bar on
	// roll: the hinge belongs to the ULNA while the sensor rides the
	// pronating distal forearm, so the axis only means something once the
	// user's pronation settles.
	int curlStrokes= 0;
	float hingeSpreadDegrees= 0.f;

	// Angle between the two measured axes. Anatomically near 90; well below
	// that means the two motions were not independent (usually the shoulder
	// turning during the curl) and the roll is being extrapolated.
	float interAxisAngleDegrees= 0.f;

	// -- Measured forearm length (elbow to sensor), from the centripetal fit -
	bool bLengthMeasured= false;
	float forearmLengthMeters= 0.f;
	float lengthFitCorrelation= 0.f;

	ePalmarSource palmarSource= ePalmarSource::None;

	// How many tracked frames the pose average was built from, and how much
	// those samples disagreed (degrees). No longer an INPUT to the geometry -
	// a wrist that would not hold still is exactly what broke that - but it
	// still says whether vision was watching, and it is the diagnostic that
	// identified the problem.
	int poseSamples= 0;
	float poseSpreadDegrees= 0.f;

	// True when every gate passed and the mounting is worth saving
	bool bMotionUsable= false;
};

// Dominant eigenvector of a symmetric angular-velocity scatter sum(w w^T),
// plus how dominant it is (lambda1 / trace): 1 = all rotation about a single
// axis, 1/3 = isotropic and therefore uninformative. Free function so the
// mounting math can be tested without a physical device attached.
glm::vec3 imuDominantRotationAxis(const glm::mat3& scatter, float& outDominance);

// Fits the centripetal signature of a curl to find which end of the long axis
// points at the HAND, and how far the sensor sits from the elbow.
//
// The sensor orbits the elbow, so the accelerometer carries a term pointing
// PROXIMALLY whose size grows with the square of the rotation rate - the same
// sign whichever way the arm happens to be swinging. That is the one
// distal/proximal cue in the data that does not depend on the user following
// an instruction, and gravity cannot supply it: flexing is symmetric under
// swapping the two ends.
//
// outSignedRadius is positive when longAxis points distally, and its
// magnitude is the elbow-to-sensor distance in meters - which has to come out
// a forearm length, so the fit checks itself. Returns false when the window
// held too little rotation to fit anything.
bool imuFitCentripetalRadius(const std::vector<MotionSample>& curl, const glm::vec3& longAxis,
							 const glm::vec3& hingeAxis, float& outSignedRadius, float& outCorrelation);

// Solves forearm -> sensor from a recorded twist and a recorded curl.
//
// Two rotations that are not parallel determine a frame outright:
//  - TWIST (pronation/supination) turns the forearm about its long axis, so
//    the dominant axis of that window IS forearm +X in the sensor's own
//    frame. It is invariant to how the arm is held, and it is the only
//    degree of freedom the elbow estimate consumes.
//  - CURL (elbow flexion) turns it about the hinge, which is close to
//    perpendicular to the long axis. Orthogonalized against +X that gives
//    +Y, and the cross product closes the frame.
//
// Nothing here touches world space, so filter drift, unobservable yaw and
// wrist bend cannot reach the result. That is the point: the vision pose
// average this replaces was corrupted by a wrist that would not hold still
// during the capture, which no amount of averaging could fix.
//
// palmarHint resolves the one bit the motion cannot - which side of the hinge
// the palm is on. It is only ever compared against two candidates 180 degrees
// apart, so a hint tens of degrees off still chooses correctly. That margin
// is why a pose average far too noisy to BE the mounting is still perfectly
// good at picking between these two. Pass null when none is available; the
// solve then reports ePalmarSource::None rather than guessing.
void imuSolveMountingFromMotions(const std::vector<MotionSample>& twist,
								 const std::vector<MotionSample>& curl, const glm::quat* palmarHint,
								 ePalmarSource hintSource, MountingCaptureResult& outResult);

// Scores accumulated rotation statistics as a forearm-twist measurement.
// pathRadians is sum(|w| dt), net is sum(w dt), scatter is sum(w w^T dt).
// Free functions so the status readout, the capture gate and the tests all
// judge a twist by the same rules.
void imuEvaluateTwist(const glm::mat3& scatter, float pathRadians, const glm::vec3& net,
					  float& outDominance, float& outProgress, float& outReversal);
// True when a twist is good enough to define the forearm axis. All three
// conditions are needed: enough rotation (a rank-1 scatter is free for tiny
// motions), enough reversal (a steady turn or an uncorrected rate offset is
// also rank-1), and enough single-axis dominance (arm-waving is not a twist).
bool imuIsTwistUsable(float dominance, float progress, float reversal);

// Signed rotation of `rotation` about the given unit axis, degrees. Pulls the
// axial component out of a measured wrist joint.
float imuSignedComponentDegrees(const glm::quat& rotation, const glm::vec3& axis);
