#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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

	// -- Live twist measurement (what the mounting wizard watches) -----
	//
	// Three separate numbers because there are three separate ways for a twist
	// to be useless, and they need different corrections from the user.
	//
	// How single-axis the recent motion is (0..1, -1 = nothing measured).
	// NOT sufficient on its own: a scatter built from a moment's motion is
	// rank-1, so this reads ~1.0 for free. It only means something once
	// twistProgress is full.
	float armAxisDominance= -1.f;

	// SIGNED axial residual in the measured wrist joint, degrees, -999 until
	// measured. Anatomically this must be zero: the wrist cannot rotate about
	// the forearm's long axis. Anything else is mounting roll error, and
	// unlike forearmAxisConsistency this is meaningful during ANY motion
	// rather than only during a deliberate twist. A left wrist reading 45 deg
	// here is what a badly rolled mounting looks like.
	float wristAxialTwistDegrees= -999.f;
	// How much twisting has accumulated, 0..1 against the amount needed
	float twistProgress= 0.f;
	// How much the motion REVERSES: 1 = perfectly back-and-forth, 0 = a
	// one-way turn. Pronation/supination oscillates, so this separates real
	// twisting from a steady turn or a constant rate offset - both of which
	// also produce a rank-1 scatter pointing somewhere meaningless.
	float twistReversal= 0.f;

	glm::vec3 gyroBiasDegreesPerSecond{0.f};
	// The bias estimate is pinned at its bound - the filter diverged and
	// everything this device reports is suspect
	bool biasSaturated= false;
	// Static bias calibration: 0..1 while running, -1 when not running.
	// biasCalibrationDisturbed means the controller was moved and the
	// measurement restarted.
	float biasCalibrationProgress= -1.f;
	bool biasCalibrationDisturbed= false;

	float yawSigmaRadians= 0.f;

	// -- Filter internals, split out from the composed forearm orientation --
	//
	// The dump only ever carried q_sensor * mounting, so a frozen filter and a
	// mounting that cancels it are indistinguishable. Recording the filter's
	// own orientation separates the two.
	glm::quat filterOrientation{1.f, 0.f, 0.f, 0.f};
	float tiltSigmaRadians= 0.f;
	float gravityAcceptRatio= -1.f;
	float visionYawCorrectionDegrees= 0.f;

	std::string deviceName;

	// Increments on every resetMountingMotion(). A caller that just requested
	// a reset can tell an already-refreshed status from a stale one, instead
	// of latching a twist measurement made before the reset landed.
	uint32_t motionEpoch= 0;
};

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

// Which calibration motion is being recorded, if any
enum class eMountingMotion
{
	None,
	Twist,
	Curl,
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

class ImuService
{
public:
	ImuService();
	~ImuService();

	bool startup();
	void shutdown();
	void setConfig(const ImuServiceConfig& config);
	const ImuServiceConfig& getConfig() const { return m_config; }

	// Asks the discovery worker to re-scan for controllers and open any that
	// aren't streaming yet. Returns immediately: enumeration and the Bluetooth
	// open handshake block for hundreds of milliseconds, and update() runs on
	// the vision thread's frame loop, where that would starve every camera.
	// Results are adopted by the next update().
	void requestDiscovery();

	// Drains every device's buffered samples and advances its filter.
	// Call once per pipeline tick (the samples carry their own timestamps,
	// so a slow caller loses no information - only output freshness).
	void update();

	// Anchors yaw for one side from a vision-measured palm orientation.
	// Ignored until that side is calibrated (the mounting rotation is what
	// relates the palm to the sensor at all).
	void applyVisionPalmOrientation(eHandSide side, const glm::quat& palmOrientationWorld);

	// Measures the anatomically impossible axial component of the wrist joint,
	// reported as ImuSideStatus::wristAxialTwistDegrees.
	//
	// The wrist has no axial degree of freedom - pronation happens in the
	// FOREARM, the radius crossing the ulna - so whatever this reads is
	// mounting roll error rather than anatomy. That makes it the health check
	// on the roll the curl measured, which is why it is worth computing even
	// though nothing acts on it automatically: an earlier version DID act on
	// it, and correcting a measured roll using the noisiest signal in the
	// system could only make it worse.
	//
	// Kept separate from applyVisionPalmOrientation because the two use the
	// same input for different purposes: that one lets vision correct the
	// FILTER's yaw, this one only reports on the MOUNTING. Confidence gates
	// it - a shaky palm estimate says nothing about a calibration.
	void updateWristAxialResidual(eHandSide side, const glm::quat& palmOrientationWorld,
								  float confidence);

	// Forearm orientation in world space. False when that side has no
	// calibrated, streaming, converged device.
	// NOT const: also feeds the mounting-quality metric below, which needs to
	// watch the orientation actually being published.
	bool getForearmOrientation(eHandSide side, glm::quat& outForearmToWorld);

	// Accumulates the pose-derived mounting, inverse(q_sensor) * q_palm, one
	// sample per tracked frame.
	//
	// Under the correct model that quantity is CONSTANT - both terms rotate
	// together with the arm - so the only thing that perturbs it is wrist
	// bend, which is bounded and averages out. That is what makes averaging
	// valid here, and it is why this beats sampling a single held pose: a
	// mean over hundreds of samples spanning many arm orientations cannot
	// flip the way one instant can, and the arm-axis sign is resolved
	// against it.
	void accumulatePoseMounting(eHandSide side, const glm::quat& palmOrientationWorld, float confidence);

	// Starts recording one of the calibration motions on every device,
	// discarding whatever that window held before. Also resets the live
	// decaying scatter, so the progress bars measure the stage being asked
	// for rather than the one before it.
	void beginMotionRecording(eMountingMotion motion);
	// Stops recording without discarding what was captured
	void endMotionRecording();
	eMountingMotion getMotionRecording() const { return m_motionRecording; }

	// Captures the mounting rotation for one side from the two recorded
	// motions (see imuSolveMountingFromMotions).
	//
	// Takes no pose argument: the geometry comes entirely from the recorded
	// motions, and the accumulated pose average is used only to pick which
	// side of the hinge axis the palm is on. The result reports every gate it
	// checked; a caller must refuse a capture whose bMotionUsable is false
	// rather than bake in a bad mounting.
	bool captureMounting(eHandSide side, MountingCaptureResult& outResult);

	// Discards the accumulated angular-velocity scatter on every device, so a
	// fresh calibration measures only the twisting the user does from here on
	// (the scatter decays on its own, but a wizard should not start with a
	// half-full history of whatever the arms happened to be doing before).
	void resetMountingMotion();

	// STATIC GYRO BIAS CALIBRATION. With the controllers resting untouched,
	// true angular velocity is zero, so the raw gyro reading IS the bias -
	// measured directly on all three axes.
	//
	// This is not redundant with the filter's online estimate: gravity only
	// makes the bias observable about the TILT axes. The component about the
	// gravity axis is not inertially observable at all, and it is exactly the
	// one that shows up later as yaw drift.
	void beginBiasCalibration();
	void cancelBiasCalibration();
	// True while any device is still collecting
	bool isBiasCalibrationRunning() const;

	ImuSideStatus getSideStatus(eHandSide side) const;

	// The most recent mounting capture for one side, with the quality numbers
	// it was built from. Retained because those numbers were previously
	// discarded the moment the wizard closed, leaving no way to judge a bad
	// calibration after the fact.
	const MountingCaptureResult& getLastCapture(eHandSide side) const
	{
		return m_lastCapture[(int)side];
	}

	// Recent RAW samples for one side, oldest first, exactly as the device
	// reported them - before the mirror correction, the bias and the filter.
	//
	// Recorded so the sensor's axis convention can be MEASURED against vision
	// rather than inferred: every candidate axis mapping can be replayed
	// offline against the same capture and scored on how well the integrated
	// rotation tracks the camera-observed palm. Inferring it from correlation
	// signatures produced two confidently wrong answers.
	void getRawSampleHistory(eHandSide side, std::vector<ImuSample>& outSamples) const;

private:
	struct DeviceEntry
	{
		// Shared with the device manager: discovery runs on its own thread and
		// must never free a device this entry is still draining
		std::shared_ptr<IImuDevice> device;
		ImuOrientationFilter filter;
		double lastSampleTimestampMs= -1.0;
		int reopenCooldownFrames= 0;
		// The discovery worker is close()/open()ing this device right now, so
		// nothing here touches it until the worker reports back. The filter
		// state stays put, so a reconnect keeps its converged orientation.
		bool bAwaitingReopen= false;

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
		// Total rotation travelled, sum(|w| dt) - "how much twisting happened"
		float rotationPathRadians= 0.f;
		// Net rotation, sum(w dt). Back-and-forth twisting cancels out here
		// while the path keeps growing; a one-way turn or a constant rate
		// offset makes the two equal.
		glm::vec3 rotationNet{0.f};

		// Running mean of inverse(q_sensor) * q_palm over the calibration
		// window, hemisphere-aligned. Replaces the single held pose.
		glm::vec4 poseMountingSum{0.f};
		int poseMountingSamples= 0;
		float poseSpreadSumDegrees= 0.f;

		// Rolling window of raw samples for the axis-convention diagnostic.
		// ~30 seconds at the Joy-Con's 200 Hz.
		std::deque<ImuSample> rawHistory;

		// The two calibration motions, recorded in full while the wizard asks
		// for them. Separate from the decaying scatter above, which stays the
		// LIVE readout the progress bars watch: a recording must not fade out
		// from under a user who is still performing it.
		std::vector<MotionSample> twistRecording;
		std::vector<MotionSample> curlRecording;

		// Rolling mean of the wrist's axial residual (see
		// updateWristAxialResidual) - a mounting-roll health check, not a
		// correction
		float twistResidualDegreesEma= 0.f;
		int twistSamples= 0;

		// Static bias calibration in progress
		bool bCalibratingBias= false;
		bool bBiasDisturbed= false;
		glm::dvec3 biasSum{0.0};
		int biasSampleCount= 0;
		double biasSeconds= 0.0;
	};

	// Feeds one sample into a device's static bias measurement, restarting it
	// if the controller was disturbed
	void accumulateBiasCalibration(DeviceEntry& entry, const ImuSample& sample, float dtSeconds);


	// Index into m_devices for a wrist, honoring swapSides; -1 when none
	int findDeviceIndexForSide(eHandSide side) const;

	// -- Discovery worker ---------------------------------------------------
	// HID enumeration and the Bluetooth open handshake both block for
	// hundreds of milliseconds. They run here so update() never does.
	void discoveryLoop();
	// Hands a silent device to the worker for a close/open cycle
	void requestReopen(const std::shared_ptr<IImuDevice>& device);
	// Adopts published discovery results + completed reopens (vision thread)
	void adoptDiscoveryResults();

	ImuServiceConfig m_config;
	// Touched ONLY by the discovery worker once started (created before the
	// worker starts, destroyed after it joins)
	std::unique_ptr<class JoyconDeviceManager> m_deviceManager;
	// Vision-thread state
	std::vector<std::unique_ptr<DeviceEntry>> m_devices;
	std::vector<ImuSample> m_sampleScratch;
	bool m_bStarted= false;
	eMountingMotion m_motionRecording= eMountingMotion::None;
	uint32_t m_motionEpoch= 0;
	MountingCaptureResult m_lastCapture[2];
	int m_rescanCooldownFrames= 0;

	std::thread m_discoveryThread;
	mutable std::mutex m_discoveryMutex;
	std::condition_variable m_discoveryCondition;
	bool m_bDiscoveryRequested= false;
	bool m_bDiscoveryExit= false;
	// Worker -> vision thread handoff (guarded by m_discoveryMutex; the lock
	// is only ever held for the swap, never across the blocking calls)
	std::vector<std::shared_ptr<IImuDevice>> m_discoveredDevices;
	bool m_bDiscoveryResultReady= false;
	std::vector<std::shared_ptr<IImuDevice>> m_reopenRequests;
	std::vector<std::string> m_reopenCompletedPaths;
};
