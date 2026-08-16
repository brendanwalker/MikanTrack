#pragma once

#include <array>
#include <cassert>
#include <string>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"

#include "MikanVideoSourceTypes.h"
#include "OscOutputMode.h"
#include "TrackingTypes.h"

// Persisted application settings, stored as JSON at
// %APPDATA%/MikanMediaPipe/config.json

struct VideoConfig
{
	std::string deviceName;
	std::string devicePath;
	std::string modeName;
};

struct IntrinsicsConfig
{
	bool present= false;
	MikanMonoIntrinsics intrinsics;
	double reprojectionError= 0.0;
};

struct ExtrinsicsConfig
{
	bool present= false;
	glm::dmat4 markerFromCamera{1.0};
	// Quality of this camera's own pose solve: mean reprojection error of the
	// averaged board corners it was solved from. Low here only means the pose
	// fits THIS camera's detections - see ExtrinsicsQualityConfig for whether
	// the cameras agree with each other.
	double patternReprojectionErrorPx= 0.0;
	int patternCornerCount= 0;
	// Origin aruco marker this camera was calibrated against. A charuco board
	// was tried and rejected: at the 60-70 cm working distance its 24 mm
	// squares (the largest that fits on letter paper) do not resolve enough
	// corner detail to detect, while one large aruco square does.
	int markerId= 0;
	double markerLengthMM= 100.0;
};

// Cross-camera extrinsics quality from the last calibration, measured by
// triangulating the calibration board's corners and comparing the
// reconstruction against the board's known geometry. Reports the WORST camera
// pair, because that pair is what limits fused tracking.
struct ExtrinsicsQualityConfig
{
	bool present= false;
	int pairCount= 0;
	int worstPairCameraA= -1;
	int worstPairCameraB= -1;
	int worstPairSharedCorners= 0;
	// Comparable to the hand pipeline's triResidualRmsPx
	double worstPairReprojectionRmsPx= 0.0;
	// Mean absolute error of triangulated inter-corner distances
	double worstPairSpacingErrorMm= 0.0;
	// Reconstruction scale (1.0 = correct); catches a wrong baseline
	double worstPairSpacingScale= 1.0;
	double worstPairPlanarityRmsMm= 0.0;
};

struct HandScaleConfig
{
	bool present= false;
	double refLengthMeters= 0.08; // wrist -> middle-MCP distance
};

// The user's own hand geometry, measured from stereo triangulation (see
// HandBoneCalibrator). NOT per camera: bones are a physical property, unlike
// the rest angles above, which correct each camera's view-dependent model
// bias. When present it supersedes the landmark model's proportions both as
// the PnP object model and as the skeleton streamed to clients.
//
// neutralDirInPalm is deliberately not stored - it is derived from the base
// positions on load, so the streamed skeleton's zero keeps meaning one thing.
struct HandSkeletonConfig
{
	bool present[2]= {false, false}; // indexed by eHandSide
	std::array<HandSkeleton, 2> skeleton{};
};

struct CharucoBoardConfig
{
	int cols= 11;
	int rows= 8;
	double squareLengthMM= 16.0;
	double markerLengthMM= 12.0;
};

struct TrackingConfig
{
	bool flipHandedness= true;
	int detectorIntervalFrames= 30;
	// Relaxed palm-detection cutoff used while reacquiring a recently-lost
	// hand (a slot is free + a slot died within ~1.5s). Recall when it
	// matters, strict precision otherwise; fusion's gates absorb the extra
	// false positives.
	float palmScoreThresholdRelaxed= 0.25f;
	// When one camera tracks a hand another camera lost, project it into the
	// lost camera's image and try the landmark model there directly
	// RealSense cameras: use the hardware depth stream for the palm transform
	// (metric measurement replaces the monocular PnP estimate)
	bool useRealSenseDepth= true;
	std::string onnxEp= "directml"; // "directml" | "cpu"
};

struct OscConfig
{
	bool enabled= true;
	// Which wire format is live. The two are mutually exclusive.
	eOscOutputMode outputMode= eOscOutputMode::Mikan;
	std::string targetIp= "127.0.0.1";
	int targetPort= 8000;
	// VMC's conventional Performer -> Marionette port. Held separately from
	// targetPort so switching modes cannot silently aim a stream at a listener
	// that speaks the other format.
	int vmcPort= 39539;
	int maxRateHz= 60;
	// Withhold a hand's pose messages (and report it untracked) below this
	// fused confidence, so clients can hold/blend instead of jittering
	float minConfidence= 0.f;
	// Keep sending the last good pose (confidence decaying to zero) for this
	// long after a dropout before reporting tracked=0 - bridges brief losses
	// so clients don't slam to their rest-pose blend. 0 = report immediately.
	float holdOnDropoutMs= 250.f;
	// Log every palm transform as it goes onto the wire, for diffing against
	// what a client reports receiving. Off by default: this is one line per
	// hand per frame.
	bool logPalmFrames= false;

	// -- VMC mode only ------------------------------------------------------
	// Neck -> head bone offset. A VMC receiver replaces the translation of
	// every bone it is sent, so the head needs one, and nothing on this rig
	// measures a neck.
	float vmcHeadOffsetMeters= 0.08f;
	// Keep streaming a lost hand's last bones instead of going silent, which
	// would drop that arm back to the avatar's rest T-pose
	bool vmcFreezeOnLoss= true;
};

// Everything specific to one physical camera. Each camera calibrates
// intrinsics + extrinsics against the same printed marker, so all cameras
// share one world frame (which is what makes multi-camera fusion possible).
// Angles this camera reports for a hand held in its rest pose. Subtracted
// from later measurements so the rest pose reads zeros. PER CAMERA because
// MediaPipe's model landmarks are view-dependent - two cameras watching one
// physical hand disagree about articulation by tens of degrees.
struct RestAnglesConfig
{
	bool present[2]= {false, false}; // indexed by eHandSide
	std::array<std::array<FingerAngles, FINGER_COUNT>, 2> angles{};
};

// Fitted per-side Gaussian prior over the 20 RAW finger angles (mean +
// row-major precision matrix), produced by --fit-angle-prior from the user's
// own stereo-quality recordings. Consumed by the hand state estimator as a
// weak Mahalanobis pull toward the user's real pose distribution.
struct AnglePriorConfig
{
	static constexpr int k_angleCount= FINGER_COUNT * 4;
	bool present[2]= {false, false}; // indexed by eHandSide
	std::array<std::array<float, k_angleCount>, 2> mean{};
	std::array<std::array<float, k_angleCount * k_angleCount>, 2> precision{};
};

// Body-pose stage, opt-in per camera. The person detector only fires on
// cameras that see the user upright, so this stays off for overhead cameras
// and costs them nothing.
struct BodyPoseCameraConfig
{
	bool enabled= false;
	// Pose models run every Nth frame on this camera
	int poseFrameDivider= 2;
	// Rebuild the region of interest from the image every Nth model frame,
	// regardless of model confidence
	int detectorIntervalFrames= 20;
};

struct CameraProfile
{
	VideoConfig video;
	IntrinsicsConfig intrinsics;
	ExtrinsicsConfig extrinsics;
	RestAnglesConfig restAngles;
	BodyPoseCameraConfig bodyPose;
};

// Wrist-strapped inertial trackers (Joy-Cons today, SlimeVR later).
// The mounting rotation is captured per side with the wrist held straight,
// which defines the forearm frame as "the palm frame at neutral wrist".
struct ImuConfig
{
	bool enabled= true;
	// Vision yaw-anchor strength, radians. Loose on purpose: vision sees the
	// palm, the sensor rides the forearm, and the wrist joint between them is
	// unknown - so vision should correct slow yaw drift without fighting real
	// wrist motion.
	float visionYawSigma= 0.35f;
	// Swap which controller drives which wrist
	bool swapSides= false;
	// Captured mounting rotation (forearm -> sensor), indexed by eHandSide
	bool mountingPresent[2]= {false, false};
	std::array<glm::quat, 2> forearmToSensor{glm::quat(1.f, 0.f, 0.f, 0.f), glm::quat(1.f, 0.f, 0.f, 0.f)};
};

// Body proportions shared by every elbow consumer (IMU-measured forearms and
// the vision body-pose solver alike)
struct BodyConfig
{
	// Forearm length, used to place the elbow estimate back along the forearm
	// direction from the wrist (visualization + IK hint). The direction is
	// measured; only this length is assumed. The IMU mounting wizard's curl
	// stage overwrites it with a measured value.
	float forearmLengthMeters= 0.25f;
	// Elbow-to-shoulder length. Only disambiguates which of the two
	// ray-sphere elbow solutions is real; it never places a joint by itself.
	float upperArmLengthMeters= 0.30f;
	// Derive the upper arm from the shoulder width instead of measuring it.
	// Measuring it needs the arm held straight and square to the camera,
	// which is hard to hit at a desk and wrong by 20% when missed - and a
	// wrong upper arm makes the elbow solve pick the wrong bend. Body
	// proportions are stable enough that a multiple of a width that IS easy
	// to measure beats a length that is not.
	bool bDeriveUpperArmFromShoulderWidth= true;
	// Upper arm as a multiple of the LANDMARK shoulder width. Not the
	// anatomical ratio: the model's shoulder points sit inside the real
	// joints (measured at ~0.74 of a biacromial breadth), so the familiar
	// "arm is about 1.5 shoulder widths" becomes ~2.0 of THIS width, and the
	// upper arm alone lands near 1.05.
	float upperArmPerShoulderWidth= 1.05f;
	// Distance between the shoulder joints. With both shoulder rays known,
	// this fixes their depth without the pose model's own metric guess, which
	// measured unusable per frame (its forearm length swung nearly 3x within
	// one session).
	float shoulderWidthMeters= 0.40f;
	// Ear-to-ear width, which fixes head depth the same way
	float headWidthMeters= 0.15f;
	// Ear-axis midpoint to nose tip, which fixes head yaw and pitch
	float noseForwardMeters= 0.11f;
};

// Tracking recordings store landmarks, which are abstract enough to share.
// Raw frames are video of a room, so storing them is OPT-IN and local: this
// defaults off and stays off unless someone deliberately turns it on.
struct RecordingConfig
{
	// Store the frames the models consumed alongside the landmark recording.
	// The reason to want it: a landmark recording cannot answer "would a
	// different pose model have done better", because the model's input is
	// gone. Costs roughly 3-6 MB/s per camera.
	bool recordRawFrames= false;
	int jpegQuality= 85;
};

struct FusionConfig
{
	// A camera's last result older than this is excluded from fusion
	double stalenessWindowMs= 66.0;
	// Two cameras' world wrists further apart than this can't be the same
	// physical hand (cross-camera handedness-conflict gate)
	float wristMatchMaxDistM= 0.25f;
	// Drop a camera's observation outright below this confidence
	// (presence x measured stability). 0 = rely on soft weighting only.
	float minCameraConfidence= 0.f;
	// Palm jitter (mm) at which an observation counts as half as trustworthy
	float jitterReferenceMm= 15.f;
	// Stereo landmark triangulation (two cameras -> world landmarks straight
	// from image geometry; the network's monocular depth stays out of it)
	bool triangulationEnabled= true;
	// RMS reprojection residual above which a triangulated pairing is
	// rejected as two different physical hands
	float triangulationMaxResidualPx= 25.f;
	// Residual (px) at which a triangulated pose counts as half as
	// trustworthy - a TRUE pose-quality signal (unlike presence, which stays
	// high on ill-conditioned views), folded into the fused confidence
	float residualReferencePx= 8.f;

	// Angle-space multi-view state estimator: one temporally continuous state
	// per hand fit to all fresh cameras' 2D landmarks (the output pose path).
	// Temporal-prior process sigmas (how far each block may move per second)
	float estimatorPalmPosSigmaMPerS= 0.3f;
	float estimatorPalmRotSigmaRadPerS= 2.f;
	float estimatorAngleSigmaRadPerS= 2.5f;
	// Measurement weighting
	float estimatorPixelSigmaPx= 2.f;
	float estimatorHuberDeltaPx= 10.f;
	int estimatorMaxIterations= 4;
	// Post-fit mean residual above this holds the previous state; a streak
	// of them drops the state for a reseed (divergence guard)
	float estimatorMaxResidualPx= 25.f;
	// Palm steps implying more than this speed are correspondence errors,
	// not motion - they take the bad-fit hold path (physical innovation gate)
	float estimatorMaxStepMetersPerS= 4.f;
	// Anatomical joint-limit prior inside the estimator fit (one-sided soft
	// penalties beyond the per-DoF ranges; zero cost inside them)
	bool estimatorJointLimitsEnabled= true;
	float estimatorJointLimitSigmaRad= 0.05f;
	// Scale on the fitted angle prior's precision (see AnglePriorConfig);
	// 1 = trust the fitted distribution as-is, smaller = softer
	float estimatorAnglePriorWeight= 0.3f;
};

// Declared here (rather than each caller assembling its own) so the live
// solve and a replayed one cannot drift apart
struct BodyDimensions;
BodyDimensions makeBodyDimensions(const class AppConfig& config);

// Same drift-proofing for the fusion config: the vision thread, the replay
// engine and the replay self-test all build it from an AppConfig through this
// one mapping
struct HandFusionConfig;
HandFusionConfig makeHandFusionConfig(const class AppConfig& config);

class AppConfig
{
public:
	// invariant: never empty (default-constructed with one camera)
	std::vector<CameraProfile> cameras= std::vector<CameraProfile>(1);
	HandScaleConfig handScale;
	HandSkeletonConfig handSkeleton;
	CharucoBoardConfig charucoBoard;
	TrackingConfig tracking;
	OscConfig osc;
	FusionConfig fusion;
	ImuConfig imu;
	BodyConfig body;
	RecordingConfig recording;
	// Rest-pose zero for the stereo-TRIANGULATED path (one set, not per
	// camera: triangulated geometry has no per-camera model bias to fold in).
	// Captured alongside the per-camera rest angles.
	RestAnglesConfig fusedRestAngles;
	// Fitted angle prior for the hand state estimator (--fit-angle-prior)
	AnglePriorConfig anglePrior;

	// Cross-camera extrinsics quality from the last calibration session
	ExtrinsicsQualityConfig extrinsicsQuality;

	CameraProfile& camera(size_t index)
	{
		assert(index < cameras.size());
		return cameras[index];
	}
	const CameraProfile& camera(size_t index) const
	{
		assert(index < cameras.size());
		return cameras[index];
	}
	size_t cameraCount() const { return cameras.size(); }

	// Loads from the default config path; returns false (and keeps defaults)
	// if missing/corrupt. Migrates v1 single-camera configs into cameras[0].
	// Guarantees cameras.size() >= 1 afterwards.
	bool load();
	bool save() const;
	// Parses the same schema load() reads from an in-memory JSON string.
	// Tracking recordings embed a config snapshot in their header; replay
	// reconstructs the recorded AppConfig through this.
	bool loadFromJsonString(const std::string& jsonText);
	// Serialized live config (same schema save() writes) - diagnostic dumps
	// embed this so a dump always reflects unsaved in-UI changes too
	std::string toJsonString() const;

	void markDirty() { m_bDirty= true; }
	// Saves at most once per cooldown period when dirty; call from the main loop
	void updateAutoSave(float deltaSeconds);

	static std::string getConfigFilePath();
	// Fresh timestamped folder path for a diagnostic dump:
	// <config dir>/dumps/<yyyy-mm-dd_hh-mm-ss> (not created here)
	static std::string makeDumpDirectoryPath();
	// Fresh timestamped file path for a tracking recording:
	// <config dir>/recordings/<yyyy-mm-dd_hh-mm-ss>.jsonl (directory created)
	static std::string makeRecordingFilePath();
	static std::string getRecordingsDirectoryPath();

private:
	bool m_bDirty= false;
	float m_secondsSinceDirty= 0.f;
};
