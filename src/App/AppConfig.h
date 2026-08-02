#pragma once

#include <array>
#include <cassert>
#include <string>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"

#include "MikanVideoSourceTypes.h"
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
	int markerId= 0;
	double markerLengthMM= 100.0;
};

struct HandScaleConfig
{
	bool present= false;
	double refLengthMeters= 0.08; // wrist -> middle-MCP distance
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
	// With two calibrated cameras, continuously refine the hand scale from
	// stereo wrist triangulation (overrides the wizard-measured scale live)
	bool autoHandScaleFromStereo= true;
	// When one camera tracks a hand another camera lost, project it into the
	// lost camera's image and try the landmark model there directly
	bool crossCameraSeeding= true;
	// RealSense cameras: use the hardware depth stream for the palm transform
	// (metric measurement replaces the monocular PnP estimate)
	bool useRealSenseDepth= true;
	// Post-fusion one-euro smoothing, split by signal: palm transform latency
	// is visible (the hand drags through space) while finger articulation
	// latency isn't - so the palm gets a high cutoff (responsive) and the
	// angles a low one (steady). One-euro: cutoff= minCutoff + beta * |dx|.
	float palmMinCutoff= 3.0f;
	float palmBeta= 0.1f;
	float angleMinCutoff= 0.75f;
	float angleBeta= 0.02f;
	bool smoothingEnabled= true;
	std::string onnxEp= "directml"; // "directml" | "cpu"
};

struct OscConfig
{
	bool enabled= true;
	std::string targetIp= "127.0.0.1";
	int targetPort= 8000;
	int maxRateHz= 60;
	// Withhold a hand's pose messages (and report it untracked) below this
	// fused confidence, so clients can hold/blend instead of jittering
	float minConfidence= 0.f;
	// Keep sending the last good pose (confidence decaying to zero) for this
	// long after a dropout before reporting tracked=0 - bridges brief losses
	// so clients don't slam to their rest-pose blend. 0 = report immediately.
	float holdOnDropoutMs= 250.f;
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

struct CameraProfile
{
	VideoConfig video;
	IntrinsicsConfig intrinsics;
	ExtrinsicsConfig extrinsics;
	RestAnglesConfig restAngles;
};

struct FusionConfig
{
	// A camera's last result older than this is excluded from fusion
	double stalenessWindowMs= 66.0;
	// Two cameras' world wrists further apart than this can't be the same
	// physical hand (cross-camera handedness-conflict gate)
	float wristMatchMaxDistM= 0.25f;
	// Spatial side prior for users who never cross their hands: world axis
	// (marker frame) pointing toward the RIGHT hand's side of the desk.
	// 0=off, 1=+X, 2=-X, 3=+Y, 4=-Y
	int spatialSidePriorAxis= 0;
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
};

class AppConfig
{
public:
	// invariant: never empty (default-constructed with one camera)
	std::vector<CameraProfile> cameras= std::vector<CameraProfile>(1);
	HandScaleConfig handScale;
	CharucoBoardConfig charucoBoard;
	TrackingConfig tracking;
	OscConfig osc;
	FusionConfig fusion;
	// Rest-pose zero for the stereo-TRIANGULATED path (one set, not per
	// camera: triangulated geometry has no per-camera model bias to fold in).
	// Captured alongside the per-camera rest angles.
	RestAnglesConfig fusedRestAngles;

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

private:
	bool m_bDirty= false;
	float m_secondsSinceDirty= 0.f;
};
