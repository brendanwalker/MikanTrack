#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "HandFusion.h" // CameraFrameResult, FusionDiagnostics
#include "HandTrackingPipeline.h" // HandSeedStats
#include "TrackingTypes.h"

// Diagnostic dump system (F9): a rolling history of compact tracking state
// recorded every fusion iteration on the vision thread, written to a
// timestamped folder on demand together with the current camera frames (raw
// + annotated PNGs), full landmark snapshots and the live config. Built for
// after-the-fact diagnosis of transient tracking failures (clap recovery,
// side swaps) - the frames LEADING UP to a failure are the diagnosis.
//
// THREAD AFFINITY: record() and write() run on the vision thread; there is
// no internal locking.

// Compact per-side state kept in the history ring
struct DiagHandState
{
	bool tracked= false;
	int slotId= -1; // pipeline slot identity (watch for ROI hijacks at hand crossings)
	float presence= 0.f;
	float handednessScore= 0.f;
	float rightProb= 0.5f; // flip-adjusted classifier opinion (what fusion votes with)
	int labeledSide= -1;   // camera records: that camera's own label; fused: assigned side
	float visibility= 0.f;
	float confidence= 0.f; // presence x measured stability (what OSC gates on)
	float fkReprojectionPx= 0.f; // FK-vs-2D-landmark agreement (parameterization fidelity)
	bool stereoTriangulated= false; // pose from cross-camera landmark triangulation
	bool hasWorldPose= false;
	glm::vec3 palmPositionWorld{0.f};
	glm::quat palmOrientationWorld{1.f, 0.f, 0.f, 0.f};
	bool hasForearmPose= false; // wrist IMU calibrated + streaming
	glm::quat forearmOrientationWorld{1.f, 0.f, 0.f, 0.f};
	glm::vec2 wristPx{0.f};
	std::array<FingerAngles, FINGER_COUNT> fingers{};
	// Lighting/exposure statistics of the hand ROI (camera records only; the
	// fused pose has no single image behind it)
	HandImageQuality imageQuality;
};

// Per-side wrist IMU state. Recorded every frame because the questions it
// answers are about CHANGE over time - when a controller went silent, whether
// the mounting quality degraded mid-session - and because a mounting that is
// subtly wrong can only be judged against the motion that was happening at
// the time.
struct DiagImuState
{
	bool deviceConnected= false;
	bool streaming= false;
	bool calibrated= false;
	bool orientationValid= false;
	float sampleRateHz= 0.f;
	double millisecondsSinceLastSample= -1.0;
	// Twisting the forearm must leave the forearm frame's +X fixed; this is
	// how well it does (-1 = not enough motion yet). Low here with the elbow
	// misbehaving means the MOUNTING is wrong, not the tracking.
	float forearmAxisConsistency= -1.f;
	// How single-axis the recent motion has been - i.e. whether the above
	// number was measured against real twisting or against arm-waving
	float armAxisDominance= -1.f;
	float twistProgress= 0.f;
	float twistReversal= 0.f;
	// SIGNED axial residual of the wrist joint, degrees (-999 = not measured).
	// Anatomically zero, because the wrist has no axial degree of freedom, so
	// anything else is mounting roll error - and unlike forearmAxisConsistency
	// it means something during any motion rather than only a twist.
	float wristAxialTwistDegrees= -999.f;
	glm::vec3 gyroBiasDegreesPerSecond{0.f};
	// Bias pinned at its bound = the filter diverged; everything else this
	// device reported is suspect, so this is the first thing to check
	bool biasSaturated= false;
	float yawSigmaRadians= 0.f;
	// The filter's OWN orientation, separate from the mounting composed onto
	// it. Without this a frozen filter and a cancelling mounting look the same.
	glm::quat filterOrientation{1.f, 0.f, 0.f, 0.f};
	float tiltSigmaRadians= 0.f;
	// Fraction of accelerometer samples the gravity gate accepted. Near zero
	// means the filter is running open loop on the gyro.
	float gravityAcceptRatio= -1.f;
	float visionYawCorrectionDegrees= 0.f;
};

struct DiagCameraState
{
	bool valid= false;
	double timestampMs= 0.0;
	float captureFps= 0.f;
	float inferenceMs= 0.f;
	// Whole-frame luminance oscillation (flicker beat / AE hunting)
	float lumaInstability= 0.f;
	float lumaFlickerHz= 0.f;
	DiagHandState sides[2];
};

struct DiagFrameRecord
{
	double fuseTimestampMs= 0.0;
	std::vector<DiagCameraState> cameras;
	DiagHandState fused[2];
	int dominantCamera[2]= {-1, -1};
	float autoScaleFactor= 1.f;
	FusionDiagnostics fusion;
	DiagImuState imu[2];
};

// Raw IMU samples for the axis-convention diagnostic: exactly what the device
// sent, before any correction, bias or filtering. Paired with the palm
// quaternions already in the history, this lets every candidate axis mapping
// be replayed offline and scored against what the cameras actually saw -
// which is the only way to settle a sensor frame, since a frame error can be
// perfectly self-consistent between gyro and accelerometer.
struct DiagImuRawSample
{
	double timestampMs= 0.0;
	glm::vec3 acceleration{0.f};
	glm::vec3 angularVelocity{0.f};
};

// The mounting capture a side is currently running on, and the quality
// numbers behind it. Written once per dump rather than per frame - it only
// changes when someone recalibrates - so a bad calibration can be judged
// after the fact instead of being redone blind.
struct DiagImuCapture
{
	bool present= false;
	glm::quat forearmToSensor{1.f, 0.f, 0.f, 0.f};
	bool motionUsable= false;
	int poseSamples= 0;
	float poseSpreadDegrees= 0.f;
	float axisDominance= 0.f;
	float twistProgress= 0.f;
	float twistReversal= 0.f;
	float curlDominance= 0.f;
	float curlProgress= 0.f;
	float curlReversal= 0.f;
	int curlStrokes= 0;
	float hingeSpreadDegrees= 0.f;
	float interAxisAngleDegrees= 0.f;
	bool lengthMeasured= false;
	float forearmLengthMeters= 0.f;
	float lengthFitCorrelation= 0.f;
	int palmarSource= 0;
};

// Everything write() needs about one camera at dump time
struct DiagCameraSnapshot
{
	const CameraFrameResult* lastResult= nullptr;
	const cv::Mat* frame= nullptr; // last processed BGR frame (may be null/empty)
	float deviceFps= 0.f;
	uint64_t droppedFrames= 0;
	const char* activeEp= "none";
	bool trackingEnabled= false;
	// Cross-camera seed accounting (cumulative, so it belongs in the snapshot
	// rather than the per-frame history). Null when the camera has no pipeline.
	const HandSeedStats* seedStats= nullptr;
};

class DiagnosticDump
{
public:
	// Appends one history record (call after every fusion iteration)
	void record(const std::vector<const CameraFrameResult*>& cameraResults,
				const TrackingFrameResult& fused,
				const FusionDiagnostics& fusionDiagnostics,
				const int dominantCamera[2],
				float autoScaleFactor,
				const DiagImuState imu[2]);

	// Writes dump.json + per-camera raw/annotated PNGs into dumpDir (created
	// if needed). Returns false on any I/O failure (partial output possible).
	bool write(const std::string& dumpDir,
			   const std::vector<DiagCameraSnapshot>& cameras,
			   const TrackingFrameResult& latestFused,
			   const std::string& configJsonString,
			   const std::vector<DiagImuRawSample> rawImu[2],
			   const DiagImuCapture lastCapture[2]) const;

private:
	// ~4-8s of history depending on the fusion rate; records are ~1-2 KB
	static constexpr size_t kMaxHistory= 240;

	std::deque<DiagFrameRecord> m_history;
};
