#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "TrackingTypes.h"

// Tracking recording format: everything the post-MediaPipe stages consume,
// captured per fusion iteration, plus the fused output and a checksum, so the
// whole LandmarkTo3D -> HandFusion -> smoothing pipeline can be re-run offline
// bit-exactly. JSONL on disk: one header line, one line per frame, one footer
// line. nlohmann's float round-trip is lossless, which is what makes replayed
// checksums comparable to recorded ones.
//
// Non-finite floats serialize as null (nlohmann behavior) and load back as 0;
// a NaN that reached the recorded values means live consumed it too, so the
// resulting checksum mismatch is the honest outcome, not corruption.

// Pipeline-output fields for one hand, exactly what LandmarkTo3D::process
// consumes (plus imageQuality, carried for timeline display/analysis only)
struct RecordedHandInput
{
	bool tracked= false;
	int side= 0;
	int slotId= -1;
	float presence= 0.f;
	float handednessScore= 0.f;
	float rightProb= 0.5f;
	std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints{};
	std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints{};
	HandImageQuality imageQuality;
};

// One camera that popped a frame this iteration (stale cameras record
// nothing; replay carries their previous mirror forward)
struct RecordedCameraInput
{
	int cameraIndex= -1;
	double timestampMs= 0.0;
	// bProducedTracking: false = the camera ticked but tracking was disabled
	// or the pipeline was down; replay updates the mirror's timestamp only
	bool valid= false;
	// The hand scale actually in effect for this camera's LandmarkTo3D call
	// (records the auto-scale EMA feedback loop as a plain input)
	float refLengthMeters= 0.f;

	int64_t frameIndex= -1;
	int frameWidth= 0;
	int frameHeight= 0;
	float captureFps= 0.f;
	float inferenceMs= 0.f;
	float lumaInstability= 0.f;
	float lumaFlickerHz= 0.f;

	std::array<RecordedHandInput, 2> hands;

	// Body-pose observation from this camera's opt-in body-pose stage. The
	// stage runs upstream of this tap, so divider cadence (held re-emits) is
	// baked in and replay never re-runs the pose models.
	bool bHaveBodyPose= false;
	BodyPoseObservation body;
};

// Fused pose snapshot taken immediately after fuse(), BEFORE the IMU forearm
// fill. Full skeleton included so the recorded output renders via FK.
struct RecordedPoseOutput
{
	bool tracked= false;
	int side= 0;
	float presence= 0.f;
	float visibility= 0.f;
	float confidence= 0.f;
	float fkReprojectionPx= 0.f;
	bool stereoTriangulated= false;
	bool hasCameraPose= false;
	glm::vec3 palmPositionCamera{0.f};
	glm::quat palmOrientationCamera{1.f, 0.f, 0.f, 0.f};
	bool hasWorldPose= false;
	glm::vec3 palmPositionWorld{0.f};
	glm::quat palmOrientationWorld{1.f, 0.f, 0.f, 0.f};
	std::array<FingerAngles, FINGER_COUNT> fingers{};
	HandSkeleton skeleton;
};

// The IMU forearm output actually published this frame. Replay overlays it
// (fusion never reads IMU; the EKF is deliberately not re-run offline).
struct RecordedImuOutput
{
	bool hasForearmPose= false;
	glm::quat forearmOrientationWorld{1.f, 0.f, 0.f, 0.f};
	float forearmConfidence= 0.f;
};

struct RecordedFrame
{
	int64_t seq= -1;
	// The newestTimestampMs passed to fuse() (recorded, never re-derived)
	double nowTimestampMs= 0.0;
	// false = the no-world passthrough path (camera 0's result passed through
	// without a fuse() call)
	bool bFused= false;
	// Cameras that popped a frame this iteration, ascending camera index (the
	// live loop's sequential processing order)
	std::vector<RecordedCameraInput> freshCameras;
	std::array<RecordedPoseOutput, 2> outPoses;
	std::array<RecordedImuOutput, 2> imu;
	uint64_t checksum= 0;
};

struct RecordingHeader
{
	int formatVersion= 0;
	// The AppConfig snapshot (same schema config.json uses), reparsable via
	// AppConfig::loadFromJsonString
	std::string appConfigJsonText;
	int cameraCount= 0;
};

struct RecordingFooter
{
	int64_t frameCount= 0;
	bool bAborted= false;
	std::string reason;
};

namespace TrackingRecording
{
constexpr int k_formatVersion= 1;

// FNV-1a 64 over the fused output's raw little-endian float bytes, per side
// Left then Right: 1 byte tracked; if tracked, 1 byte hasWorldPose, then 28
// floats in fixed order: palm position xyz, palm orientation xyzw (world when
// hasWorldPose, else camera), the 20 finger angles, confidence. Computed on
// the PRE-IMU fused result by both the live tap and replay.
uint64_t computeFusedChecksum(const TrackingFrameResult& preImuFused);

// Fills the pose snapshots from a (pre-IMU) fused result
void snapshotFusedOutput(const TrackingFrameResult& fused,
						 std::array<RecordedPoseOutput, 2>& outPoses);

// Raw frames live in a sibling directory named after the recording, so a
// recording and its frames travel together and either can be deleted alone:
// <dir>/2026-01-01_12-00-00.jsonl -> <dir>/2026-01-01_12-00-00_frames
std::string makeFrameDirectoryPath(const std::string& recordingFilePath);

nlohmann::json headerToJson(const RecordingHeader& header);
bool headerFromJson(const nlohmann::json& j, RecordingHeader& outHeader);
nlohmann::json frameToJson(const RecordedFrame& frame);
bool frameFromJson(const nlohmann::json& j, RecordedFrame& outFrame);
nlohmann::json footerToJson(const RecordingFooter& footer);
bool footerFromJson(const nlohmann::json& j, RecordingFooter& outFooter);
} // namespace TrackingRecording
