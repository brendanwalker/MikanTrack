#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include "opencv2/core/mat.hpp"

#include "HandFusion.h" // CameraFrameResult, FusionDiagnostics
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
	glm::vec2 wristPx{0.f};
	std::array<FingerAngles, FINGER_COUNT> fingers{};
};

struct DiagCameraState
{
	bool valid= false;
	double timestampMs= 0.0;
	float captureFps= 0.f;
	float inferenceMs= 0.f;
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
};

class DiagnosticDump
{
public:
	// Appends one history record (call after every fusion iteration)
	void record(const std::vector<const CameraFrameResult*>& cameraResults,
				const TrackingFrameResult& fused,
				const FusionDiagnostics& fusionDiagnostics,
				const int dominantCamera[2],
				float autoScaleFactor);

	// Writes dump.json + per-camera raw/annotated PNGs into dumpDir (created
	// if needed). Returns false on any I/O failure (partial output possible).
	bool write(const std::string& dumpDir,
			   const std::vector<DiagCameraSnapshot>& cameras,
			   const TrackingFrameResult& latestFused,
			   const std::string& configJsonString) const;

private:
	// ~4-8s of history depending on the fusion rate; records are ~1-2 KB
	static constexpr size_t kMaxHistory= 240;

	std::deque<DiagFrameRecord> m_history;
};
