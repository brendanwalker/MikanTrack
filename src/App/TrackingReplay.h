#pragma once

#include <memory>
#include <string>
#include <vector>

#include "AppConfig.h"
#include "BodyPoseSolver.h"
#include "HandFusion.h"
#include "LandmarkTo3D.h"
#include "TrackingRecording.h"
#include "TrackingTypes.h"

// Offline deterministic re-simulation of a tracking recording: loads the
// JSONL file, reconstructs the recorded AppConfig, then re-runs the exact
// LandmarkTo3D -> HandFusion -> smoothing sequence the vision thread ran, on
// its own instances, caching per-frame outputs for timeline scrubbing.
//
// Determinism contract: same binary + recorded inputs => bit-identical fused
// output, verified per frame by the FNV-1a checksum recorded live. The IMU
// EKF is not re-run; the recorded forearm output is overlaid for display.
//
// THREAD AFFINITY: single-threaded (main thread / CLI); no locking.
class TrackingReplay
{
public:
	// Everything cached for one recorded frame after a replay pass
	struct ReplayFrame
	{
		// The recorded inputs + live outputs (owned by m_recordedFrames)
		const RecordedFrame* recorded= nullptr;

		// Replayed fused output with the recorded IMU forearm overlay applied
		// (what the timeline displays for the "Replayed" view)
		TrackingFrameResult replayedFused;
		// Each camera's replayed per-camera result as of this frame (stale
		// cameras carry the previous fresh result forward, like live)
		std::vector<TrackingFrameResult> perCamera;
		std::vector<bool> perCameraValid;

		uint64_t replayChecksum= 0;
		bool bChecksumMatch= false;

		// What-if pass output (valid when bHasWhatIf)
		TrackingFrameResult whatIfFused;
		// The what-if pass's own per-camera results, the twin of perCamera.
		// Kept so a tool can ask what a single camera made of the frame under
		// each set of parameters, not just what fusion did with them.
		std::vector<TrackingFrameResult> whatIfPerCamera;
		bool bHasWhatIf= false;

		// Regenerated fusion diagnostics (populated only inside the
		// setCaptureDiagnostics range - they are large)
		FusionDiagnostics diagnostics;
		bool bHasDiagnostics= false;

		// The what-if pass's own diagnostics over the same capture range, so a
		// tool can attribute what-if behavior (estimator reseeds/holds) rather
		// than infer it from the output alone
		FusionDiagnostics whatIfDiagnostics;
		bool bHasWhatIfDiagnostics= false;
	};

	// Parameters for the what-if pass, seeded from the recording's config
	struct WhatIfParams
	{
		HandFusionConfig fusionConfig;
		// Multiplier on each frame's recorded effective ref length
		float refLengthScale= 1.f;
		// Off = ignore the recording's measured hand skeleton, falling back to
		// the landmark model's own proportions (the A/B for a bone calibration)
		bool bApplyCalibratedSkeleton= true;
		// Replace each camera's extrinsics with these (indexed by camera).
		// Sound to override offline: recorded landmarks live in each camera's
		// own undistorted image space, and extrinsics apply strictly after
		// camera-space projection - so two calibrations can be A/B'd against
		// the same recorded hands (--replay-extrinsics).
		bool bOverrideExtrinsics= false;
		std::vector<ExtrinsicsConfig> extrinsicsOverride;
	};

	// Loads and parses the whole file (header + frames; a missing footer is
	// tolerated as a crash-truncated recording). Does not replay yet.
	bool load(const std::string& path, std::string& outError);

	int getFrameCount() const { return (int)m_frames.size(); }
	const AppConfig& getRecordedConfig() const { return m_recordedConfig; }
	const RecordingFooter& getFooter() const { return m_footer; }
	bool hasFooter() const { return m_bHasFooter; }
	const std::string& getFilePath() const { return m_filePath; }

	// Replays every frame from frame 0, filling the cache and the checksum
	// comparison. Optionally captures fusion diagnostics for a frame range
	// (set before calling).
	void runAll();
	void setCaptureDiagnostics(bool bCapture, int firstFrame= 0, int lastFrame= -1);

	const ReplayFrame& getFrame(int index) const { return m_frames[index]; }
	const std::vector<int>& getDivergentFrames() const { return m_divergentFrames; }
	bool didRun() const { return m_bDidRun; }

	// Seeds what-if parameters from the recording's config (the same mapping
	// the vision thread applies)
	WhatIfParams makeDefaultWhatIfParams() const;
	// Re-runs the whole recording with edited parameters into each frame's
	// whatIfFused slot; the plain replay results are untouched
	void runWhatIf(const WhatIfParams& params);
	void clearWhatIf();
	bool hasWhatIf() const { return m_bHasWhatIf; }

private:
	// Builds the HandFusionConfig the recorded AppConfig implies (mirrors
	// VisionThread::refreshConfigOnThread)
	HandFusionConfig buildFusionConfig() const;
	// One full deterministic pass; whatIfParams null = faithful replay
	void runPass(const WhatIfParams* whatIfParams);

	std::string m_filePath;
	AppConfig m_recordedConfig;
	RecordingHeader m_header;
	RecordingFooter m_footer;
	bool m_bHasFooter= false;

	std::vector<RecordedFrame> m_recordedFrames;
	std::vector<ReplayFrame> m_frames;
	std::vector<int> m_divergentFrames;
	bool m_bDidRun= false;
	bool m_bHasWhatIf= false;

	bool m_bCaptureDiagnostics= false;
	int m_diagnosticsFirstFrame= 0;
	int m_diagnosticsLastFrame= -1;

	// Replay instances (rebuilt at the start of every pass)
	HandFusion m_fusion;
	BodyPoseSolver m_bodyPoseSolver;
	std::vector<std::unique_ptr<LandmarkTo3D>> m_landmarkTo3D;
	std::vector<CameraFrameResult> m_mirrors;
};
