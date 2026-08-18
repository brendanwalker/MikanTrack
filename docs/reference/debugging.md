# Debugging

Logging, the command registry, and the diagnostic instruments used to investigate a tracking problem: the on-demand state dump (F9) and deterministic record/replay (F10). Build and toolchain details are in [build.md](./build.md), the full command-line flag list in [commands.md](./commands.md), the app's architecture in [architecture.md](./architecture.md), coordinate and unit conventions in [conventions.md](./conventions.md), and OSC/VMC wire details in [wire-protocol.md](./wire-protocol.md). Hand and body tracking specifics live in [hand-tracking.md](./hand-tracking.md) and [body-pose.md](./body-pose.md), camera calibration in [calibration.md](./calibration.md), and IMU mounting/axis calibration in [imu.md](./imu.md). Past investigations and the reasoning behind non-obvious design choices are in [LEARNINGS.md](../../LEARNINGS.md).

---

## Logging

The logger lives in `src/Utility/Logger.h` / `src/Utility/Logger.cpp`. Levels are `LogSeverityLevel` trace/debug/info/warning/error/fatal. Log with the `MIKAN_LOG_<LEVEL>("Scope::function") << ...` stream macros on the main thread, and the `MIKAN_MT_LOG_*` variants off it (`ThreadSafeLoggerStream` takes a mutex). Each process calls `log_init(LoggerSettings)` once; the settings choose the minimum level, an optional log file (`log_filename`, opened as a relative path), whether a Win32 GUI process allocates a console (`enable_console`), and an optional `log_callback`.

Every log line always reaches stdout (stderr at error and above) and the log file, if one was configured, through `log_default_callback`. A caller-supplied `log_callback` is invoked in addition to that, never instead of it: `LoggerStreamImpl::write_line` calls `log_default_callback` first and then calls the configured callback only when it differs from the default. A custom sink (the in-app log panel) can add a destination; it cannot silently swallow the file or stdout output for the code paths that forgot to also write there.

`MikanTrack.exe` is a `WIN32`-subsystem executable (`add_executable(MikanTrack WIN32 ...)` in the top-level `CMakeLists.txt`), so it has no attached console and printing to stdout goes nowhere. `App::startup` (`src/App/App.cpp`) calls `log_init` with `enable_console= false` and `log_filename= "MikanTrack.log"`, so the only way to read a normal run's log is the file, written next to the process's working directory. It also sets `log_callback= &LogPanel::logCallback` (`src/UI/LogPanel.*`), so the same lines populate the in-app Log panel.

The command-line self-tests and tools (below) each get their own logger via `TestRegistry::tryRun`: minimum level info, console enabled, and a log filename built from the flag by stripping leading dashes and appending `.log`. `--replay-verify` writes `replay-verify.log`, `--test-fusion` writes `test-fusion.log`, and so on for every registered command.

---

## Test registry

`src/Tests/TestRegistry.h` / `TestRegistry.cpp`. Each self-test, hardware test, and headless tool lives in its own file under `src/Tests` and registers itself with `MIKAN_REGISTER_TEST(flag, description, category, function)`, placed directly below the command's function so the flag, description, and implementation stay together.

`eTestCategory` has three values:
- `SelfTest`: deterministic, no hardware, no input files. Safe to run as a batch.

- `Hardware`: needs a camera or controller physically connected to mean anything (for example `--test-imuaxes`, `--test-joycon`).

- `Tool`: a headless diagnostic that operates on a file the user names (for example `--replay-verify`, `--calibrate-bones`).

Registrations land in a function-local static (`getMutableRegistry()` inside `TestRegistry.cpp`), not a file-scope global, because registration runs from static initializers scattered across translation units whose construction order relative to a file-scope container is undefined.

`src/Tests` has one command per file: `TestFusion.cpp` (`--test-fusion`), `TestBodyPose.cpp` (`--test-bodypose`), `TestReplay.cpp` (`--test-replay`), `TestDump.cpp` (`--test-dump`), `TestFrameRecorder.cpp` (`--test-framerecorder`), `ToolReplayVerify.cpp` (`--replay-verify`), `ToolReplayDump.cpp` (`--replay-dump`), `ToolReplayBodyPose.cpp` (`--replay-bodypose`), `ToolReplayExtrinsics.cpp` (`--replay-extrinsics`), `ToolCalibrateBones.cpp` (`--calibrate-bones`), and more than twenty others. All of them share `src/Tests/TestCommon.h`, one deliberately generous include block covering most of the app, so each file stays a function plus a registration line. The full flag list is in [commands.md](./commands.md).

`main.cpp` checks `--list-tests` first (prints every registered command grouped by category via `TestRegistry::printAvailable()`) and otherwise falls through to `TestRegistry::tryRun`, which owns the per-command logger setup/teardown described above and returns the process exit code.

Adding a test means adding one file under `src/Tests` plus a CMake re-run: the target's sources come from `file(GLOB TESTS_SRC src/Tests/*.h src/Tests/*.cpp)` in the top-level `CMakeLists.txt`, and a plain (non `CONFIGURE_DEPENDS`) glob does not pick up a new file until CMake reconfigures.

---

## The diagnostic dump (F9)

Trigger: `F9` anywhere in the app (`MainWindow::update`, `src/UI/MainWindow.cpp`), or the "Dump tracking state (F9)" button in the Tracking panel's Diagnostics section (`SettingsPanels.cpp`). Both call `visionThread->requestDiagnosticDump(config->makeDumpDirectoryPath())`.

Output directory: `AppConfig::makeDumpDirectoryPath()` (`src/App/AppConfig.cpp`) returns `<project folder>/dumps/<yyyy-mm-dd_hh-mm-ss>/` under the active project. `DiagnosticDump::write` (`src/App/DiagnosticDump.cpp`) fills that directory with `dump.json` plus per-camera PNGs:

- `config`: the live `AppConfig` snapshot, which may differ from the saved `project.json`.
- `cameras[]`: per camera, `trackingEnabled`, `activeEp`, `deviceFps`, `droppedFrames`, cumulative `seedStats` (cross-camera search-hint accounting), `hasExtrinsics`, a full landmark `snapshot` of that camera's last result, and `images` (`camN_raw.png` plus `camN_annotated.png`, the latter drawn by `annotateFrame` with landmarks, bones, ROI boxes, and per-hand labels overlaid).
- `fusedSnapshot`: a full landmark snapshot of the latest fused output, same schema as a per-camera snapshot.
- `history[]`: the rolling ring, oldest first. `DiagnosticDump::record` appends one compact entry every fusion iteration on the vision thread (`kMaxHistory= 240`, roughly 4-8s of history depending on the fusion rate). Each entry carries per-camera compact hand/body state, the fused `left`/`right` state, `dominantCamera`, `autoScaleFactor`, `fusion` (that iteration's `FusionDiagnostics`), and per-side `imu` state.
- `imuRaw`: raw accelerometer/gyro samples per side, deliberately uncorrected, so any candidate sensor axis mapping can be replayed offline and scored against the palm quaternions already in `history`.
- `imuLastCapture`: the mounting calibration currently in force (`forearmToSensor` plus the capture-quality numbers it was measured from), written once per dump rather than per frame since it only changes on a recalibration.

The ring is recorded every fusion iteration on the vision thread and is serviced even when cameras are stopped: `VisionThread::threadLoop`'s idle path (`!bAnyNewResult`) still checks the dump request flag, so `F9` writes something useful with no camera producing frames.

How to read a dump:

- `history` is oldest-first: the frames leading up to a failure sit near the end of the array.

- `fusion.clusters[].affinity` (keys `left`/`right`, each with `vote`/`temporal`/`spatial`/`total`) explains why a cluster was assigned the side it got. `assignedSide == -1` means the cluster was dropped (more clusters than hands) or the assignment was refused.

- The per-camera `snapshot`/annotated-PNG labels and the top-level `fusedSnapshot` are different failure domains. A camera can mislabel or lose a hand in its own preview while fusion still recovers the correct fused output from another camera, and the reverse can happen too. A wrong overlay does not by itself mean the fused output is wrong: check `fusedSnapshot` (or `history[].fused`) before concluding fusion broke.

- Per-pose fields worth reading first: `fkReprojectionPx` (how well the fitted parametric hand model reprojects onto the observed 2D landmarks, the angles-vs-reality fidelity metric, 0 when not evaluated, such as on a fused pose with no single owning camera), `stereoTriangulated`, and for a triangulated cluster the `fusion.clusters[].triResidualRmsPx` / `triResidualMaxPx` (cross-camera reprojection residual; a pairing above `triangulationMaxResidualPx` is vetoed as a wrong cross-camera match rather than a real hand).

---

## Record/replay (F10)

Trigger: `F10` anywhere, or the Timeline panel's "Start/Stop Recording (F10)" button (`src/UI/TimelinePanel.cpp`), calling `VisionThread::requestRecordingStart(config->makeRecordingFilePath())` / `requestRecordingStop()`.

Format: JSONL at `<project folder>/recordings/<yyyy-mm-dd_hh-mm-ss>.jsonl` (`AppConfig::makeRecordingFilePath`): one header line, one line per frame, one footer line (`src/App/TrackingRecording.h`). A missing or torn footer is tolerated as a crash-truncated recording; `TrackingReplay::load` stops cleanly at the first unparseable line.

What gets recorded (`RecordedFrame`): the per-hand fields `LandmarkTo3D::process` consumes (`imagePoints`, `modelPoints`, presence/handedness scores, `imageQuality`) for every camera that popped a fresh frame that iteration, post hand-model, pre-3D-projection and pre-fusion. It also carries the effective `refLengthMeters` (the auto-scale EMA recorded as a plain input rather than re-derived), the opt-in body-pose observation when that stage ran, and the fused output taken immediately after `fuse()` plus an FNV-1a-64 `checksum`. Both the snapshot and checksum are computed PRE-IMU and PRE-body-pose-solver (`TrackingRecording::computeFusedChecksum`), matching the point at which the live vision thread taps them.

Overflow behavior differs between the two recorders, deliberately:

- `TrackingRecorder` (`src/App/TrackingRecorder.*`), the JSONL landmark recorder: a full write queue (`k_maxQueuedFrames= 2048`, roughly 20s at 120 iterations/s) ABORTS the recording instead of dropping a frame. A silent gap would break replay determinism for every later frame, which is strictly worse than an honest truncated file.

- `FrameRecorder` (`src/App/FrameRecorder.*`), the opt-in raw-frame recorder: a full queue (`k_maxQueuedFrames= 64`) DROPS the frame and counts it. Frames are evidence for offline model comparison, not pipeline inputs, so losing one costs coverage of a moment, never replay correctness.

Starting a recording resets transient tracking state (`VisionThread::handleRecordingStartOnThread`): `HandFusion::resetTransientState`, `BodyPoseSolver::reset`, each camera's `LandmarkTo3D::resetTransientState`, and every camera's `lastResult`, so replay's freshly constructed instances start from the same zero state live did. `m_autoScaleFactor` is deliberately left untouched. A config refresh while recording finalizes the in-flight recording first, with reason `"config changed"`: a hard discontinuity the recording's single config-header snapshot cannot represent.

Opt-in raw frame recording: `AppConfig::recording.recordRawFrames`, off by default. The Timeline panel's checkbox carries an explicit privacy warning, because raw frames are video of the room, whereas a landmark recording is abstract enough to share. Frames land in `<recording>_frames/cam<N>_<frameIndex>.jpg` (`TrackingRecording::makeFrameDirectoryPath`), keyed by the same `frameIndex` the JSONL records per camera so a frame and its landmarks join up. This is what makes an offline model A/B possible: the checksummed replay re-runs every stage after inference, so it cannot judge whether a different pose model would have done better, because its input is gone. With frames on disk, `--replay-bodypose <recording.jsonl> [cameraIndex]` (`ToolReplayBodyPose.cpp`) re-runs the body-pose stage over them and reports per-joint step/score statistics.

Replay (`TrackingReplay`, `src/App/TrackingReplay.*`) loads the file, reconstructs the recorded `AppConfig` from the header, then re-runs `LandmarkTo3D` -> `HandFusion` bit-exactly on fresh instances per pass, matching the checksum the live tap recorded (PRE-IMU, PRE-body-solver). The recorded IMU forearm output is overlaid rather than re-run (the EKF is deliberately not replayed), and `BodyPoseSolver` then runs fresh over the replayed candidates. Re-running the solver, instead of replaying its recorded output, is what makes body-pose-solver changes A/B-able against old recordings, at the cost of the solver's output sitting outside the checksum contract.

THE RULES:

- `--replay-verify <recording.jsonl>` (`ToolReplayVerify.cpp`) exits 0 only when every frame's replayed checksum matches the recorded one. Always read `replay-verify.log` rather than trusting a shell pipeline's exit status alone: the log lists every divergent frame with both checksums in hex.

- A checksum divergence between an old recording and the current binary, after a change that touches `LandmarkTo3D` or `HandFusion`, is the diff of that change, not corruption. It is expected, and it is exactly what makes such a change visible.

- Checksums are a same-binary guarantee only. Nothing about a match or mismatch survives a rebuild that changes those stages' output; a divergent recording after an unrelated rebuild is not itself evidence of a bug.

- `--replay-dump <recording.jsonl> <firstFrame> <lastFrame> [outPath]` (`ToolReplayDump.cpp`) always replays from frame 0 for determinism, but only regenerates full `FusionDiagnostics` (the same schema the F9 dump's `history[].fusion` uses) inside the requested range, alongside the replayed fused poses, body-pose output, and per-frame checksum comparison. Default output path is `<recording>.frames_<first>_<last>.json`.

What-if replay: the recording's header line embeds the full config used to build the fusion settings (`RecordingHeader::appConfigJsonText`). Header-patching a copy of the recording, or using the Timeline panel's What-if controls (which seed a `TrackingReplay::WhatIfParams` from that same config), re-runs the identical recorded inputs under different fusion, hand-scale, or extrinsics parameters: an A/B of a config change against identical inputs with zero code changes. `--replay-extrinsics <recording.jsonl> <config.json>` (`ToolReplayExtrinsics.cpp`) specializes this to camera extrinsics: it replays once with the recording's own extrinsics and once with a candidate `config.json`'s, and compares triangulation success and fused confidence on the same recorded hands.

---

## The hitch watchdog

`VisionThread::reportHitchIfSlow` (`src/App/VisionThread.cpp`) runs at the end of every loop iteration, including the idle (no fresh camera frame) path. Threshold: `k_hitchThresholdMs= 50.0`. Every camera shares this one thread, so any phase that overruns the frame budget starves all of them at once: frames keep arriving from the driver, find no free block, and are dropped, which looks downstream like a synchronized multi-camera tracking gap or a USB fault rather than what it is.

Named phases (`VisionThread::eVisionPhase`): `ConfigRefresh` (config refresh / recording start-stop), `Capture` (frame pop, undistort, ML inference, 3D projection), `Imu` (wrist IMU sample drain, discovery, forearm publish), `Fusion` (cross-camera fusion, smoothing, auto-scale bookkeeping), `Osc`, `Diagnostics` (dump history record, recording enqueue, dump writes).

Over threshold, `MIKAN_MT_LOG_WARNING` logs the total time, the worst phase and its time, and the full per-phase breakdown; `m_hitchCount` increments and `m_lastHitchMs` / `m_lastHitchPhase` are published for the Tracking panel's watchdog readout. A stall now names its phase instead of leaving a generic "tracking froze" symptom to diagnose from scratch.

Rule this watchdog enforces: anything that can block for hundreds of milliseconds must not run on the vision thread. HID enumeration and the Bluetooth open handshake are the concrete case already built this way: `ImuService::update()` runs on every vision-thread iteration and must never block on them, so device discovery runs on a separate worker (`ImuService::discoveryLoop`) instead.

---

## Image-quality metrics (`HandRoiQuality`)

`HandRoiQuality::analyzeHand` (`src/Vision/HandRoiQuality.*`) runs on the vision thread against the exact frame the model consumed (after undistortion), for every tracked hand, every frame, at about 0.2 ms per camera.

Metrics fill `HandImageQuality` (`src/Vision/TrackingTypes.h`): `meanLuma`, `shadowClipRatio` / `highlightClipRatio`, `contrast` (gray-level standard deviation inside the ROI), `backgroundSeparation` (ROI mean versus the surrounding ring mean), `sharpness` (variance of the Laplacian after a 3x3 median filter), and `noise` (mean `|gray - 3x3 median|` residual). A per-camera `LumaFlickerTracker` separately tracks whole-frame flicker (`lumaInstability`, `lumaFlickerHz`).

All heavy statistics run on a nearest-neighbor decimated crop, capped at `kAnalysisMaxDim= 160`, which is what makes the values resolution-independent and comparable across cameras of different native resolution. The ROI is the landmark bounding box expanded by `kRoiExpand= 1.5`; the background ring is a further `kRingExpand= 1.4` band around that.

Sharpness and noise are split deliberately: raw Laplacian variance alone scores a noisy frame as sharp, because sensor noise reads as edges. Noise is measured separately (the median-filtered residual), and sharpness is measured on the median-filtered image so it is not rewarding noise. Sharpness only means something when it DROPS: a well-lit, low-jitter camera reads in the low thousands, and a higher value than that is just texture or scale, not "better."

These metrics show up in two places: the Tracking panel's per-camera table (`SettingsPanels::drawImageQualitySection`, worst tracked hand per camera, roughly a 1s EMA, each row's tooltip naming the camera-setting or lighting change to make), and the F9 dump / recording's per-hand `imageQuality` field, which carries the exact raw per-frame series behind that live readout.

---

## The hold-still jitter test

Lives in the Tracking panel's Diagnostics section (`SettingsPanels.cpp`, "Hold-Still Jitter Test" button), with its transient state in `TrackingPanelState::holdStill*`.

Sequence: a `k_holdStillCountdownSeconds= 3` countdown (time to get the hands back into position), then a `k_holdStillSampleSeconds= 3` sampling window that accumulates every NEW fused result (deduplicated on `fusedResult.timestampMs`, since fused results arrive slower than UI frames and re-adding a held UI frame would fake stability). For each tracked, world-posed hand it rebuilds all 21 world-space joints via `HandPoseModel::buildFingerJoints` (the same forward kinematics a client rebuilds) and accumulates a sum and sum-of-squares per joint.

Result: per side, the mean per-landmark standard deviation over the window, in millimeters, requiring at least `k_holdStillMinSamples= 30` fused samples to be meaningful. With the hands genuinely still, that deviation IS tracking noise, which makes it the one repeatable A/B number for a camera-settings or lighting change: run it once before, once after, compare.

The rule this test exists to satisfy: a free-form (moving) capture cannot A/B jitter. `HandFusion`'s own jitter estimate is a constant-velocity residual (`HandFusionConfig::jitterReferenceM`, surfaced per observation as `FusionDiagnostics::Observation::jitterMm`), and real fast motion produces a large residual by the same arithmetic that flags actual noise, so the two are indistinguishable from that signal alone. Only a genuinely still hand removes the ambiguity, which is what this test buys that a normal recording cannot.

Division of labor with the image-quality table: `HandRoiQuality` A/Bs what a single frame looks like (exposure, noise, sharpness). The hold-still test A/Bs what tracking does with a held-still hand over time (jitter). They diagnose different failure classes and are read together, not as substitutes for each other.

---

## UDP/OSC

OSC and VMC output go over UDP (`src/Osc/OscStreamer.*`, `src/Osc/UdpSocket.*`). The wire-level traps worth knowing before debugging a receiver, port exclusivity and receive-buffer sizing among them, are documented in [wire-protocol.md](./wire-protocol.md).
