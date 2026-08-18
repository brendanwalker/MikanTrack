# Architecture

Architectural map of the MikanTrack source tree: what each `src/` subtree does, the frame anatomy of one vision-thread iteration, the threading model, and the config system. MikanTrack is a single Windows executable (target `MikanTrack` in `CMakeLists.txt`) that captures webcams, runs MediaPipe-derived ONNX models through DirectML, fuses multi-camera results, and streams poses over OSC. It does not use the MediaPipe framework itself, and it does not talk to MikanXR at runtime; some sources were copied from that repo, inventoried in `NOTICE.md` at the repo root. Build invocations are covered in [build.md](./build.md), commands in [commands.md](./commands.md), coordinate conventions in [conventions.md](./conventions.md), the tracking pipelines in [hand-tracking.md](./hand-tracking.md) and [body-pose.md](./body-pose.md), calibration in [calibration.md](./calibration.md), the wrist IMU in [imu.md](./imu.md), the OSC output in [wire-protocol.md](./wire-protocol.md), and the diagnostic tooling in [debugging.md](./debugging.md). Experiment history lives at the repo root in [LEARNINGS.md](../../LEARNINGS.md).

---

## Dependency shape

Everything compiles into one executable. There is no enforced layering: every `src/` directory is on the include path (`target_include_directories` in `CMakeLists.txt`), and includes are bare filenames. The layering below is convention, derived from what each subtree actually includes.

Tracking data flows `Video` -> `Vision` -> `Tracking` -> `Osc`. `App` orchestrates that flow (the vision thread lives there), `UI` and `Render` read published snapshots on the main thread, and `Math` and `Utility` sit at the bottom with no in-repo dependencies above them.

## Module map

- `src/App`: process orchestration and the record/replay/diagnostics machinery. `App` (`App.h`) owns startup/shutdown, the SDL window and GL context, the ~90 Hz main loop (`FrameTimer(11)` in `App::exec`), and the MainMenu/Project state machine. `VisionThread` is the hub of the app: the inference thread that drains cameras, runs the ML pipelines, fuses, and streams (see Frame anatomy below). `AppConfig` is the per-project settings model, `GlobalSettings` the slim app-global one, and `ProjectManager` the project lifecycle (see Config below). `TrackingRecording`/`TrackingRecorder`/`TrackingReplay`/`FrameRecorder`/`DiagnosticDump`/`TrackingJson` are the recording and diagnostics inventory (see below).

- `src/Video`: webcam capture. `VideoCaptureSystem` is the app-facing facade; `VideoFrame.h` defines `VideoFrameBlock`, the heap-owned raw frame copy. `src/Video/Interfaces/` holds the device interfaces (`IUsbVideoDevice`, `IUsbVideoDeviceManager`) and `src/Video/WMF/` the Windows Media Foundation backend (`MikanWMFVideoDevice`, `MikanWMFVideoDeviceManager`, `DeviceHotplugNotifier`), copied from MikanXR's MikanWMFVideo plugin with its vendor-MFT blacklists and hang workarounds intact (see `NOTICE.md`).

- `src/Vision`: ONNX inference, 2D outputs only. `OnnxSession` wraps ONNX Runtime with DirectML or CPU execution providers. `HandTrackingPipeline` runs `PalmDetector` (with `SsdAnchors`) and `HandLandmarkModel` to produce per-camera 2D hand landmarks; `BodyPoseTracker` runs `PoseDetector` and `RtmPoseBodyModel` for the opt-in per-camera body observation. `HandRoiQuality` scores lighting/exposure on the exact ROI the model consumed. `TrackingTypes.h` and `BodyPoseTypes.h` define the result structs shared downstream.

- `src/Tracking`: 3D lifting, fusion, and state estimation. `LandmarkTo3D` back-projects 2D landmarks into camera space; `SpaceTransforms` (`applyWorldTransform`) moves camera-space results into the marker-anchored world frame. `HandFusion` clusters and fuses all cameras' results and owns `HandStateEstimator`, the 26-DoF angle-space multi-view fit. `BodyPoseSolver` solves elbows, shoulders, and head from the per-camera body observations. `HandPoseModel`, `HandBoneCalibrator`, and `OneEuroFilter` support skeleton parameterization, bone measurement, and smoothing.

- `src/Calibration`: OpenCV calibration math. `MonoLensDistortionCalibrator` (intrinsics), `CalibrationPatternFinder` and its `_Charuco`/`_Aruco` variants, `PatternPoseSampler` and `ExtrinsicsValidation` (extrinsics), `AnglePriorCalibrator` and `BodyDimensionCalibrator`, plus `CameraMath`, `MathOpenCV`, and `CVVideoFrameProcessor` (undistortion), several copied from MikanXR's editor calibration tree. See [calibration.md](./calibration.md).

- `src/Imu`: wrist inertial trackers. `ImuService` owns the devices, one `ImuOrientationFilter` per device, and the mounting calibration that turns a sensor orientation into a forearm orientation. `src/Imu/Joycon/` is the HID backend (`JoyconDevice`, `JoyconDeviceManager`). See [imu.md](./imu.md).

- `src/Osc`: network output. `OscStreamer` encodes the fused frame in the Mikan or VMC schema (`eOscOutputMode`, `VmcRetarget`) via `OscWriter` over `UdpSocket`. See [wire-protocol.md](./wire-protocol.md).

- `src/Render`: minimal GL helpers for the 3D scene view: `GlFrameBuffer`, `GlTexture`, `GlLineRenderer`, `DebugDraw`, `OrbitCamera`, `Colors.h`. No scene graph.

- `src/UI`: Dear ImGui panels and wizards, main thread only (see UI below).

- `src/Math`: GLM helpers (`MathGLM`, `MathUtility`, `Transform.h`), `MathTypeConversion`, and plain-struct retypes of MikanXR client types (`MikanMathTypes.h`, `MikanVideoSourceTypes.h`, `MikanCameraTypes.h`).

- `src/Utility`: `Logger`, `PathUtils`, `StringUtils`, `ThreadUtils`, `WorkerThread`, `MemoryUtils.h`. `src/Utility/easy/profiler.h` is a no-op stub of the easy_profiler macros (`EASY_FUNCTION`, `EASY_BLOCK`, ...) so files copied from MikanXR compile unmodified; there is no easy_profiler in this repo.

- `src/Tests`: command-line self-tests and headless diagnostic tools, one per file, compiled into the main executable. Each registers itself with `TestRegistry`; `src/main.cpp` routes recognized CLI flags to them (`--list-tests` prints the catalog) and otherwise launches the app. See [debugging.md](./debugging.md).

---

## Frame anatomy

Where a frame goes, one `VisionThread::threadLoop` iteration (`src/App/VisionThread.cpp`):

1. Upstream, each camera's Media Foundation callback thread copies the raw frame into a `VideoFrameBlock` from that slot's freelist and pushes it onto the slot's SPSC `frameQueue` (`VideoCaptureSystem::CameraSlot::notifyVideoFrameReceived`, `moodycamel::ReaderWriterQueue`). No free block means the frame is dropped and counted.
2. The iteration starts by consuming pending requests: a config refresh (which first finalizes any in-progress recording, then runs `refreshConfigOnThread`), a recording stop, then a recording start, in that order so the recording header snapshot reflects the refresh.
3. Each camera is processed sequentially through `processCameraFrame`: `tryPopFrame` (latest-wins, stale blocks recycled), `VideoCaptureSystem::convertFrameToBGR`, optional undistortion through `CVVideoFrameProcessor` when intrinsics are calibrated, and the luminance-flicker diagnostic.
4. With more than one camera, `seedSearchHints` projects the previous iteration's fused world poses into this camera's image as pipeline search hints. Seeding happens here, immediately before inference, because `setSearchHints` replaces the hint queue and hints are only consumed inside `process()`.
5. `HandTrackingPipeline::process` produces 2D hand landmarks. The opt-in `BodyPoseTracker` then runs on the same undistorted frame. `LandmarkTo3D::process` lifts landmarks into camera space, and `applyWorldTransform` (`SpaceTransforms`) moves them into world space when extrinsics are present. The camera's result lands in its `CameraContext::lastResult` and its preview is published latest-wins.
6. If no camera produced a frame, the loop services any pending diagnostic dump, sleeps 2 ms, and continues.
7. Every camera's `lastResult` becomes a fusion candidate. When any candidate has a valid world pose, `HandFusion::fuse` runs (internally driving `HandStateEstimator`); otherwise camera 0's camera-space result passes through unchanged.
8. Recording tap: the fused output is snapshotted and checksummed immediately after `fuse`, before the IMU forearm fill and before the body solver. Replay checksums at this same point, which is the invariant that makes recorded and replayed checksums comparable.
9. `ImuService` drains every buffered inertial sample, takes the fused palm orientation as a yaw anchor, and publishes the forearm orientation onto the poses. The IMU EKF output is recorded separately; it is never replayed.
10. `BodyPoseSolver::solve` runs after the IMU fill (IMU wins for sides it claims), producing elbows, shoulders, and head.
11. Fusion bookkeeping: seed state for the next iteration, dominant-camera and per-camera observation-confidence publishes, and the stereo auto hand-scale EMA.
12. `OscStreamer::sendFrame` streams the frame, then the fused result is published latest-wins for the main thread.
13. The tail of the loop services calibration captures (bone calibration window, rest pose), records the diagnostics history ring, assembles the recording frame and hands it to the writer thread, and services dump requests.

The loop watches itself: `eVisionPhase` names the phases (`ConfigRefresh`, `Capture`, `Imu`, `Fusion`, `Osc`, `Diagnostics`), and any iteration exceeding `k_hitchThresholdMs` (50 ms, `VisionThread.cpp`) logs a per-phase millisecond breakdown attributing the hitch to the worst phase (`reportHitchIfSlow`). A hitch starves every camera at once, so the symptom downstream is a synchronized multi-camera tracking gap that would otherwise look like a USB fault.

---

## Threading model

One vision thread processes all cameras sequentially. This is deliberate: DirectML serializes on a single GPU queue anyway, and the per-camera `CameraContext` boundary keeps a later thread-per-camera upgrade possible. ONNX Runtime sessions are created and destroyed on the vision thread only.

The main thread runs the SDL/ImGui UI (`App::tick` at a ~90 Hz cap) and pumps `VideoCaptureSystem::update`, which owns the hotplug HWND. Handoffs between the two are mutex-guarded latest-wins snapshots plus atomics:

- `fetchPreviewFrame` and `fetchFusedResult` copy the newest frame/result and clear a freshness flag
- status reads (`getDominantCamera`, `getObservationConfidence`, hitch counters, recording counters) are plain atomics
- wizard interactions are atomic request flags serviced on the vision thread with mutex-guarded result structs fetched later (`fetchRestPoseCapture`, `fetchBoneCalibration`, `fetchImuMountingCapture`)

The remaining threads: one Media Foundation callback thread per streaming camera, one HID read thread per Joy-Con (`JoyconDevice`, SPSC sample queue into the vision thread), the `ImuService` discovery worker (HID enumeration and the Bluetooth open handshake block for hundreds of milliseconds, so they never run on the frame loop), and the `TrackingRecorder` and `FrameRecorder` writer threads.

Changing the camera count requires a vision-thread restart through `App::applyCameraCountChange` (stop the thread, resize the capture slots, start it again): the per-camera `CameraContext` vector is sized in `VisionThread::start` and never resized while the thread runs, which is what lets main-thread accessors index it without locking. Per-camera setting changes only need `requestConfigRefresh()`. A config refresh finalizes an in-progress recording, because the recording header's config snapshot must describe every frame in the file.

---

## Config

Settings live in two files:

- `AppConfig` (`src/App/AppConfig.h`/`.cpp`): the per-project settings model, stored as `project.json` in the project folder. Edits call `markDirty()`; `updateAutoSave` on the main loop saves at most once per cooldown (`k_autoSaveCooldownSeconds`, 3 s). `toJsonString`/`loadFromJsonString` serialize the same schema in memory, which is how recording headers snapshot the live config and how replay reconstructs it.

- `GlobalSettings` (`src/App/GlobalSettings.h`/`.cpp`): app-global state at `%APPDATA%/MikanTrack/config.json`, currently just the last-loaded project path. That file previously held the whole `AppConfig`; on first run `ProjectManager::migrateLegacyConfigIfNeeded` pivots a legacy file into a `Default` project, copying the config, moving `recordings/` and `dumps/` along, and leaving a `config.json.bak` beside the slim replacement.

A project is a folder under `%USERPROFILE%/Documents/MikanTrack/<name>/` holding `project.json` plus that project's `recordings/` and `dumps/`. `ProjectManager` (`src/App/ProjectManager.h`/`.cpp`) creates and loads projects. Loading applies the file into the existing `AppConfig` object IN PLACE (after resetting it to defaults), because the UI panels, wizards, and the vision thread all hold raw pointers to that object. The vision thread is always stopped before the config mutates (`App::activateProject` / `App::returnToMainMenu` are the only mutation points), since it reads the config without synchronization. Headless tools that want the active project's config go through `ProjectManager::loadActiveProjectConfig`.

The free functions `makeHandFusionConfig` and `makeBodyDimensions` (declared in `AppConfig.h`) are the single mappings from an `AppConfig` to the fusion config and body dimensions. They exist so the live vision thread, the replay engine, and the replay self-test cannot assemble different configs from the same source.

---

## Video capture

`VideoCaptureSystem` (`src/Video/VideoCaptureSystem.h`) is the facade over Media Foundation. It owns MF startup/shutdown, the WMF device manager with hotplug notification, and one `CameraSlot` per configured camera. Each slot is its own `IUsbVideoDeviceListener` because the WMF frame callback does not identify its source device: the listener object's identity is the camera identity. Each slot owns `k_frameBlockCount` (6) `VideoFrameBlock`s recycled through an SPSC freelist, paired with the SPSC frame queue going the other way, so the MF callback never allocates. Device management calls belong to the main thread; `tryPopFrame`/`releaseFrame` belong to the single inference thread.

The `src/Video/WMF/` backend was copied from MikanXR's MikanWMFVideo plugin (`NOTICE.md`) with the module scaffolding removed. Its vendor-MFT blacklists and hang workarounds stay intact. Frame-pipeline details are in [debugging.md](./debugging.md) and the source headers.

---

## Record, replay, and diagnostics

Deep detail belongs to [debugging.md](./debugging.md); this is the inventory, all in `src/App/`.

- `TrackingRecording`: the recording format: JSONL with one header line, one line per fused frame, one footer. Each frame carries every input to the post-inference stages plus the fused output and an FNV-1a 64 checksum over the fused output's raw float bytes, computed pre-IMU by both the live tap and replay.

- `TrackingRecorder`: asynchronous JSONL writer. The vision thread moves a POD frame into a bounded queue; the writer thread serializes. Queue overflow ABORTS the recording rather than dropping frames, because a silent gap would break replay determinism for every subsequent frame.

- `TrackingReplay`: offline deterministic re-run of a recording: reconstructs the recorded `AppConfig`, then re-runs `LandmarkTo3D` -> `HandFusion` on fresh instances, verifying each frame against the recorded checksum bit-exactly. `BodyPoseSolver` re-runs after the checksum point, deliberately outside the contract so solver changes can be A/B'd against old recordings. The IMU EKF is not re-run; the recorded forearm output is overlaid. Supports what-if passes with altered fusion parameters or extrinsics.

- `FrameRecorder`: opt-in raw JPEG capture of the exact frames the models consumed, alongside a tracking recording. Off by default because frames are video of a room. Unlike `TrackingRecorder`, overflow DROPS frames: frames are not inputs to the replayed pipeline, so a gap costs coverage rather than correctness.

- `DiagnosticDump`: F9 writes a rolling per-frame history of compact tracking state, current camera frames (raw and annotated), and the live config to a timestamped folder. Built for after-the-fact diagnosis of transient failures, where the frames leading up to the failure are the diagnosis.

- `TrackingJson`: the shared JSON (de)serialization for tracking types, used by the dump, the recorder/replay, and the headless replay tools, so all of them speak one schema.

---

## UI and Render

`App` runs a two-state machine (`App::eAppState`). The app boots into `MainMenu`, where `MainWindow` draws only `MainMenuScreen` (Resume / New Project / Load Project / Exit) and no cameras or vision thread run. Entering a project (`App::activateProject`) starts the vision thread, opens the configured cameras, and switches to `Project`, the full tracking UI. `File > Close Project` returns to the menu; `File > Load Project...` switches projects in place. Both are deferred to the top of the next frame, never applied mid-frame, because panels size their per-camera state from the config at the start of each update. Load Project uses the native open-file dialog (`thirdparty/tinyfiledialogs`), filtered to `project.json` and seeded to the projects root.

A freshly created project starts `SetupFlow` (`src/UI/SetupFlow.h`), the guided setup chain: modal prompts (tracking-setup choice, camera selection, board and marker printing, output protocol) interleaved with the calibration wizards in dependency order. The variant chosen in the first prompt decides the chain: the Joy-Con variant adds the mounting wizard, the tri-camera variant enables body pose on the front camera and adds the body wizard, and the plain dual variant has neither. The camera-selection prompt opens each device as it is picked and shows a live preview inline, since a device name alone rarely says which physical camera is which. Each wizard reports `eWizardResult` (`src/UI/WizardResult.h`, Completed at its accept path, Cancelled otherwise) so the flow can branch; manual wizard launches are suppressed while the flow is active, and the Video Preview tab is focused whenever a camera calibration wizard starts. The flow only ever runs on a just-created project, so cancelling anywhere (always through a confirmation) deletes that project from disk (`App::discardNewProjectAndReturnToMenu`, `ProjectManager::deleteProject`) and points Resume back at the project that preceded it. The menu's Resume option is existence-checked, so a project deleted on disk is not offered.

`MainWindow` (`src/UI/MainWindow.h`) owns the ImGui dockspace, menu bar, panels, and wizards, and fetches the per-camera previews and fused result each update. The panels are `VideoPreviewPanel`, `Scene3dPanel`, `DevicePanel`, `CalibrationPanel`, and `TimelinePanel` (owned instances), plus the `LogPanel` singleton and the `SettingsPanels` namespace of free functions whose frame-to-frame state (`TrackingPanelState`) lives in `MainWindow`. The wizards are `IntrinsicsWizard`, `ExtrinsicsWizard`, `MountingWizard`, `BodyCalibrationWizard`, and `HandCalibrationWizard`.

`Scene3dPanel` renders the marker-plane grid, per-camera frustums, and the fused hand/arm skeletons (with optional dimmed per-camera skeletons) into an FBO using `src/Render/`. The skeleton it draws is the same FK reconstruction a client rebuilds from the streamed parameters.

Global hotkeys, handled in `MainWindow.cpp`: F9 requests a diagnostic dump, F10 toggles the tracking recording.
