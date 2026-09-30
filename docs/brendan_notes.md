TLAs
 * EMA - Exponentially Moving Average
 * ONNX (Open Neural Network Exchange) is an open-source intermediate representation format designed to make machine learning models portable across different frameworks and hardware. Originally developed by Microsoft and Facebook (Meta), it standardizes deep learning computational graphs into a unified file structure (.onnx), decoupling model training from deployment.

Questions
What is an "ORT sessions" (in camera thread)?
An ORT session is an ONNX Runtime Ort::Session, a loaded model plus its execution-provider state. The DirectML provider allocates D3D12 resources tied to the creating context, which is why sessions are created and destroyed only on the vision thread.

Presentation Order
* App Flow
  - New Project Wizard
    - Camera Intrinsics
    - CharUCo Marker
    - Camera Extrinsics  
    - Aruco Marker
      - Why two markers?
    - Hand Calibration
    - Body Calibration
  - Recording and the imporatance of determinism

* AI Tool generation
 - Importance of disclosure 
 - Power and Water Usage
 - Open Sourcing
 - Post Hoc learning and documentation
 - AI iterative development process
 - How to expose application state to AI tools and diagnose problems

* Tracking pipeline
 - VisionThread
   - Read USB Camera frame
   - Convert to BGR 
   - Undistort
   - Update luma flicker analytics
   - Run ML pipeline
      - Compute hand search seed from prior hand3d location projected from 2d (VisionThread::seedSearchHints) 
   - 3D projection
   - Fuse results 

* Previous Attempts
 - Softmax dual overhead camera
 - Stereo cameras
 - JoyCon for elbows
 - Triple Camera
 - Hand model

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