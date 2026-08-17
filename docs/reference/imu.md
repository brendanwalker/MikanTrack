# IMU

Wrist-worn IMUs supply forearm orientation, fused with vision to extend hand tracking to the elbow. This doc covers the IMU subsystem end to end: the design model, the vendor-neutral device interface, the Joy-Con backend, the orientation filter, and how `ImuService` threads into the vision pipeline. See [calibration.md](./calibration.md) for the `MountingWizard` capture flow, [conventions.md](./conventions.md) for the world-space and frame conventions the filter and mounting math assume, [body-pose.md](./body-pose.md) for how the vision body-pose solver fills the elbow when no IMU is present, and [wire-protocol.md](./wire-protocol.md) for the OSC forearm output. [LEARNINGS.md](../../LEARNINGS.md) section 11 records the experiment history behind this design.

---

## Design model

A wrist strap sits proximal to the wrist joint, so the sensor rotates with the FOREARM, not the hand. `HandPose::forearmOrientationWorld` in `src/Vision/TrackingTypes.h` is kept as its own frame, separate from `palmOrientationWorld`, precisely because treating it as palm orientation would be wrong whenever the wrist is flexed. `HandPose::getWristRotation()` reads the wrist joint angle as `inverse(forearmOrientationWorld) * palmOrientationWorld`, and `HandPose::getElbowPositionWorld()` extrapolates the elbow straight back along the forearm's -X axis.

The system is orientation-only. `ImuOrientationFilter`'s state is a 3-parameter orientation error plus a 3-axis gyro bias (`ImuOrientationFilter.h`); there is no velocity or position state anywhere in `src/Imu`. Double-integrating raw accelerometer noise into position drifts unboundedly within seconds, and vision already supplies world-space palm position at millimeter accuracy (`HandPose::palmPositionWorld`), so the IMU's only job is the orientation vision cannot hold steady over time: forearm tilt and (loosely) yaw.

The forearm frame is defined so that it equals the palm frame when the wrist is held straight (`ImuService.h`, `ImuOrientationFilter.cpp` and `BodyPoseSolver.cpp` all state this independently: "equals the palm frame at a neutral wrist"). That makes the wrist joint rotation identity at neutral, which is the semantics the OSC schema promises, and it is what lets a single mounting rotation, `forearmToSensor` per side, absorb every physical difference between the two controllers: which way each Joy-Con's IMU chip is placed, how the strap sits, which wrist it is on. Nothing about sensor axes is hardcoded in `JoyconDevice` or `ImuService`; it all comes out of this one calibrated quaternion.

The mounting capture itself is not a single held pose. `MountingWizard` (`src/UI/MountingWizard.h/.cpp`) records two motions per side, a twist (pronation/supination) and a curl (elbow flexion), and `imuSolveMountingFromMotions` (`ImuService.cpp`) fits the forearm frame from their rotation-axis geometry alone: the twist's dominant rotation axis IS the forearm's long axis in the sensor's own frame, and the curl's hinge axis (after splitting into half-strokes and dropping the unsettled opening one) closes the frame's remaining roll. A centripetal fit on the curl (`imuFitCentripetalRadius`) resolves which end of the long axis points at the hand and measures the elbow-to-sensor length in the same pass. Vision contributes exactly one bit: `accumulatePoseMounting` maintains a running average of `inverse(sensorOrientation) * palmOrientation` across ordinary tracked frames, and that average is used only as a hint to pick which of the two candidate frames (180 degrees apart) is the palmar one. See calibration.md for the wizard's stage sequence and quality gates.

## The vendor-neutral seam

`ImuSample` (`src/Imu/ImuTypes.h`) is the seam between backends and the fusion layer: a timestamp (`steady_clock` milliseconds, the same base as camera frame timestamps), angular velocity in radians/second, and acceleration in meters/second^2, all in the sensor's own right-handed axes as mounted. A backend is not responsible for reorienting anything; the sensor-to-forearm rotation is the separate calibrated `forearmToSensor` quantity.

`IImuDevice` (`src/Imu/IImuDevice.h`) is one streaming tracker: identity (`getDevicePath`, `getFriendlyName`, `getSide`/`setSide`), open/close/streaming state, `fetchSamples` to drain buffered samples in chronological order, and diagnostics (`getMillisecondsSinceLastSample`, `getSampleRateHz`, `getBatteryLevel`).

`IImuDeviceManager` enumerates and owns the devices for one backend: `startup`, `shutdown`, `refreshConnectedDevices`, and indexed access via `getDeviceByIndex`, which returns a `shared_ptr<IImuDevice>`. The header states plainly that discovery blocks: HID/Bluetooth enumeration and the device-open handshake take hundreds of milliseconds on Windows, so refreshing must never run on a thread with a frame deadline. Any future backend (the header names hidraw/BlueZ and IOKit as examples, and a SlimeVR-style tracker as a future device kind) inherits that contract through the same interface. Joy-Con is the only backend implemented today.

## Joy-Con backend

`JoyconDevice` (`src/Imu/Joycon/JoyconDevice.h/.cpp`) talks to a Nintendo Switch Joy-Con or Pro Controller over Bluetooth HID on Windows. `JoyconDeviceManager` (`src/Imu/Joycon/JoyconDeviceManager.h/.cpp`) enumerates paired HID devices matching Nintendo's vendor ID (`k_vendorId= 0x057E`) and the Joy-Con/Pro Controller product IDs via `SetupDiGetClassDevs`/`HidD_GetAttributes`; pairing itself happens in Windows Bluetooth settings.

`JoyconDevice::open()` runs the handshake in order: subcommand `0x08` to leave shipment (low-power) mode, subcommand `0x40` with arg `0x01` to enable the IMU, then subcommand `0x03` with arg `0x30` to switch the input report mode to standard-full-with-IMU. Input report `0x30` carries three 6-axis samples per report, 5 ms apart (`k_imuSampleSpacingMs`), which at the report cadence works out to about 200 Hz. Raw int16 values convert to SI units with fixed scale factors: `k_accelScaleG= 0.000244f` (g per LSB, +/-8g range) times gravity, and `k_gyroScaleDps= 0.070f` (degrees/second per LSB, +/-2000 dps) times pi/180. Factory calibration readback from SPI flash is deliberately not implemented; gyro bias is instead estimated online by the fusion filter, and accelerometer scale error is second-order because only the gravity direction is used.

I/O runs on a dedicated read thread (`readThreadLoop`) using an overlapped `ReadFile` (the handle is opened with `FILE_FLAG_OVERLAPPED`) so a blocking HID read never stalls the caller. A neutral-rumble output report is sent as a keepalive roughly every second, because a Joy-Con that hears nothing back from the host eventually sleeps even while it keeps streaming input reports. Decoded samples are timestamped from `steady_clock` (`nowSteadyMs`), back-dated within a report so the newest of the three samples lands at the arrival time, and queued through a `moodycamel::ReaderWriterQueue<ImuSample>` for the fusion thread to drain with `fetchSamples`.

`JoyconDevice.h` documents that Nintendo's own notes, and phase-1 capture, show the two Joy-Cons' IMU chips mounted with an axis reversed relative to each other (L reads +Z up at rest, R reads -Z up). That difference does not need special-case handling in the backend as long as it is a rotation: the per-side mounting calibration absorbs any fixed rotation between sensor and forearm. It would need handling only if a controller's reported frame were left-handed (a reflection, determinant -1), which no quaternion can represent; the `--test-imuaxes` tool exists to measure which case applies per controller by checking the gyro against the accelerometer.

## Orientation filter

`ImuOrientationFilter` (`src/Imu/ImuOrientationFilter.h/.cpp`) is an error-state Kalman filter (ESKF) estimating the sensor's orientation in the Z-up world frame, plus its gyro bias. The nominal quaternion (`m_orientation`, sensor-to-world) is kept outside the covariance; the filter runs its 6x6 covariance over a minimal error state instead: 3 components of body-frame orientation error plus the 3-axis gyro bias. This avoids the singular covariance and renormalization hacks a direct EKF over quaternion components would need. The small matrices (`Mat3`, `Mat63`, `Mat6`, all `std::array<float, N>`, row-major) are hand-rolled; there is no Eigen dependency.

`applyCorrection` is the shared correction step for every measurement type. It computes the Kalman gain, injects the resulting error into the nominal quaternion and bias, and updates the covariance in Joseph form (`P = (I-KH) P (I-KH)^T + K R K^T`), which stays symmetric positive definite under floating-point error, unlike the textbook `P = (I-KH) P` form.

Two measurement paths feed it:

- `updateWithGravity` corrects tilt (roll/pitch) from the accelerometer. It is gated: the update is applied only when the measured specific-force magnitude is within `accelGate` of standard gravity (`k_gravityMetersPerSecond2`), because under real acceleration the reading is gravity plus linear acceleration and would otherwise drag the tilt estimate around. `m_gravityAcceptEma` tracks the acceptance rate so a filter running open-loop (never accepting) or one mistaking motion for gravity (accepting too much during hard motion) is diagnosable.

- `updateWithOrientation` corrects all 3 axes from an absolute reference orientation, but this full-orientation path is not the one `ImuService` calls in production. Instead `updateWithYawReference` applies an anisotropic measurement noise built from `bodyUp` (world-up expressed in the body frame): tight along that axis, effectively infinite (`kIgnoredSigma= 100` rad) perpendicular to it, so the correction collapses onto yaw alone and leaves tilt to gravity.

The reason for that split is observability. With gravity alone, the gyro bias component about the two tilt axes converges, but the bias component about the gravity axis is not observable at all: nothing in a 6-axis IMU (no magnetometer) can see rotation about world-up, so yaw is unobservable from inertial data and would otherwise drift forever. `predict()` injects extra process noise along the current body-frame world-up direction every step specifically so the yaw covariance does not collapse and lock out later corrections. The system is therefore vision-anchored: `ImuService::applyVisionPalmOrientation` converts the vision-measured palm orientation into the equivalent sensor orientation via `inverse(forearmToSensor)` and calls `updateWithYawReference` with it. Only yaw is trusted from that reference, not the full orientation, because a real wrist joint sits between the sensor (which rides the forearm) and the vision measurement (which is the palm); a full-orientation update would fight genuine wrist motion instead of correcting drift.

`m_visionYawCorrectionEma` tracks the rolling magnitude of the yaw correction actually applied, surfaced through `ImuSideStatus::visionYawCorrectionDegrees`: sustained large values mean vision and the gyro disagree systematically, a different fault from either source drifting alone. `isTiltConverged()` and `isBiasSaturated()` are similarly surfaced diagnostics rather than internal-only state, because a frozen or diverging filter otherwise looks like a stable, confident orientation from the outside.

## Service threading and integration

`ImuService` (`src/Imu/ImuService.h/.cpp`) owns the set of IMU devices, one `ImuOrientationFilter` per device, and the per-side mounting calibration that turns a sensor orientation into a forearm orientation.

Device discovery runs on its own worker thread (`discoveryLoop`), started by `startup()` and driven by `requestDiscovery()`/`requestReopen()`. HID enumeration and the Bluetooth open handshake block for hundreds of milliseconds, and `ImuService::update()` is called from the vision thread's frame loop, where that would stall every camera; only the discovery request itself happens inline, and results are adopted on the next `update()` via `adoptDiscoveryResults()`. Devices are held by `shared_ptr<IImuDevice>` end to end so that discovery dropping a device it can no longer see never frees an object the vision thread is still draining mid-frame; the last reference closes it.

`ImuService::update()` drains every device's buffered samples once per pipeline tick. Because each `ImuSample` carries its own timestamp, integrating a whole backlog at once through `ImuOrientationFilter::processSample` is exact: a caller running slower than the sensor's 200 Hz loses only output freshness, not information. `VisionThread::processFrame` (`src/App/VisionThread.cpp`) is that caller: after fusion produces a frame's `HandPose`s, it calls `applyVisionPalmOrientation` for each tracked, world-posed side (the vision yaw anchor), then `getForearmOrientation` to read back `q_sensorToWorld * forearmToSensor` and fill `outputResult.poses[sideIndex].hasForearmPose` / `forearmOrientationWorld` / `forearmConfidence`. The EKF's output is used as-is, with no additional smoothing layered on top, which would cascade two filters onto one signal.

The vision body-pose solver runs after this fill and defers to it: `BodyPoseSolver.cpp` skips computing a forearm for any side where `pose.hasForearmPose` is already true, with the comment "IMU wins: a measured forearm direction beats a monocular one." A side with no IMU (or an unconverged/uncalibrated one) still gets an elbow from the body-pose camera when one is enabled; see body-pose.md.

## Config and UI surface

`ImuConfig` (`src/App/AppConfig.h`) is the persisted config block, held on `AppConfig::imu`:

- `bool enabled` (default true)
- `float visionYawSigma` (default `0.35f`), the yaw-anchor measurement noise radians passed to `updateWithYawReference`
- `bool swapSides` (default false), swaps which physical controller drives which wrist
- `bool mountingPresent[2]`, indexed by `eHandSide`, whether a mounting has been captured for that side
- `std::array<glm::quat, 2> forearmToSensor`, the captured mounting rotation per side

The Tracking panel's Wrist IMU section (`SettingsPanels::drawTrackingPanel`, `src/UI/SettingsPanels.cpp`) exposes the enable checkbox, the swap-wrists checkbox, and the entry point into `MountingWizard`; its tooltip states the forearm frame is streamed on `/mikan/hand/{side}/forearm` (see wire-protocol.md for the OSC schema).

Test flags registered in `src/Tests`:

- `--test-imufilter`: "IMU orientation EKF, observability, gating" (SelfTest category)
- `--test-imudiscovery`: "IMU device discovery never blocks the caller" (SelfTest category)
- `--test-joycon`: "Joy-Con sample decode + live HID streaming" (Hardware category)
- `--test-imuaxes`: "Live Joy-Con axis convention measurement" (Hardware category)
- `--test-imupair`: "Solves the transform between two rigidly coupled IMUs from a dump" (Tool category)
