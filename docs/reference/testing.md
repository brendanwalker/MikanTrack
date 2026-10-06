# Testing

The whole automated test surface is command-line flags on `MikanTrack.exe`: there is no separate test binary and no external test framework. This doc is the map of what exists, how to run all of it in one pass, what each command actually asserts, and where the coverage stops. The flag cheat sheet is in [commands.md](./commands.md), the registry mechanics and the record/replay instrument in [debugging.md](./debugging.md), and the build and working-directory rules in [build.md](./build.md).

---

## Running the tests

Build first. There is one target, and the tests live inside it:

```bash
cmake --build build --target MikanTrack --config Release --parallel
```

Then run from the exe's own directory. Model paths are relative and fail silently from anywhere else:

```bash
cd build/Release && ./MikanTrack.exe --test-fusion
```

Every command follows the same contract:

- exit code 0 means pass, nonzero means fail

- it writes `<flag without the dashes>.log` next to the exe (`--test-fusion` writes `test-fusion.log`), overwritten on each run, in addition to console output

- the log lists every individual check with its measured numbers, so read the log rather than the exit code alone. A shell pipeline can mask an exit code, and the measured values are the part that tells you how close to a threshold a passing check sits.

`./MikanTrack.exe --list-tests` prints the live list grouped by category. It is generated from the registry, so it is never stale, which the tables below can be.

### Running every self-test in one pass

The 24 self-tests are deterministic and need no hardware or input files, so they run as a batch. This drives the list from `--list-tests` rather than a hardcoded list, so a newly added test is picked up automatically:

```bash
cd build/Release && ./MikanTrack.exe --list-tests | awk '/^self-tests:/{f=1;next} /^[a-z]/{f=0} f && $1 ~ /^--/{print $1}' | while read -r flag; do ./MikanTrack.exe "$flag" >/dev/null 2>&1 || echo "FAIL $flag"; done; echo done
```

The full batch takes about 15 seconds and prints nothing but `done` when everything passes. For a failing flag, read its `.log` in the same directory.

Side effects of a batch run: each test overwrites its own `.log` next to the exe, and three of them write scratch files under `%TEMP%` (`mikan-test-replay.jsonl`, `mikan_framerecorder_test/`, `mikanmediapipe_test_dump/`). Nothing touches a project folder, a saved config, or any recording.

---

## Self-tests: what each one verifies

### Hand pipeline

- `--test-handpose`: `HandPoseModel`. Builds a synthetic right hand by forward kinematics over a hand-authored skeleton, then extracts the palm frame and finger angles back out of the resulting 21 landmarks and checks the round trip closes. Covers moderate curl, a deep fist, thumb opposition, a wrong handedness label, skeleton length recovery, the angle sign conventions (zero-angle bone along the neutral direction, positive proximal, positive lateral, zero-inter collinearity), and FK reprojection error.

- `--test-handestimator`: `HandStateEstimator`, the angle-space multi-view state estimator, in 13 lettered cases. Synthetic cameras project a known FK hand and the estimator has to recover and track it from 2D landmarks alone: clean two-camera recovery from a perturbed seed, tracking a moving hand under pixel noise, camera dropout without a step, a mono stretch where lateral motion tracks and depth stays leashed, bit-identical determinism across two runs, the divergence guard dropping garbage correspondence, an unchanged observation not moving the state, mono depth-bias drift being resisted, the joint-limit prior, DIP-PIP coupling, the fitted angle prior discriminating a snap from a pose change, the innovation gate holding teleports while tracking fast real motion, and rows weighted by measured confidence rather than presence.

- `--test-fusion`: `HandFusion` end to end, in cases (a) through (t). Two-camera fusion beating both noisy inputs, dominant-camera choice, staleness passthrough, handedness-mislabel recovery, stereo hand-scale correction, ray-aware clustering, the spatial side prior, joint cluster pairing (a regression from a captured clap dump), the decisive-disagreement veto, slot side-collision ordering, stability weighting, stereo landmark triangulation with its residual gate and veto, solo-cluster rescue, triangulation pair choice by geometry, camera loss with mono depth error, and side-assignment refusal with post-window reacquisition.

- `--test-pnp`: monocular PnP in `LandmarkTo3D`. One synthetic all-21 pose recovered to sub-millimetre, plus a real frame captured live where a flat waving hand made the object model near-planar and the old ITERATIVE cold start collapsed the depth. That second case is a pinned regression for the SQPnP cold path.

- `--test-seeding`: the positional half of the cross-camera seed decision, `HandTrackingPipeline::isSeedRedundant`. Hints inside a tracked box, hints inside the projection-error margin, hints beyond it, inflation applied about the box centre rather than the origin, and two tracked boxes each able to absorb a hint. The rest of the seeding path needs the ONNX models, so only this part is isolated.

- `--test-bonecalib`: `HandBoneCalibrator`. Noise-free recovery of known phalanx and base lengths from triangulated landmark samples, recovery under 4 mm landmark noise with the length bias and spread reported, the object model PnP builds from a calibrated skeleton, and the minimum-sample gate.

- `--test-angleprior`: `AnglePriorCalibrator`. Samples a known generative model with two latent synergy factors and checks the calibrator recovers the mean and the correlation structure, since the structure is what makes a coordinated pose change cheap and an uncorrelated single-DoF snap expensive under the Mahalanobis metric. Also covers the minimum-sample gate and the variance floor that keeps an unexercised DoF soft.

- `--test-roiquality`: `HandRoiQuality` and the per-camera `LumaFlickerTracker`. Uniform and clipped ROIs, noise separated from sharpness, resolution independence across a scaled frame, a degenerate ROI rejected, and flicker instability measured on a rippling frame against a steady one.

### Body pose

- `--test-bodypose`: `BodyPoseSolver`, the largest self-test at roughly 60 checks. Everything the solver produces comes from 2D rays plus known body lengths, so each case is constructed exactly and checked to sub-millimetre: known-separation depth, shoulders from their own rays plus the calibrated width, the elbow as a ray against the bone circle, the occluded elbow carried by continuity, the negative control proving that holding the elbow position instead would break the bone length, a fast swing returning to the right root, a contradicting ray not overruling continuity, the shoulder ray breaking the cold-start tie, grazing-ray clamping and continuity across tangency, held observations still publishing, IMU forearm precedence left untouched, the head from two ear rays plus head width including yaw recovery and the implausible-range gate, the visibility gates, and the landmark-mask round trip.

- `--test-bodycalib`: `BodyDimensionCalibrator`. Projects a synthetic body of known landmark separations, deliberately away from the anatomical defaults, and checks the calibrator measures those separations back: shoulder width, head width, nose-forward offset, upper arm derived from shoulder width, the raised-hand acceptance gate rejecting hands out on the desk, a solve from one raised hand, the too-few-samples gate, and the face-on head rejection.

### IMU

- `--test-imufilter`: `ImuOrientationFilter`, the orientation EKF, in cases (a) through (q). Static gravity-only bias learning including the deliberately unobservable gravity-axis component, the same case aided by vision, sustained rotation tracking, the accelerometer motion gate, yaw observability before and after vision, the mounting calibration round trip, motion-based arm-axis recovery, the two-motion mounting solve, twist gating, the gyro bias bound, the axial residual as a mounting-roll health check, pose-averaging the mounting, yaw covariance re-inflating while coasting, palmar-side temporal stability, flip hysteresis, and the curl strength gate.

- `--test-imudiscovery`: the invariant that `ImuService` device discovery never blocks its caller, since `ImuService::update()` runs on the vision thread. Asserts startup under 50 ms, worst single update under 20 ms, shutdown joining within 3 s, and a clean restart cycle. Hardware-independent by design: HID enumeration runs whether or not a controller is paired. See the known flake below.

### Calibration

- `--test-charuco`: the intrinsics wizard path. Bare Charuco finder construction, an ONNX pipeline startup that mirrors live app state, `MonoLensDistortionCalibrator` construction and `update()` against synthetic frames, capture progress on a rendered board including a second capture after an immediate fast board move, and synthetic intrinsics recovery checked against the true horizontal FOV and a sub-hundredth-pixel reprojection error. The ONNX step is a smoke check that the pipeline starts alongside the calibrator, not an inference assertion.

- `--test-extrinsics`: the extrinsics wizard math. An Aruco marker rendered into a synthetic camera 0.8 m above the world origin, then marker distance, recovered camera height, `ExtrinsicsWizard::raycastPixelOntoPlane` onto the table, and the marker axes fixing the world frame rather than only the origin. A second section covers `ExtrinsicsValidation` cross-camera scoring: perfect observations score near zero, and a one-degree orientation error on one camera is flagged by the spacing check even though reprojection alone nearly hides it.

### Recording, replay, diagnostics

- `--test-replay`: record/replay determinism. Runs 60 synthetic frames through the same call sequence `VisionThread` uses, records them, then asserts the replay matches bit-exactly, that replayed poses match float-exactly, that a what-if config change does diverge (a negative control against a replay that ignores its parameters), and that a corrupted checksum is reported rather than passed.

- `--test-framerecorder`: the opt-in raw-frame recorder without cameras. The frame directory being the recording's sibling, every queued frame written and byte-accounted, naming by camera and frame index so a frame joins its landmarks, reload at the same size and within JPEG loss, a 4000-frame burst dropping rather than stalling the caller, and an unstarted recorder writing nothing at all (the default, and the privacy-relevant case).

- `--test-dump`: the F9 diagnostic dump writer and its JSON schema, built from a synthetic camera result and written to a temp directory.

### Output and wire

- `--selftest`: the OSC writer. Packet round trip across the message and bundle forms, address and string lengths at every 4-byte padding boundary, negative int32, reuse after `clear()`, the dropout hold-and-decay behavior, and the wrist-joint rotation the mounting wizard and the OSC panel read.

- `--test-face`: the iFacialMocap receiver. The ARKit name table (52 unique names, `_L`/`_R` folding, unknown names), the v1 `-` and v2 `&` blendshape formats with negative values, head bones needing all six values and eye bones three, values a datagram omits keeping their state, the own-handshake receive filter, empty and garbage input, a loopback round trip through `FaceService` (looped-back handshake dropped, last datagram wins, rebind on port change), and the `UdpSocket` bound receive path.

- `--test-vmc`: the VMC retarget, the streamer's datagram layout, and the face and avatar messages, about 70 checks. The world-to-Unity basis change including rotation sense and properness, the rest palm frames being palms-down T-pose hands of the correct chirality, the rest pose emitting only identity rotations, a bent finger breaking that identity without disturbing its neighbours, and a chain round trip that composes the emitted bones the way a receiver does and rebuilds the measured shoulder, elbow, wrist, palm orientation, and FK finger joints. Also degraded cases (no elbow, no arm, invalid side, invalid head), the freeze-on-loss behavior, the OSC bundle decode against the spec, bone name spelling and uniqueness, and argument order. The face half: `/VMC/Ext/Blend/Val` as `,sf` under the avatar's preset and ARKit morph names with the emotion presets and the redundant two-eyed blink left out, `/VMC/Ext/Blend/Apply` after the last value, the one relaxing frame of zeros when the stream stops and silence after it, the 52 ARKit names with no avatar, `/VMC/Ext/VRM` as `,sss` throttled to once a second, the perfect-sync rule and clamping in the face map, the phone head's axes and composition order in the head frame, and the anchor reproducing the camera head it was captured against.

### Avatar

- `--test-vrm`: the VRM loader and `AvatarSkeleton` on files built in memory, a minimal humanoid in both VRM generations with one skinned triangle and a 2x2 PNG. Refusal of a plain glTF, a truncated file and a file missing its hips; version detection with `VRMC_vrm` winning over `VRM`; the 1.0 thumb rename and the 0.x one-to-one map, unknown names warned and ignored; strided, normalized, sparse and byte/short/int accessor encodings with weight renormalization; a matrix node and a rotated rest node composing to the same world positions as the plain rig; the skin's inverse binds mapping a vertex onto itself at rest and like a CPU reference when a joint turns, with the mesh node's own transform ignored; the avatar-to-world conversion for both generations; the derived lengths, parent resolution past a missing optional bone, world parent offsets under a rotated node, the palm frames matching `VmcRetarget::restPalmFrame` with the tracked chirality sign, phalanx lengths and the avatar rest fingers as neutral directions; PNG decode with alpha, sampler wrap, the default alpha cutoff, the MToon 1.0 fields and the 0.x property migration including render queue ranking; the two generations producing one world rest skeleton; morph target names, the 0.x and 1.0 expressions (preset mapping, weight scaling, node-to-mesh resolution, out-of-range binds dropped with warnings, the find helpers); and SHA-256 against the "abc" vector and a model's own bytes.

- `--test-avatar-retarget`: `AvatarRetarget` and the avatar-driven VMC stream on the shared synthetic rig (`src/Tests/SyntheticAvatar.h`), in cases (a) through (i). A user resting in the avatar's own T-pose leaves every placed bone at identity, every posed bone on its rest position, the node globals at rest, and the VMC stream at 39 identity bones carrying the avatar's own offsets; a reach scales by the arm ratio and clamps at full reach with a straight arm; the two-bone solve keeps both lengths, bends in the plane of the measured elbow and, without one, bends down with the forearm roll following the palm; the posed hand reproduces the measured palm and a measured pronation reappears as forearm roll; curled fingers point where the avatar-hand FK puts them with the other fingers untouched and the thumb's proximal angle moving its metacarpal; the head delta is the measured yaw; the root follows both shoulders with the shoulder-line yaw (a whole-body yaw landing in the root, not the arms), filters a jump, holds across a lost shoulder, and falls back to the fixed root; degradation without a shoulder, an invalid side, or an avatar lacking fingers and clavicles; VRM 0.x, 1.0 and a rotated-rest rig posing identically, bit-identical output on repeated input, and the VMC stream composing back to the posed arm and fingers.

- `--test-avatar-rig`: `AvatarRigSettings` and what each adjustment does to the solve on the shared synthetic rig, in cases (a) through (h). The sidecar round-trips through a non-ASCII path with an unmapped override kept distinct from no override, a missing sidecar yields defaults, and a corrupt one is rejected. Overrides map and unmap by name, an unknown name and an unmapped required bone keep the file's mapping, and each rejection and a shared node warns. On a rig whose file swaps the left index and middle, overriding them back clears the skeleton warning and restores the index sign. A 30 degree hand trim turns the rest hand by exactly that about its palm axis, a forearm roll trim rolls the forearm by its angle, and a head trim leaves its own inverse. Curl and splay gains of 0.5 pose the index exactly as the halved angles do, and a disabled finger is absent. An explicit thumb pronation moves the swapped rig's thumb curl to the side the correct rig uses. The elbow hint turns with the root yaw, and the confidence blend bends toward the hint at zero, the measurement at the threshold, and their midpoint at half.

### App plumbing

- `--loc-test`: the localization tables, without a GL context. Load-time validation warnings (key parity both ways, printf specifier mismatches, embedded `##`, `_meta` problems) are hard failures here, window stable IDs must be unique and non-empty because the English text is the ImGui window ID, the unknown-key fallback must pass the key through, and every codepoint of every language must sit inside the glyph ranges the font atlas actually bakes.

---

## Hardware tests

These need a device physically connected to mean anything, so they stay out of the batch:

- `--test-joycon`: Joy-Con sample decode against hand-built report bytes with known values, then live HID streaming.

- `--test-imuaxes`: measures whether the gyro axes are consistent with the accelerometer axes on a live controller, by scoring every signed permutation against `dg/dt = -w x g`. A mounting calibration can absorb a fixed rotation but not an axis permutation or sign flip inside the chip.

- `--test-vrm-samples`: loads the shipped VRoid sample avatars from `models/avatars/` and checks the full humanoid map, valid hand rests, a T-pose with palms down, and bounded skins, plus the `aa` and `blink` expressions and a SHA-256. Skips with a warning when the files are absent.

- `--test-posemodel`: loads the body-pose ONNX models and cross-checks the decode against keypoints from the Python reference implementation. It needs `models/` and a working execution provider rather than a camera, so it is the one that tells you an inference-stack change is at fault rather than tracking logic.

---

## Tools with regression value

The `Tool` category commands take a file argument, so they are not part of any automated pass, but two of them are the strongest regression instruments in the repo when a recording is available:

- `--replay-verify <recording.jsonl>` re-runs a recording and exits 0 only when every frame's replayed checksum matches the recorded one. This is the real coverage for `LandmarkTo3D` and `HandFusion` against live data rather than synthetic fixtures. Checksums are a same-binary guarantee: a divergence after a change that touches those stages is the diff of the change, not corruption, and a divergence after an unrelated rebuild is not by itself evidence of a bug.

- `--replay-popmetrics <recording.jsonl>... [prior-config.json]` reports pop statistics of a recording, baseline against the hand estimator, which measures a smoothing or estimator change rather than only detecting that it changed something.

The remaining tools (`--replay-dump`, `--replay-bodypose`, `--replay-extrinsics`, `--calibrate-bones`, `--fit-angle-prior`, `--test-imupair`, `--vrm-info`) are diagnostics and fitting utilities, not pass/fail checks. `--render-avatar <file.vrm> [out.png]` renders an avatar's rest pose through the real scene renderer from a hidden window, the regression instrument for the skinned renderer: compare its PNG before and after a shader or draw-order change. `--export-board` and `--export-marker` write a PNG and open it in the system viewer, so keep them out of any batch.

---

## Before and after a refactor

1. Run the full self-test batch on the pre-change build and confirm it is green. A test that was already failing is not evidence about the refactor.
2. If a recording exists and the change touches `LandmarkTo3D` or `HandFusion`, run `--replay-verify` on it before the change so the recorded checksums are known to match the current binary.
3. Make the change.
4. Re-run the batch. `--test-handestimator` case (e) and `--test-replay` case (a) are the two determinism assertions: if a refactor introduced wall-clock, iteration-order, or floating-point-order dependence into a replayed stage, they are where it surfaces.
5. Re-run `--replay-verify`. A pure refactor of a replayed stage must leave the checksums matching, which makes this the sharpest available test that behavior did not change. A divergence means the refactor was not behavior-preserving.

---

## What the tests do not cover

Worth knowing before relying on a green batch:

- `VisionThread` orchestration. `--test-replay` reproduces its call sequence in a single-threaded test, so the sequencing is covered but the threading, the frame-pop path, the hitch watchdog, and the config-refresh handling are not.

- Video capture. Nothing under `src/Video` (the WMF backend, device enumeration, mode selection) has a test.

- Inference itself. `PalmDetector`, `HandLandmarkModel`, `SsdAnchors`, and `OnnxSession` are exercised only as a startup smoke check by `--test-charuco`, and only `--test-posemodel` numerically checks a decode, on the body models. Every hand-side test starts from landmarks that a test fixture generated, not from a model.

- All rendering and UI. `src/Render` and `src/UI` have no coverage beyond the localization tables and the wizard math that `--test-extrinsics` calls directly.

- Config and project persistence. `AppConfig`, `ProjectManager`, and `GlobalSettings` round trips are not tested, though the replay test does reconstruct an `AppConfig` from a recording header.

- Assorted library code with no direct test: `OneEuroFilter`, most of `src/Math`, `src/Utility` (`PathUtils`, `StringUtils`, `WorkerThread`), and the `UdpSocket` send path to a remote host (`--test-vmc` decodes the bytes the streamer builds, but never puts them on a socket).

---

## Known flake

`--test-imudiscovery` fails when HID enumeration is cold. Case (a) allows `ImuService::shutdown()` 3 s to join a worker that may be mid-enumeration. With no Joy-Cons paired and no recent enumeration, that join has been measured between 4.5 s and 10.1 s, which is the first run of a batch every time on at least one dev machine. Run it again immediately and the join settles at about 3 ms and passes.

The flake signature is a `shutdown must join promptly` line in `test-imudiscovery.log` with the startup and worst-update times well inside their thresholds. That is the enumeration being slow, not a regression: re-run it. A failure naming `discovery must not block the caller` is the invariant this test exists for and is never the flake.

---

## Adding a test

One file under `src/Tests`, holding the function and the `MIKAN_REGISTER_TEST(flag, description, category, function)` line directly below it, including `TestCommon.h` for the shared include block. Then re-run CMake: the target's sources come from a plain `file(GLOB)`, so a new file does not link until a reconfigure. The registry mechanics are in [debugging.md](./debugging.md).

Pick the category honestly. `SelfTest` means deterministic with no hardware and no input files, which is what makes the batch above safe to run unattended, so anything that needs a device or a file belongs in `Hardware` or `Tool` instead.
