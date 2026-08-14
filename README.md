# MikanMediaPipe

<!-- AI_USAGE_BADGES:BEGIN -->
![AI tokens](https://img.shields.io/badge/AI_tokens-4.2M_out_%2F_1.3B_read-blueviolet) ![est. energy](https://img.shields.io/badge/est._energy-~71_kWh-yellow) ![est. water](https://img.shields.io/badge/est._water-~214_L-blue)
(estimates, see [TOKEN_STATS.md](TOKEN_STATS.md))
<!-- AI_USAGE_BADGES:END -->

Standalone Windows app for GPU hand tracking from one or more webcams,
streaming a parametric hand model (palm transform + finger bend angles) over
OSC (built for consumption by Unreal Engine's OSC plugin).

Runs the Google MediaPipe hand models (palm detection + hand landmark) via
**ONNX Runtime with the DirectML execution provider** — real GPU inference on
any Windows GPU, no Bazel, no MediaPipe framework build. The two-stage
detector→landmark graph logic (SSD anchors, weighted NMS, rotated crops,
frame-to-frame ROI tracking) is implemented in C++, ported from the
[OpenCV Zoo](https://github.com/opencv/opencv_zoo) reference demos. Webcam
capture, calibration and app scaffolding are borrowed from
[MikanXR](https://github.com/MikanXR/MikanXR) (see `NOTICE.md`).

Body pose (measured elbows, shoulders, head pose) is an OPT-IN per-camera
stage: enable it on a camera that sees you upright (a person detector never
fires on top-down views, so overhead cameras leave it off). Two backends are
selectable per camera. RTMPose (default) is top-down - this app supplies the
person box and each joint is scored independently, so joints outside the
frame read as low confidence. BlazePose owns its own crop and always emits a
whole body, which suits a fully visible person but invents a lower body for
someone truncated at a desk.

Everything above the wrist is solved from 2D rays plus known body lengths,
never the pose model's own metric 3D (measured unusable per frame): the elbow
is the elbow ray against a forearm-length sphere around the FUSED wrist, the
shoulder chains onward from it, and the head takes its depth from the
apparent ear separation. A calibrated wrist IMU still wins for the forearm
when present. Rigs with only overhead cameras solve arms client-side with
Two-Bone IK from the palm transform.

## Features

- Webcam capture via Media Foundation (device/mode selection, hotplug,
  hardware-decode policy with vendor-MFT hang workarounds, NV12/YUY2 passthrough)
- Two-hand tracking as a parametric model: 6-DoF palm transform (solvePnP
  over the articulated metric hand) + per-finger bend angles — low-noise,
  scale-aware, EKF/IMU-fusion-ready
- **Multi-camera fusion**: add a second camera at a different angle (Device
  panel -> Add Camera), calibrate it against the same printed marker, and the
  visibility-weighted pose/angle blend rides through hand poses that defeat
  a single view (e.g. clapping edge-on to an overhead camera). Left/right is
  resolved at the fusion level (view-ray-aware world-space clustering + votes +
  a fixed "right hand toward -Y" spatial prior, guaranteed by the labelled
  calibration board), the two views continuously refine the hand scale by stereo
  triangulation, and a hand tracked by one camera but lost by another is
  projected into the lost camera's image to re-seed its search directly.
  Prefer 720p per camera and separate USB controllers for two streams.
- Live preview with landmark overlay; alternate 3D scene view rendering the
  forward-kinematics hand reconstruction (exactly what OSC clients rebuild),
  camera frustums, marker grid, orbit camera
- **Body measurement wizard** (Tracking panel -> Measure My Body): the elbow,
  shoulder and head estimates rest on lengths that are NOT anatomical - they
  are distances between the pose model's own landmarks, which sit inside the
  real joints by an amount that varies per person and per model. Raise ONE hand
  beside your shoulder: that puts a metric wrist at roughly the depth of your
  torso, and the fused wrists (the only metric, world-anchored points, measured
  by the other cameras) become the ruler for the shoulder and head widths. A
  second step turns the head, which swings the nose's forward offset into the
  image where it can be measured. Only WIDTHS are measured; the upper arm is
  taken as a multiple of the shoulder width, because measuring it directly
  needs the arm straight and square to the camera - a pose that is hard to hold
  at a desk and reads 20% short when missed, which is enough to make the elbow
  bend the wrong way.
- **Charuco intrinsics calibration** wizard (partial-board captures supported)
- **Charuco extrinsics** wizard: the SAME printed charuco board defines the
  tracking world - its center is the origin, and the FORWARD/RIGHT labels on
  the sheet pin the axes (+X forward, +Y left, +Z up; the right hand is
  always toward -Y, a fixed convention the L/R hand assignment relies on).
  All cameras calibrate in one session against one placement, and the wizard
  validates the result by triangulating the board's corners across each
  camera pair and checking the reconstruction against the board's known
  geometry (reprojection px, corner-spacing error in mm, scale, flatness) -
  the numbers are saved with the config as a baseline. Hand scale is
  measured continuously from stereo/depth while tracking runs.
- OSC 1.0 output over UDP unicast, one bundle per frame (rate-limited)
- Dear ImGui (docking) UI; config persisted to `%APPDATA%/MikanMediaPipe/config.json`
- **Image quality diagnostics** (Tracking panel): per-camera statistics of
  each hand's region in the exact image the model consumed - luminance,
  highlight/shadow clipping, contrast, hand-vs-background separation,
  sharpness, sensor noise, plus whole-frame flicker/auto-exposure-hunting
  detection. Landmark jitter is almost always a lighting or camera-settings
  problem; this readout names the knob (hover each metric for what to
  adjust). The **hold-still jitter test** (Diagnostics section) turns a few
  seconds of holding still into one mm number per hand, for A/B-ing exposure,
  gain, lighting and background changes.
- **Diagnostic dump (F9)**: writes the last few seconds of tracking/fusion
  history (per-camera hand states with image-quality statistics, fusion
  clusters and the L/R side-assignment scores), the live config and each
  camera's current frame (raw + annotated PNG) to
  `%APPDATA%/MikanMediaPipe/dumps/<timestamp>/` - hit it the moment
  tracking misbehaves and attach the folder to a bug report
- **Frame-loop hitch watchdog**: every camera is served by one vision thread,
  so any phase that overruns the frame budget drops frames on all of them at
  once and reads downstream as a camera or USB fault. Iterations over 50 ms are
  logged with a per-phase breakdown (capture/inference, IMU, fusion, OSC,
  diagnostics) and counted in the Tracking panel. Device discovery, which used
  to block that thread for ~200 ms at a time, runs on its own worker.
- **Tracking recording + deterministic replay (F10 / Timeline panel)**:
  records every input to the post-MediaPipe stages (per-camera landmarks,
  depth samples, timestamps - no video, ~1 MB/s) as JSONL under
  `%APPDATA%/MikanMediaPipe/recordings/`, then re-runs the whole
  triangulation/fusion/smoothing pipeline offline, verified bit-exact by
  per-frame checksums. The Timeline panel scrubs/steps/plays a recording in
  the 3D scene view, marks any divergent frames, and re-simulates the same
  incident with edited fusion parameters ("what-if") to judge a candidate
  fix against the recorded original. Headless:
  `MikanMediaPipe --replay-verify <file>` re-verifies a recording;
  `MikanMediaPipe --replay-dump <file> <first> <last>` emits a frame range
  with regenerated fusion diagnostics for offline analysis. Starting a
  recording resets transient tracking state (brief blip); editing tracking
  settings mid-recording finalizes the file. Checksums only verify against
  the same build that recorded them.
- **Raw frame recording (opt-in, off by default)**: a landmark recording
  replays every stage after inference, so it cannot answer whether a
  different pose model would have done better - that model's input is gone.
  Ticking "Also record raw camera frames" in the Timeline panel additionally
  writes the exact images the models consumed as JPEGs beside the recording,
  which makes `MikanMediaPipe --replay-bodypose <file> [blazepose|rtmpose]`
  a real measurement (per-joint scores, 2D jitter, box source, inference
  cost) rather than a live impression. **This is video of your room**, which
  is why it defaults off and is a local choice; it costs roughly 3-6 MB per
  second per camera, and frames are dropped rather than stalling tracking if
  the encoder falls behind.

## Building

Windows 10/11, Visual Studio 2022, CMake >= 3.15.

```bat
git clone <this repo>
cd MikanMediaPipe
InitialSetup_x64.bat            :: downloads deps/ (SDL2, OpenCV, GLEW, ONNX Runtime, DirectML) + models/
GenerateProjectFiles_X64_VS2022.bat
cmake --build build --config Release
build\Release\MikanMediaPipe.exe
```

`MikanMediaPipe.exe --list-tests` lists every self-test and headless
diagnostic tool, grouped by whether it needs hardware or an input file. Each
one lives in its own file under `src/Tests` and registers itself, so adding a
test means adding a file. Run one by passing its flag (for example
`--test-fusion`); it logs to `<flag>.log` next to the exe and exits non-zero on
failure.

## Calibration workflow

1. **Intrinsics** (Calibration menu → Intrinsics Wizard): export the charuco
   board PNG, print at 100% scale (verify a square with a ruler), capture it
   from 12 poses. Target reprojection error < 0.5 px.
2. **Extrinsics** (requires intrinsics on every camera): lay the SAME charuco
   board flat where the tracking origin should be, FORWARD label pointing away
   from you (the board center becomes the world origin), and capture - every
   camera samples the one placement together. Review shows per-camera
   reprojection plus the cross-camera agreement metrics; accept saves all
   cameras atomically. The board can be removed afterwards (but re-run the
   wizard if a camera moves). `--replay-extrinsics <recording> <config.json>`
   A/Bs a new calibration against a recorded session offline.

Without calibration the app still tracks and streams, but only image-space
data is meaningful (no metric 3D / world space).

## OSC output

Default target `127.0.0.1:8000` (configurable). One OSC 1.0 bundle per frame.
Hands are streamed as a PARAMETRIC model - palm transform + finger bend
angles - rather than raw landmarks: angles come from the network's local
articulation (its most reliable output) and are depth-noise-free, and
poses/angles fuse cleanly across cameras where landmark blending distorted
bones. Elbow, shoulder and head addresses are always sent; their trailing
confidence carries validity (0 = do not use), so they never go silent.

| Address | Types | Meaning |
|---|---|---|
| `/mikan/frame` | `iifi` | frameId, timestampMs, fps, sendSequence |
| `/mikan/hand/{left,right}/tracked` | `iff` | tracked (0/1), presence, confidence |
| `/mikan/hand/{left,right}/elbow` | `4f` | elbow position xyz (m) + confidence; from the wrist IMU forearm when calibrated, else the vision body-pose solve |
| `/mikan/hand/{left,right}/shoulder` | `4f` | shoulder position xyz (m) + confidence; vision body pose |
| `/mikan/hand/{left,right}/palm` | `7f` | palm position xyz (m) + orientation quaternion xyzw |
| `/mikan/hand/{left,right}/wrist` | `i8f` | valid (0/1), forearm orientation xyzw (world), wrist joint rotation xyzw (palm in the forearm frame) |
| `/mikan/hand/{left,right}/fingers` | `20f` | per finger (thumb..pinky): lateral, proximalBend, intermediateBend, distalBend (DEGREES, 0 = the rest pose) |
| `/mikan/hand/{left,right}/skeleton` | `45f` | per finger: base position in palm frame xyz + phalanx lengths [proximal, intermediate, distal] (m) + neutral (zero-angle) direction in palm frame xyz; sent at 1 Hz |
| `/mikan/body/head` | `8f` | head position xyz (m) + orientation xyzw (+X facing, +Y person's left, +Z up) + confidence; vision body pose |
| `/mikan/info` | `ss` | space/units/palm-frame convention, app version (1 Hz) |

**Palm frame** (Ultraleap-compatible): origin at the palm center (midway
wrist to middle knuckle), **+X toward the fingers**, **+Z out of the palmar
surface**, +Y completing right-handed. Positions are in the marker-anchored
world frame (right-handed, meters, +Z up out of the table); before extrinsics
calibration they fall back to OpenCV camera space (`/mikan/info` says which).

**Angle conventions** (all in the palm frame):

- **Zero = the rest pose.** Forward kinematics starts from the per-finger
  `neutralDirInPalm` streamed in the skeleton message (the flat-hand default:
  four fingers parallel to palm +X, thumb along its own metacarpal) - use it,
  don't derive one. **Tracking panel -> Capture Rest Pose** then records what
  each camera reports for your hands held flat and subtracts it, so your rest
  pose reads zeros on all four angles. Without it a hand hovering over a
  keyboard reports 20-50 degrees of knuckle flexion, which is *correct* but
  rarely what a client wants as its origin.

  The capture is **per camera, and needs every camera to see both hands**,
  because MediaPipe's model landmarks are view-dependent: two cameras
  watching the same physical hand disagree about its articulation by tens of
  degrees (41 degrees on one finger, measured on this rig). Removing each
  camera's own bias is also what makes their angles comparable enough for
  fusion to blend them rather than average two different biases.
- **`lateral`** rotates about palm **+Z, positive counter-clockwise**, i.e.
  toward palm **+Y = cross(palmZ, palmX)**. Purely geometric and identical
  for both hands (the palm frame carries the chirality), so on a right hand
  positive splays toward the pinky and on a left hand toward the thumb.
- **`proximal`** is positive **curling toward the palm** (the +Z side).
- **`intermediate` and `distal` are relative to their PARENT BONE**, not to
  the palm: `intermediate` is the middle bone's bend from the proximal bone,
  `distal` the tip bone's bend from the intermediate bone. Zero means
  collinear with the parent. They chain, so an evenly curling finger reads
  three similar values.

**Client-side hand reconstruction**: place each finger base at its skeleton
offset in the palm frame, start from that finger's streamed
`neutralDirInPalm`, apply lateral rotation about palm +Z, then bend the three
phalanx segments about the finger's lateral axis by the three bend angles. **Thumb exception**: the thumb's
intermediate/distal bends rotate about its hinge PRONATED 1.2 rad (~69 deg)
about the thumb metacarpal direction (positive pronation on a right hand,
negative on a left) - the thumb rests twisted relative to the fingers, so
its flexion sweeps across the palm toward the pinky rather than curling
toward the palm plane. The app's own 3D view renders exactly this
reconstruction, so it shows what your client will see.

**Skeleton/bone lengths** come from MediaPipe's metric hand model scaled by
the calibrated hand scale - no separate bone calibration needed.

**Confidence** is `presence x stability`, where stability is measured from the
observed palm jitter (the constant-velocity residual) rather than taken from
the network. MediaPipe's own presence score answers "is a hand here" and stays
near 1.0 on a badly conditioned edge-on view whose depth swings by centimeters,
so it is not usable as a trust signal on its own. Fusion weights each camera by
`confidence x how face-on the palm is`, so a camera with a poor view of a hand
stops polluting the fused pose; the streamed confidence is the best
contributing camera's. Set **OSC panel -> Min confidence** to withhold
`/palm` and `/fingers` below a threshold - the hand is then streamed as
`tracked=0` and the client should hold its last good pose or blend to a rest
pose. Tune with the live per-camera confidence table in the Tracking panel.

### Consuming in Unreal Engine

UE is left-handed, Z-up, centimeters. Convert per landmark:

```
UE.X = 100 * mikan.Y
UE.Y = 100 * mikan.X
UE.Z = 100 * mikan.Z
```

(the axis swap performs the handedness flip; rotate the palm quaternion
accordingly). In UE: enable the **OSC plugin**, create an OSC Server bound to
the configured port, and drive your hand rig from the palm transform + finger
angles - the same representation the Ultraleap SDK feeds it. Solve elbows
with Two-Bone IK from the palm transform.

## Notes

- The overlay HUD shows the active execution provider: green `DirectML` = GPU
  inference; yellow `CPU` = fallback (forceable with `"onnxEp": "cpu"` in config).
- MediaPipe's handedness assumes a mirrored selfie view; the **Flip handedness**
  toggle (default on) corrects for a normal non-mirrored camera. Verify on your
  rig and flip if L/R are swapped.
- Depth accuracy from a single camera is in the ±2 cm class; it depends on a
  good intrinsics calibration and an accurate hand-scale measurement.
