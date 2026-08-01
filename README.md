# MikanMediaPipe

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

Elbows are deliberately NOT estimated or streamed: a hand-only view doesn't
contain enough information for a useful estimate (BlazePose was tried and
removed — its person detector never fires on top-down views). Solve arms
client-side with Two-Bone IK from the palm transform.

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
  an optional "right hand toward +X" spatial prior for users who don't cross
  their hands), the two views continuously refine the hand scale by stereo
  triangulation, and a hand tracked by one camera but lost by another is
  projected into the lost camera's image to re-seed its search directly.
  Prefer 720p per camera and separate USB controllers for two streams.
- Live preview with landmark overlay; alternate 3D scene view rendering the
  forward-kinematics hand reconstruction (exactly what OSC clients rebuild),
  camera frustums, marker grid, orbit camera
- **Charuco intrinsics calibration** wizard (partial-board captures supported)
- **Aruco extrinsics + hand-scale** wizard: a printed marker on the table
  defines the world origin; laying your hand flat next to it measures your
  real wrist→knuckle length, which is what converts 2D landmarks + relative
  depth into metric 3D
- OSC 1.0 output over UDP unicast, one bundle per frame (rate-limited)
- Dear ImGui (docking) UI; config persisted to `%APPDATA%/MikanMediaPipe/config.json`
- **Diagnostic dump (F9)**: writes the last few seconds of tracking/fusion
  history (per-camera hand states, fusion clusters and the L/R side-assignment
  scores), the live config and each camera's current frame (raw + annotated
  PNG) to `%APPDATA%/MikanMediaPipe/dumps/<timestamp>/` - hit it the moment
  tracking misbehaves and attach the folder to a bug report

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

`MikanMediaPipe.exe --selftest` runs the OSC encoder golden-byte tests.

## Calibration workflow

1. **Intrinsics** (Calibration menu → Intrinsics Wizard): export the charuco
   board PNG, print at 100% scale (verify a square with a ruler), capture it
   from 12 poses. Target reprojection error < 0.5 px.
2. **Extrinsics + hand scale** (requires intrinsics): print the aruco marker
   (default 100 mm, ID 0), tape it flat on the table where the world origin
   should be, capture the camera pose, then lay a hand flat next to the marker
   to measure hand scale. The marker can be removed afterwards (but re-run the
   wizard if the camera moves).

Without calibration the app still tracks and streams, but only image-space
data is meaningful (no metric 3D / world space).

## OSC output

Default target `127.0.0.1:8000` (configurable). One OSC 1.0 bundle per frame.
Hands are streamed as a PARAMETRIC model - palm transform + finger bend
angles - rather than raw landmarks: angles come from the network's local
articulation (its most reliable output) and are depth-noise-free, and
poses/angles fuse cleanly across cameras where landmark blending distorted
bones. Elbows are not streamed - solve arms client-side with IK from the
palm transform.

| Address | Types | Meaning |
|---|---|---|
| `/mikan/frame` | `iif` | frameId, timestampMs, fps |
| `/mikan/hand/{left,right}/tracked` | `iff` | tracked (0/1), presence, confidence |
| `/mikan/hand/{left,right}/palm` | `7f` | palm position xyz (m) + orientation quaternion xyzw |
| `/mikan/hand/{left,right}/fingers` | `20f` | per finger (thumb..pinky): lateral, proximalBend, intermediateBend, distalBend (radians, 0 = the rest pose) |
| `/mikan/hand/{left,right}/skeleton` | `45f` | per finger: base position in palm frame xyz + phalanx lengths [proximal, intermediate, distal] (m) + neutral (zero-angle) direction in palm frame xyz; sent at 1 Hz |
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
