# MikanMediaPipe

Standalone Windows app for GPU hand + forearm tracking from an overhead webcam,
streaming world-space landmark positions over OSC (built for consumption by
Unreal Engine's OSC plugin).

Runs the Google MediaPipe hand/pose models (palm detection, hand landmark,
person detection, BlazePose) via **ONNX Runtime with the DirectML execution
provider** — real GPU inference on any Windows GPU, no Bazel, no MediaPipe
framework build. The two-stage detector→landmark graph logic (SSD anchors,
weighted NMS, rotated crops, frame-to-frame ROI tracking) is implemented in
C++, ported from the [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
reference demos. Webcam capture, calibration and app scaffolding are borrowed
from [MikanXR](https://github.com/MikanXR/MikanXR) (see `NOTICE.md`).

## Features

- Webcam capture via Media Foundation (device/mode selection, hotplug,
  hardware-decode policy with vendor-MFT hang workarounds, NV12/YUY2 passthrough)
- Two-hand tracking (21 landmarks each) + elbows/forearms from BlazePose,
  with a hand-derived forearm fallback for overhead framing where BlazePose
  gets unreliable (fallback forearms are drawn dashed and sent with low confidence)
- Live preview with skeleton wireframe overlay; alternate 3D scene view
  (marker grid, camera frustum, 3D skeletons, orbit camera)
- **Charuco intrinsics calibration** wizard (partial-board captures supported)
- **Aruco extrinsics + hand-scale** wizard: a printed marker on the table
  defines the world origin; laying your hand flat next to it measures your
  real wrist→knuckle length, which is what converts 2D landmarks + relative
  depth into metric 3D
- OSC 1.0 output over UDP unicast, one bundle per frame (rate-limited)
- Dear ImGui (docking) UI; config persisted to `%APPDATA%/MikanMediaPipe/config.json`

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

Default target `127.0.0.1:8000` (configurable). One OSC 1.0 bundle per frame:

| Address | Types | Meaning |
|---|---|---|
| `/mikan/frame` | `iif` | frameId, timestampMs, fps |
| `/mikan/hand/{left,right}/tracked` | `if` | tracked (0/1), presence |
| `/mikan/hand/{left,right}/wrist` | `fff` | wrist position |
| `/mikan/hand/{left,right}/palm` | `fff` | palm centroid (MCPs 5/9/13/17) |
| `/mikan/hand/{left,right}/landmarks` | `fff`×21 | all 21 landmarks, MediaPipe order |
| `/mikan/arm/{left,right}/elbow` | `ffff` | elbow xyz + confidence |
| `/mikan/arm/{left,right}/forearm` | `ffffff` | elbow xyz, wrist xyz |
| `/mikan/info` | `ss` | space/units convention, app version (1 Hz) |

**Coordinate convention** (`space=marker`): right-handed, **meters**, origin at
the marker center, +X along the marker's right edge, +Y along its top edge,
**+Z up out of the table**. Before extrinsics calibration, positions are in
OpenCV camera space (`space=camera`: +X right, +Y down, +Z away from camera).

### Consuming in Unreal Engine

UE is left-handed, Z-up, centimeters. Convert per landmark:

```
UE.X = 100 * mikan.Y
UE.Y = 100 * mikan.X
UE.Z = 100 * mikan.Z
```

(the axis swap performs the handedness flip). In UE: enable the **OSC plugin**,
create an OSC Server bound to the configured port, and bind addresses with
`,fff` messages to your actors. Low-confidence elbows (fallback mode sends
confidence 0.1) should be blended or ignored.

## Notes

- The overlay HUD shows the active execution provider: green `DirectML` = GPU
  inference; yellow `CPU` = fallback (forceable with `"onnxEp": "cpu"` in config).
- MediaPipe's handedness assumes a mirrored selfie view; the **Flip handedness**
  toggle (default on) corrects for a normal non-mirrored camera. Verify on your
  rig and flip if L/R are swapped.
- Depth accuracy from a single camera is in the ±2 cm class; it depends on a
  good intrinsics calibration and an accurate hand-scale measurement.
