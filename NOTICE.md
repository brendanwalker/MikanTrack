# Third-party and borrowed code

## Code copied from MikanXR

The following sources were copied (and locally adapted) from the MikanXR
project (https://github.com/MikanXR/MikanXR) by the same author, rather than
linked, so this app builds standalone.

| File(s) here | MikanXR origin |
|---|---|
| `src/Video/Interfaces/IVideoDevice.h`, `IUsbVideoDevice.h`, `IUsbVideoDeviceManager.h` | `src/Libraries/MikanCoreApp/Public/` (verbatim) |
| `src/Video/WMF/*` | `src/Plugins/MikanWMFVideo/Private/` (module scaffolding removed; capture engine, device enumeration, and vendor-MFT workarounds preserved). Device enumeration originally adapted from Evgeny Pereguda's CodeProject Media Foundation capture articles — see file headers. |
| `src/Utility/Logger.*`, `WorkerThread.*`, `ThreadUtils.*` | `src/Libraries/MikanCoreApp/` |
| `src/Utility/PathUtils.*`, `StringUtils.*`, `MemoryUtils.h` | `src/Libraries/MikanUtility/` |
| `src/App/FrameTimer.*` | `src/Editor/AppCore/` (verbatim) |
| `src/Math/MathGLM.*`, `MathMikan.*`, `MathUtility.*`, `Transform.h` | `src/Libraries/MikanMath/` |
| `src/Math/MathTypeConversion.*` | `src/Editor/Math/` (editor-only conversions removed) |
| `src/Math/MikanMathTypes.h`, `MikanVideoSourceTypes.h` | plain-struct retypes of `src/Libraries/MikanClientAPI/Public/` types (reflection/serialization removed) |
| `src/Calibration/CameraMath.*` | `src/Editor/Math/` (partial-board charuco fix applied) |
| `src/Calibration/OpenCVFwd.h`, `MathOpenCV.*` | `src/Editor/OpenCV/` (verbatim) |
| `src/Calibration/CalibrationPatternFinder*`, `MonoLensDistortionCalibrator.*`, `ArucoMarkerPoseSampler.*`, `CVVideoFrameProcessor.*` | `src/Editor/Calibration/` (editor/ECS/renderer coupling removed) |
| `src/Render/DebugDraw.*` | ported from `src/Editor/Renderer/MikanLineRenderer.*` |
| `src/Render/OrbitCamera.*` | orbit-camera math ported from `src/Editor/Renderer/MikanCamera.*` |
| `src/Render/Colors.h` | `src/Editor/Renderer/` |
| `src/Osc/UdpSocket.*` | adapted from `src/Libraries/MikanDMX/Private/UdpMulticastSocket.*` |
| `tools/7zip/7za.exe` | 7-Zip standalone console (see `tools/7zip/License.txt`) |

## Third-party libraries

- **Dear ImGui** (docking branch) — MIT — `thirdparty/imgui` (submodule)
- **tinyfiledialogs** 3.8.8 — zlib — `thirdparty/tinyfiledialogs` (vendored, two files)
- **Mochiy Pop One** (UI font) — SIL OFL 1.1 — `resources/fonts/` (license text alongside as `OFL.txt`)
- **glm** — MIT/Happy Bunny — `thirdparty/glm` (submodule)
- **nlohmann/json** — MIT — `thirdparty/nlohmann_json` (submodule)
- **readerwriterqueue** — BSD-2 — `thirdparty/readerwriterqueue` (submodule)
- **SDL2** 2.30.10 — zlib — prebuilt in `deps/`
- **GLEW** 2.2.0 — BSD/MIT — prebuilt in `deps/`
- **OpenCV** 4.10.0 — Apache-2.0 — prebuilt in `deps/`
- **ONNX Runtime (DirectML)** 1.20.1 — MIT — NuGet package in `deps/`
- **DirectML** 1.15.4 — Microsoft binary license — NuGet package in `deps/`

## ML models

The MediaPipe hand models and person detector (Apache-2.0, Google) are used in their
OpenCV Zoo ONNX conversions (https://github.com/opencv/opencv_zoo,
Apache-2.0), downloaded into `models/` by `InitialSetup_x64.bat`:

- `palm_detection.onnx` — palm_detection_mediapipe_2023feb
- `hand_landmark.onnx` — handpose_estimation_mediapipe_2023feb
- `person_detection.onnx` — person_detection_mediapipe_2023mar

The RTMPose body model (Apache-2.0, OpenMMLab) is the body landmark
model, downloaded from the MMPose model zoo
(https://github.com/open-mmlab/mmpose):

- `rtmpose_body.onnx` — rtmpose-m_simcc-body7_pt-body7_420e-256x192

`rtm_demo.jpg` is MMDeploy's demo image (Apache-2.0), used only as the
fixture the pose model cross-check scores against.
