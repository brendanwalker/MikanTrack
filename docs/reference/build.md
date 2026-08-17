# Build System

How MikanMediaPipe is configured and built: toolchain, dependency setup, the CMake target, and the ONNX model set it depends on. A copy-paste command cheat sheet is in [commands.md](./commands.md); how to run and interpret the two kinds of test output is in [debugging.md](./debugging.md).

---

## Toolchain

- Windows 10/11 only, MSVC via Visual Studio 2022, x64 only.
- CMake minimum 3.15 (`cmake_minimum_required` in the root `CMakeLists.txt`).
- The root `CMakeLists.txt` guards on `MSVC`: `if(NOT MSVC) message(FATAL_ERROR "MikanMediaPipe is Windows/MSVC only") endif()`. There is no non-Windows configuration path.
- C++20 (`CMAKE_CXX_STANDARD 20`), `/MP /W3`, with `NOMINMAX`, `_CRT_SECURE_NO_WARNINGS`, `WIN32_LEAN_AND_MEAN`, and `GLM_ENABLE_EXPERIMENTAL` defined globally.
- The whole project is one file: a single `CMakeLists.txt` at the repo root configuring a single target, `MikanMediaPipe`. There is no `cmake/` subtree of included modules and no multi-target library split.

---

## Build sequence

Three steps, in order:

1. `InitialSetup_x64.bat`: fetches submodules, dependencies, and models. Only needs rerunning when dependency versions change.
2. `GenerateProjectFiles_X64_VS2022.bat`: configures `build/` and produces `build/MikanMediaPipe.sln`.
3. `cmake --build build --config Release`, or open `build\MikanMediaPipe.sln` in Visual Studio 2022 and build there.

---

## First-time setup

`InitialSetup_x64.bat` (run from the repo root):

- Runs `git submodule update --init --recursive` to populate `thirdparty/`.
- Deletes any existing `deps/` folder and recreates it from scratch. This is a full wipe-and-redownload every time, roughly 1.3 GB total, with OpenCV alone around 900 MB. Never store anything by hand in `deps/`; it does not survive a re-run.
- Downloads pinned dependency versions with `curl` and extracts them with the vendored `tools/7zip/7za.exe` (zips, self-extracting `.exe` archives, and NuGet `.nupkg` files, which are zips under a different extension).
- Downloads the ONNX models into `models/`.

`deps/` and `models/` are both gitignored; nothing under either is checked in.

### Dependencies (`deps/`)

| Dependency | Version | Location | Source |
|---|---|---|---|
| SDL2 | 2.30.10 (devel VC) | `deps/SDL2-2.30.10` | GitHub release zip |
| GLEW | 2.2.0 | `deps/glew-2.2.0` | GitHub release zip |
| OpenCV | 4.10.0, `opencv_world` build | `deps/opencv` | GitHub release self-extracting `.exe`, unpacked with `7za` |
| ONNX Runtime | 1.20.1, DirectML flavor | `deps/onnxruntime` | NuGet package `microsoft.ml.onnxruntime.directml` |
| DirectML | 1.15.4 | `deps/directml` | NuGet package `microsoft.ai.directml` |

### Submodules (`thirdparty/`)

- `thirdparty/imgui`: Dear ImGui, `docking` branch, compiled directly into the exe (not a prebuilt library).
- `thirdparty/glm`: header-only.
- `thirdparty/nlohmann_json`: header-only, the include path is `thirdparty/nlohmann_json/single_include`.
- `thirdparty/readerwriterqueue`: header-only.

---

## Models (`models/`)

`InitialSetup_x64.bat` downloads four model files into `models/`. `hand_landmark.onnx`, `palm_detection.onnx`, and `person_detection.onnx` come from the OpenCV Zoo project's ONNX conversions of the MediaPipe models, fetched from Hugging Face. `rtmpose_body.onnx` comes from an OpenMMLab MMDeploy SDK zip; only its `end2end.onnx` is extracted and renamed. The setup script also fetches `rtm_demo.jpg` into the repo root, a reference photo the `--test-posemodel` numeric cross-check scores its RTMPose decode against; if it is missing, that check is skipped rather than failed.

| Model file | Consumed by |
|---|---|
| `palm_detection.onnx` | `src/Vision/PalmDetector.h`/`.cpp` |
| `hand_landmark.onnx` | `src/Vision/HandLandmarkModel.h`/`.cpp` |
| `person_detection.onnx` | `src/Vision/PoseDetector.h`/`.cpp` |
| `rtmpose_body.onnx` | `src/Vision/RtmPoseBodyModel.h`/`.cpp` |

Each model's exact tensor input/output contract (shapes, layout, normalization, decode math) is documented as a comment block directly above its wrapper class in the corresponding header, not in this doc. Read `PalmDetector.h`, `PoseDetector.h`, `HandLandmarkModel.h`, or `RtmPoseBodyModel.h` before changing how a model is invoked.

---

## Configuring and the target

`GenerateProjectFiles_X64_VS2022.bat` runs `cmake .. -G "Visual Studio 17 2022" -A x64` from a `build/` directory it creates, producing `build/MikanMediaPipe.sln`. There are no project-specific `-D` cache variables to pass; every dependency path is a fixed relative path under `deps/` or `thirdparty/` set directly in `CMakeLists.txt`.

The build defines one target: `add_executable(MikanMediaPipe WIN32 ...)`. `WIN32` means a GUI subsystem with no console window attached, so the logger writes to `.log` files next to the exe rather than stdout being visible interactively (console output still appears when the exe is launched from a terminal). ImGui's `.cpp` files are listed directly in the target's sources and compiled into the exe; there is no separate ImGui static library.

Source collection is per-directory `file(GLOB ...)` (one glob per `src/<Dir>`, e.g. `APP_SRC`, `VISION_SRC`, `TESTS_SRC`). Because glob results are captured at configure time, adding a new `.cpp` file to an existing `src/<Dir>` requires re-running CMake configure (`GenerateProjectFiles_X64_VS2022.bat` or the equivalent `cmake ..`) before it is picked up; Visual Studio's F5 build alone will not see it. Adding a whole new `src/<Dir>/` requires editing `CMakeLists.txt`: a new `file(GLOB ...)` line, adding that variable to the `add_executable` and `source_group` calls, and adding the directory to `target_include_directories`.

Every module directory under `src/` is on the include path (see the `target_include_directories` block), so includes across the codebase use bare filenames (`#include "HandLandmarkModel.h"`) rather than directory-qualified paths.

---

## Post-build steps

A `POST_BUILD` custom command on the `MikanMediaPipe` target:

- Copies the SDL2, GLEW, ONNX Runtime, DirectML, and OpenCV (`opencv_world4100.dll`, or the `d` debug variant under a Debug config) runtime DLLs next to the exe.
- `copy_directory`s `models/` and `resources/` into `<exe dir>/models` and `<exe dir>/resources`.

`VS_DEBUGGER_WORKING_DIRECTORY` is set to the exe's own directory, which is the working directory Visual Studio's F5 launches with. This matters beyond the debugger: every model load path in the code is relative (`models/palm_detection.onnx`, `models/rtmpose_body.onnx`, ...), so the exe must be run with its current working directory set to the exe's directory. Running it with the repo root as the working directory silently fails to load models (or loads the wrong stale copy if the same exe was previously run correctly from its own directory).
