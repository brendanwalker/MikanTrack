# Plan

The living plan: what is in flight now, what comes next, and the open questions. Completed items are removed rather than checked off. Resolved questions are removed. Deferred work enters here at the moment of deferral.

## Now

- [ ] Live-verify localization: full Japanese walkthrough (menu, setup flow, wizards, panels, tooltips), live en/ja switching with modals and wizards open, dock layout preserved from a pre-localization imgui.ini, and a Japanese truncation sweep of narrow labels.
- [ ] Live-verify the setup-flow usability pass: cancel mid-flow deletes the new project and Resume falls back to the previous one, camera-selection previews appear as devices are picked, the Video Preview tab fronts when intrinsics/extrinsics start, and a Resume pointing at a deleted project is hidden. Remaining from the original walkthrough: the Joy-Con and tri-camera setup variants end to end.
- [ ] Finish the MikanTrack rename outside the source tree: rename the repo working directory and the GitHub repository, then re-clone or update the remote. The code, docs, and appdata folder already use the new name.
- [ ] Live-verify the `/mikan/hand/{s}/forearm` wire change end to end with an OSC consumer (both sides built and self-tested, not yet exercised live).

## Next

Code cleanup pass, one concern per commit.

- [ ] `MainWindow::update` decoupling. It interleaves deferred project-state transitions, the vision-thread data pull, the F9/F10 hotkeys, dockspace and menu drawing, panel drawing, wizard launch requests, focus management, the forearm overlay projection, scene camera construction, and the wizard update chain. Split along those seams:
	- [ ] Pull the F9/F10 hotkeys into one table-driven handler
	- [ ] Extract the forearm overlay projection and the scene camera view construction into free functions of the config and the fused result
	- [ ] Replace the wizard update chain with one active-wizard host owning the launch-request flags, the mutual exclusion, and the Video Preview focus rising edge
	- [ ] Let `DevicePanel` register the capture-system hotplug callbacks itself instead of `MainWindow` relaying them
- [ ] `VisionThread` decomposition. `threadLoop` becomes a sequence of stage classes alongside the `CameraContext` capture and inference stage (fusion, IMU forearm fill, body solve, output, calibration captures, diagnostics and recording). Stages are stateless over `VisionThread`: the handoff state (mutexes, atomics, fetch accessors) stays on `VisionThread` and each stage receives what it needs by reference. The recording taps are the constraint: the checksum point stays immediately after `fuse`, and `--replay-verify` on the existing recordings is the acceptance test.

## Later

- [ ] Human-review the machine-translated Japanese strings in `resources/localization/ja.json` (translated per key alongside the code conversion; natural phrasing and terminology consistency were not reviewed by a native speaker).
- [ ] Pre-open-sourcing cleanup: untrack the root `test-*.log` files and `imgui.ini`, remove the stale `20230831/` extraction residue and the empty `cmake/` and `tests/` directories, delete the orphaned `models/rtmpose_hand.onnx`, and fix the stale `NOTICE.md` entry for `ArucoMarkerPoseSampler` (now `PatternPoseSampler`).
- [ ] `InitialSetup_x64.bat` downloads `rtm_demo.jpg` to the repo root but no build step copies it next to the exe, so `--test-posemodel` silently skips its numeric cross-check. Fetch it into a copied directory or copy it post-build.
- [ ] Replace the IMU discovery poll with native device-arrival notifications (Win32 `CM_Register_Notification`, filtered on the Joy-Con VID), keeping a slow poll as the safety net.
- [ ] Refit the per-user angle prior (`--fit-angle-prior`) as recording coverage grows; the shipped weighting is deliberately weak.
- [ ] `GenerateProjectFiles_X64_VS2022.bat` passes `-A x64`, which conflicts with a `build/` cache configured without an explicit platform (Visual Studio 2022 defaults to x64 either way). Decide on one invocation so a fresh clone and an existing tree agree.

## Open questions

- A fully tracked Mikan-format frame (~1.6 KB) exceeds a single 1472-byte UDP payload and would rely on IP fragmentation on a real network (localhost is unaffected). Chunk it like VMC mode, or keep the one-bundle-per-frame contract and accept fragmentation?
- Capture-phase GPU path, if the profile justifies one: OpenCV's OpenCL T-API (`cv::UMat`, no shader code, still a CPU tensor at the ONNX boundary) or a D3D12 preprocess feeding DirectML through IoBinding (custom shaders, no readback, the larger job)?
