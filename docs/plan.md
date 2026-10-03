# Plan

The living plan: what is in flight now, what comes next, and the open questions. Completed items are removed rather than checked off. Resolved questions are removed. Deferred work enters here at the moment of deferral.

## Now

- [ ] Live-verify localization: full Japanese walkthrough (menu, setup flow, wizards, panels, tooltips), live en/ja switching with modals and wizards open, dock layout preserved from a pre-localization imgui.ini, and a Japanese truncation sweep of narrow labels.
- [ ] Live-verify the setup-flow usability pass: cancel mid-flow deletes the new project and Resume falls back to the previous one, camera-selection previews appear as devices are picked, the Video Preview tab fronts when intrinsics/extrinsics start, and a Resume pointing at a deleted project is hidden. Remaining from the original walkthrough: the Joy-Con and tri-camera setup variants end to end.
- [ ] Finish the MikanTrack rename outside the source tree: rename the repo working directory and the GitHub repository, then re-clone or update the remote. The code, docs, and appdata folder already use the new name.
- [ ] Live-verify the `/mikan/hand/{s}/forearm` wire change end to end with an OSC consumer (both sides built and self-tested, not yet exercised live).
- [ ] Live-verify the Avatar panel in the app: load `models/avatars/fem_vroid.vrm` and Bonjiri through Browse, check the avatar stands on the marker grid facing +X with arms along +/-Y in the 3D Scene tab, the tracked skeleton lines draw over it, the placement controls move it, project save and reload restore it, and Close Project clears it (the renderer itself is verified by `--render-avatar`, the panel path is not).

## Next

- [ ] Live-verify the avatar retarget: with an avatar loaded and both cameras tracking, the 3D scene character follows the hands, bends its elbows toward the measured ones, turns its head, and slides onto the measured shoulders when the body-pose stage tracks them; then VMC mode against VSeeFace and VNyan with a VRoid sample and Bonjiri (arms, wrist roll, fingers, and the rest pose on a dropout). The solver and the stream are self-tested; the live feel (root follow time constant, default elbow hint and its confidence blend, reach scaling on a real body) is not.
- [ ] A `/mikan/avatar` bone block on the Mikan wire so the Unreal plugin can drop its animation blueprint retarget, planned separately once the retarget is live.
- [ ] Live-verify the avatar mapping UI: load Jasper, swap index and middle on both hands through the Mapping tab's pickers and confirm the warning clears and the thumb curls inward on the Demo preview; drag an elbow hint in the 3D scene and watch the bend follow; tune a hand trim on the Rest preview; reload the project and confirm the sidecar restored everything. The rig math is self-tested and Jasper's remap is checked through `--render-avatar`; the panel, the figure, the picker and the gizmo picking are not exercised.

## Later

- [ ] Human-review the machine-translated Japanese strings in `resources/localization/ja.json` (translated per key alongside the code conversion; natural phrasing and terminology consistency were not reviewed by a native speaker).
- [ ] Pre-open-sourcing cleanup: untrack the root `test-*.log` files and `imgui.ini`, remove the stale `20230831/` extraction residue and the empty `cmake/` and `tests/` directories, delete the orphaned `models/rtmpose_hand.onnx`, and fix the stale `NOTICE.md` entry for `ArucoMarkerPoseSampler` (now `PatternPoseSampler`).
- [ ] `InitialSetup_x64.bat` downloads `rtm_demo.jpg` to the repo root but no build step copies it next to the exe, so `--test-posemodel` silently skips its numeric cross-check. Fetch it into a copied directory or copy it post-build.
- [ ] Replace the IMU discovery poll with native device-arrival notifications (Win32 `CM_Register_Notification`, filtered on the Joy-Con VID), keeping a slow poll as the safety net.
- [ ] Refit the per-user angle prior (`--fit-angle-prior`) as recording coverage grows; the shipped weighting is deliberately weak.
- [ ] `GenerateProjectFiles_X64_VS2022.bat` passes `-A x64`, which conflicts with a `build/` cache configured without an explicit platform (Visual Studio 2022 defaults to x64 either way). Decide on one invocation so a fresh clone and an existing tree agree.
- [ ] Human-review the machine-translated Japanese strings for the Avatar panel (`avatarPanel`, `windows.avatar`, `mainWindow.viewAvatarPanel` in `resources/localization/ja.json`).
- [ ] A redistributable VRM 1.0 sample under `models/avatars/` for live and sample-test coverage of the 1.0 path (the synthetic test is the only 1.0 coverage; both VRoid samples and Bonjiri are 0.x).
- [ ] MToon features the renderer skips: outline, rim and matcap, UV animation; plus morph targets (expressions) and spring bones (hair and cloth physics), which also need the VRM extension parsing extended.
- [ ] Avatar load is synchronous on the main thread (well under a second for a VRoid file); move the parse and image decode to a worker if large models stall the UI noticeably.
- [ ] T-pose capture for the avatar rig: derive the head and hand rotation trims from a held reference pose instead of setting them by hand.
- [ ] The avatar shade for non-MToon materials and the single key light are fixed in the renderer; expose light direction and ambient if the preview needs tuning.

## Open questions

- A fully tracked Mikan-format frame (~1.6 KB) exceeds a single 1472-byte UDP payload and would rely on IP fragmentation on a real network (localhost is unaffected). Chunk it like VMC mode, or keep the one-bundle-per-frame contract and accept fragmentation?
- Capture-phase GPU path, if the profile justifies one: OpenCV's OpenCL T-API (`cv::UMat`, no shader code, still a CPU tensor at the ONNX boundary) or a D3D12 preprocess feeding DirectML through IoBinding (custom shaders, no readback, the larger job)?
