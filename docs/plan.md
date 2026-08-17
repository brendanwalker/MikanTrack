# Plan

The living plan: what is in flight now, what comes next, and the open questions. Completed items are removed rather than checked off. Resolved questions are removed. Deferred work enters here at the moment of deferral.

## Now

- [ ] Finish the MikanTrack rename outside the source tree: rename the repo working directory and the GitHub repository, then re-clone or update the remote. The code, docs, and appdata folder already use the new name.
- [ ] Live-verify the `/mikan/hand/{s}/forearm` wire change end to end with an OSC consumer (both sides built and self-tested, not yet exercised live).

## Later

- [ ] Pre-open-sourcing cleanup: untrack the root `test-*.log` files and `imgui.ini`, remove the stale `20230831/` extraction residue and the empty `cmake/` and `tests/` directories, delete the orphaned `models/rtmpose_hand.onnx`, and fix the stale `NOTICE.md` entry for `ArucoMarkerPoseSampler` (now `PatternPoseSampler`).
- [ ] `InitialSetup_x64.bat` downloads `rtm_demo.jpg` to the repo root but no build step copies it next to the exe, so `--test-posemodel` silently skips its numeric cross-check. Fetch it into a copied directory or copy it post-build.
- [ ] Replace the IMU discovery poll with native device-arrival notifications (Win32 `CM_Register_Notification`, filtered on the Joy-Con VID), keeping a slow poll as the safety net.
- [ ] Refit the per-user angle prior (`--fit-angle-prior`) as recording coverage grows; the shipped weighting is deliberately weak.
- [ ] `GenerateProjectFiles_X64_VS2022.bat` passes `-A x64`, which conflicts with a `build/` cache configured without an explicit platform (Visual Studio 2022 defaults to x64 either way). Decide on one invocation so a fresh clone and an existing tree agree.

## Open questions

- A fully tracked Mikan-format frame (~1.6 KB) exceeds a single 1472-byte UDP payload and would rely on IP fragmentation on a real network (localhost is unaffected). Chunk it like VMC mode, or keep the one-bundle-per-frame contract and accept fragmentation?
