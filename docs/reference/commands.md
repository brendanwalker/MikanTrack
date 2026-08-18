# Command Cheat Sheet

Handy commands for working in the MikanTrack repo, all run from the repo root unless noted. Background on the build system, dependency versions, and the model set is in [build.md](./build.md); how to read test/log output and the replay tooling workflows are in [debugging.md](./debugging.md).

---

## One-time setup

```
InitialSetup_x64.bat
```

Populates git submodules, wipes and re-downloads `deps/` (roughly 1.3 GB), and downloads the ONNX models into `models/`. See [build.md](./build.md) for exactly what it fetches.

---

## Generate project files

```
GenerateProjectFiles_X64_VS2022.bat
```

Configures `build/` with the `Visual Studio 17 2022` generator and produces `build/MikanTrack.sln`. Rerun after `InitialSetup_x64.bat`, and rerun any time a `.cpp` file is added to an existing `src/<Dir>` (the source list is a configure-time glob).

---

## Build

```
cmake --build build --target MikanTrack --config Release --parallel
```

Or open `build\MikanTrack.sln` in Visual Studio 2022 and build there. There is only the one target, `MikanTrack`.

---

## Run the app and its self-tests

The app exe doubles as the test runner: every self-test, hardware check, and headless tool is a flag handled before `App::exec` ever starts, and `MikanTrack.exe` must be run with its own directory as the working directory (see [build.md](./build.md)).

```
MikanTrack.exe --list-tests
```

Prints every registered command grouped by category (self-tests, hardware required, tools). Each command logs to `<flag-without-dashes>.log` next to the exe in addition to console output (`--test-fusion` writes `test-fusion.log`), and returns exit code 0 on pass.

### Self-tests (deterministic, no hardware or input files needed)

- `--loc-test`: localization tables: key parity, printf specifiers, window IDs, glyph coverage
- `--selftest`: OSC writer packet round-trip
- `--test-angleprior`: finger-angle prior fitting (mean, correlations, floors)
- `--test-bodycalib`: body dimension calibration recovers known landmark separations
- `--test-bodypose`: body-pose solver: rays + known lengths, root choice, hold, gates
- `--test-bonecalib`: hand bone calibration from triangulated landmarks
- `--test-charuco`: Charuco board finder + synthetic intrinsics recovery
- `--test-dump`: diagnostic dump writer + schema
- `--test-extrinsics`: Aruco marker pose + table-plane raycast
- `--test-framerecorder`: raw frame recording: round trip, naming, overflow drops
- `--test-fusion`: multi-camera fusion, clustering, triangulation, holds
- `--test-handestimator`: angle-space multi-view hand state estimator
- `--test-handpose`: palm frame, finger angles, FK round-trip, conventions
- `--test-imudiscovery`: IMU device discovery never blocks the caller
- `--test-imufilter`: IMU orientation EKF, observability, gating
- `--test-pnp`: monocular PnP recovers a synthetic hand pose
- `--test-replay`: record/replay determinism, checksums, what-if
- `--test-roiquality`: hand ROI image-quality metrics + flicker tracker
- `--test-seeding`: cross-camera seed redundancy gate
- `--test-vmc`: VMC retarget: axis conversion, rest identity, chain round trip

### Hardware (need a camera or controller physically connected to mean anything)

- `--test-imuaxes`: live Joy-Con axis convention measurement
- `--test-joycon`: Joy-Con sample decode + live HID streaming
- `--test-posemodel`: body pose ONNX models load + match the reference decode

### Tools (headless diagnostics that operate on a file the user names)

- `--calibrate-bones`: measure hand bone lengths from a recording's stereo frames
- `--export-board`: writes + opens the printable Charuco board PNG
- `--export-marker`: writes + opens the printable origin Aruco marker PNG
- `--fit-angle-prior`: fit the finger-angle prior from recordings' stereo frames
- `--replay-bodypose`: re-run body pose over a recording's raw frames
- `--replay-dump`: emits a recording frame range with regenerated fusion diagnostics
- `--replay-extrinsics`: A/B a candidate extrinsics config against a recording
- `--replay-popmetrics`: pop statistics of a recording: baseline vs hand estimator
- `--replay-verify`: re-runs a recording and verifies every frame checksum
- `--test-imupair`: solves the transform between two rigidly coupled IMUs from a dump (categorized as a Tool despite the `--test-` name, since it takes a dump file argument rather than running standalone)

---

## Token stats

```
python tools/token_stats.py
```

Regenerates `TOKEN_STATS.md` and the AI-usage badge line in `README.md` from local Claude Code session transcripts.
