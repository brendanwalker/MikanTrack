# Experiments and Learnings

MikanMediaPipe went from empty repo to a working multi-camera hand tracker between late July and mid August 2026, developed almost entirely in AI pair-programming sessions (Claude Fable 5 and Opus 5, see `TOKEN_STATS.md`). Most major experiments ran on their own branch and were merged when they won or parked when they lost. This document records what was tried and what each experiment taught us. It is a companion to `README.md`, which describes what the app IS. This describes how it got that way, including the dead ends.

## Timeline at a glance

| Branch | When | Experiment | Outcome |
|---|---|---|---|
| (main) | Jul 29-30 | ONNX Runtime + DirectML instead of the MediaPipe framework | Foundation, kept |
| (main) | Jul 30 | BlazePose for elbow tracking | Removed, never fired on overhead views |
| multi_camera | Jul 30-31 | Multi-camera fusion of landmark positions | Kept, then superseded by pose-level fusion |
| multi_camera_body | Jul 30 | Hand-seeded body-pose ROI fallback | Failed live, removed |
| (main) | Jul 31 | Parametric hand model (palm transform + finger angles) | Kept, became the core representation |
| (main) | Aug 1 | Diagnostic dump system + measured-jitter confidence | Kept, changed how every later bug was fixed |
| multi_camera_3d | Aug 1 | Stereo triangulation as the primary 3D lift | Kept |
| rtm-pose | Aug 1 | RTMPose hand landmarks as 2D refinement | Won its A/B, parked anyway (cost/benefit) |
| stereo_camera | Aug 2 | RealSense D455 hardware depth | Kept as an option, rig later moved to dual wide-FOV webcams |
| imu | Aug 2-9 | Wrist IMUs (Joy-Cons) for forearm orientation | Kept |
| image_metrics | Aug 9-10 | Image-quality diagnostics + record/replay | Kept |
| extrinsic_calibration | Aug 10-11 | Charuco-board extrinsics, all cameras in one session | Kept |
| (main) | Aug 11 | Depth A/B tool, marker-scale discovery, bone calibration | Kept |
| seed_fix | Aug 12 | Cross-camera seeding instrumentation + fixes | Kept, seeding now unconditional |

## 1. ONNX Runtime + DirectML instead of the MediaPipe framework

The obvious path was building Google's MediaPipe C++ framework. It was rejected before writing code: on native Windows it is CPU-only and requires a Bazel build. Instead we took the OpenCV Zoo ONNX conversions of the MediaPipe models and ran them through ONNX Runtime with the DirectML execution provider, reimplementing the two-stage detector-to-landmark graph (SSD anchors, weighted NMS, rotated crops, ROI tracking) in C++ by porting the opencv_zoo Python demos.

The port was validated numerically against the Python reference to 0.0002 px agreement before any live testing. That upfront rigor paid off repeatedly: every later tracking bug could be assumed to be OUR logic or OUR calibration, never the inference stack.

Learned: porting a model graph is far cheaper than binding a framework, if and only if you cross-validate numerically against the reference implementation first. One trap worth recording: `cv::copyMakeBorder` on a ROI view silently reads pixels outside the ROI unless `BORDER_ISOLATED` is set.

## 2. BlazePose elbows: a model outside its training distribution

We wanted elbow positions for client-side arm IK, so we ran BlazePose alongside the hand models. Its person detector simply never fires on a top-down desk view of a torso-less human. A fallback that seeded the pose ROI from the tracked hands (branch multi_camera_body) also failed live. BlazePose was removed entirely; purely geometric elbow estimates (forearm direction from hand orientation, length from hand scale, table-plane clamp) did better, and eventually even those were dropped in favor of streaming just the palm transform and letting the client solve arms with Two-Bone IK. The wrist IMU work later restored a real forearm measurement.

Learned: a model trained on upright people contributes nothing to an overhead rig, no matter how it is coaxed. When the sensor geometry is outside a model's training distribution, simple geometry beats the model. Also: not estimating something is a valid design. Deleting the elbow estimate made the OSC contract more useful, not less.

## 3. Multi-camera fusion and the hand-identity wars

Adding a second camera was the single most consequential feature, and hand identity (which physical hand is left vs right) was its hardest problem. It took roughly five distinct diagnoses, each driven by a diagnostic dump of a live failure, to get stable:

- Per-camera left/right labels are unreliable whenever a camera sees only one hand. Sides must be assigned at the fusion level by clustering observations in world space. Trusting per-camera labels collapses both physical hands into one slot.
- After a clap, both of one camera's tracking slots converge on the same physical hand and confidently track it, leaving the separated hand untracked with no free slot. Duplicate-slot detection (hand-box IoU over consecutive frames) is required.
- Monocular depth error lies along the camera's view ray and reaches 15-25 cm on this rig, comparable to the separation between hands. Position alone therefore CANNOT solve cross-camera correspondence when hands are near each other. Clustering must be ray-aware, and pairing must be a joint assignment that weighs label votes, not greedy nearest-neighbor.
- Slot labels get physically swapped by ROI hijacks. Labels are bookkeeping, not evidence. The classifier's raw score on the hands a camera sees well IS evidence, even while its labels are wrong.
- MediaPipe's presence score measures "is a hand here", never "is this the left or right hand" and never "is this pose good". Resolving a side collision by presence displaced a decisively-classified hand on what amounted to noise. Decisiveness of the handedness score is the right tiebreak.

Learned, beyond the mechanics: every one of these was diagnosed from a recorded dump, not from staring at live behavior, and two early hypotheses (a mirrored-camera theory, a "fusion is broken" report that was actually a preview-overlay artifact) were disproven by dump data. The general rule that emerged: per-camera overlays show per-camera state; always check the fused output before believing a failure report.

## 4. The parametric hand model

The first fusion approach blended 21 3D landmark positions per hand across cameras. When cameras disagreed about articulation, blending positions distorted bone lengths and produced hands no skeleton could make. The replacement, modeled on Ultraleap's representation, is a palm transform plus per-finger bend angles over a fixed skeleton, with fusion operating at the pose level (position blend, hemisphere-aligned quaternion blend, angle blend, later angle SELECTION). Forward kinematics from the streamed values is what clients rebuild, and the app's 3D view renders exactly that reconstruction.

Learned: choose a representation in which every reachable value is a valid hand. Blending is then safe by construction. The sign conventions (which way is positive curl, what zero means) were the genuinely hard part and only became stable once round-trip tests (extract angles, FK back, compare: 2e-7 rad) and an end-to-end FK reprojection metric pinned them down. That reprojection metric, kept as a live per-frame value, later became the key diagnostic separating "the angles are wrong" from "the hand really is bent".

## 5. Confidence is measured, not reported

A recurring, load-bearing negative result, found independently three times:

- MediaPipe presence does not correlate with pose quality. Binning frame-to-frame jitter by presence showed 12 mm median jitter at presence 0.85-0.95 and 26 mm at presence above 0.95. An edge-on, dim hand keeps presence near 0.9 while its depth solve is ill-conditioned.
- RTMPose's own confidence scores ran around 0.3 on landmarks that were WINNING the measured reprojection contest against MediaPipe's.
- Calibration reprojection error looked clean (0.42 px) on an extrinsics solve carrying a 1-degree orientation error that warped reconstructed geometry by 2.2 mm of corner spacing. Reconstruction-scale metrics catch what reprojection metrics cannot.

The fix in all cases was to measure quality directly: per-camera jitter (constant-velocity residual) drives confidence, triangulation residual drives stereo trust and can VETO a pairing, and FK reprojection measures articulation fidelity. Model confidence outputs answer "did the model find something", never "is the answer good".

## 6. The models are view-dependent and the hand is not canonical

Two related discoveries about MediaPipe's metric hand output:

- For the same physical hand at the same instant, two cameras' "world landmarks" disagree about articulation by 25-41 degrees. Rest-pose calibration therefore has to be captured per camera, and the bias was later confirmed to be pose-dependent, not a constant offset.
- MediaPipe's canonical hand model does not match a real hand's proportions. Measured against stereo triangulation of the user's actual hand, the model's proximal phalanges were short by a factor of about 1.8, and the model has an index distal phalanx longer than its proximal, which no human hand has. Since solvePnP rescales the model on a single bone and then fits all 21 points, a proportionally wrong model biases every monocular depth estimate. Per-bone skeleton calibration from stereo data fixed it, and immediately lifted the cross-camera seeding success rate from 32% to 88%.

Learned: a "metric" model output is an opinion, not a measurement. Calibrate anything personal (rest pose, bone lengths, hand scale) from your own sensors. Related human fact: a hand resting on a keyboard genuinely holds 20-50 degrees of MCP flexion, so a rest-calibrated zero can never be a flat hand, by anatomy rather than by bug.

## 7. Stereo triangulation as the primary 3D lift

A study of the SystemAnimatorOnline project prompted a rework of the 3D pipeline: triangulate the 21 2D image points across cameras as the PRIMARY path, and never let network-estimated depth into the 3D solve when two views are available. The triangulation reprojection residual doubles as a correspondence test (high residual vetoes the pairing as two different hands) and as a confidence input. Monocular PnP became strictly a fallback, and articulation from multiple cameras is selected (with hysteresis) rather than blended.

Learned: when you have two calibrated views, classical geometry beats learned depth in accuracy, and its residual is a trustworthy self-diagnostic, which learned depth never provides. Splitting smoothing into separate palm and finger-angle filters (different cutoffs) came from the same study and removed a long-standing tuning tension.

## 8. RTMPose: winning the A/B and losing the cost-benefit

RTMPose-m hand landmarks were integrated as an optional 2D refinement over MediaPipe's, with MediaPipe keeping detection, ROI tracking and handedness. The A/B methodology matured over three rounds:

- Round 1 mixed RTMPose and MediaPipe points within a single triangulation pair. The two models place joint centers by different conventions, and the mismatch read as correspondence error. Comparisons must keep sources pure per solve and select per frame by measured residual.
- Round 2 appeared to show MediaPipe winning, but the confidence gate (0.35) sat mid-distribution for RTMPose and censored most of its frames: gate-survivorship bias. One camera-hand combination never passed the gate at all.
- Round 3, with the gate floored, showed RTMPose decisively better: it won 84-96% of per-frame residual contests, with the winning-side median residual at 1.84 px.

RTMPose was parked anyway. It roughly tripled per-camera inference cost (6.5 to 23 ms), and the user judged the tracking improvement too modest to pay for. The branch survives if wanted later.

Learned: an experiment can succeed technically and still lose on cost. Also two transferable A/B rules: audit what a quality gate censors before trusting a comparison, and expect different models to disagree on landmark conventions even when both are "the same" landmarks.

## 9. Calibration: where the silent errors live

Camera calibration produced more subtle bugs than any other subsystem:

- OpenCV's `CALIB_RATIONAL_MODEL` with k3/k4/k5 fixed leaves k6 free: a lone r^6 denominator term that fits sub-pixel while producing a degenerate undistortion (reported FOV of 180 or 8.6 degrees). The same latent bug existed in MikanXR. Plausibility gates on recovered FOV are now part of the wizard.
- Focal length is unobservable from fronto-parallel board views. The solver pins fx at exactly width/2 (a suspiciously round 90-degree FOV) instead of failing. The wizard now requires tilted captures with measurable keystone.
- Recalibrating at the same resolution kept stale undistortion maps, so a good calibration LOOKED broken. Rebuild derived state unconditionally on recalibration.
- Extrinsics captured per camera in separate sessions, with the marker nudged between, produce a silent world-frame disagreement that masquerades as tracking noise. The wizard was rewritten to calibrate all cameras in one session against one board placement.
- The biggest one: world scale was 30% short for most of the project, because extrinsics had been solved against a printed aruco marker ASSUMED to be 100 mm that actually measured 130 mm. Every internal quality metric validated the wrong scale, because they all compared the reconstruction against the same assumed size that produced it. The error was caught only when hardware depth (factory-calibrated, board-independent) disagreed with the stereo world by a consistent factor, and settled by the user measuring a hand bone with a physical ruler: ruler 85 mm, depth 82 mm, stereo 68 mm. The two board-free measurements agreed; the board-derived one was wrong. Fixing the marker size snapped triangulated bone lengths to within 1.5% of the ruler.

Learned: calibration quality metrics that share an assumption with the calibration are self-referential and will happily validate a wrong world. Keep at least one measurement path that does not depend on the calibration (hardware depth, a physical ruler), and when two internal numbers disagree, reach for it immediately. Also, a diagnosis arc from this incident: the depth camera was first accused of reading 24% long, then the desk was accused of contaminating depth samples, and both theories fit the histograms. Only the independent measurement discriminated. Two wrong-but-plausible readings preceded the right one, and the dump-everything habit is what kept each theory cheaply testable.

## 10. Hardware depth: real value, initially masked

A RealSense D455 was added as a backend (dynamically loaded SDK, depth-resolved palm anchors replacing PnP on its camera). Three results:

- A single overhead D455 replacing the two-webcam rig produced mushier fingers. Depth helps the palm; articulation still needs two angular viewpoints. The two-camera rig stayed.
- The first depth on/off A/B (a replay tool that re-runs a recorded session both ways over identical motion) showed depth buying almost nothing. That result was an artifact of the 30% world-scale error above, which was punishing the depth path for being right.
- On the corrected rig, the same A/B showed depth's real contribution: about 17% more tracked frames, 5 points of stereo rate, 15% confidence, and 17-19% lower p90 jitter, concentrated in the 10-15% of frames where fusion falls back to one camera.

The rig nevertheless moved to dual wide-FOV webcams. The user's live observation that a 90-degree FOV camera "surprisingly" improved tracking held up: field of view keeps both hands in both frusta, which raises the stereo rate, and stereo frames never read depth anyway. After bone calibration and the seeding fixes, the dual wide-FOV rig hit 93.3% stereo share against the RealSense rig's ~95%, without the depth hardware.

Learned: quantify what a sensor buys before paying for it, on a correctly calibrated rig, with identical-motion replays. And FOV is upstream of almost everything in a stereo hand tracker: a frame that keeps both hands in both views avoids the entire monocular fallback problem that depth existed to soften.

## 11. Wrist IMUs: model the joint you actually measure

Joy-Cons on wrist straps feed a 6-axis orientation ESKF per wrist. Design decisions that mattered:

- A wrist-strap IMU rides the FOREARM, not the palm. Modeling it as forearm orientation, with the wrist joint recovered as palm-relative-to-forearm, restored the forearm information vision measures poorly and enabled elbow visualization again.
- Orientation only. Accelerometer double-integration diverges, and vision already delivers millimeter-level position.
- Without a magnetometer, yaw is unobservable, and (same physics) so is gyro bias about the gravity axis. The filter test suite documents this observability structure explicitly. Vision anchors yaw through an anisotropic measurement update that corrects ONLY yaw, because vision measures the palm and a full-orientation correction would fight real wrist motion through the unknown joint between the sensors.
- When the filter owns orientation, the downstream one-euro smoothing for that quantity is disabled. Cascading filters double-smooths and adds lag.
- Mounting calibration evolved from "hold a pose" to solving the strap orientation from a twist-and-curl motion, scored against real forearm twist. Held poses under-constrain the solve.

Measured along the way: Joy-Cons stream at exactly 200 Hz over Bluetooth HID, resting gyro bias on one unit implied 114 degrees/minute of yaw drift (stable, hence estimable), and the left and right units mount their IMUs with opposite orientations, which is why mounting calibration cannot be hardcoded.

## 12. Look at the image: quality metrics for the WHY layer

By August the project had good outcome metrics (jitter, residuals, reprojection) but nothing explaining WHY a camera was bad. Per-hand image statistics (luminance, clipping, contrast, background separation, sharpness, sensor noise, flicker detection), computed on the exact pixels the model consumed, filled that layer. First live use attributed one camera's 4x worse jitter to dim exposure and 2x sensor noise; turning up a ring light halved that camera's jitter p90 in a controlled before/after.

Methodology lessons: raw Laplacian variance scores a noisy frame as "sharp", so sharpness (after median filtering) and noise (the median residual) must be split into separate metrics. And free-form hand motion cannot A/B jitter, because the jitter metric reads real acceleration as noise; a scripted hold-still test exists for exactly that comparison, while the image metrics are motion-independent.

## 13. Record/replay: the highest-leverage tool in the project

Deterministic recording of all post-inference inputs, with bit-exact offline re-simulation (checksummed, verified per frame), turned live-only tracking bugs into desk work:

- The first replay-driven fix: a stereo "pop" the user reported as frame 348 of a recording was diagnosed in minutes (a frame stutter pushed one camera past the staleness window, dropping fusion to the wrong monocular camera) and the fix was validated by replaying the SAME recording: a 212 mm pop became 5 mm.
- The stutter itself traced to something nobody suspected: IMU device rescans running Windows HID enumeration (measured 237 ms) on the vision thread every 151 iterations. It was never USB. The periodicity was visible only because recordings made the loop timing inspectable. Discovery moved to a worker thread, and a per-phase hitch watchdog now names any future stall.
- What-if replay (same recording, edited config, different extrinsics, depth on/off) became the standard A/B harness. Both the depth verdict and the marker-scale discovery ran on it.

Learned: in a real-time perception system, determinism is a feature you build early and lean on constantly. The prerequisite discipline (no wall clock or randomness in the replayed stages, explicit transient-state reset) is cheap compared to what it buys. One expectation to set: after a legitimate fusion change, mass checksum divergence against old recordings IS the diff of the fix, not a bug.

## 14. Cross-camera seeding: instrument the funnel before tuning it

Seeding (projecting a hand one camera lost into the other camera's image to skip the palm detector) existed for weeks and appeared to work. Reacquisition latency histograms said otherwise: a cluster of tails sat exactly at the palm-detector period, meaning seeding had silently not fired. Rather than tuning, we added counters at every gate a seed can die at (candidates, offered, redundant, no-free-slot, applied, model-rejected). One live session of counters exposed two independent defects: most hints were overwritten before the pipeline ever evaluated them (a producer/consumer rate mismatch), and the survivors lost the free slot to the palm detector that ran first. Fixing both took reacquisition from 160 ms median to 32 ms (one frame). The counters then caught a third issue for free: seed projections missing the hand, which turned out to be the bone-calibration error, not the seeding code.

Learned: for any multi-stage pipeline with drop points, instrument the funnel first. Each fix here was obvious once the counters localized the loss, and unmeasurable before. Postscript: once seeding was verified, its config toggle and UI were deleted. Settled experiments do not keep their switches, a cleanup rule applied across the project (depth estimators, blend-vs-select, smoothing variants all had their losing arms removed).

## Cross-cutting lessons

1. Measured quality beats reported confidence. Presence scores, model confidences and self-consistent calibration metrics all failed as quality signals. Reprojection residuals, measured jitter and reconstruction-geometry checks did not.
2. Keep one measurement path independent of your calibration. The 30% world-scale error survived every internal metric and fell to a ruler.
3. Record everything, replay deterministically. The dump system (built week one) and the replay system (built week two) drove nearly every diagnosis afterward. The pattern was always: user cites a timestamp or frame, read the data, form a theory, test it against the same data.
4. Labels are bookkeeping, evidence is measurement. Slot labels, preview overlays and classifier argmaxes all lied at some point; raw scores and geometry did not.
5. A/Bs have their own failure modes: confidence gates censor (survivorship bias), free motion confounds jitter, mixed conventions read as error, and a miscalibrated rig can invert a verdict. Identical-motion replays and scripted tests are the antidotes.
6. Models carry their training distribution with them: overhead views broke the person detector and degraded the handedness classifier, and the canonical hand matched nobody's hand. Geometry plus per-user calibration filled every one of those gaps.
7. Winning the benchmark is not winning the argument. RTMPose won its A/B and was still parked on cost. The two-camera rig beat a depth camera on articulation. Field of view beat both.
8. Delete the losing arm. Every settled A/B ended with the toggle and the dead code removed, which is why the codebase still reads like a design rather than an archaeology site.
