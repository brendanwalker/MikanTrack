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
| stereo_camera | Aug 2 | RealSense D455 hardware depth | Kept as an option; deleted Aug 16 once the estimator's mono prior covered its niche |
| imu | Aug 2-9 | Wrist IMUs (Joy-Cons) for forearm orientation | Kept |
| image_metrics | Aug 9-10 | Image-quality diagnostics + record/replay | Kept |
| extrinsic_calibration | Aug 10-11 | Charuco-board extrinsics, all cameras in one session | Kept |
| (main) | Aug 11 | Depth A/B tool, marker-scale discovery, bone calibration | Kept |
| seed_fix | Aug 12 | Cross-camera seeding instrumentation + fixes | Kept, seeding now unconditional |
| body_pose | Aug 12-14 | Opt-in per-camera body pose (elbows, shoulders, head) | Kept; landmark model swapped BlazePose -> RTMPose |
| biomech | Aug 15-16 | Angle-space multi-view state estimator + biomechanical priors | Live-verified, made permanent; superseded holds/hysteresis/one-euro deleted |

## 1. ONNX Runtime + DirectML instead of the MediaPipe framework

The obvious path was building Google's MediaPipe C++ framework. It was rejected before writing code: on native Windows it is CPU-only and requires a Bazel build. Instead we took the OpenCV Zoo ONNX conversions of the MediaPipe models and ran them through ONNX Runtime with the DirectML execution provider, reimplementing the two-stage detector-to-landmark graph (SSD anchors, weighted NMS, rotated crops, ROI tracking) in C++ by porting the opencv_zoo Python demos.

The port was validated numerically against the Python reference to 0.0002 px agreement before any live testing. That upfront rigor paid off repeatedly: every later tracking bug could be assumed to be OUR logic or OUR calibration, never the inference stack.

Learned: porting a model graph is far cheaper than binding a framework, if and only if you cross-validate numerically against the reference implementation first. One trap worth recording: `cv::copyMakeBorder` on a ROI view silently reads pixels outside the ROI unless `BORDER_ISOLATED` is set.

## 2. BlazePose elbows: a model outside its training distribution

We wanted elbow positions for client-side arm IK, so we ran BlazePose alongside the hand models. Its person detector simply never fires on a top-down desk view of a torso-less human. A fallback that seeded the pose ROI from the tracked hands (branch multi_camera_body) also failed live. BlazePose was removed entirely; purely geometric elbow estimates (forearm direction from hand orientation, length from hand scale, table-plane clamp) did better, and eventually even those were dropped in favor of streaming just the palm transform and letting the client solve arms with Two-Bone IK. The wrist IMU work later restored a real forearm measurement.

Learned: a model trained on upright people contributes nothing to an overhead rig, no matter how it is coaxed. When the sensor geometry is outside a model's training distribution, simple geometry beats the model. Also: not estimating something is a valid design. Deleting the elbow estimate made the OSC contract more useful, not less.

Postscript (branch `body_pose`, Aug 12-13): a third, front-facing camera was added to the rig, which puts an upright person squarely back inside the model's training distribution. BlazePose returned as an OPT-IN PER-CAMERA stage - the overhead cameras leave it off and pay nothing - now feeding measured elbows, shoulders and head pose. The removal was still correct: the rejection was of the model on that geometry, not of the model. Three lessons landed from the first two live sessions.

First, the elbow could not be re-solved on every fused frame: the pose models run near 10 Hz while wrists update at 30 Hz, so re-intersecting a held observation against a moved wrist slid the elbow along a ray cast up to 200 ms earlier. A repeated observation now holds its forearm direction and lets the elbow ride the fresh wrist.

Second, confidence-from-measured-jitter has to be normalized by the sampling interval: the hands' plain-distance residual assumes a fixed rate, and applied to a variable ~10 Hz signal it mostly measured how long the gap was, scoring every elbow at 0.01. Dividing by dt squared turns it into an acceleration, which means the same thing at any cadence.

Third, and the one that was actually causing the visible popping: anchoring the elbow to a forearm-length sphere around the wrist leaves a camera ray intersecting that sphere TWICE, roughly 400 mm apart, and we were choosing between them with the model's own elbow-versus-wrist depth. That sign flips on 28-42% of model frames because its magnitude (24-46 mm median) sits inside its own noise, so the elbow teleported at random. Raising a threshold on it does not help - even the top 5-19% of samples by magnitude still flip 10-17% of the time. Continuity decides it instead (the forearm cannot swing 400 mm between model results, giving 0-1% flips), with the model depth demoted to seeding the first solve and an independent shoulder-length check as a sustained-evidence escape hatch. The general lesson: when a geometric solve has a discrete ambiguity, resolve it with the strongest signal available, and prefer temporal continuity over a per-frame measurement whose noise is comparable to the quantity being measured.

Then the model itself lost. Even with the elbow solve fixed, the landmarks under it were poor, and the reason was structural rather than tunable: BlazePose is holistic, deriving its own crop from a hip center and a "full body" point. A person truncated at a desk has no correct crop, so it fabricated a lower body and dragged the arms with it, while scoring the shoulders 0.9998 as they jittered 30-40 px. The landmark model was swapped for RTMPose-m (body7), which is top-down: this app supplies the person box, and each of the 17 COCO keypoints is scored independently, so joints outside the frame read as low confidence instead of being invented. Same frames, same session, the two measured: 99% of frames usable against 70%, elbow 2D step medians of 6.7 and 7.4 px against 53 and 74 px, nose 3.9 px against 23 px. BlazePose was then removed outright. Two things made that decision cheap. Raw frame recording (opt-in, off by default) preserves the model's actual input, so "would a different model have done better" became a replay measurement rather than an argument. And the box handed to the top-down model has to be built from the person detector's KEYPOINTS, not its own box, which is a face box - feeding that to RTMPose cropped the arms away.

Learned: a confidence score is a statement about the model's own crop, not about the answer. When a model owns its region of interest, its assumptions about the scene are non-negotiable, and a scene that violates them cannot be fixed downstream.

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

Postscript (Aug 16): with the state estimator permanent - whose anisotropic mono prior softens the same monocular-depth problem the sensor targeted - the RealSense backend, the depth-sampling path, and their recording/replay plumbing were deleted entirely under the settled-experiment rule. The measured verdicts above are the record of what the hardware was worth.

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

## 15. The waving hand: a solver degeneracy hiding behind a plausible theory

A third camera joined the rig and the right hand would not stay tracked while waving (53.7% fused tracking, dropouts up to 1.7s). The replay harness localized the loss precisely: the camera with the BEST right-hand view (presence 0.99, decisive handedness) was contributing nothing to fusion, because its monocular PnP solve collapsed to a 2-4 cm or negative depth on 412 frames and the sanity gate rejected every one. The first theory (a poisoned palmar-side memory mirroring the object model) was plausible, mechanistic, and testable, and a flip-and-retry experiment refuted it: 6 rescues out of 412. The real cause was geometry: a flat waving hand posed by FK is a NEAR-planar PnP object model, and ITERATIVE's cold-start DLT is ill-conditioned near planarity. Exactly planar routes to homography initialization and is fine; near-planar is the worst case. The failure self-sustained because each rejected frame cleared the warm start, so every solve was a cold solve. Switching the cold path to SQPnP removed 406 of 412 collapses. Synthetic flat hands with noise would not reproduce the collapse, so the regression test embeds a real collapsed frame from the recording, which fails under ITERATIVE and solves to a plausible depth under SQPnP.

The second, compounding cause was configuration: fusion.minCameraConfidence had been raised to 0.2 (default 0), and a fast-waving hand's constant-velocity jitter residual crushes confidence exactly when tracking it matters most, demoting high-presence observations to the rescue pool. Replay what-ifs attributed the two cleanly: config alone 54% to 88%, SQPnP on top 95.3%, dropouts 20 to 7. A live session with both fixes confirmed the replay's prediction: 92.8% right-hand tracking, and every remaining gap showed the hand at frame edges in all three views at once.

Learned: when a solve fails persistently, suspect the solver's degenerate configurations before the inputs. And a hard confidence cutoff turns a soft-weighting system into a gate that censors precisely the fast-motion frames the jitter metric mismeasures (lesson 5's censoring bias, in config form).

## 16. The state estimator: replacing selection seams with one continuous fit

The remaining pops all shared one shape: a discrete per-frame selection changing between frames (tri-vs-mono path, stereo pair choice, articulation source, palmar sign), each patched by its own hold or hysteresis. Branch `biomech` replaced the selections with one temporally continuous 26-DoF state per hand (palm transform + 20 raw finger angles) fit every fuse to ALL cameras' 2D landmarks by FK reprojection, damped Gauss-Newton with a constant-position prior, the classic path demoted to seed and fallback. Measured over a 5-recording corpus with a purpose-built pop-metrics replay tool: palmar flips 8 to 0, palm pops over 100 mm 5 to 0, worst single pop 288 to 90 mm, and the FK reprojection residual (the anti-cheat metric, since a frozen hand has zero pops) HALVED at the same time. The estimator both explains the pixels better and pops less, which is what distinguishes a structural fix from a smoothing knob.

Three designs died on measurement inside one session. Fitting only the freshest camera's rows ping-ponged the state between the cameras' systematically disagreeing solutions at fuse cadence (9-10 mm median step on a quiet recording); the fix was fitting every observation inside the staleness window every fuse, age-weighted, which is sound because a per-fuse MAP re-fit of unchanged rows from a converged state stays converged. An instant divergence drop turned every marginal frame into a drop/classic-snap/reseed pop cycle; one bad fit now holds the previous state and only a sustained streak reseeds. And a mono stretch drifted with the single camera's scale-implied depth then snapped 106 mm when a second camera returned; the position prior now goes anisotropic under a single-camera fit, tight along that camera's view ray - the continuous generalization of the old tri position hold, and the anisotropic-covariance step the EKF roadmap note predicted.

The first out-of-sample live recording added a fourth: a solo wrong-side observation (a mislabeled cluster force-assigned by elimination while the real hand was unobserved for one fuse) teleported the palm 0.7 m at a 15 px fit residual - perfectly consistent pixels of the wrong hand, invisible to every quality gate. The fix is physical rather than statistical: a dt-scaled innovation gate (4 m/s ceiling) routes impossible steps into the bad-fit hold path, so a genuine relocation still lands via the reseed within a few fuses and the gate can delay truth but never censor it (the failure mode that killed the hard confidence cutoff in lesson 15).

The deeper issue was upstream: side assignment handing a cluster to a side by ELIMINATION even when its own affinity said no. A survey across three recordings made the rule obvious - healthy assignments never scored below +2.05 total affinity, while all 14 negative-affinity assignments were mislabeled or phantom clusters that stepped the fused hand 256-782 mm, every one a transient in the middle of a well-tracked stretch, none a reacquisition. Fusion now refuses an assignment whose raw affinity is negative WHILE THE SIDE WAS RECENTLY TRACKED (250 ms window): the side goes untracked for a fuse instead of adopting the wrong cluster. The window is the load-bearing half - after a real dropout the temporal prior is stale testimony and the same cluster is a legitimate reacquisition, so refusal expires and a deadlock is structurally impossible. The refusal fixed the CLASSIC path's worst pops too (288 mm max step to 149 on the hardest recording), for a tracked-percentage cost of 0.1-0.7. Learned: when a forced choice has no good option, refusing to choose for a frame is cheaper than either error - but only with an expiry that keeps the refusal from becoming a censor.

Biomechanical priors then landed as residuals on the state, cheapest layer first. Anatomical joint limits (one-sided soft penalties, zero cost in range) collapsed the worst implausible excursions from 34-94 degrees beyond range to 13-27 at zero FK-residual cost. A DIP-PIP coupling fallback holds a garbaged fingertip to anatomical coherence (46 to 20 degrees of deviation in the synthetic test). A per-user angle prior (mean + shrunk precision over the 20 raw angles, fit by `--fit-angle-prior` from the user's own stereo frames) cut single-DoF angle jumps a further 20% on the busiest recording by making coordinated pose changes cheap and uncorrelated snaps expensive; sweeping its weight 0.3 to 1.0 changed little, so it ships weak. A musculoskeletal layer (muscle dynamics) was surveyed and deliberately skipped: it buys torque plausibility, which a kinematic tracking prior at 30 Hz never consumes.

Learned: when every mitigation is a hold or a hysteresis on a discrete switch, the switch itself is the bug - estimate one continuous state and the seams disappear along with their patches. And a prior's test scenario must match its failure mode: the fitted angle prior looked useless against a persistent stereo-consistent snap (correctly - that could be a real unusual pose) and earned its keep exactly on view-inconsistent corruption. A synthetic training set with independent per-finger motion taught the prior that lone-finger curls are normal, and it then correctly refused to fix them; the test had to encode real hands' shared-synergy structure before it could measure anything.

A hard pose - both index fingers pointing and touching - then exposed two things the estimator had inherited wrong. On that rig cam0 and cam1 see the two index tips 1-2 px apart (cam2, the front camera, separates them by 33+ px), so their landmark sets mix the hands; cam0's right-hand palm jitter measured 85-126 mm against cam2's 2.8 mm, and the fusion confidence signal knew it (0.12 vs 0.98). The estimator did not: it weighted measurement rows by the model's reported PRESENCE, which is high on all three. The project's most-repeated lesson, violated in new code. Symptom chain: garbage rows held the fit residual at 25-30 px (healthy is 5-8), which tripped the divergence guard, which after five held frames dropped the state, which fell back to the classic path - and the classic path has no joint limits, so a -148 degree middle-finger angle reached the output. The limits were never violated by the estimator; they were bypassed by its fallback.

Weighting rows by measured confidence fixed the fit, but half the bug survived: the guard judged an UNWEIGHTED mean residual, so a camera the fit had correctly discounted could still trip it. Weighting the residual by the same row weights closed it. Measured on that recording: right-hand angle jumps 125 to 23, bad-fit holds 110 to 36, classic fallbacks to zero, the -148 degree excursion gone (worst violation now 9.9 degrees). One measurement caveat worth keeping: the FK-reprojection anti-cheat metric got WORSE (17 to 21 px) because it averages agreement over all cameras, and agreeing with a camera whose landmarks belong to the other hand is not a virtue - a metric that assumes every camera is valid ground truth stops being one when a camera is provably wrong.

The same pass caught a self-inflicted regression: the estimator had been overwriting the fused `worldPoints` with its own forward kinematics, and bone calibration samples that field - so calibration had quietly become circular, re-measuring the skeleton it was handed instead of the hand. Nothing rendered from the field (the 3D view builds FK from the streamed pose), so the fix was simply to stop overwriting it and let bone calibration ask for the triangulated landmarks explicitly. Learned: when a new estimator starts writing into a field, check who was READING it - a fitted quantity silently replacing a measured one is invisible until a calibration stops converging.

After live verification the estimator became the permanent output path, and the settled-experiment cleanup rule deleted its losing arms wholesale: the experimental toggle, the tri angle and palm-depth holds, the articulation-source hysteresis, and the post-fusion one-euro cascade for hands (dead by construction once estimator poses bypassed it) - along with their config keys, UI, and tests, whose guarantees moved to the estimator's own test suite. The classic extraction survives only as the estimator's seed, its fallback, and the output for uncalibrated cameras.

## 17. A missing joint became a broken joint one bone down

Elbows cut out whenever one hand crosses in front of the other elbow, and in VSeeFace the hand appeared to spin about 180 degrees around the wrist each time - while the app's own 3D view stayed perfectly stable. Two facts explain the whole thing. The solver takes forearm ROLL from the palm and only DIRECTION from the elbow, so a missing elbow cannot produce a twist. And the VMC retarget gated both arm bones on having an elbow while streaming the hand unconditionally: a receiver leaves an unstreamed bone at the avatar's REST pose, which for an arm is the T-pose, so the arm snapped sideways while the hand kept its correct world orientation, and the arm's entire rotation surfaced at the only joint left to absorb it. The app's 3D view was unaffected because it draws the hand from the pose and never walks the arm chain - the same output looking fine in one consumer and broken in another was the clue that the bug was in the retarget, not the tracking.

Two fixes, and the second matters more than the first. The arm chain is now always streamed: with no elbow the upper arm aims straight at the wrist and the forearm takes the hand's own frame, so the wrist reads neutral instead of carrying the arm. And the elbow is no longer dropped when its landmark is occluded - it is re-solved on the same bone circle the measured path uses (shoulder sphere meeting wrist sphere), with the point picked by continuity instead of by the missing landmark ray. Holding the last elbow POSITION was the obvious alternative and is wrong for a measurable reason: the wrist keeps moving, so a held position silently changes the forearm's length (37 mm in the test's own numbers). Re-solving on the circle holds both bones exactly by construction - measured live at 0.8 mm upper-arm error across 395 dead-reckoned frames - because the circle IS the two length constraints. Confidence decays over a one-second budget and deliberately excludes the jitter tracker, since a dead-reckoned elbow is smooth by construction and scoring its smoothness would report high confidence exactly when the value is inferred. Result on the recording: left-arm dropouts 7 to 0, forearm validity 88.6% to 98.2%, with the only remaining gap before the first elbow was ever seen.

Learned: when a consumer composes a hierarchy, a missing estimate is not a local absence - it lands on whichever child is still being streamed. And when the thing you cannot see is constrained by lengths you know, re-solve it under those constraints rather than holding its last value; holding is only safe for quantities nothing else moves.

## 18. A smoothing filter inside a discrete choice

The left elbow spent the last 2.5 seconds of a session parked at shoulder height, 25 cm from where the recorded frames put it, and it stayed there while the 2D landmark returned to exactly the pixel it had started at. Starting a new recording always cured it, which is the signature of state rather than input. The elbow is picked from the circle where the two bone spheres meet, and both points on that circle are anatomically valid arms, so something has to choose: continuity does, by predicting the elbow from the direction held since the last model frame. That held direction was the SMOOTHED one, which is also the output. Through a fast swing the smoothed direction still points at the pose the arm has already left, so the root matching the old pose keeps winning, and once the choice lands on the wrong point the filter converges onto it and endorses it harder. Separating the two - predict from the unsmoothed direction, publish the smoothed one - restored the arm: median elbow height went from 31 mm below the shoulder to 282 mm, against the other arm's 307 mm, and every other recording in the corpus replayed bit-identical.

The second half of the fix was tried, measured, and thrown away. If continuity can take a wrong root, it seems obvious that the image should be able to take it back: let the elbow ray overrule the held choice when the other circle point is decisively nearer it for several consecutive frames. It fires correctly on the recording that motivated it, where the correct point sat 10-19 mm from the ray against the held point's 24-32 mm for 35 straight frames. On an earlier recording, with both elbows plainly down in the frames, the same ray preferred the shoulder-height root for 7 consecutive frames by 15-36 mm - and there is nothing to threshold on, because the two cases' margins and their absolute fits overlap. Shipping it would have cost that arm 32 jumps over 100 mm to buy nothing anywhere else in the corpus.

Learned: never put a smoothing filter in the feedback path of a discrete decision - smoothing is for what you publish, not for what you decide from, and a lagging predictor does not merely delay a decision, it can latch the wrong one permanently. And an arbiter has to be shown to arbitrate: the elbow ray is the only new evidence each frame, which made it look like the natural tiebreaker, but measured against ground truth it is right often enough to convince and wrong often enough to be unusable.

## Cross-cutting lessons

1. Measured quality beats reported confidence. Presence scores, model confidences and self-consistent calibration metrics all failed as quality signals. Reprojection residuals, measured jitter and reconstruction-geometry checks did not.
2. Keep one measurement path independent of your calibration. The 30% world-scale error survived every internal metric and fell to a ruler.
3. Record everything, replay deterministically. The dump system (built week one) and the replay system (built week two) drove nearly every diagnosis afterward. The pattern was always: user cites a timestamp or frame, read the data, form a theory, test it against the same data.
4. Labels are bookkeeping, evidence is measurement. Slot labels, preview overlays and classifier argmaxes all lied at some point; raw scores and geometry did not.
5. A/Bs have their own failure modes: confidence gates censor (survivorship bias), free motion confounds jitter, mixed conventions read as error, and a miscalibrated rig can invert a verdict. Identical-motion replays and scripted tests are the antidotes.
6. Models carry their training distribution with them: overhead views broke the person detector and degraded the handedness classifier, and the canonical hand matched nobody's hand. Geometry plus per-user calibration filled every one of those gaps.
7. Winning the benchmark is not winning the argument. RTMPose won its A/B and was still parked on cost. The two-camera rig beat a depth camera on articulation. Field of view beat both.
8. Delete the losing arm. Every settled A/B ended with the toggle and the dead code removed, which is why the codebase still reads like a design rather than an archaeology site.
