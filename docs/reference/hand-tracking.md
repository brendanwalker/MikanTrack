# Hand tracking

The hand pipeline end to end: the two ONNX models and their inference wrapper, the per-camera MediaPipe-style tracking graph, the monocular 3D lift, the parametric hand pose model, multi-camera fusion, and the state estimator that produces the streamed pose. See [architecture.md](./architecture.md) for the threads these stages run on, [calibration.md](./calibration.md) for the intrinsics, extrinsics, hand scale, and bone measurements the 3D stages consume, [conventions.md](./conventions.md) for the coordinate frames, [body-pose.md](./body-pose.md) for the separate elbow/shoulder/head stage, [imu.md](./imu.md) for the forearm channel, [wire-protocol.md](./wire-protocol.md) for how the fused pose leaves over OSC, and [debugging.md](./debugging.md) for the diagnostic surfaces. Experiment history behind the design choices lives in [LEARNINGS.md](../../LEARNINGS.md).

---

## Models and inference

`OnnxSession` (`src/Vision/OnnxSession.h`) is the only ONNX Runtime touchpoint. `create(modelPath, preferredEp)` with `"directml"` tries the DirectML execution provider on device 0 with the session options DML requires (memory pattern disabled, sequential execution) and falls back to a plain CPU session on any failure. All sessions live on the single inference (vision) thread, sharing a process-lifetime `Ort::Env`.

Two helpers matter to callers:

- `findOutputByLastDim` maps detector outputs (score tensor vs box+keypoint regressor) by shape instead of graph order, because the opencv_zoo ONNX conversions do not order outputs consistently across models
- `runOutputs` runs a subset of outputs, and ONNX Runtime prunes the graph to what was asked for, so branches feeding only unrequested outputs never execute

The palm and person detection models do not embed SSD anchors. `src/Vision/SsdAnchors.h` regenerates them by porting MediaPipe's `ssd_anchors_calculator.cc` (fixed-anchor-size variant, so only normalized anchor centers are needed). `makePalmDetectorAnchorConfig` produces the palm table: 192x192 input, strides {8,16,16,16}, 2016 anchors, matching the table hardcoded in `mp_palmdet.py`. `weightedNonMaxSuppression` is MediaPipe-style weighted NMS: overlapping candidates are blended into the highest-scoring one with score-weighted averaging. The zoo demos use hard NMS at the same 0.3 IoU; weighted blending matches upstream MediaPipe and is smoother for single-object-per-region use.

`PalmDetector` (`src/Vision/PalmDetector.h`) ports `mp_palmdet.py`. Contract for `palm_detection.onnx`:

- input `input_1` `[1,192,192,3]` NHWC RGB float, divided by 255, aspect-preserving letterbox with black padding
- output `Identity` `[1,2016,18]` per-anchor regressors: box center/size in 192-px input units plus 7 keypoints
- output `Identity_1` `[1,2016,1]` per-anchor score logits (sigmoid applied)

Defaults are score threshold 0.5 and NMS IoU 0.3 (`m_scoreThreshold`, `m_nmsIouThreshold`). Each `PalmDetection` carries the box, 7 palm keypoints (wrist, four finger MCPs, thumb CMC and MCP), and the MediaPipe ROI rotation derived from wrist to middle-MCP.

`HandLandmarkModel` (`src/Vision/HandLandmarkModel.h`) ports `mp_handpose.py`. Contract for `hand_landmark.onnx`:

- input `input_1` `[1,224,224,3]` NHWC RGB float divided by 255, a rotated palm crop (palm box enlarged 4x, rotated by the wrist to middle-MCP angle, re-cropped from the rotated keypoint bbox shifted [0,-0.4] and enlarged 3x, resized to 224 with `INTER_AREA`)
- output `Identity` `[1,63]` 21 landmarks as (x,y,z), x/y in 224-px crop space, z relative to the wrist
- output `Identity_1` `[1,1]` presence/confidence, already in [0,1]
- output `Identity_2` `[1,1]` handedness, 0=left to 1=right in model terms (the model assumes a mirrored/selfie view)
- output `Identity_3` `[1,63]` model-space "world" landmarks, meters, hand-centered

`HandLandmarkResult` maps the crop-space landmarks back to full-frame pixels (`imagePoints`) and keeps the metric model landmarks (`modelPoints`). Both models preallocate their input buffers and OpenCV scratch mats, so the steady-state frame path does no heap allocation.

## The per-camera pipeline

`HandTrackingPipeline` (`src/Vision/HandTrackingPipeline.h`) is the MediaPipe tracking graph rebuilt in C++ on the inference thread. It owns two hand slots (`HandSlot`), fills the image-space fields of `TrackingFrameResult`, and leaves camera space, world space, and the parametric pose to the Tracking module.

Detector cadence: the palm detector is the expensive model, so it only runs when a slot is free, and otherwise every `detectorIntervalFrames` (default 30) as a drift guard. A tracked slot skips detection entirely by synthesizing its next ROI from the previous frame's landmarks (`HandRoi::fromLandmarks`). The crop math uses only the keypoints' bounding geometry plus the wrist to middle-MCP direction, so a landmark-derived ROI behaves identically to a detector-derived one.

Reacquisition: when the active slot count drops, the detector runs with the relaxed cutoff `palmScoreThresholdRelaxed` (default 0.25 instead of `palmScoreThreshold` 0.5) for `relaxedDetectorWindowFrames` (default 90). This trades precision for recall exactly when a known hand vanished, and is safe because the fusion gates downstream (clustering, residual veto, temporal and spatial priors) catch the extra false positives.

Slot lifecycle: a slot deactivates after `handPresenceLostFrames` consecutive frames below `handPresenceThreshold`. New palm detections that overlap an active slot's hand box above `slotDedupeIouThreshold` are dropped. After a clap or overlap both slots can converge on the same physical hand, blocking the free slot the separated hand needs, so sustained near-identical hand boxes (IoU above `slotDuplicateIouThreshold` for `slotDuplicateKillFrames` frames) kill the lower-presence slot.

Handedness: the model's raw handedness score assumes a selfie view, so `flipHandedness` (default true for an unmirrored webcam feed) converts it into `rightProb`, the flip-adjusted probability of a right hand. A slot's published side label is sticky: it flips only after `handednessSwitchFrames` (default 15) consecutive contradictions. When both slots claim the same side, `preferredSlotOrder` resolves the collision by classifier decisiveness, `|rightProb - 0.5|`, with presence only as the tiebreak. Presence measures tracking quality and says nothing about which hand is which. The key principle downstream: per-camera L/R labels are slot bookkeeping, not evidence. A slot can be displaced to the wrong side by a collision, so fusion's side votes read the flip-adjusted `rightProb`, never the label.

Cross-camera seeding: when another camera's fused hand is missing here, its world pose projects into this camera as a `HandSearchHint` (palm center, direction, palm size in pixels), which seeds a direct landmark-model attempt with no palm detection. Hint-seeded slots are speculative: if the first landmark pass finds no confident hand they drop immediately. `isSeedRedundant` rejects hints landing on an already-tracked hand positionally rather than by label, with boxes inflated because the projected center carries the other camera's depth error. The `HandSeedStats` funnel counts every hint's fate (candidates, projection skips, offered, too small, redundant, no free slot, applied, rejected by model, accepted), because seeding runs upstream of the tracking recording and cannot be replayed offline.

## Monocular 3D lift

`LandmarkTo3D` (`src/Tracking/LandmarkTo3D.h`) lifts one camera's image-space landmarks into camera-space meters. Input pixels are in undistorted image space (the vision thread undistorts before inference), and camera space is the OpenCV convention: +X right, +Y down, +Z forward.

The lift is `cv::solvePnP` against a dynamic per-frame metric object model. Articulation is baked into the object model, so PnP recovers only the global rigid pose. The object model is the user's calibrated skeleton (`setCalibratedSkeleton`, from `HandBoneCalibrator`) posed by the landmark model's angles via forward kinematics when a calibration exists, and otherwise the landmark model's own metric hand rescaled so wrist to middle-MCP matches the hand scale (`m_refLengthMeters`, seeded from the `handScale.refLengthMeters` config value and live-refined by stereo). The distinction matters: the model's proportions are not the user's, and rescaling on one bone leaves an object with too little spread, which PnP can only fit by placing it too close.

The solve is warm-started per side from the previous frame's rvec/tvec with `SOLVEPNP_ITERATIVE`, and cold-started with `SOLVEPNP_SQPNP`. ITERATIVE's cold-start DLT is ill-conditioned on near-planar point sets, and a flat hand posed by FK is exactly that, collapsing to near-zero or negative depth repeatedly until the hand curls. SQPnP has no planar degeneracy. A rare solve failure just skips camera space for the frame, and fusion's staleness window absorbs the gap.

`LandmarkTo3D` also fills the parametric `HandPose` (palm transform, finger angles, skeleton) and its `fkReprojectionPx` validation metric, the mean pixel error between the 2D landmarks and the FK hand rebuilt from the extracted pose.

## The hand pose model

`HandPoseModel` (`src/Tracking/HandPoseModel.h`) is a pure math namespace over any 21-landmark set in a right-handed metric space. It defines the parametric representation everything downstream fuses and streams:

- `computePalmFrame`: palm center origin, +X toward the fingers, +Z out of the palmar surface, +Y completing right-handed (Ultraleap-compatible)
- `computeFingerAngles`: per finger a `FingerAngles` of [`lateral`, `proximal`, `intermediate`, `distal`], all zero in the rest pose, with the two distal bends measured relative to their parent bone
- `computeSkeleton`: `HandSkeleton`, finger base positions in the palm frame plus phalanx lengths and the flat-hand neutral directions
- `buildFingerJoints`: forward kinematics that exactly inverts `computeFingerAngles` given the same skeleton

The thumb rests pronated relative to the fingers, so its flexion hinge is the finger-style hinge rotated about the thumb bone by `kThumbPronationRad` (1.2 rad), applied identically in extraction and FK so the 4-angle schema round-trips.

The palmar sign (which side of the palm plane is the palm) is the fragile bit. `PalmarSideMemory` carries the previous answer forward, one instance per tracked hand per space, reset on reacquisition. Evidence is scored from finger curl plus a thumb-plane term (the thumb metacarpal sits palmar of the wrist-index-pinky plane). Curl evidence may only contradict the memory above the decisive threshold `kCurlOverride` (1.0), because near-flat hands genuinely hyperextend 10 to 20 degrees and the sign of a small curl sum means nothing. Even decisive contradiction must persist for `kFlipEvidenceCount` (5) distinct observations before the sign flips. The thumb term is weak but flexion-independent, so it seeds a fresh hand when curl is inconclusive rather than overturning one already tracked. Deciding the sign independently every frame is what let a palm frame invert mid-capture; see [LEARNINGS.md](../../LEARNINGS.md) for the measurements.

## Multi-camera fusion

`HandFusion` (`src/Tracking/HandFusion.h`, `src/Tracking/HandFusion.cpp`) fuses every camera's `CameraFrameResult` into one world-space `TrackingFrameResult`. Left/right assignment happens here, not per camera, because a camera that sees only one hand routinely mislabels it.

Clustering: all fresh world-tracked observations are clustered by palm proximity, one cluster per physical hand, by joint per-camera assignment over `pairCost`. The position term is ray-aware (`kRayDepthSlackM`, 0.5 m of slack along the observing camera's view ray), because monocular depth error lies along the ray while lateral accuracy is much better. Vote coherence discounts or bumps the cost, and two decisively opposed votes veto a merge outright unless the points practically coincide.

Side assignment: clusters get sides by minimizing pair cost over `sideAffinity`, which sums three terms per side:

- weighted classifier votes, each observation's `signedVote` built from the flip-adjusted `rightProb` and weighted by presence times decisiveness
- temporal continuity against the last fused palm position per side
- a spatial prior along `k_worldRightAxis`, the constant `(0, -1, 0)`: the printed calibration board fixes the world frame (+X forward, +Z up), so the user's right hand lives toward -Y by construction

An assignment whose winning raw affinity is negative while that side was recently tracked is refused (`assignmentRefused`): the side goes untracked this fuse rather than adopting a wrong cluster by elimination. While only one hand is tracked, `m_lastSoloSide` adds hysteresis so near-tied affinities cannot flip-flop the label.

Triangulation: for a cluster with two or more intrinsics-bearing observations, `triangulateCluster` picks the best camera pair scored per pair, `presence_A * presence_B * sin(parallax)`. Depth error goes as 1/sin(parallax), so two well-scored cameras sitting close together reconstruct worse than a lesser pair with a wide baseline. Per-camera visibility is deliberately excluded from the pair score: a camera seeing the palm edge-on is the one resolving the depth its face-on partner cannot, so multiplying two face-on scores selects for redundant viewpoints. The winning pair two-ray triangulates all 21 landmarks and measures the RMS reprojection residual against both views. A residual above `triangulationMaxResidualPx` (25 px) vetoes the pairing: it means the two observations are two different physical hands, and a veto also skips blending, keeping the best single observation instead of manufacturing a hand between them.

Solo-cluster rescue: mono depth error routinely exceeds the clustering gates (0.28 m measured during a pointing gesture), stranding one camera's observation outside the other's cluster and killing triangulation. `rescueSoloClusters` probes unpaired and below-presence observations from other cameras against a solo cluster, letting the reprojection residual, not the mono positions, decide the pairing.

Stereo hand scale: each camera's estimated palm depth scales linearly with the configured hand scale, but its view-ray direction does not, so `updateStereoScale` two-ray triangulates the palm from a two-camera cluster and derives a correction factor for the configured scale. `VisionThread` folds the sample into a slow EMA (`kScaleEmaAlpha` 0.02) and pushes the corrected length into every camera's `LandmarkTo3D`, unless both hands have calibrated skeletons, which supersede it.

Rest zero: the rest-pose offset (`fusedRestAngles` in `HandFusionConfig`, captured from the user's rest pose) is subtracted once at fusion output, in `applyFusedRestOffset` and the estimator's equivalent, so streamed zero means one thing across the estimator, triangulated, and monocular paths. The state itself stays in raw angles.

## The state estimator

`HandStateEstimator` (`src/Tracking/HandStateEstimator.h`, `.cpp`) produces the output pose whenever at least one camera has intrinsics. The classic per-frame extraction (triangulate, blend, monocular fallback) still runs every fuse: it advances the shared bookkeeping, seeds the estimator on cold start, stands in when a fit is refused, and remains the output when no camera has intrinsics. It replaced post-fusion one-euro smoothing, whose per-frame winner selections (tri vs mono, pair choice, palmar re-decision) were the discrete switches users saw as pops; see [LEARNINGS.md](../../LEARNINGS.md).

The state is 26 DoF per side (`kStateDim`): palm position (3), palm orientation as a MEKF-style error quaternion (3), and 20 raw finger angles. Each fuse predicts (constant position), then fits the state to all fresh cameras' 2D `imagePoints` at once by damped Gauss-Newton over the FK reprojection residual, with a numeric central-difference Jacobian, Huber-weighted pixel residuals (`huberDeltaPx` 10), and a hand-rolled Cholesky over the 26x26 normal equations. Bone lengths hold by construction, since the fit moves palm and angles through FK. MediaPipe's view-dependent monocular articulation (`modelPoints`) is not in the loop at all.

Rows are weighted by sqrt of the observation's measured confidence times an age factor, never by model presence. Presence stays high on a camera whose landmarks are swinging, and weighting by it let a 126 mm-jitter camera drag the fit. Every observation inside `measurementWindowMs` (66 ms) enters every fit, with deliberately no per-camera freshness dedupe: cameras deliver in turn, and fitting only the freshest camera's pixels ping-pongs the state between their systematically disagreeing solutions.

The priors and guards:

- a temporal prior holds the state near its prediction, scaled per second by `palmPosSigmaMPerS`, `palmRotSigmaRadPerS`, and `angleSigmaRadPerS`. It is a soft leash: strong measurements override it freely, so it binds only when the measurement is weak, which is exactly when the classic pipeline pops.

- single-camera fits make the position prior anisotropic via `monoDepthPriorFactor` (0.2): tight along the observing camera's view ray, normal laterally. A mono fit's only depth signal is apparent scale, which carries the per-camera bias stereo exists to remove.

- one-sided quadratic joint-limit penalties (`jointLimitsEnabled`, `jointLimitSigmaRad`, limits from `angleLimits`) cost nothing inside the anatomical range and absorb single-finger landmark snaps.

- a weak DIP-PIP coupling residual (`couplingSigmaRad`) pulls distal toward 0.67 times intermediate, superseded automatically by a fitted angle prior.

- an optional fitted per-user angle prior (`HandStateEstimatorConfig::AnglePrior`, the `anglePrior` block in the app config, fit offline by `--fit-angle-prior` from recordings) adds a Mahalanobis pull toward the user's own pose distribution, telling a coordinated pose change from a single-DoF landmark snap.

- a dt-scaled innovation gate (`maxStepMetersPerS`, 4 m/s) rejects fitted palm steps implying impossible speed. Those are correspondence errors whose pixels fit fine, and gated steps take the bad-fit hold path, so a genuine relocation still lands via reseed within a few fuses.

- the divergence guard holds then drops: a fit whose confidence-weighted mean pixel residual exceeds `maxResidualPx` (25 px) streams the previous state instead, and only a streak of `maxBadFitStreak` (5) drops the state for a reseed. An instant drop turned every marginal frame into a drop, snap, reseed pop cycle on real recordings.

A side unobserved for `resetGapMs` (264 ms) resets, mirroring fusion's staleness handling. The whole solver is deterministic by construction, with no wall clock, no randomness, and a fixed iteration cap (`maxIterations` 4), so replaying a recording reproduces the live output bit-exactly.

## Confidence

Model confidence outputs answer "is a hand here", never "is this pose good". Presence stays high on an edge-on hand whose depth solve swings by centimeters. Quality is therefore measured, not reported:

- per-observation confidence is presence times a jitter stability factor, where jitter is the EMA of the constant-velocity residual `|p(t) - 2p(t-1) + p(t-2)|` of the palm (zero for smooth motion of any speed, large for frame-to-frame noise) and `stabilityFactor` is a soft inverse-variance weight around `jitterReferenceM`
- reprojection residuals arbitrate geometry: the triangulation veto, the estimator's divergence guard, and the `residualFactor` term in triangulated and estimator confidence all judge how well a pose explains what the cameras actually saw
- fused `HandPose::confidence` is presence times output stability times the residual factor on the estimator and triangulated paths, and the best single view's confidence on the blend path, since a second worse view adds information but cannot make the estimate less trustworthy

`HandPose::confidence` is the number to gate output on. See [debugging.md](./debugging.md) for the diagnostic surfaces (`FusionDiagnostics`, the seed funnel, jitter and residual dumps) and [LEARNINGS.md](../../LEARNINGS.md) for the measurements behind each of these choices.
