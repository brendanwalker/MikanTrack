# Body pose

The opt-in per-camera body-pose stage and everything downstream of it: the ONNX models (`src/Vision/PoseDetector.h`, `src/Vision/RtmPoseBodyModel.h`), the per-camera tracker that runs them on a cadence (`src/Vision/BodyPoseTracker.h`), and the post-fusion world solver that turns 2D landmarks into elbow, shoulder, and head estimates (`src/Tracking/BodyPoseSolver.h`). Body pose layers on top of the hand system: see [hand-tracking.md](./hand-tracking.md) for the hand pipeline whose fused wrists anchor the solve, [architecture.md](./architecture.md) for where this sits in the vision-thread frame anatomy, [imu.md](./imu.md) for the IMU forearm overlay that takes precedence over it, [calibration.md](./calibration.md) for the body measurement wizard that supplies its lengths, [wire-protocol.md](./wire-protocol.md) for the OSC messages it feeds, and [conventions.md](./conventions.md) for the coordinate spaces. Experiment history, including the model selection A/Bs, lives in [LEARNINGS.md](../../LEARNINGS.md).

---

## Models

Two ONNX models, loaded from `models/` (`person_detection.onnx` and `rtmpose_body.onnx`, fetched by `InitialSetup_x64.bat`). Load is fail-soft: `BodyPoseTracker::load` returns false when either is missing and the stage becomes a no-op.

### Person detector

`PoseDetector` (`src/Vision/PoseDetector.h`) ports opencv_zoo's `mp_persondet.py`, the MediaPipe person detector, to ONNX Runtime. Contract for `person_detection.onnx`:

- input `input_1:0`: `[1,3,224,224]` NCHW RGB float, normalized `(x/255 - 0.5) * 2` to `[-1,1]` before the aspect-preserving letterbox (pad value 0 is mid-gray)
- output `Identity:0`: `[1,2254,12]` per-anchor box regressors plus 4 keypoints, in 224-px input units
- output `Identity_1:0`: `[1,2254,1]` score logits

The score output comes first in the ONNX graph, so outputs are mapped by last dimension, not position. Anchors (2254, `SsdAnchors`) are generated at load. Score threshold is 0.5, weighted-NMS IoU threshold 0.3.

The detector's output box is a face box, not a person box. Only its keypoints are consumed downstream: per `PersonDetection`, keypoint 0 is the hip center, 1 the full-body ROI point, 2 the shoulder center, 3 the upper-body ROI point. The person box for the landmark model is built from keypoints 0 and 1 as a square about the hip center reaching the full-body point, which is the crop convention this detector was built for. Feeding the face box to the landmark model crops the arms away.

### RTMPose body landmarks

`RtmPoseBodyModel` (`src/Vision/RtmPoseBodyModel.h`) runs OpenMMLab's RTMPose-m body7 SimCC model (the `rtmpose-m_simcc-body7` MMDeploy end2end export). It is top-down: the caller supplies the person box, and each keypoint is predicted independently inside it with its own score. There is no full-body prior, so a joint outside the frame degrades to a low score instead of being invented. BlazePose landmarking was evaluated and removed as a poor fit: it always emits one coherent full body, so a subject truncated at a desk makes it fabricate the unseen legs, and its self-referential crop cannot recover (see [LEARNINGS.md](../../LEARNINGS.md)).

Contract for `rtmpose_body.onnx`:

- input `input`: `[1,3,256,192]` NCHW RGB, normalized `(x - mean) / std` with the ImageNet statistics (`k_mean`, `k_std` in `RtmPoseBodyModel.cpp`), from a 1.25-padded aspect-corrected affine crop of the person box
- output `simcc_x`: `[1,17,384]` per-keypoint x class scores (192 * 2)
- output `simcc_y`: `[1,17,512]` per-keypoint y class scores (256 * 2)

Decode is argmax over each axis divided by the 2.0 SimCC split ratio, mapped back through the inverse crop affine. A keypoint's score is `min(peakX, peakY)`. SimCC peaks are not probabilities; the tracker clamps them to `[0,1]` so downstream visibility gates keep their meaning. The model emits the 17-keypoint COCO order (`eCocoKeypoint`, `COCO_KEYPOINT_COUNT`).

## The tracker

`BodyPoseTracker` (`src/Vision/BodyPoseTracker.h`) owns the detect-vs-track cadence per camera and fills one `BodyPoseObservation` per processed frame.

### The canonical landmark space

`BodyPoseObservation` (`src/Vision/BodyPoseTypes.h`) uses the 33-slot BlazePose index space (`ePoseLandmark`, `POSE_LANDMARK_COUNT`) as the canonical layout, so every consumer (solver, overlays, recordings, dumps) is backend-independent. `k_cocoToPoseLandmark` in `BodyPoseTracker.cpp` maps RTMPose's 17 COCO keypoints into it. Slots the backend does not fill are reported through `providedMask`, a 64-bit mask with one bit per landmark, rather than being invented. Landmark positions are undistorted full-frame pixels. `visibility` per landmark carries the clamped model score, and the observation's `confidence` is the mean score over the 17 provided keypoints. `eBodyBoxSource` records where each model frame's person box came from:

- `Detector`: the person detector fired this model frame
- `Tracked`: grown from the previous model frame's confident keypoints
- `FullFrame`: nothing else available, the whole image
- `None`: no model run

### Cadence and boxes

Models run every Nth processed frame (`BodyPoseTrackerConfig::frameDivider`). Divided-out frames re-emit the held observation with its `modelFrameIndex` unchanged, so downstream jitter tracking can tell a fresh model result from a re-emit, and the cadence is baked into recordings without replay needing the divider.

On a model frame, `acquireBox` prefers the tracked box: the confident-keypoint extent of the previous model result (gate `keypointScoreThreshold`, default 0.3), expanded by `kTrackedBoxExpansion` (1.15). Because the box is built only from keypoints that scored, a truncated subject produces a smaller box, not a fabricated one. Every `detectorIntervalFrames` model frames (default 20) the detector is forced to re-run as a drift guard, regardless of model confidence, because a box only ever rebuilt from the previous keypoints can feed its own drift. A failed forced re-detect keeps the working tracked box: the periodic detect is a guard, not a requirement. With no tracked box and no detection, the tracker falls back to the full frame, which a top-down model still reads for a centered subject. Boxes smaller than `kMinBoxSize` (24 px) are treated as collapsed.

## The world solver

`BodyPoseSolver` (`src/Tracking/BodyPoseSolver.h`) runs after fusion and fills the fused result's forearm (elbow), shoulder, and head outputs. Everything is solved from 2D rays plus known lengths. The pose model's own metric 3D is deliberately not consumed: measured per-frame it is unusable, and the top-down backend produces none. The fused wrists are the metric anchor, the best-measured points in the system. The solver is deterministic: no wall clock and no config reads, state advances on the observation's `modelFrameIndex` and the fused timestamp, so a replayed recording produces bit-identical output.

Only a new `modelFrameIndex` triggers a re-solve (`solveFromObservation`). Re-solving a repeated observation against a moved wrist would slide the estimate along a stale ray, so repeated observations hold: the forearm direction is held and the elbow rides the fresh wrist through it, while shoulders and head hold their world positions (they hang off the torso, not the wrist). A model gap longer than `kMaxHoldMs` (500 ms) resets the solver. Estimated depths outside `kMinBodyDepthMeters` (0.15) to `kMaxBodyDepthMeters` (2.0) are rejected: any solve that divides a known separation by an apparent one blows up when the model collapses that pair, and the window keeps a wild value out of the estimates that hang off it.

### Shoulders

Shoulders are placed first and independently of the arms, from their own two rays plus the calibrated shoulder separation. `depthFromKnownSeparation` exploits that both rays leave the same optical center: equal range t on both rays gives `|t*a - t*b| = separation`, which fixes the depth with no model 3D. Solving them first makes them an independent arbiter for the elbow instead of a quantity the elbow choice can drag along. When one shoulder is unseen the pair no longer fixes a depth, and the held torso-fixed positions stand.

### Elbows

With a shoulder and a fused wrist known, `solveElbowOnBoneCircle` constructs the circle where the upper-arm sphere about the shoulder intersects the forearm sphere about the wrist. Both bone lengths then hold by construction, and the 2D elbow landmark is demoted from defining the elbow to picking a point on that circle, a far weaker demand on a noisy landmark and one that cannot produce an anatomically impossible arm. The pick minimizes distance to the camera ray around the circle by a 64-point sampled scan plus ternary refinement per minimum (`kCircleSamples`, `kRefineSteps`), which stays deterministic for replay. A wrist modestly out of reach straightens the arm to a single point, keeping the estimate continuous through full extension. A wrist wildly out of reach (`kMaxReachOvershoot`, 1.15 of both bones laid end to end) means the shoulder is wrong, and the solve declines rather than flinging the elbow. Without a shoulder, `intersectRaySphere` falls back to the elbow ray against the forearm sphere about the wrist alone, with a tangency clamp for the same continuity reason.

The circle yields up to two candidates, both anatomically valid, so the choice is only which valid arm. Continuity wins outright when there is a held direction, and it predicts from the raw unfiltered forearm direction, never the smoothed one: a lagging predictor inside a discrete decision keeps endorsing the pose the arm has already left, and once the wrong root wins the filter converges onto it. With no history, the prior is that the elbow trails the wrist relative to the camera. The stored quantity is the forearm direction (wrist to elbow), not the elbow position, so the elbow rides a wrist that updates at full camera rate between model results.

### Dead reckoning

An occluded elbow landmark does not drop the arm. `solveElbowNearestTo` re-solves the same bone circle from the still-measured shoulder and wrist and picks the circle point nearest the predicted elbow, closed form (radial projection onto the circle's plane). Re-solving rather than holding the last elbow position is the point: the wrist keeps moving, and a held position would stretch the forearm. Dead reckoning is capped at `kMaxDeadReckonMs` (1000 ms) with confidence decaying linearly over the gap. The jitter tracker is deliberately excluded from a dead-reckoned confidence: an inferred elbow is smooth by construction, and scoring smoothness there would report a confident arm precisely when it is being invented.

### Head

Head depth comes from the apparent ear separation against the calibrated head width, the same `depthFromKnownSeparation` construction as the shoulders. The nose is then placed by `intersectRaySphere` on its own ray against a sphere of radius `noseForwardMeters` about the ear midpoint, taking the near intersection (a face points toward the camera that sees it). The nose's depth carries the whole head yaw and pitch signal; placing it at the ear depth would flatten the face into the ear plane. In the output orthonormalization the forward axis is primary: yaw lives in the nose's offset along the ear axis, and rebuilding forward perpendicular to the ears would be exactly the projection that deletes it. The two direction series are filtered independently, so orthogonality is restored once, on the way out in `applyEstimates`.

### Filtering and confidence

One-euro filtering (`src/Tracking/OneEuroFilter.h`) is applied to the estimates, not the outputs: the forearm direction and the torso-anchored positions are the noisy monocular quantities, while the wrist they hang off is already fused and filtered. Filtering the elbow position instead would lag it behind its own wrist. Filter dt is the interval between model results, which is what the filtered series is sampled at. Parameters are the `kPositionMinCutoffHz` / `kPositionBeta` / `kDirectionMinCutoffHz` / `kDirectionBeta` / `kDerivativeCutoffHz` constants in `BodyPoseSolver.cpp`, slower than the hand palm because these are IK hints refreshed at a fraction of the camera rate.

Landmark visibility gates but never scores: `kMinVisibility` (0.5) decides whether a landmark is usable, and confidence is measured stability instead. `JitterTracker` keeps a constant-velocity residual EMA per solved joint, divided by dt squared so it reads as an acceleration and means the same thing at every cadence, scored against `kAccelReference` (25 m/s^2). Per joint:

- elbow: fused hand confidence times jitter stability, or times the linear dead-reckon decay when inferred
- shoulder: jitter stability
- head: jitter stability times the minimum landmark visibility of nose and ears, because a wrong head that holds still is perfectly stable and the landmark scores are the part of that judgement jitter cannot see

## Integration and calibration inputs

### Where it runs

The per-camera stage runs inside each camera's processing in `VisionThread.cpp`: `BodyPoseTracker::process` fills `result.body` on the camera's `CameraFrameResult`, so observations ride the same fusion candidate mirrors, recordings, and replay as the hand results. The world solve runs once per fuse iteration on the vision thread, after the fused-output recording checksum tap (which is pre-IMU), after the IMU forearm fill, and after the IMU recording tap. The ordering keeps recordings carrying pure IMU output so replay can re-run the solver for what-if comparisons. IMU precedence is enforced in `applyEstimates`: a side whose `hasForearmPose` is already set (wrist IMU) is left untouched, because a measured direction beats a monocular inference. The shoulder and head outputs have no IMU counterpart and always apply. See [architecture.md](./architecture.md) for the full frame anatomy and [imu.md](./imu.md) for the forearm overlay.

### Configuration

The stage is opt-in per camera via `BodyPoseCameraConfig` on `CameraProfile` (`src/App/AppConfig.h`), serialized under the camera's `bodyPose` JSON object:

- `bodyPose.enabled` (default false)
- `bodyPose.poseFrameDivider` (default 2)
- `bodyPose.detectorIntervalFrames` (default 20)

Cameras with the stage disabled pay nothing. The person detector only fires on cameras that see the user upright, so it stays off for overhead cameras.

### Body dimensions

The solve's lengths come from `BodyConfig` (`src/App/AppConfig.h`), mapped through `makeBodyDimensions` in `AppConfig.cpp` into the `BodyDimensions` struct the solver consumes. One mapping is shared by the live solve, the replay engine, and the replay self-test so they cannot drift apart. Fields:

- `forearmLengthMeters`: shared with the IMU elbow path, and overwritten with a measured value by the IMU mounting wizard's curl stage
- `shoulderWidthMeters`: fixes shoulder depth from the two shoulder rays
- `upperArmLengthMeters` and `upperArmPerShoulderWidth`: with `bDeriveUpperArmFromShoulderWidth` (default true) the upper arm is `shoulderWidthMeters * upperArmPerShoulderWidth` rather than the stored length, because measuring it directly needs a hard pose and a multiple of a width does not
- `headWidthMeters`: ear-to-ear width, fixes head depth
- `noseForwardMeters`: ear-axis midpoint to nose tip, fixes head yaw and pitch

Every length here is a landmark-space quantity, and this is the trap: the model's shoulder and ear landmarks sit inboard of the anatomical joints by an amount that varies with the person and the model, so assumed anatomical dimensions do not fit landmark separations. An anatomical shoulder width plugged into the depth solve places a shoulder wildly wrong, and the familiar anatomical arm-to-shoulder ratios do not transfer (`upperArmPerShoulderWidth` is a landmark-width multiple, not an anatomical one). Every length the solve uses must either be calibrated in landmark space or be anchored to a measured quantity, and the fused wrists are that metric anchor.

`BodyDimensionCalibrator` (`src/Calibration/BodyDimensionCalibrator.h`) is how the widths get measured, driven by the body measurement wizard (`src/UI/BodyCalibrationWizard.cpp`). It carries the fused wrist's metric scale onto the body landmarks: the user raises one hand near the shoulder, which puts a stereo-triangulated wrist at roughly the torso's depth, and the shoulder and ear separations are measured at that depth. The raised-hand check is `k_maxRaisedHandOffset` in the image, and results are medians over at least `k_minSamples` samples with a spread readout. The nose-forward distance needs its own head-turn step, since facing the camera the nose sits on the ear midpoint and says nothing about how far it juts forward. Only widths are measured. See [calibration.md](./calibration.md) for the wizard flow.

### Outputs

The solved joints leave over OSC via `OscStreamer` (`src/Osc/OscStreamer.h`):

- `/mikan/hand/{left,right}/elbow`: `,ffff` position xyz plus confidence
- `/mikan/hand/{left,right}/shoulder`: `,ffff` position xyz plus confidence
- `/mikan/body/head`: `,ffffffff` position xyz plus orientation xyzw plus confidence

The elbow position is derived from the forearm frame the solver fills on `poses[]`: the frame's +X runs from the elbow toward the hand, matching the palm frame at a neutral wrist, with roll taken from the palm. See [wire-protocol.md](./wire-protocol.md) for the full message contract.
