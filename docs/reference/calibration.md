# Calibration

The calibration subsystem in `src/Calibration`, built on OpenCV, plus the ImGui wizard flows in `src/UI` that drive it. Covers per-camera lens intrinsics, whole-rig extrinsics against a printed origin marker, hand bone and rest-pose calibration, body dimension measurement, wrist IMU mounting, and the live hand scale. Calibration quality is tracking quality: every metric solve downstream stands on these results. See [conventions.md](./conventions.md) for the coordinate conventions the extrinsics pin, [hand-tracking.md](./hand-tracking.md) and [body-pose.md](./body-pose.md) for the pipelines that consume the results, [imu.md](./imu.md) for the orientation filter behind the mounting calibration, and [debugging.md](./debugging.md) for the replay tooling the offline calibrators run on. Experiment history, including the rejected approaches referenced below, lives in [LEARNINGS.md](../../LEARNINGS.md).

---

## Shared infrastructure

- `CalibrationPatternFinder` (`src/Calibration/CalibrationPatternFinder.h`) is the pattern-detection base class with two subclasses: `CalibrationPatternFinder_Charuco` (`cv::aruco::CharucoDetector`, used by intrinsics) and `CalibrationPatternFinder_Aruco` (`cv::aruco::ArucoDetector`, used by extrinsics). Both use the `DICT_6X6` dictionary.
- `PatternPoseSampler` (`src/Calibration/PatternPoseSampler.h`) samples one camera's view of a static pattern for extrinsics. It pools per-corner pixel observations across captures (keyed by pattern point ID) and solves one pose from the averages, rather than averaging per-frame poses, which attacks detection noise where it enters and yields one meaningful reprojection error.
- `CVVideoFrameProcessor` (`src/Calibration/CVVideoFrameProcessor.cpp`) bakes intrinsics into X/Y remap maps via `cv::initUndistortRectifyMap` and applies them with `cv::remap`. `VisionThread` recreates it unconditionally on every config refresh: gating on frame dimensions alone once kept a previous calibration's maps alive after recalibrating at the same resolution.
- `PatternExport` (`src/Calibration/PatternExport.cpp`) writes the printable patterns to `resources/calibration` and opens them with `PathUtils::openFileWithDefaultApplication`. The wizards expose this as export buttons, and the same generators run headless as `--export-board [cols rows squareMM markerMM]` (`src/Tests/ToolExportBoard.cpp`) and `--export-marker [markerId markerLengthMM]` (`src/Tests/ToolExportMarker.cpp`), defaulting to the saved config's parameters. Both sheets must print at 100% scale: a rescaled print silently rescales the whole tracking world.

---

## Lens intrinsics (per camera)

`IntrinsicsWizard` (`src/UI/IntrinsicsWizard.cpp`) drives `MonoLensDistortionCalibrator` for one camera at a time. On entry it disables tracking and undistortion for that camera (samples need raw distorted frames); the other cameras keep tracking. The user shows a printed charuco board, whose geometry comes from the `CharucoBoardConfig` block and is edited in the wizard.

Sample gating, in `MonoLensDistortionCalibrator::update` and `CalibrationPatternFinder_Charuco::findNewCalibrationPattern`:

- Full board required. Partial detections technically calibrate (the solve matches corners by charuco ID), but they are exactly the marginal-condition samples that fed wild focal-length solves, so every corner must be detected.

- A sample auto-captures after the board holds steady for `k_imagePointStabilityDuration` (1 s) in a new location (`k_defaultMinSeperationDist`, 10 px average corner movement).

- Tilt quota: focal length is only observable from perspective, so at least `k_minTiltedSampleCount` (4) of the 12 samples must show keystone at or above `k_tiltKeystoneThreshold` (1.10, the ratio of opposite outer edge lengths). Once the flat-board quota fills, fronto-parallel boards are rejected and the wizard prompts for tilt.

The solve is `computeMonoLensCameraCalibration` (`src/Calibration/CameraMath.cpp`): `cv::initCameraMatrix2D` then `cv::calibrateCamera` with `CALIB_FIX_ASPECT_RATIO + CALIB_FIX_PRINCIPAL_POINT + CALIB_ZERO_TANGENT_DIST + CALIB_RATIONAL_MODEL + CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6`, so only k1/k2 are estimated. k6 must be fixed along with k3/k4/k5: under the rational model a free k6 is a lone r^6 denominator degree of freedom that fits the captured corners to sub-pixel error while going wild outside them, after which `cv::getOptimalNewCameraMatrix` emits a degenerate undistorted matrix. The result then passes plausibility gates before it is accepted: hfov must land in a 20 to 160 degree window and the undistorted/distorted fx ratio in 0.2 to 5.0, because a degenerate capture set reports sub-pixel reprojection error while the recovered focal length is nonsense. Failing either gate fails the calibration loudly instead of persisting intrinsics that poison every downstream solve.

On accept the wizard writes `IntrinsicsConfig` for that camera plus the board parameters, saves the config, and re-enables undistortion so the user can verify straight lines in the live preview. The undistortion maps are rebuilt from the new intrinsics unconditionally (see shared infrastructure above).

---

## Extrinsics (all cameras at once)

`ExtrinsicsWizard` (`src/UI/ExtrinsicsWizard.cpp`) calibrates every camera in one session against a single printed ArUco marker lying flat on the table. Every camera needs calibrated intrinsics first; the wizard refuses to start otherwise, since the pose solve runs on undistorted points through the calibrated camera matrix. A charuco board was evaluated for this target and rejected as a fit: at the 60 to 70 cm desk working distance the largest squares that fit on letter paper do not resolve enough corner detail to detect, while one large ArUco square does (see the `ExtrinsicsConfig` comment in `src/App/AppConfig.h` and [LEARNINGS.md](../../LEARNINGS.md)).

The marker is the world anchor: it defines the tracking world's origin and its axes, and it stays on the table during tracking. The printed sheet carries FORWARD/LEFT/RIGHT axis labels because laying it down turned produces tracking that is self-consistent and points the wrong way. The world frame it pins (+X forward, +Z up out of the table) and the OpenCV-convention camera space the stored transform maps from are documented in [conventions.md](./conventions.md); the stored `markerFromCamera` is built in `computeWorldFromCameraPose` and right-multiplied by the wizard's `k_cvFromGlFlip` so it maps CV-convention camera points directly to world.

Capture is simultaneous and atomic:

- Every camera runs its own `PatternPoseSampler` (12 pooled samples) against the same physical marker placement, and the wizard only advances to review once every camera has a solved pose.

- Accept writes all cameras' `ExtrinsicsConfig` blocks in one save. Per-camera sessions with a marker nudged between them would silently disagree about the world frame, which is why there is no single-camera extrinsics path.

Review shows two layers of quality. Per camera, `PatternPoseSampler::getMeanReprojectionErrorPx` reports how well the pose fits that camera's own detections, which says nothing about cross-camera agreement because each `cv::solvePnP` minimizes its own reprojection. Cross-camera, `evaluateExtrinsicsPair` (`src/Calibration/ExtrinsicsValidation.cpp`) triangulates the marker corners both cameras saw (at least 4 shared) using the just-solved extrinsics and compares the reconstruction against the marker's known geometry:

- `reprojectionRmsPx`: triangulated corners reprojected into both views, the same residual the hand pipeline reports, so the numbers compare directly
- `spacingErrorMm`: triangulated inter-corner distances against the known distances
- `spacingScale`: reconstruction scale, 1.0 is correct
- `planarityRmsMm`: the marker is flat, so this is pure measurement error

Spacing is the metric that catches warped reconstructions: a small orientation error on one camera (a wrong baseline) shows up as spacing and scale error while reprojection stays low, so low rms alone does not clear a calibration. With only four marker corners these numbers catch a grossly wrong relative pose rather than finely grading a good one; the UI colors a pair good below 2 px rms and 3 mm spacing error. The worst pair's metrics persist as `ExtrinsicsQualityConfig` so a later "is my calibration still good" question has a baseline.

---

## Hand calibration (bones, then rest)

`HandCalibrationWizard` (`src/UI/HandCalibrationWizard.cpp`) runs two stages in a fixed order: bone measurement, then rest-pose capture. The order is enforced because saving measured bones moves the thumb's raw-angle zero. The thumb's neutral direction derives from the skeleton's base positions (`HandPoseModel::makeDefaultNeutralDirections`, rebuilt on load rather than stored, per the `HandSkeletonConfig` comment), so a rest pose captured before the bone save would be measured against a zero that no longer exists.

Bones: the vision thread runs `HandBoneCalibrator` (`src/Tracking/HandBoneCalibrator.h`) over a 10 second window, feeding it every stereo-triangulated frame of each hand. It takes per-component medians of `HandPoseModel::computeSkeleton` across the samples, medians rather than means because a bone lying along a camera's view ray triangulates poorly and produces outliers rather than noise. A side needs `HandBoneCalibrator::k_minSamples` (30) stereo samples or its capture is refused. The wizard reviews the result (per-bone spread above 4 mm warns of an unlucky window) before writing anything; accept writes `HandSkeletonConfig` and sets `handScale.refLengthMeters` from the measured wrist-to-middle-MCP reference bone, so the reference the rest of the app reads comes from the measurement.

Rest: captured from the fused stereo path only, so both hands must be stereo-tracked. The captured angles store as `fusedRestAngles` (one set, not per camera: triangulated geometry has no per-camera model bias to fold in) and are subtracted once at fusion output (`HandFusion::applyFusedRestOffset`; the estimator's internal state stays raw). After this, zero angles mean the user's rest hand on the wire, while the skeleton's neutral directions still describe the idealized flat hand, so a client rendering all-zero angles shows the flat-hand convention, not the user's exact rest pose.

Two offline tools cover the same ground from recordings, and neither mutates the live config: each writes the recording's embedded config plus its fitted result to an explicit output path, which is merged into the project's `project.json` by hand.

- `--calibrate-bones <recording.jsonl> [output-config.json]` (`src/Tests/ToolCalibrateBones.cpp`) replays a recording, runs the same `HandBoneCalibrator`, and prints the measured skeleton next to the landmark model's proportions.
- `--fit-angle-prior <recording.jsonl>... [output-config.json]` (`src/Tests/ToolFitAnglePrior.cpp`) fits `AnglePriorConfig` via `AnglePriorCalibrator` (`src/Calibration/AnglePriorCalibrator.h`): a per-side Gaussian (mean plus shrinkage-regularized precision matrix) over the 20 raw finger angles, from confident stereo frames only. It needs `AnglePriorCalibrator::k_minSamples` (300) samples per side, and the hand state estimator consumes the result as a weak Mahalanobis pull toward the user's real pose distribution.

---

## Body measurement

`BodyCalibrationWizard` (`src/UI/BodyCalibrationWizard.cpp`) drives `BodyDimensionCalibrator` (`src/Calibration/BodyDimensionCalibrator.h`) on the one camera with body pose enabled, which needs both intrinsics and extrinsics. The fused wrists are the only metric anchor in the frame: they are stereo-triangulated by the hand pipeline, world-anchored, and independent of both the body-pose camera and every length being measured, so everything here is that scale carried onto the body landmarks.

The measured numbers are landmark-space, not anatomical. The pose model's shoulder and ear landmarks sit inboard of the real joints by an amount that varies with the person and the model (assuming an anatomical 0.40 m between the shoulder landmarks once put a live shoulder 0.8 m too far away), so lengths must be calibrated in landmark space and never assumed from anthropometry.

Only widths are measured: shoulder width, head width, and nose-forward offset. The upper arm is derived as `upperArmPerShoulderWidth` times the measured shoulder width (see `BodyConfig`); direct upper-arm measurement was evaluated and dropped as a fit because it needed the arm straight and square to the camera, a pose desk framing defeats, and missing it under-measured the arm by 20 percent, enough to make the elbow solve pick the wrong bend (see [LEARNINGS.md](../../LEARNINGS.md)). Dropping it collapses the capture pose to "raise one hand".

- Frontal stage: the user faces the camera and raises one hand beside the shoulder, which puts a metric wrist at roughly the torso's depth. The acceptance gate is image-space, normalized by the shoulders' pixel separation: a hand counts only within `BodyDimensionCalibrator::k_maxRaisedHandOffset` (0.80) of that separation from its own shoulder, because a hand resting on the desk sits well in front of the torso and would scale the widths wrong. Either hand can carry a frame.

- Head-turn stage: facing the camera the nose sits on the ear midpoint and says nothing about how far it juts forward, so the user turns their head to swing that offset into the image plane. This stage is skippable; the nose offset only sets head yaw and pitch.

`BodyDimensionCalibrator::solve` takes medians over the samples (`k_minSamples` is 20 per gate) and reports spread so a moved pose is visible at review. Accept writes the `BodyConfig` widths, the derived upper arm, and the nose offset when measured.

---

## Wrist IMU mounting

`MountingWizard` (`src/UI/MountingWizard.cpp`) measures how each controller is strapped to the wrist, producing the per-side `forearmToSensor` rotation in `ImuConfig` that turns a sensor orientation into a forearm orientation. The forearm frame is defined as the palm frame at neutral wrist (+X toward the hand), which is what makes the streamed wrist joint rotation identity when the hand is straight; the orientation filter consuming the result is documented in [imu.md](./imu.md).

The capture is motion-based, and the geometry comes entirely from the IMU:

- Rest: controllers flat on the desk while gyro bias is measured (`requestImuBiasCalibration`). Skippable; the filter estimates bias online except about the vertical axis.
- Twist: pronation/supination turns the forearm about its long axis and nothing else, so the dominant rotation axis of the window is the forearm axis in the sensor's own frame.
- Curl: elbow flexion adds a second axis near perpendicular to the first, which fixes the roll a twist alone cannot see. As a by-product the curl measures the elbow-to-controller distance, which overwrites `body.forearmLengthMeters`.

Vision decides exactly one bit, which side of the hinge axis the palm is on, a choice between two candidates 180 degrees apart that survives a palm estimate far too noisy to be a mounting. There is deliberately no held pose: earlier versions took the roll and axis sign from vision at a single instant, then from an average over the twist, and both failed because a wrist that does not hold still corrupts the average as thoroughly as one frame (see the `MountingWizard.h` header comment and [LEARNINGS.md](../../LEARNINGS.md)). Each side's capture passes per-gate checks (axis dominance, twist and curl reversal, inter-axis angle, hinge spread, length-fit correlation, palmar source) and a failed side saves nothing. The review stage verifies live: with hands straight in line with the forearms, the wrist bend readout should stay near zero through a forearm twist.

---

## Live hand scale

The wrist-to-middle-MCP reference length that scales the monocular PnP object model is `handScale.refLengthMeters`, and it is corrected live rather than by a wizard. When fusion produces a stereo wrist triangulation, `HandFusion::getStereoScaleSample` yields a scale sample, and `VisionThread` folds it into `m_autoScaleFactor` as a slow EMA (`kScaleEmaAlpha`, 0.02); the effective reference length pushed into every camera's `LandmarkTo3D` is the config baseline times that factor. The factor is session-local and never persisted: the settings panel shows the measured length as a readout with nothing to press, since the EMA re-converges within seconds each session. The baseline itself is written by bone calibration (live wizard or `--calibrate-bones`), and once both hands have a calibrated skeleton the EMA is parked at 1.0: the measured skeleton is the hand's geometry, and two mechanisms setting scale at once would only fight. A config refresh also resets the factor to 1.0 so a freshly saved baseline is not double-corrected.

---

## Where everything persists

All calibration results live in the active project's `project.json` (`AppConfig`, serialization in `src/App/AppConfig.cpp`; projects live under `%USERPROFILE%/Documents/MikanTrack/`). The intrinsics, extrinsics, and hand wizards save immediately on accept; the body and mounting wizards mark the config dirty for the auto-save (3 second cooldown).

- Intrinsics: per camera under `cameras[i].intrinsics` (`present`, `reprojectionError`, `width`, `height`, `hfov`, `vfov`, `distortedCameraMatrix`, `undistortedCameraMatrix`, `distortion`), plus the board geometry under `charucoBoard` (`cols`, `rows`, `squareMm`, `markerMm`)
- Extrinsics: per camera under `cameras[i].extrinsics` (`present`, `markerFromCamera`, `patternReprojectionErrorPx`, `patternCornerCount`, `markerId`, `markerLengthMm`), plus the cross-camera worst-pair metrics under top-level `extrinsicsQuality` (`worstPairReprojectionRmsPx`, `worstPairSpacingErrorMm`, `worstPairSpacingScale`, `worstPairPlanarityRmsMm`, and the pair's identity)
- Hand bones: `handSkeleton.left`/`handSkeleton.right` (finger bases plus phalanx lengths; neutral directions are rebuilt on load), and the reference bone under `handScale` (`present`, `refLengthMeters`)
- Hand rest pose: `fusedRestAngles.left`/`fusedRestAngles.right`
- Angle prior: `anglePrior.left`/`anglePrior.right` (`mean`, `precision`)
- Body: the `body` block (`shoulderWidthMeters`, `headWidthMeters`, `noseForwardMeters`, `upperArmLengthMeters`, `upperArmPerShoulderWidth`, `deriveUpperArmFromShoulderWidth`, `forearmLengthMeters`)
- IMU mounting: `imu.mountingLeft`/`imu.mountingRight` (the `forearmToSensor` quaternions), with the curl-measured forearm length under `body.forearmLengthMeters`
