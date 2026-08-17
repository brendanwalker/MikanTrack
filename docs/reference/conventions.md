# Conventions

The coordinate and math conventions every subsystem assumes, in one place: the spaces a landmark passes through, the world frame, the palm and forearm frames, angle sign conventions, and units. The app bridges OpenCV calibration, MediaPipe-style model outputs, GL rendering, and OSC consumers (Unreal, Unity-convention VMC receivers), and each disagrees with the others somewhere; this doc names each disagreement and its conversion point in code. See [hand-tracking.md](./hand-tracking.md) for the pipeline that moves data through these spaces, [calibration.md](./calibration.md) for how the world frame gets anchored, and [wire-protocol.md](./wire-protocol.md) for what crosses the wire.

---

## Math vocabulary

- GLM is the working math library everywhere: `glm::vec3`, `glm::quat`, `glm::mat4` (float), with `glm::dmat4` doubles for calibration transforms such as `markerFromCamera`.
- OpenCV types (`cv::Mat`, solvePnP inputs/outputs) appear only at the calibration and pose-solve boundaries.
- There is no Eigen. The IMU filter hand-rolls its small fixed-size matrices (`src/Imu/ImuOrientationFilter.cpp`).
- GLM matrices are column major, vectors are column vectors, transforms apply as `M * v` and chains read right to left.

## The spaces a landmark passes through

- **Image space**: full-frame pixels in the UNDISTORTED image. Undistortion happens once, before the models, in `CVVideoFrameProcessor`; everything downstream (landmarks, rays, reprojection residuals) assumes undistorted pixels against the undistorted camera matrix. A landmark's `z` is MediaPipe relative depth, in the same scale as the pixel x/y and relative to the wrist (`TrackedHand::imagePoints` in `src/Vision/TrackingTypes.h`).
- **Model space**: the landmark network's metric hand, meters, hand-centered (`TrackedHand::modelPoints`). It supplies articulation and scale, never world position.
- **Camera space**: OpenCV convention, +X right, +Y down, +Z forward out of the lens, meters, origin at the optical center. Produced by `LandmarkTo3D` (`TrackedHand::cameraPoints`, `HandPose::palmPositionCamera`).
- **World space**: marker-anchored, right-handed, meters. +X forward, +Y toward the person's left, +Z up out of the table. The printed calibration board pins the axes (its labels spell FORWARD and RIGHT), so the person's right hand always lives toward -Y; `k_worldRightAxis` in `src/Tracking/HandFusion.h` hard-codes that and the left/right hand prior relies on it. Deliberately not configurable.

## Camera-to-world and the missing GL flip

The single conversion point from camera space to world space is `applyWorldTransform` in `src/Tracking/SpaceTransforms.h`: `worldPoint = markerFromCamera * cameraPoint`, where `markerFromCamera` is the per-camera extrinsic stored by calibration and maps OpenCV-convention camera points directly into world space (see [calibration.md](./calibration.md) for its construction). No OpenCV-to-GL axis flip is applied anywhere in the tracking path. A renderer or engine that wants a GL-style camera (+Y up, -Z forward) applies its own conversion after this transform; the app's one rendering conversion point is `k_glFromCvFlip` in `src/UI/Scene3dPanel.cpp`. `SpaceTransforms` also owns the shared ray helpers `cameraPositionWorld` and `pixelRayDirWorld` (undistorted camera matrix in, world ray out) that fusion and the body solver both use.

## Palm, forearm, and head frames

- **Palm frame** (Ultraleap-compatible, stated on `HandPose` in `src/Vision/TrackingTypes.h`): origin at the palm center (midpoint of wrist and middle MCP), +X toward the fingers, +Z out of the palmar surface (chirality-corrected per hand), +Y completing right-handed.
- **Wrist joint**: half a palm back from the palm center along palm -X. The half-palm distance is `skeleton.baseInPalm[Middle].x` (`HandPose::getWristPositionWorld`).
- **Forearm frame**: equals the palm frame at a neutral wrist, +X along the forearm toward the hand, so the elbow is one forearm length back along -X (`HandPose::getElbowPositionWorld`). This identity is the reference IMU mounting calibration solves for (see [imu.md](./imu.md)), and it makes the wrist joint rotation `inverse(forearm) * palm`, identity when the hand points straight along the forearm.
- **Head frame**: +X facing direction, +Y toward the person's left, +Z up, origin at the ear midpoint (see [body-pose.md](./body-pose.md)).

## Finger angle conventions

Angles are radians everywhere inside the app and DEGREES on the wire; the OSC boundary is the only conversion point. The sign conventions are stated on `FingerAngles` in `src/Vision/TrackingTypes.h`:

- `lateral` is splay about palm +Z, positive counter-clockwise (toward palm +Y). Purely geometric and identical for both hands; the palm frame carries the chirality, so positive is toward the pinky on a right hand and toward the thumb on a left.
- `proximal` is the base bone away from its neutral direction, positive curling toward the palmar side (+Z).
- `intermediate` and `distal` are each measured RELATIVE TO THE PARENT BONE, not the palm, positive toward the palm; zero means collinear with the parent. They chain, so an evenly curling finger reads three similar values.
- The thumb's intermediate/distal hinge is pronated `kThumbPronationRad` (1.2 rad) about its metacarpal direction (`src/Tracking/HandPoseModel.cpp`), positive on a right hand and negative on a left; the chirality sign is read from `baseInPalm[Index].y`.

Zero is the rest pose. `HandSkeleton::neutralDirInPalm` is ALWAYS the flat-hand default (four fingers parallel to palm +X, thumb along its own metacarpal): it is where forward kinematics starts and it must mean one thing on the wire. A captured rest pose does not change it; the capture is stored as a per-side angle offset (`fusedRestAngles` in `AppConfig`) and subtracted once at fusion output (`HandFusion.cpp`). Consequently zero angles render an idealized flat hand, not the user's exact resting hand, and the thumb's zero moves when the skeleton is recalibrated, which is why bone calibration must precede rest capture (see [calibration.md](./calibration.md)).

## Units

- Positions are meters in every metric space, including the wire.
- Image quantities are pixels; reprojection and residual metrics are pixels.
- Angles are radians internally, degrees on the wire.
- Millimeters appear only in human-facing readouts (calibration spacing errors, jitter test results).

## Sides and handedness

`eHandSide` (Left/Right) indexes every per-hand array. A per-camera pipeline's side label is slot bookkeeping and can be displaced when two slots claim the same side; the classifier's actual opinion is `TrackedHand::rightProb` (flip-adjusted probability of being a right hand), and fusion-level side assignment treats only that, plus geometry, as evidence. See [hand-tracking.md](./hand-tracking.md).
