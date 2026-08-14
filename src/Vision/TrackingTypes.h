#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "glm/ext/quaternion_common.hpp"
#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"

#include "BodyPoseTypes.h"

// Shared result types passed from the vision/inference thread to the render
// thread and the OSC streamer.

// MediaPipe hand landmark indices (21 points)
enum class eHandLandmark : int
{
	WRIST= 0,
	THUMB_CMC= 1,
	THUMB_MCP= 2,
	THUMB_IP= 3,
	THUMB_TIP= 4,
	INDEX_MCP= 5,
	INDEX_PIP= 6,
	INDEX_DIP= 7,
	INDEX_TIP= 8,
	MIDDLE_MCP= 9,
	MIDDLE_PIP= 10,
	MIDDLE_DIP= 11,
	MIDDLE_TIP= 12,
	RING_MCP= 13,
	RING_PIP= 14,
	RING_DIP= 15,
	RING_TIP= 16,
	PINKY_MCP= 17,
	PINKY_PIP= 18,
	PINKY_DIP= 19,
	PINKY_TIP= 20,
};
constexpr int HAND_LANDMARK_COUNT= 21;

// Bone connection table for wireframe drawing (MediaPipe HAND_CONNECTIONS)
constexpr int HAND_CONNECTION_COUNT= 21;
constexpr int HAND_CONNECTIONS[HAND_CONNECTION_COUNT][2]= {
	{0, 1},  {1, 2},  {2, 3},  {3, 4},        // thumb
	{0, 5},  {5, 6},  {6, 7},  {7, 8},        // index
	{5, 9},  {9, 10}, {10, 11}, {11, 12},     // middle
	{9, 13}, {13, 14}, {14, 15}, {15, 16},    // ring
	{13, 17}, {17, 18}, {18, 19}, {19, 20},   // pinky
	{0, 17},                                  // palm edge
};

enum class eHandSide : int
{
	Left= 0,
	Right= 1,
	Count= 2,
};

// A palm/person detection in full-frame pixel space (debug overlay + ROI seed)
struct DetectionBox
{
	// Oriented box corners in full-frame pixels, CCW
	std::array<glm::vec2, 4> corners;
	float rotationRadians= 0.f;
	float score= 0.f;
};

// Image statistics of a hand's ROI in the frame the model consumed -
// diagnostics that explain landmark jitter in terms of the knobs that fix it
// (exposure, gain, lighting, background). All statistics are computed on a
// nearest-neighbor decimated grayscale crop with a fixed ~160px working size,
// so values are comparable across camera resolutions and ROI sizes. Filled by
// HandRoiQuality on the vision thread.
struct HandImageQuality
{
	bool valid= false;
	float meanLuma= 0.f;             // mean gray level in the ROI, 0-255
	float shadowClipRatio= 0.f;      // fraction of ROI pixels stuck at black (<= 2)
	float highlightClipRatio= 0.f;   // fraction of ROI pixels blown out (>= 253)
	float contrast= 0.f;             // gray-level stddev inside the ROI (skin texture)
	float backgroundSeparation= 0.f; // |ROI mean - surrounding ring mean|, gray levels
	float sharpness= 0.f;            // variance of Laplacian AFTER a 3x3 median, so it
	                                 // measures blur without rewarding sensor noise
	float noise= 0.f;                // mean |gray - 3x3 median| residual (sensor noise)
};

struct TrackedHand
{
	bool tracked= false;
	eHandSide side= eHandSide::Left;
	int slotId= -1;
	float presence= 0.f;        // landmark model presence/confidence score
	float handednessScore= 0.f; // raw model handedness (per opencv_zoo mp_handpose.py:
	                            // 0=left .. 1=right in model terms, assuming a
	                            // mirrored/selfie view; see flipHandedness)
	// Flip-adjusted probability this is a RIGHT hand. THIS is the classifier's
	// actual opinion - `side` can be displaced by slot bookkeeping (two slots
	// claiming the same side) and must not be treated as evidence
	float rightProb= 0.5f;

	// Landmarks in full-frame pixels; z is MediaPipe relative depth
	// (same scale as x/y pixels, relative to the wrist)
	std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints;

	// Model-space "world" landmarks (meters, hand-centered) from the landmark
	// model, used for foreshortening correction and scale estimation
	std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints;

	// Camera-space landmarks in meters (valid when hasCameraSpace)
	bool hasCameraSpace= false;
	std::array<glm::vec3, HAND_LANDMARK_COUNT> cameraPoints;

	// World (marker-anchored) landmarks in meters (valid when hasWorldSpace)
	bool hasWorldSpace= false;
	std::array<glm::vec3, HAND_LANDMARK_COUNT> worldPoints;

	// Lighting/exposure diagnostics for this hand's ROI (valid when tracked
	// and the ROI was large enough to analyze)
	HandImageQuality imageQuality;
};

// -- Parametric hand representation -----
// Palm transform + per-finger bend angles (Ultraleap-style), computed from
// the landmark model's metric hand. Angles live in the model's LOCAL
// articulation - the network's most reliable output - and are scale- and
// depth-invariant, so all depth noise concentrates in the palm transform.
// This is also the natural state for cross-camera fusion (poses/angles
// compose; blending raw landmarks distorts bones) and for a future EKF
// (position + error quaternion + angle vector).

constexpr int FINGER_COUNT= 5;
enum class eFinger : int
{
	Thumb= 0,
	Index= 1,
	Middle= 2,
	Ring= 3,
	Pinky= 4,
};

// MediaPipe landmark indices of each finger's 4 joints, base -> tip.
// (Thumb: CMC, MCP, IP, TIP; fingers: MCP, PIP, DIP, TIP)
constexpr int FINGER_JOINTS[FINGER_COUNT][4]= {
	{1, 2, 3, 4},     // thumb
	{5, 6, 7, 8},     // index
	{9, 10, 11, 12},  // middle
	{13, 14, 15, 16}, // ring
	{17, 18, 19, 20}, // pinky
};

// Bend angles for one finger, radians, all ZERO in the rest pose (see
// HandSkeleton::neutralDirInPalm). Sign conventions, in the palm frame:
//   lateral:      splay about palm +Z, POSITIVE COUNTER-CLOCKWISE, i.e.
//                 toward palm +Y = cross(palmZ, palmX). Purely geometric -
//                 the same rotation direction for either hand (so on a right
//                 hand positive is toward the pinky, on a left toward the
//                 thumb; the palm frame itself carries the chirality).
//   proximal:     base bone away from the neutral direction, POSITIVE
//                 CURLING TOWARD THE PALM (+Z, the palmar side)
//   intermediate: bend of the middle bone RELATIVE TO THE PROXIMAL BONE,
//                 positive toward the palm (0 = collinear with it)
//   distal:       bend of the tip bone RELATIVE TO THE INTERMEDIATE BONE,
//                 positive toward the palm (0 = collinear with it)
// The three bends chain: each is measured against its parent bone, not
// against the palm, so a finger curling evenly reads three similar values.
struct FingerAngles
{
	float lateral= 0.f;
	float proximal= 0.f;
	float intermediate= 0.f;
	float distal= 0.f;
};

// Slowly-varying skeleton geometry in the palm frame, metric (scaled by the
// calibrated hand scale; MediaPipe's model is canonical average-hand scale)
struct HandSkeleton
{
	// Each finger's base joint position in the palm frame
	std::array<glm::vec3, FINGER_COUNT> baseInPalm{};
	// Phalanx lengths base->tip: [proximal, intermediate, distal]
	std::array<std::array<float, 3>, FINGER_COUNT> phalanxLengths{};
	// Direction each finger's proximal phalanx points when all four of its
	// angles are zero, in the palm frame - where forward kinematics starts.
	// Always the flat-hand convention: the four fingers parallel to palm +X,
	// the thumb along its own metacarpal. (A captured rest pose does NOT
	// change this; it is applied as an angle offset instead, because it is
	// per-camera and the streamed skeleton has to mean one thing.)
	std::array<glm::vec3, FINGER_COUNT> neutralDirInPalm{};
};

// Palm frame convention (Ultraleap-compatible):
//   origin: palm center (midpoint of wrist and middle MCP)
//   +X: toward the fingers (wrist -> middle MCP)
//   +Z: out of the palmar surface (chirality-corrected per hand)
//   +Y: completes right-handed
struct HandPose
{
	bool tracked= false;
	eHandSide side= eHandSide::Left;
	float presence= 0.f;
	float visibility= 0.f; // palm face-on factor for the observing camera

	// How much this pose can be trusted, [0,1] = presence x measured
	// stability. Unlike presence (which only answers "is a hand here" and
	// stays high on an ill-conditioned edge-on view), this tracks the
	// observed pose noise, so it is the number to gate output on. After
	// fusion it is the best contributing camera's confidence.
	float confidence= 0.f;

	// Validation: mean pixel distance between this camera's 2D landmarks and
	// the forward-kinematics hand rebuilt from THIS pose (palm transform +
	// angles + skeleton) projected back into the image. Small = the
	// parametric model faithfully represents what was actually seen.
	// 0 when not evaluated (e.g. after fusion, which has no single camera).
	float fkReprojectionPx= 0.f;

	// True when this pose came from cross-camera triangulation of the 2D
	// landmarks (stereo geometry) rather than a single camera's monocular
	// depth estimate - the network's depth output is not in the loop at all
	bool stereoTriangulated= false;

	bool hasCameraPose= false;
	glm::vec3 palmPositionCamera{0.f}; // OpenCV camera convention, meters
	glm::quat palmOrientationCamera{1.f, 0.f, 0.f, 0.f};

	bool hasWorldPose= false;
	glm::vec3 palmPositionWorld{0.f}; // marker-anchored Z-up world, meters
	glm::quat palmOrientationWorld{1.f, 0.f, 0.f, 0.f};

	// FOREARM orientation from a wrist-strapped IMU (world space). A wrist
	// strap sits PROXIMAL to the wrist joint, so the sensor rotates with the
	// forearm, not the hand - treating it as palm orientation would be wrong
	// exactly when the wrist is flexed. Keeping it as its own frame is what
	// makes the wrist joint angle measurable (see getWristRotation) and
	// restores the forearm direction that arm IK wants.
	bool hasForearmPose= false;
	glm::quat forearmOrientationWorld{1.f, 0.f, 0.f, 0.f};

	// How much the forearm estimate - and so the elbow derived from it - can
	// be trusted, [0,1], 0 when there is no forearm pose at all. Distinct from
	// `confidence`, which scores the HAND: the elbow additionally depends on
	// the IMU's mounting calibration, and a mounting that is subtly wrong
	// leaves the palm perfect while swinging the elbow around it.
	float forearmConfidence= 0.f;

	// Shoulder estimate from the vision body-pose solver, world space. Unlike
	// the forearm it has no IMU source: valid only when a body-pose camera saw
	// the shoulder and a world-anchored wrist existed to anchor it.
	bool hasShoulder= false;
	glm::vec3 shoulderPositionWorld{0.f};
	float shoulderConfidence= 0.f;

	// Wrist joint rotation: the palm expressed IN the forearm frame, i.e.
	// the local rotation a skeleton hierarchy applies to the hand bone.
	// Identity when the palm points straight along the forearm. Only
	// meaningful when hasForearmPose && hasWorldPose.
	glm::quat getWristRotation() const
	{
		return glm::inverse(forearmOrientationWorld) * palmOrientationWorld;
	}

	// Wrist JOINT position in world space. The palm origin is the palm
	// CENTER (midway between the wrist and the middle MCP), so the wrist sits
	// half a palm back along the palm's -X axis.
	glm::vec3 getWristPositionWorld() const
	{
		const float halfPalm= skeleton.baseInPalm[(int)eFinger::Middle].x;
		return palmPositionWorld + palmOrientationWorld * glm::vec3(-halfPalm, 0.f, 0.f);
	}

	// Elbow estimate: straight back along the forearm from the wrist. The
	// forearm frame's +X points toward the hand (it equals the palm frame at
	// a neutral wrist), so the elbow lies along -X. Only meaningful when
	// hasForearmPose - this is a rigid extrapolation from a MEASURED forearm
	// direction, not an inference, so its error is just the length guess.
	glm::vec3 getElbowPositionWorld(float forearmLengthMeters) const
	{
		return getWristPositionWorld() -
			forearmOrientationWorld * glm::vec3(1.f, 0.f, 0.f) * forearmLengthMeters;
	}

	std::array<FingerAngles, FINGER_COUNT> fingers{};
	HandSkeleton skeleton;
};

struct TrackingFrameResult
{
	int64_t frameIndex= -1;
	double timestampMs= 0.0;
	int frameWidth= 0;
	int frameHeight= 0;

	float captureFps= 0.f;
	float inferenceMs= 0.f;

	// Whole-frame temporal luminance stability (decimated frame mean tracked
	// over the last few seconds): detrended AC RMS as a fraction of the mean
	// level. High = light flicker beating against the shutter, or auto-exposure
	// hunting - both destabilize landmarks while every static-frame statistic
	// looks fine.
	float lumaInstability= 0.f;
	// Dominant oscillation frequency behind lumaInstability, Hz (0 until one
	// stands out from the broadband motion background)
	float lumaFlickerHz= 0.f;

	// Indexed by eHandSide. hands carries the raw landmark data (overlays,
	// debug, stereo scale); poses is the parametric output that gets fused
	// and streamed.
	std::array<TrackedHand, 2> hands;
	std::array<HandPose, 2> poses;

	// Raw BlazePose observation from this camera's opt-in body-pose stage
	// (invalid on cameras without the stage). On the FUSED result this is
	// unused; the solved outputs live on poses[] (forearm/shoulder) and head.
	BodyPoseObservation body;

	// Head pose from the vision body-pose solver, world space. Body-level
	// rather than per-side; only ever filled on the fused result.
	struct HeadPose
	{
		bool valid= false;
		glm::vec3 positionWorld{0.f};
		glm::quat orientationWorld{1.f, 0.f, 0.f, 0.f};
		float confidence= 0.f;
	};
	HeadPose head;

	// Debug: raw palm detector output + active hand ROI boxes
	std::vector<DetectionBox> palmDetections;
};
