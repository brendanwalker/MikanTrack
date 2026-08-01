#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"

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

struct TrackedHand
{
	bool tracked= false;
	eHandSide side= eHandSide::Left;
	int slotId= -1;
	float presence= 0.f;        // landmark model presence/confidence score
	float handednessScore= 0.f; // raw model handedness (per opencv_zoo mp_handpose.py:
	                            // 0=left .. 1=right in model terms, assuming a
	                            // mirrored/selfie view; see flipHandedness)

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

// Bend angles for one finger, radians, relative to the neutral (straight,
// along the metacarpal direction) pose:
//   lateral:      signed splay in the palm plane (+ toward the thumb side)
//   proximal:     base-bone curl toward the palm (+ = curling in)
//   intermediate: hinge angle vs the proximal bone (0 = straight)
//   distal:       hinge angle vs the intermediate bone (0 = straight)
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

	bool hasCameraPose= false;
	glm::vec3 palmPositionCamera{0.f}; // OpenCV camera convention, meters
	glm::quat palmOrientationCamera{1.f, 0.f, 0.f, 0.f};

	bool hasWorldPose= false;
	glm::vec3 palmPositionWorld{0.f}; // marker-anchored Z-up world, meters
	glm::quat palmOrientationWorld{1.f, 0.f, 0.f, 0.f};

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

	// Indexed by eHandSide. hands carries the raw landmark data (overlays,
	// debug, stereo scale); poses is the parametric output that gets fused
	// and streamed.
	std::array<TrackedHand, 2> hands;
	std::array<HandPose, 2> poses;

	// Debug: raw palm detector output + active hand ROI boxes
	std::vector<DetectionBox> palmDetections;
};
