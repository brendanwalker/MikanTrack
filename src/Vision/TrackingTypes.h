#pragma once

#include <array>
#include <cstdint>
#include <vector>

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

struct TrackedArm
{
	bool valid= false;
	// Elbows are geometric estimates extended from the hand orientation
	// (BlazePose measurement was removed - it never fires on overhead rigs);
	// kept in the schema so OSC consumers can distinguish estimate quality
	bool fromFallback= true;
	float confidence= 0.f;

	glm::vec2 elbowPixel{0.f};
	glm::vec2 wristPixel{0.f};

	bool hasCameraSpace= false;
	glm::vec3 elbowCamera{0.f};
	glm::vec3 wristCamera{0.f};

	bool hasWorldSpace= false;
	glm::vec3 elbowWorld{0.f};
	glm::vec3 wristWorld{0.f};
};

struct TrackingFrameResult
{
	int64_t frameIndex= -1;
	double timestampMs= 0.0;
	int frameWidth= 0;
	int frameHeight= 0;

	float captureFps= 0.f;
	float inferenceMs= 0.f;

	// Indexed by eHandSide
	std::array<TrackedHand, 2> hands;
	std::array<TrackedArm, 2> arms;

	// Debug: raw palm detector output + active hand ROI boxes
	std::vector<DetectionBox> palmDetections;
};
