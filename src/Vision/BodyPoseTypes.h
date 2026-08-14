#pragma once

#include <array>
#include <cstdint>

#include "glm/ext/vector_float3.hpp"

// BlazePose landmark indices (33-point topology). Only the indices the app
// consumes are named; the full set rides through observations and recordings
// so later consumers (VMC output) don't force a schema change.
enum class ePoseLandmark : int
{
	NOSE= 0,
	LEFT_EYE_INNER= 1,
	LEFT_EYE= 2,
	LEFT_EYE_OUTER= 3,
	RIGHT_EYE_INNER= 4,
	RIGHT_EYE= 5,
	RIGHT_EYE_OUTER= 6,
	LEFT_EAR= 7,
	RIGHT_EAR= 8,
	LEFT_SHOULDER= 11,
	RIGHT_SHOULDER= 12,
	LEFT_ELBOW= 13,
	RIGHT_ELBOW= 14,
	LEFT_WRIST= 15,
	RIGHT_WRIST= 16,
};

constexpr int POSE_LANDMARK_COUNT= 33;

// BlazePose's own skeleton topology, for overlays. Shared so the live preview
// and the diagnostic dump cannot draw two different skeletons.
constexpr int POSE_CONNECTION_COUNT= 35;
constexpr int POSE_CONNECTIONS[POSE_CONNECTION_COUNT][2]= {
	// face
	{0, 1}, {1, 2}, {2, 3}, {3, 7}, {0, 4}, {4, 5}, {5, 6}, {6, 8}, {9, 10},
	// arms
	{11, 13}, {13, 15}, {15, 17}, {15, 19}, {15, 21}, {17, 19},
	{12, 14}, {14, 16}, {16, 18}, {16, 20}, {16, 22}, {18, 20},
	// torso
	{11, 12}, {11, 23}, {12, 24}, {23, 24},
	// legs
	{23, 25}, {25, 27}, {27, 29}, {27, 31}, {29, 31},
	{24, 26}, {26, 28}, {28, 30}, {28, 32}, {30, 32},
};

// One camera's body-pose result for a frame: the per-camera body-pose stage
// fills it, and it rides TrackingFrameResult through the fusion candidate
// mirrors, recordings, replay, and diagnostics. Landmark sides are
// body-semantic (they name the person's own left/right).
// Where the person box for a model frame came from. Recorded because a
// top-down backend is only as good as its box, and "the detector missed" and
// "the tracked box drifted" fail identically from the outside.
enum class eBodyBoxSource : int
{
	None= 0,
	Detector= 1,   // the person detector fired this model frame
	Tracked= 2,    // grown from the previous model frame's keypoints
	FullFrame= 3,  // nothing else available; the whole image
};

struct BodyPoseObservation
{
	bool valid= false;
	eBodyBoxSource boxSource= eBodyBoxSource::None;

	// Tracker frame index of the model run that produced these landmarks.
	// Divided-out frames re-emit the held observation with this unchanged, so
	// downstream jitter tracking can tell a fresh model result from a re-emit.
	int64_t modelFrameIndex= -1;

	// Landmark-model confidence [0,1]. A gate, not a quality signal: world
	// estimates derive their confidence from measured stability instead.
	float confidence= 0.f;

	// Undistorted full-frame px; z is model-relative depth (hip-centered,
	// x/y-pixel scale)
	std::array<glm::vec3, POSE_LANDMARK_COUNT> imagePoints{};

	// Bit per landmark: which slots this backend actually fills. The 33-slot
	// BlazePose layout is the canonical index space, so a backend with a
	// different topology (RTMPose emits COCO's 17) leaves the rest untouched
	// at the image origin - and an overlay that drew them would pin a phantom
	// landmark in the corner of the frame.
	uint32_t providedMask= 0;
	bool isProvided(int landmarkIndex) const
	{
		return (providedMask & (1u << landmarkIndex)) != 0;
	}

	// sigmoid [0,1]: within frame and not occluded
	std::array<float, POSE_LANDMARK_COUNT> visibility{};
};
