#pragma once

#include <vector>

#include "glm/ext/vector_float2.hpp"

// MediaPipe SSD anchor generation + detection decode helpers.
//
// The opencv_zoo palm/person detection ONNX models do NOT embed anchors; the
// demo scripts (mp_palmdet.py / mp_persondet.py) carry a hardcoded anchor
// table that is exactly the output of MediaPipe's SsdAnchorsCalculator with
// fixed_anchor_size=true. Because anchor sizes are fixed, only the normalized
// (xCenter, yCenter) of each anchor participates in the decode, so anchors
// are represented here as glm::vec2 centers.
//
// Verified against the demo tables:
//   palm:   192x192, strides {8,16,16,16},  minScale 0.1484375, maxScale 0.75
//           -> 24*24*2 + 12*12*6 = 2016 anchors
//   person: 224x224, strides {8,16,32,32,32}, minScale 0.1484375, maxScale 0.75
//           -> 28*28*2 + 14*14*2 + 7*7*6 = 2254 anchors
struct SsdAnchorsConfig
{
	int inputWidth= 192;
	int inputHeight= 192;
	std::vector<int> strides;
	float minScale= 0.1484375f;
	float maxScale= 0.75f;
	float anchorOffsetX= 0.5f;
	float anchorOffsetY= 0.5f;
	std::vector<float> aspectRatios{1.f};
	// > 0 adds one extra interpolated-scale anchor per layer (MediaPipe default 1.0)
	float interpolatedScaleAspectRatio= 1.f;
};

// Anchor config matching mp_palmdet.py's embedded table (palm_detection.onnx)
SsdAnchorsConfig makePalmDetectorAnchorConfig();
// Anchor config matching mp_persondet.py's embedded table (person_detection.onnx)
SsdAnchorsConfig makePersonDetectorAnchorConfig();

// Ports mediapipe/calculators/tflite/ssd_anchors_calculator.cc
// (fixed_anchor_size variant; only centers are produced)
std::vector<glm::vec2> generateSsdAnchors(const SsdAnchorsConfig& config);

// One decoded SSD detection in full-frame pixel space
struct SsdDetection
{
	glm::vec2 boxMin{0.f};
	glm::vec2 boxMax{0.f};
	std::vector<glm::vec2> keypoints;
	float score= 0.f;
};

// MediaPipe-style weighted non-max suppression: overlapping candidates
// (IoU > iouThreshold) are blended into the highest-scoring candidate,
// with box corners and keypoints averaged weighted by score.
// ioDetections is replaced by the suppressed set, sorted by score descending.
// (The zoo demos use hard NMS via cv.dnn.NMSBoxes with the same 0.3 IoU
// threshold; weighted blending matches upstream MediaPipe and is strictly
// smoother for single-object-per-region use.)
void weightedNonMaxSuppression(std::vector<SsdDetection>& ioDetections, float iouThreshold);
