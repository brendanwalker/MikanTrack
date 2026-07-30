#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "opencv2/core/mat.hpp"

#include "OnnxSession.h"
#include "SsdAnchors.h"
#include "TrackingTypes.h"

// One palm found in the full frame.
// Keypoint order follows the MediaPipe palm detector (matches the semantics
// mp_handpose.py assigns via PALM_LANDMARK_IDS= [0,5,9,13,17,1,2]):
//   0 wrist center, 1 index MCP, 2 middle MCP, 3 ring MCP, 4 pinky MCP,
//   5 thumb CMC, 6 thumb MCP
struct PalmDetection
{
	glm::vec2 boxMin{0.f}; // palm bounding box, full-frame px
	glm::vec2 boxMax{0.f};
	std::array<glm::vec2, 7> keypoints{};
	float score= 0.f;
	// MediaPipe ROI rotation: pi/2 - atan2(-(kp2.y-kp0.y), kp2.x-kp0.x),
	// normalized to [-pi, pi)
	float rotationRadians= 0.f;
};

// Ports opencv_zoo mp_palmdet.py to ONNX Runtime.
// Model contract (palm_detection.onnx):
//   input  input_1    [1,192,192,3] NHWC RGB float, /255, aspect-preserving
//                     letterbox with black padding
//   output Identity   [1,2016,18]  per-anchor regressors: box cx,cy,w,h in
//                     192-px input units + 7 keypoints (x,y)
//   output Identity_1 [1,2016,1]   per-anchor score logits (sigmoid applied)
// Anchors: generated (2016, see SsdAnchors). Score threshold 0.5, NMS IoU 0.3.
class PalmDetector
{
public:
	bool load(const std::string& modelPath, const std::string& preferredEp);
	bool isLoaded() const { return m_session.isValid(); }
	const char* activeEp() const { return m_session.activeEp(); }

	// Full-frame BGR -> palm detections (weighted-NMS'd, score descending)
	void detect(const cv::Mat& bgrFrame, std::vector<PalmDetection>& outDetections);

	// Axis-aligned detection box as a debug overlay box
	static DetectionBox toDebugBox(const PalmDetection& detection);

	// MediaPipe ROI rotation from wrist center -> middle finger MCP
	static float computeRotation(const glm::vec2& wristCenter, const glm::vec2& middleMcp);

private:
	OnnxSession m_session;
	std::vector<glm::vec2> m_anchors;
	int m_boxOutputIndex= -1;
	int m_scoreOutputIndex= -1;

	float m_scoreThreshold= 0.5f;
	float m_nmsIouThreshold= 0.3f;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<float> m_inputBuffer;
	cv::Mat m_resizedMat;
	cv::Mat m_paddedMat;
	cv::Mat m_rgbMat;
	std::vector<SsdDetection> m_rawDetections;
};
