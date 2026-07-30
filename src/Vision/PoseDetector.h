#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "opencv2/core/mat.hpp"

#include "OnnxSession.h"
#include "SsdAnchors.h"
#include "TrackingTypes.h"

// One person found in the full frame.
// Keypoint semantics per mp_persondet.py:
//   0 hip center, 1 full-body ROI point, 2 shoulder center, 3 upper-body ROI point
// (the box is the detector's "face bbox"; only the keypoints are consumed
// downstream by the pose landmark stage)
struct PersonDetection
{
	glm::vec2 boxMin{0.f}; // full-frame px
	glm::vec2 boxMax{0.f};
	std::array<glm::vec2, 4> keypoints{};
	float score= 0.f;
};

// Ports opencv_zoo mp_persondet.py to ONNX Runtime.
// Model contract (person_detection.onnx):
//   input  input_1:0    [1,3,224,224] NCHW RGB float, (x/255 - 0.5) * 2 ->
//                       [-1,1], normalization applied BEFORE the aspect-
//                       preserving letterbox (pad value 0 == mid-gray)
//   output Identity:0   [1,2254,12] per-anchor regressors: box cx,cy,w,h in
//                       224-px input units + 4 keypoints (x,y)
//   output Identity_1:0 [1,2254,1]  score logits (clamped sigmoid applied)
// NOTE: the score output comes FIRST in the ONNX graph; outputs are mapped by
// last dimension, not position. Anchors generated (2254, see SsdAnchors).
// Score threshold 0.5, NMS IoU 0.3.
class PoseDetector
{
public:
	bool load(const std::string& modelPath, const std::string& preferredEp);
	bool isLoaded() const { return m_session.isValid(); }
	const char* activeEp() const { return m_session.activeEp(); }

	// Full-frame BGR -> person detections (weighted-NMS'd, score descending)
	void detect(const cv::Mat& bgrFrame, std::vector<PersonDetection>& outDetections);

	// Axis-aligned detection box (hip-centered ROI square) as a debug box
	static DetectionBox toDebugBox(const PersonDetection& detection);

private:
	OnnxSession m_session;
	std::vector<glm::vec2> m_anchors;
	int m_boxOutputIndex= -1;
	int m_scoreOutputIndex= -1;

	float m_scoreThreshold= 0.5f;
	float m_nmsIouThreshold= 0.3f;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<float> m_inputBuffer;
	cv::Mat m_rgbMat;
	cv::Mat m_normalizedMat;
	cv::Mat m_resizedMat;
	cv::Mat m_paddedMat;
	std::array<cv::Mat, 3> m_channelPlanes;
	std::vector<SsdDetection> m_rawDetections;
};
