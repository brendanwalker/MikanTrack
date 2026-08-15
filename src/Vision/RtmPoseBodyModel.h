#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "opencv2/core/mat.hpp"

#include "OnnxSession.h"

// COCO 17-keypoint order, which is what the RTMPose body models emit
enum class eCocoKeypoint : int
{
	NOSE= 0,
	LEFT_EYE= 1,
	RIGHT_EYE= 2,
	LEFT_EAR= 3,
	RIGHT_EAR= 4,
	LEFT_SHOULDER= 5,
	RIGHT_SHOULDER= 6,
	LEFT_ELBOW= 7,
	RIGHT_ELBOW= 8,
	LEFT_WRIST= 9,
	RIGHT_WRIST= 10,
	LEFT_HIP= 11,
	RIGHT_HIP= 12,
	LEFT_KNEE= 13,
	RIGHT_KNEE= 14,
	LEFT_ANKLE= 15,
	RIGHT_ANKLE= 16,
};
constexpr int COCO_KEYPOINT_COUNT= 17;

struct RtmPoseResult
{
	bool valid= false;
	std::array<glm::vec2, COCO_KEYPOINT_COUNT> points{}; // full-frame px
	std::array<float, COCO_KEYPOINT_COUNT> scores{};     // SimCC peak, ~[0,1]
};

// RTMPose body (SimCC) via ONNX Runtime.
//
// TOP-DOWN, which is the reason this model was chosen: the caller supplies the
// person box and the model predicts each keypoint independently inside it.
// The holistic model tried before it owned its own region of interest, derived
// from a hip center and a "full body" point, and always emitted one coherent
// full-body skeleton - so a person truncated at a desk had no correct crop and
// got a fabricated lower body that dragged the arms with it.
//
// Model contract (rtmpose_body.onnx, MMDeploy end2end export):
//   input  input    [1,3,256,192] NCHW RGB, (x - mean) / std with the
//                   ImageNet statistics below, from a 1.25-padded
//                   aspect-corrected affine crop of the person box
//   output simcc_x  [1,17,384]  per-keypoint x class scores (192 * 2)
//   output simcc_y  [1,17,512]  per-keypoint y class scores (256 * 2)
// Decode is argmax over each axis divided by the 2.0 split ratio, mapped back
// through the inverse affine; the score is min(peakX, peakY).
class RtmPoseBodyModel
{
public:
	bool load(const std::string& modelPath, const std::string& preferredEp);
	bool isLoaded() const { return m_session.isValid(); }
	const char* activeEp() const { return m_session.activeEp(); }

	// boxMin/boxMax: the person box in full-frame px
	void estimate(const cv::Mat& bgrFrame, const glm::vec2& boxMin, const glm::vec2& boxMax,
				  RtmPoseResult& outResult);

private:
	OnnxSession m_session;
	int m_simccXOutputIndex= 0;
	int m_simccYOutputIndex= 1;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<float> m_inputBuffer;
	cv::Mat m_cropMat;
	cv::Mat m_rgbMat;
	std::array<cv::Mat, 3> m_channelPlanes;
};
