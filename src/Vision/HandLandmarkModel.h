#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "opencv2/core/mat.hpp"

#include "OnnxSession.h"
#include "PalmDetector.h"
#include "TrackingTypes.h"

// Palm region fed to the landmark model. Keypoint order matches
// PalmDetection::keypoints (mp_handpose.py PALM_LANDMARK_IDS= [0,5,9,13,17,1,2]):
//   0 wrist, 1 index MCP, 2 middle MCP, 3 ring MCP, 4 pinky MCP,
//   5 thumb CMC, 6 thumb MCP
// The crop math only uses the keypoints' bounding geometry plus kp0->kp2 for
// rotation, so a ROI synthesized from previous-frame landmarks behaves the
// same as one from the detector.
struct HandRoi
{
	glm::vec2 palmBoxMin{0.f}; // full-frame px
	glm::vec2 palmBoxMax{0.f};
	std::array<glm::vec2, 7> palmKeypoints{};

	static HandRoi fromPalmDetection(const PalmDetection& detection);
	// Tracked-mode ROI: palm keypoints pulled out of the previous frame's 21
	// landmarks; palm box is their bounding box (the demo's crop math re-derives
	// the final crop from the rotated keypoint bbox, so this stays faithful)
	static HandRoi fromLandmarks(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& imagePoints);
};

struct HandLandmarkResult
{
	bool valid= false;
	float confidence= 0.f; // presence/confidence score, already in [0,1]
	float handedness= 0.f; // raw model score; per mp_handpose.py: 0=left .. 1=right
	                       // (model terms, assumes mirrored/selfie input)

	// Full-frame px; z is relative to the wrist, in the same scale as x/y px
	std::array<glm::vec3, HAND_LANDMARK_COUNT> imagePoints{};
	// Model/world-space landmarks in meters, hand-centered. x/y are de-rotated
	// into frame orientation (per the demo); z is the raw model depth.
	std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints{};

	// Demo's output hand box (landmark bbox shifted [0,-0.1], enlarged 1.65):
	// used for slot IoU dedupe against new palm detections
	glm::vec2 handBoxMin{0.f};
	glm::vec2 handBoxMax{0.f};

	// The rotated crop region actually used, mapped back to full-frame px
	DetectionBox usedRoi;
};

// Ports opencv_zoo mp_handpose.py to ONNX Runtime.
// Model contract (hand_landmark.onnx):
//   input  input_1    [1,224,224,3] NHWC RGB float /255, rotated palm crop
//   output Identity   [1,63] 21x(x,y,z), x/y in 224-px crop space, z rel wrist
//   output Identity_1 [1,1]  presence/confidence (already 0..1)
//   output Identity_2 [1,1]  handedness (0=left .. 1=right, model terms)
//   output Identity_3 [1,63] 21x(x,y,z) world landmarks, meters, hand-centered
// Crop math: pre-crop palm box enlarged 4x (diagonal-square padded), rotate by
// wrist->middle-MCP angle, re-crop rotated keypoint bbox shifted [0,-0.4] and
// enlarged 3x (square padded), resize to 224 with INTER_AREA.
class HandLandmarkModel
{
public:
	bool load(const std::string& modelPath, const std::string& preferredEp);
	bool isLoaded() const { return m_session.isValid(); }
	const char* activeEp() const { return m_session.activeEp(); }

	// Runs the landmark model on the palm ROI within the full BGR frame.
	// outResult.valid is false when the crop degenerates (ROI off-frame).
	void estimate(const cv::Mat& bgrFrame, const HandRoi& roi, HandLandmarkResult& outResult);

private:
	OnnxSession m_session;
	int m_landmarkOutputIndex= 0;
	int m_confidenceOutputIndex= 1;
	int m_handednessOutputIndex= 2;
	int m_worldLandmarkOutputIndex= 3;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<float> m_inputBuffer;
	cv::Mat m_cropMat;
	cv::Mat m_rgbMat;
	cv::Mat m_rotatedMat;
	cv::Mat m_finalCropMat;
	cv::Mat m_resizedMat;
};
