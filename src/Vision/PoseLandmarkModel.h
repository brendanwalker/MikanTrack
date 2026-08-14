#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "opencv2/core/mat.hpp"

#include "BodyPoseTypes.h"
#include "OnnxSession.h"
#include "PoseDetector.h"
#include "TrackingTypes.h"

// Person region fed to the pose landmark model: the hip-center point plus the
// "full body" ROI point that defines the crop radius and rotation
// (mp_pose.py consumes exactly these two of the detector's 4 keypoints).
struct PoseRoi
{
	glm::vec2 hipCenter{0.f};      // full-frame px
	glm::vec2 fullBodyPoint{0.f};  // full-frame px

	static PoseRoi fromPersonDetection(const PersonDetection& detection);
};

struct PoseLandmarkResult
{
	bool valid= false;
	float confidence= 0.f; // pose presence score, already in [0,1]

	// Full-frame px; z is relative to the hip center, x/y-pixel scale
	std::array<glm::vec3, POSE_LANDMARK_COUNT> imagePoints{};
	// sigmoid([0,1]): visibility= within frame and not occluded, presence= within frame
	std::array<float, POSE_LANDMARK_COUNT> visibility{};
	std::array<float, POSE_LANDMARK_COUNT> presence{};
	// World landmarks in meters, hip-centered; x/y de-rotated into frame orientation
	std::array<glm::vec3, POSE_LANDMARK_COUNT> worldPoints{};

	// Auxiliary landmarks 33/34 (body ROI center + scale point) mapped to
	// full-frame px: usable as next frame's PoseRoi (tracking mode)
	glm::vec2 auxRoiCenter{0.f};
	glm::vec2 auxRoiScalePoint{0.f};

	// The crop region actually used, mapped back to full-frame px
	DetectionBox usedRoi;
};

// Ports opencv_zoo mp_pose.py to ONNX Runtime.
// Model contract (pose_landmark.onnx):
//   input  input_1    [1,256,256,3] NHWC RGB float /255, rotated person crop
//   output Identity   [1,195] 39x(x,y,z,visibilityLogit,presenceLogit);
//                     x/y in 256-px crop space, z relative to hip;
//                     39= 33 landmarks + 6 auxiliary points
//   output Identity_1 [1,1]   confidence (already 0..1)
//   output Identity_2 [1,256,256,1] segmentation mask (unused here)
//   output Identity_3 [1,64,64,39]  landmark refine heatmap (unused here)
//   output Identity_4 [1,117] 39x(x,y,z) world landmarks, meters, hip-centered
// Crop math: square ROI of radius |hip - fullBodyPoint| around the hip,
// rotated so hip->fullBodyPoint points up, resized to 256.
class PoseLandmarkModel
{
public:
	bool load(const std::string& modelPath, const std::string& preferredEp);
	bool isLoaded() const { return m_session.isValid(); }
	const char* activeEp() const { return m_session.activeEp(); }

	// Runs the pose landmark model on the person ROI within the full BGR frame
	void estimate(const cv::Mat& bgrFrame, const PoseRoi& roi, PoseLandmarkResult& outResult);

private:
	OnnxSession m_session;
	int m_landmarkOutputIndex= 0;
	int m_confidenceOutputIndex= 1;
	int m_worldLandmarkOutputIndex= 4;

	// Preallocated scratch (steady-state: no per-frame heap churn)
	std::vector<float> m_inputBuffer;
	cv::Mat m_paddedMat;
	cv::Mat m_rotatedMat;
	cv::Mat m_resizedMat;
	cv::Mat m_rgbFloatMat;
};
