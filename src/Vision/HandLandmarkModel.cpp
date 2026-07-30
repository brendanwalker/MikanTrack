#include "HandLandmarkModel.h"

#include <algorithm>
#include <cmath>

#include "glm/common.hpp"
#include "glm/ext/vector_int2.hpp"
#include "opencv2/imgproc.hpp"

#include "Logger.h"

static constexpr int kHandInputSize= 224;

// mp_handpose.py crop constants
static constexpr float kPalmBoxPreEnlargeFactor= 4.f;
static constexpr float kPalmBoxShiftY= -0.4f;
static constexpr float kPalmBoxEnlargeFactor= 3.f;
static constexpr float kHandBoxShiftY= -0.1f;
static constexpr float kHandBoxEnlargeFactor= 1.65f;

// The 7 palm keypoints expressed as hand landmark ids (PALM_LANDMARK_IDS)
static constexpr int kPalmLandmarkIds[7]= {0, 5, 9, 13, 17, 1, 2};

HandRoi HandRoi::fromPalmDetection(const PalmDetection& detection)
{
	HandRoi roi;
	roi.palmBoxMin= detection.boxMin;
	roi.palmBoxMax= detection.boxMax;
	roi.palmKeypoints= detection.keypoints;
	return roi;
}

HandRoi HandRoi::fromLandmarks(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& imagePoints)
{
	HandRoi roi;
	for (int i= 0; i < 7; ++i)
		roi.palmKeypoints[i]= glm::vec2(imagePoints[kPalmLandmarkIds[i]]);

	roi.palmBoxMin= roi.palmKeypoints[0];
	roi.palmBoxMax= roi.palmKeypoints[0];
	for (int i= 1; i < 7; ++i)
	{
		roi.palmBoxMin= glm::min(roi.palmBoxMin, roi.palmKeypoints[i]);
		roi.palmBoxMax= glm::max(roi.palmBoxMax, roi.palmKeypoints[i]);
	}
	return roi;
}

bool HandLandmarkModel::load(const std::string& modelPath, const std::string& preferredEp)
{
	if (!m_session.create(modelPath, preferredEp))
		return false;

	// Map outputs by name (Identity/Identity_1/Identity_2/Identity_3, matching
	// the demo's tuple order: landmarks, conf, handedness, world landmarks);
	// fall back to positional order if the names differ.
	auto findByName= [this](const char* name, int fallback) -> int {
		for (size_t i= 0; i < m_session.getOutputCount(); ++i)
		{
			if (m_session.getOutputName(i) == name)
				return (int)i;
		}
		MIKAN_MT_LOG_WARNING("HandLandmarkModel::load")
			<< "Output '" << name << "' not found, assuming index " << fallback;
		return fallback;
	};
	m_landmarkOutputIndex= findByName("Identity", 0);
	m_confidenceOutputIndex= findByName("Identity_1", 1);
	m_handednessOutputIndex= findByName("Identity_2", 2);
	m_worldLandmarkOutputIndex= findByName("Identity_3", 3);

	m_inputBuffer.resize((size_t)kHandInputSize * kHandInputSize * 3);
	return true;
}

// Port of mp_handpose.py::_cropAndPadFromPalm.
// Shifts/enlarges the palm box, clips it to the image, crops, and pads the
// crop to a square (side= diagonal when forRotation so corners survive the
// later rotation). Returns false on a degenerate (empty) crop.
static bool cropAndPadFromPalm(
	const cv::Mat& image,
	glm::vec2 boxMin, glm::vec2 boxMax,
	bool forRotation,
	cv::Mat& outImage,
	glm::ivec2& outClippedBoxMin, glm::ivec2& outClippedBoxMax,
	glm::ivec2& outBias)
{
	// shift bounding box
	const glm::vec2 wh= boxMax - boxMin;
	const glm::vec2 shift= forRotation ? glm::vec2(0.f) : glm::vec2(0.f, kPalmBoxShiftY) * wh;
	boxMin+= shift;
	boxMax+= shift;

	// enlarge bounding box
	const glm::vec2 center= (boxMin + boxMax) * 0.5f;
	const float enlarge= forRotation ? kPalmBoxPreEnlargeFactor : kPalmBoxEnlargeFactor;
	const glm::vec2 halfSize= (boxMax - boxMin) * enlarge * 0.5f;

	// int truncation matches numpy astype(np.int32)
	glm::ivec2 intMin((int)(center.x - halfSize.x), (int)(center.y - halfSize.y));
	glm::ivec2 intMax((int)(center.x + halfSize.x), (int)(center.y + halfSize.y));

	// clip to image bounds
	intMin.x= std::clamp(intMin.x, 0, image.cols);
	intMin.y= std::clamp(intMin.y, 0, image.rows);
	intMax.x= std::clamp(intMax.x, 0, image.cols);
	intMax.y= std::clamp(intMax.y, 0, image.rows);

	const int cropW= intMax.x - intMin.x;
	const int cropH= intMax.y - intMin.y;
	if (cropW <= 0 || cropH <= 0)
		return false;

	const cv::Mat crop= image(cv::Rect(intMin.x, intMin.y, cropW, cropH));

	// pad to square so corner pixels survive rotation / keep aspect
	const int sideLen=
		forRotation
		? (int)std::sqrt((double)cropW * cropW + (double)cropH * cropH)
		: std::max(cropW, cropH);
	const int padW= sideLen - cropW;
	const int padH= sideLen - cropH;
	const int left= padW / 2;
	const int top= padH / 2;
	// BORDER_ISOLATED: crop is a ROI view; without it BORDER_CONSTANT would
	// pull real out-of-ROI pixels instead of black (the demo pads numpy
	// slices, which always pad black)
	cv::copyMakeBorder(
		crop, outImage,
		top, padH - top, left, padW - left,
		cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0, 0, 0));

	outClippedBoxMin= intMin;
	outClippedBoxMax= intMax;
	outBias= glm::ivec2(intMin.x - left, intMin.y - top);
	return true;
}

void HandLandmarkModel::estimate(
	const cv::Mat& bgrFrame,
	const HandRoi& roi,
	HandLandmarkResult& outResult)
{
	outResult= HandLandmarkResult();
	if (!m_session.isValid() || bgrFrame.empty())
		return;

	// -- Preprocess (mp_handpose.py::_preprocess) --
	// 1) generous pre-crop around the palm box (diagonal-square padded)
	glm::ivec2 clippedBoxMin, clippedBoxMax, bias;
	if (!cropAndPadFromPalm(bgrFrame, roi.palmBoxMin, roi.palmBoxMax, true,
							m_cropMat, clippedBoxMin, clippedBoxMax, bias))
		return;

	cv::cvtColor(m_cropMat, m_rgbMat, cv::COLOR_BGR2RGB);
	const glm::ivec2 padBias= bias;

	// palm box + keypoints in crop coordinates
	const glm::vec2 cropBoxMin= glm::vec2(clippedBoxMin - padBias);
	const glm::vec2 cropBoxMax= glm::vec2(clippedBoxMax - padBias);
	std::array<glm::vec2, 7> cropKeypoints;
	for (int i= 0; i < 7; ++i)
		cropKeypoints[i]= roi.palmKeypoints[i] - glm::vec2(padBias);

	// 2) rotation from wrist (kp0) -> middle MCP (kp2)
	const float radians= PalmDetector::computeRotation(cropKeypoints[0], cropKeypoints[2]);
	const double angleDeg= (double)radians * 180.0 / 3.14159265358979;

	const glm::vec2 rotationCenter= (cropBoxMin + cropBoxMax) * 0.5f;
	const cv::Mat rotationMatrix=
		cv::getRotationMatrix2D(cv::Point2f(rotationCenter.x, rotationCenter.y), angleDeg, 1.0);
	cv::warpAffine(m_rgbMat, m_rotatedMat, rotationMatrix, m_rgbMat.size());

	// rotate the palm keypoints and take their bounding box
	auto applyAffine= [&rotationMatrix](const glm::vec2& p) -> glm::vec2 {
		const double* m0= rotationMatrix.ptr<double>(0);
		const double* m1= rotationMatrix.ptr<double>(1);
		return glm::vec2(
			(float)(m0[0] * p.x + m0[1] * p.y + m0[2]),
			(float)(m1[0] * p.x + m1[1] * p.y + m1[2]));
	};
	glm::vec2 rotatedKpMin= applyAffine(cropKeypoints[0]);
	glm::vec2 rotatedKpMax= rotatedKpMin;
	for (int i= 1; i < 7; ++i)
	{
		const glm::vec2 p= applyAffine(cropKeypoints[i]);
		rotatedKpMin= glm::min(rotatedKpMin, p);
		rotatedKpMax= glm::max(rotatedKpMax, p);
	}

	// 3) final crop: rotated keypoint bbox shifted [0,-0.4], enlarged 3x
	glm::ivec2 rotatedBoxMin, rotatedBoxMax, unusedBias;
	if (!cropAndPadFromPalm(m_rotatedMat, rotatedKpMin, rotatedKpMax, false,
							m_finalCropMat, rotatedBoxMin, rotatedBoxMax, unusedBias))
		return;

	cv::resize(m_finalCropMat, m_resizedMat, cv::Size(kHandInputSize, kHandInputSize), 0.0, 0.0, cv::INTER_AREA);

	cv::Mat inputMat(kHandInputSize, kHandInputSize, CV_32FC3, m_inputBuffer.data());
	m_resizedMat.convertTo(inputMat, CV_32FC3, 1.0 / 255.0);

	const std::array<int64_t, 4> inputShape= {1, kHandInputSize, kHandInputSize, 3};
	Ort::Value inputTensor= Ort::Value::CreateTensor<float>(
		OnnxSession::getCpuMemoryInfo(),
		m_inputBuffer.data(), m_inputBuffer.size(),
		inputShape.data(), inputShape.size());

	std::vector<Ort::Value> outputs= m_session.run(&inputTensor, 1);

	// -- Postprocess (mp_handpose.py::_postprocess) --
	const float* landmarkData= outputs[m_landmarkOutputIndex].GetTensorData<float>();
	const float* worldData= outputs[m_worldLandmarkOutputIndex].GetTensorData<float>();
	outResult.confidence= outputs[m_confidenceOutputIndex].GetTensorData<float>()[0];
	outResult.handedness= outputs[m_handednessOutputIndex].GetTensorData<float>()[0];

	// scale from 224-crop space back to rotated-crop pixels
	const glm::vec2 whRotatedBox= glm::vec2(rotatedBoxMax - rotatedBoxMin);
	const float scaleFactor=
		std::max(whRotatedBox.x, whRotatedBox.y) / (float)kHandInputSize;

	// de-rotation matrix applied as row-vector * R (np.dot(lm, R[:, :2]))
	const cv::Mat coordsRotationMatrix= cv::getRotationMatrix2D(cv::Point2f(0.f, 0.f), angleDeg, 1.0);
	const double r00= coordsRotationMatrix.at<double>(0, 0);
	const double r01= coordsRotationMatrix.at<double>(0, 1);
	const double r10= coordsRotationMatrix.at<double>(1, 0);
	const double r11= coordsRotationMatrix.at<double>(1, 1);
	auto deRotate= [&](const glm::vec2& p) -> glm::vec2 {
		return glm::vec2(
			(float)(p.x * r00 + p.y * r10),
			(float)(p.x * r01 + p.y * r11));
	};

	// inverse of the first-stage rotation matrix, applied to the crop center
	const double m00= rotationMatrix.at<double>(0, 0);
	const double m01= rotationMatrix.at<double>(0, 1);
	const double m02= rotationMatrix.at<double>(0, 2);
	const double m10= rotationMatrix.at<double>(1, 0);
	const double m11= rotationMatrix.at<double>(1, 1);
	const double m12= rotationMatrix.at<double>(1, 2);
	const glm::vec2 rotatedCenter= glm::vec2(rotatedBoxMin + rotatedBoxMax) * 0.5f;
	const glm::vec2 originalCenter(
		(float)(rotatedCenter.x * m00 + rotatedCenter.y * m10 + (-(m00 * m02 + m10 * m12))),
		(float)(rotatedCenter.x * m01 + rotatedCenter.y * m11 + (-(m01 * m02 + m11 * m12))));

	const glm::vec2 finalOffset= originalCenter + glm::vec2(padBias);

	glm::vec2 lmMin(0.f), lmMax(0.f);
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		// crop-space -> centered/scaled -> de-rotated -> translated
		const glm::vec2 centered(
			(landmarkData[i * 3 + 0] - (float)kHandInputSize * 0.5f) * scaleFactor,
			(landmarkData[i * 3 + 1] - (float)kHandInputSize * 0.5f) * scaleFactor);
		const float z= landmarkData[i * 3 + 2] * scaleFactor;

		const glm::vec2 imagePos= deRotate(centered) + finalOffset;
		outResult.imagePoints[i]= glm::vec3(imagePos, z);

		// world landmarks: de-rotate x/y, keep z (meters, hand-centered)
		const glm::vec2 worldXy= deRotate(glm::vec2(worldData[i * 3 + 0], worldData[i * 3 + 1]));
		outResult.modelPoints[i]= glm::vec3(worldXy, worldData[i * 3 + 2]);

		if (i == 0)
		{
			lmMin= imagePos;
			lmMax= imagePos;
		}
		else
		{
			lmMin= glm::min(lmMin, imagePos);
			lmMax= glm::max(lmMax, imagePos);
		}
	}

	// hand bounding box: landmark bbox shifted [0,-0.1], enlarged 1.65
	{
		const glm::vec2 wh= lmMax - lmMin;
		const glm::vec2 shift= glm::vec2(0.f, kHandBoxShiftY) * wh;
		const glm::vec2 boxMin= lmMin + shift;
		const glm::vec2 boxMax= lmMax + shift;
		const glm::vec2 center= (boxMin + boxMax) * 0.5f;
		const glm::vec2 halfSize= (boxMax - boxMin) * kHandBoxEnlargeFactor * 0.5f;
		outResult.handBoxMin= center - halfSize;
		outResult.handBoxMax= center + halfSize;
	}

	// debug box: the final crop region mapped back to full-frame pixels
	{
		const glm::vec2 corners[4]= {
			glm::vec2(rotatedBoxMin),
			glm::vec2((float)rotatedBoxMin.x, (float)rotatedBoxMax.y),
			glm::vec2(rotatedBoxMax),
			glm::vec2((float)rotatedBoxMax.x, (float)rotatedBoxMin.y)};
		for (int i= 0; i < 4; ++i)
			outResult.usedRoi.corners[i]= deRotate(corners[i] - rotatedCenter) + finalOffset;
		outResult.usedRoi.rotationRadians= radians;
		outResult.usedRoi.score= outResult.confidence;
	}

	outResult.valid= true;
}
