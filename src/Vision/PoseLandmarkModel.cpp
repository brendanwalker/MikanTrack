#include "PoseLandmarkModel.h"

#include <algorithm>
#include <cmath>

#include "glm/ext/vector_int2.hpp"
#include "glm/geometric.hpp"
#include "opencv2/imgproc.hpp"

#include "Logger.h"

static constexpr int kPoseInputSize= 256;
static constexpr int kPoseRawLandmarkCount= 39; // 33 landmarks + 6 auxiliary
static constexpr int kPoseAuxRoiCenterIndex= 33;
static constexpr int kPoseAuxRoiScaleIndex= 34;
static constexpr float kPi= 3.14159265358979f;
static constexpr float kTwoPi= 6.28318530717959f;

static float stableSigmoid(float x)
{
	const float clamped= std::clamp(x, -100.f, 100.f);
	return 1.f / (1.f + std::exp(-clamped));
}

PoseRoi PoseRoi::fromPersonDetection(const PersonDetection& detection)
{
	PoseRoi roi;
	roi.hipCenter= detection.keypoints[0];
	roi.fullBodyPoint= detection.keypoints[1];
	return roi;
}

bool PoseLandmarkModel::load(const std::string& modelPath, const std::string& preferredEp)
{
	if (!m_session.create(modelPath, preferredEp))
		return false;

	// Map outputs by name (demo tuple order: landmarks, conf, mask, heatmap,
	// world landmarks); fall back to positional order
	auto findByName= [this](const char* name, int fallback) -> int {
		for (size_t i= 0; i < m_session.getOutputCount(); ++i)
		{
			if (m_session.getOutputName(i) == name)
				return (int)i;
		}
		MIKAN_MT_LOG_WARNING("PoseLandmarkModel::load")
			<< "Output '" << name << "' not found, assuming index " << fallback;
		return fallback;
	};
	m_landmarkOutputIndex= findByName("Identity", 0);
	m_confidenceOutputIndex= findByName("Identity_1", 1);
	m_worldLandmarkOutputIndex= findByName("Identity_4", 4);

	m_inputBuffer.resize((size_t)kPoseInputSize * kPoseInputSize * 3);
	return true;
}

void PoseLandmarkModel::estimate(
	const cv::Mat& bgrFrame,
	const PoseRoi& roi,
	PoseLandmarkResult& outResult)
{
	outResult= PoseLandmarkResult();
	if (!m_session.isValid() || bgrFrame.empty())
		return;

	// -- Preprocess (mp_pose.py::_preprocess) --
	// Square ROI of radius |hip - fullBodyPoint| around the hip center
	const float fullDist= glm::length(roi.hipCenter - roi.fullBodyPoint);
	if (fullDist < 2.f)
		return;

	// int truncation matches numpy astype(np.int32)
	const glm::ivec2 fullBoxMin((int)(roi.hipCenter.x - fullDist), (int)(roi.hipCenter.y - fullDist));
	const glm::ivec2 fullBoxMax((int)(roi.hipCenter.x + fullDist), (int)(roi.hipCenter.y + fullDist));

	glm::ivec2 clippedMin(
		std::clamp(fullBoxMin.x, 0, bgrFrame.cols),
		std::clamp(fullBoxMin.y, 0, bgrFrame.rows));
	glm::ivec2 clippedMax(
		std::clamp(fullBoxMax.x, 0, bgrFrame.cols),
		std::clamp(fullBoxMax.y, 0, bgrFrame.rows));

	const int cropW= clippedMax.x - clippedMin.x;
	const int cropH= clippedMax.y - clippedMin.y;
	if (cropW <= 0 || cropH <= 0)
		return;

	const cv::Mat crop= bgrFrame(cv::Rect(clippedMin.x, clippedMin.y, cropW, cropH));

	// pad the clipped edges back out so the hip stays centered
	const int left= clippedMin.x - fullBoxMin.x;
	const int top= clippedMin.y - fullBoxMin.y;
	const int right= fullBoxMax.x - clippedMax.x;
	const int bottom= fullBoxMax.y - clippedMax.y;
	// BORDER_ISOLATED: crop is a ROI view; without it BORDER_CONSTANT would
	// pull real out-of-ROI pixels instead of black (the demo pads numpy
	// slices, which always pad black)
	cv::copyMakeBorder(
		crop, m_paddedMat,
		top, bottom, left, right,
		cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(0, 0, 0));

	const glm::ivec2 padBias= fullBoxMin;

	// rotation so hip -> fullBodyPoint points "up" in the crop
	const glm::vec2 hipCrop= roi.hipCenter - glm::vec2(padBias);
	const glm::vec2 fullBodyCrop= roi.fullBodyPoint - glm::vec2(padBias);
	float radians=
		kPi * 0.5f -
		std::atan2(-(fullBodyCrop.y - hipCrop.y), fullBodyCrop.x - hipCrop.x);
	radians-= kTwoPi * std::floor((radians + kPi) / kTwoPi);
	const double angleDeg= (double)radians * 180.0 / 3.14159265358979;

	const cv::Mat rotationMatrix=
		cv::getRotationMatrix2D(cv::Point2f(hipCrop.x, hipCrop.y), angleDeg, 1.0);
	cv::warpAffine(m_paddedMat, m_rotatedMat, rotationMatrix, m_paddedMat.size());

	cv::resize(m_rotatedMat, m_resizedMat, cv::Size(kPoseInputSize, kPoseInputSize), 0.0, 0.0, cv::INTER_AREA);

	// float, BGR -> RGB, [0,1] (demo converts the float blob)
	cv::Mat inputMat(kPoseInputSize, kPoseInputSize, CV_32FC3, m_inputBuffer.data());
	m_resizedMat.convertTo(m_rgbFloatMat, CV_32FC3, 1.0 / 255.0);
	cv::cvtColor(m_rgbFloatMat, inputMat, cv::COLOR_BGR2RGB);

	const std::array<int64_t, 4> inputShape= {1, kPoseInputSize, kPoseInputSize, 3};
	Ort::Value inputTensor= Ort::Value::CreateTensor<float>(
		OnnxSession::getCpuMemoryInfo(),
		m_inputBuffer.data(), m_inputBuffer.size(),
		inputShape.data(), inputShape.size());

	// Only the three outputs this decode reads. The model also emits a
	// segmentation mask and a refine heatmap that nothing here consumes.
	// (Measured: this saves nothing on DirectML - the runtime was already
	// pruning them - but asking for unread tensors invites paying for them.)
	const int requestedOutputs[3]= {
		m_landmarkOutputIndex, m_confidenceOutputIndex, m_worldLandmarkOutputIndex};
	std::vector<Ort::Value> outputs= m_session.runOutputs(&inputTensor, 1, requestedOutputs, 3);
	if (outputs.size() != 3)
		return;

	// -- Postprocess (mp_pose.py::_postprocess) --
	const float* landmarkData= outputs[0].GetTensorData<float>();
	const float* worldData= outputs[2].GetTensorData<float>();
	outResult.confidence= outputs[1].GetTensorData<float>()[0];

	// scale from 256-crop space back to rotated-crop pixels (per axis)
	const glm::vec2 whRotatedBox((float)m_paddedMat.cols, (float)m_paddedMat.rows);
	const glm::vec2 scaleFactor= whRotatedBox / (float)kPoseInputSize;
	const float maxScaleFactor= std::max(scaleFactor.x, scaleFactor.y);

	// de-rotation applied as row-vector * R (np.dot(lm, R[:, :2]))
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

	// inverse of the first-stage rotation applied to the crop center
	const double m00= rotationMatrix.at<double>(0, 0);
	const double m01= rotationMatrix.at<double>(0, 1);
	const double m02= rotationMatrix.at<double>(0, 2);
	const double m10= rotationMatrix.at<double>(1, 0);
	const double m11= rotationMatrix.at<double>(1, 1);
	const double m12= rotationMatrix.at<double>(1, 2);
	const glm::vec2 rotatedCenter= whRotatedBox * 0.5f;
	const glm::vec2 originalCenter(
		(float)(rotatedCenter.x * m00 + rotatedCenter.y * m10 + (-(m00 * m02 + m10 * m12))),
		(float)(rotatedCenter.x * m01 + rotatedCenter.y * m11 + (-(m01 * m02 + m11 * m12))));
	const glm::vec2 finalOffset= originalCenter + glm::vec2(padBias);

	auto decodeImagePoint= [&](int rawIndex) -> glm::vec3 {
		const float* lm= landmarkData + rawIndex * 5;
		const glm::vec2 centered(
			(lm[0] - (float)kPoseInputSize * 0.5f) * scaleFactor.x,
			(lm[1] - (float)kPoseInputSize * 0.5f) * scaleFactor.y);
		const float z= lm[2] * maxScaleFactor;
		return glm::vec3(deRotate(centered) + finalOffset, z);
	};

	for (int i= 0; i < POSE_LANDMARK_COUNT; ++i)
	{
		outResult.imagePoints[i]= decodeImagePoint(i);
		outResult.visibility[i]= stableSigmoid(landmarkData[i * 5 + 3]);
		outResult.presence[i]= stableSigmoid(landmarkData[i * 5 + 4]);

		const glm::vec2 worldXy= deRotate(glm::vec2(worldData[i * 3 + 0], worldData[i * 3 + 1]));
		outResult.worldPoints[i]= glm::vec3(worldXy, worldData[i * 3 + 2]);
	}

	// auxiliary ROI landmarks for next-frame tracking
	outResult.auxRoiCenter= glm::vec2(decodeImagePoint(kPoseAuxRoiCenterIndex));
	outResult.auxRoiScalePoint= glm::vec2(decodeImagePoint(kPoseAuxRoiScaleIndex));

	// debug box: the rotated crop mapped back to full-frame pixels
	{
		const glm::vec2 corners[4]= {
			glm::vec2(0.f, 0.f),
			glm::vec2(0.f, whRotatedBox.y),
			whRotatedBox,
			glm::vec2(whRotatedBox.x, 0.f)};
		for (int i= 0; i < 4; ++i)
			outResult.usedRoi.corners[i]= deRotate(corners[i] - rotatedCenter) + finalOffset;
		outResult.usedRoi.rotationRadians= radians;
		outResult.usedRoi.score= outResult.confidence;
	}

	outResult.valid= true;
}
