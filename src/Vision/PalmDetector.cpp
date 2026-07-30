#include "PalmDetector.h"

#include <algorithm>
#include <cmath>

#include "opencv2/imgproc.hpp"

#include "Logger.h"

static constexpr int kPalmInputSize= 192;
static constexpr int kPalmKeypointCount= 7;
static constexpr int kPalmRegressorWidth= 18; // 4 box + 7*2 keypoints
static constexpr float kPi= 3.14159265358979f;
static constexpr float kTwoPi= 6.28318530717959f;

static float stableSigmoid(float x)
{
	const float clamped= std::clamp(x, -100.f, 100.f);
	return 1.f / (1.f + std::exp(-clamped));
}

bool PalmDetector::load(const std::string& modelPath, const std::string& preferredEp)
{
	if (!m_session.create(modelPath, preferredEp))
		return false;

	// Map outputs by last dimension: regressors are [1,2016,18], scores [1,2016,1]
	m_boxOutputIndex= m_session.findOutputByLastDim(kPalmRegressorWidth);
	m_scoreOutputIndex= m_session.findOutputByLastDim(1);
	if (m_boxOutputIndex < 0 || m_scoreOutputIndex < 0)
	{
		MIKAN_MT_LOG_ERROR("PalmDetector::load")
			<< "Unexpected output shapes in " << modelPath;
		return false;
	}

	m_anchors= generateSsdAnchors(makePalmDetectorAnchorConfig());
	m_inputBuffer.resize((size_t)kPalmInputSize * kPalmInputSize * 3);
	return true;
}

float PalmDetector::computeRotation(const glm::vec2& wristCenter, const glm::vec2& middleMcp)
{
	// mp_handpose.py: radians= pi/2 - atan2(-(p2.y - p1.y), p2.x - p1.x),
	// normalized to [-pi, pi)
	float radians=
		kPi * 0.5f -
		std::atan2(-(middleMcp.y - wristCenter.y), middleMcp.x - wristCenter.x);
	radians-= kTwoPi * std::floor((radians + kPi) / kTwoPi);
	return radians;
}

DetectionBox PalmDetector::toDebugBox(const PalmDetection& detection)
{
	DetectionBox box;
	box.corners[0]= detection.boxMin;
	box.corners[1]= glm::vec2(detection.boxMin.x, detection.boxMax.y);
	box.corners[2]= detection.boxMax;
	box.corners[3]= glm::vec2(detection.boxMax.x, detection.boxMin.y);
	box.rotationRadians= detection.rotationRadians;
	box.score= detection.score;
	return box;
}

void PalmDetector::detect(const cv::Mat& bgrFrame, std::vector<PalmDetection>& outDetections)
{
	outDetections.clear();
	if (!m_session.isValid() || bgrFrame.empty())
		return;

	const int frameWidth= bgrFrame.cols;
	const int frameHeight= bgrFrame.rows;

	// -- Preprocess (mp_palmdet.py::_preprocess) --
	// Aspect-preserving resize into 192x192 with centered black padding
	const double ratio= std::min(
		(double)kPalmInputSize / (double)frameHeight,
		(double)kPalmInputSize / (double)frameWidth);
	int padBiasX= 0;
	int padBiasY= 0;

	const cv::Mat* letterboxed= &bgrFrame;
	if (frameWidth != kPalmInputSize || frameHeight != kPalmInputSize)
	{
		const int ratioW= (int)((double)frameWidth * ratio);
		const int ratioH= (int)((double)frameHeight * ratio);
		cv::resize(bgrFrame, m_resizedMat, cv::Size(ratioW, ratioH));

		const int padW= kPalmInputSize - ratioW;
		const int padH= kPalmInputSize - ratioH;
		const int left= padW / 2;
		const int top= padH / 2;
		cv::copyMakeBorder(
			m_resizedMat, m_paddedMat,
			top, padH - top, left, padW - left,
			cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		letterboxed= &m_paddedMat;

		// pad bias reported in full-frame pixels
		padBiasX= (int)((double)left / ratio);
		padBiasY= (int)((double)top / ratio);
	}

	cv::cvtColor(*letterboxed, m_rgbMat, cv::COLOR_BGR2RGB);

	// NHWC float tensor over the preallocated buffer, normalized to [0,1]
	cv::Mat inputMat(kPalmInputSize, kPalmInputSize, CV_32FC3, m_inputBuffer.data());
	m_rgbMat.convertTo(inputMat, CV_32FC3, 1.0 / 255.0);

	const std::array<int64_t, 4> inputShape= {1, kPalmInputSize, kPalmInputSize, 3};
	Ort::Value inputTensor= Ort::Value::CreateTensor<float>(
		OnnxSession::getCpuMemoryInfo(),
		m_inputBuffer.data(), m_inputBuffer.size(),
		inputShape.data(), inputShape.size());

	std::vector<Ort::Value> outputs= m_session.run(&inputTensor, 1);

	// -- Postprocess (mp_palmdet.py::_postprocess) --
	const float* boxData= outputs[m_boxOutputIndex].GetTensorData<float>();
	const float* scoreData= outputs[m_scoreOutputIndex].GetTensorData<float>();
	const size_t anchorCount= std::min(
		m_anchors.size(),
		(size_t)outputs[m_scoreOutputIndex].GetTensorTypeAndShapeInfo().GetShape()[1]);

	const float scale= (float)std::max(frameWidth, frameHeight);
	const float invInputSize= 1.f / (float)kPalmInputSize;

	m_rawDetections.clear();
	for (size_t i= 0; i < anchorCount; ++i)
	{
		const float score= stableSigmoid(scoreData[i]);
		if (score < m_scoreThreshold)
			continue;

		const float* regressor= boxData + i * kPalmRegressorWidth;
		const glm::vec2& anchor= m_anchors[i];

		const float cxDelta= regressor[0] * invInputSize;
		const float cyDelta= regressor[1] * invInputSize;
		const float wDelta= regressor[2] * invInputSize;
		const float hDelta= regressor[3] * invInputSize;

		SsdDetection detection;
		detection.score= score;
		detection.boxMin= glm::vec2(
			(cxDelta - wDelta * 0.5f + anchor.x) * scale - (float)padBiasX,
			(cyDelta - hDelta * 0.5f + anchor.y) * scale - (float)padBiasY);
		detection.boxMax= glm::vec2(
			(cxDelta + wDelta * 0.5f + anchor.x) * scale - (float)padBiasX,
			(cyDelta + hDelta * 0.5f + anchor.y) * scale - (float)padBiasY);

		detection.keypoints.resize(kPalmKeypointCount);
		for (int k= 0; k < kPalmKeypointCount; ++k)
		{
			detection.keypoints[k]= glm::vec2(
				(regressor[4 + 2 * k] * invInputSize + anchor.x) * scale - (float)padBiasX,
				(regressor[5 + 2 * k] * invInputSize + anchor.y) * scale - (float)padBiasY);
		}

		m_rawDetections.push_back(std::move(detection));
	}

	weightedNonMaxSuppression(m_rawDetections, m_nmsIouThreshold);

	for (const SsdDetection& raw : m_rawDetections)
	{
		PalmDetection detection;
		detection.boxMin= raw.boxMin;
		detection.boxMax= raw.boxMax;
		detection.score= raw.score;
		for (int k= 0; k < kPalmKeypointCount; ++k)
			detection.keypoints[k]= raw.keypoints[k];
		detection.rotationRadians= computeRotation(detection.keypoints[0], detection.keypoints[2]);

		outDetections.push_back(detection);
	}
}
