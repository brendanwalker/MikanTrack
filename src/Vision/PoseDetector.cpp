#include "PoseDetector.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "opencv2/imgproc.hpp"

#include "Logger.h"

static constexpr int kPersonInputSize= 224;
static constexpr int kPersonKeypointCount= 4;
static constexpr int kPersonRegressorWidth= 12; // 4 box + 4*2 keypoints

static float stableSigmoid(float x)
{
	const float clamped= std::clamp(x, -100.f, 100.f);
	return 1.f / (1.f + std::exp(-clamped));
}

bool PoseDetector::load(const std::string& modelPath, const std::string& preferredEp)
{
	if (!m_session.create(modelPath, preferredEp))
		return false;

	// The graph lists the score output first; map by last dim instead
	m_boxOutputIndex= m_session.findOutputByLastDim(kPersonRegressorWidth);
	m_scoreOutputIndex= m_session.findOutputByLastDim(1);
	if (m_boxOutputIndex < 0 || m_scoreOutputIndex < 0)
	{
		MIKAN_MT_LOG_ERROR("PoseDetector::load")
			<< "Unexpected output shapes in " << modelPath;
		return false;
	}

	m_anchors= generateSsdAnchors(makePersonDetectorAnchorConfig());
	m_inputBuffer.resize((size_t)kPersonInputSize * kPersonInputSize * 3);

	// NCHW plane views over the shared input buffer
	const size_t planeSize= (size_t)kPersonInputSize * kPersonInputSize;
	for (int c= 0; c < 3; ++c)
	{
		m_channelPlanes[c]= cv::Mat(
			kPersonInputSize, kPersonInputSize, CV_32FC1,
			m_inputBuffer.data() + planeSize * c);
	}
	return true;
}

DetectionBox PoseDetector::toDebugBox(const PersonDetection& detection)
{
	// Draw the ROI actually consumed by the pose stage: the hip-centered
	// square reaching the full-body keypoint (mp_pose.py's person box)
	const glm::vec2& hip= detection.keypoints[0];
	const glm::vec2& fullBody= detection.keypoints[1];
	const float dist= glm::length(fullBody - hip);

	DetectionBox box;
	box.corners[0]= hip + glm::vec2(-dist, -dist);
	box.corners[1]= hip + glm::vec2(-dist, dist);
	box.corners[2]= hip + glm::vec2(dist, dist);
	box.corners[3]= hip + glm::vec2(dist, -dist);
	box.rotationRadians= 0.f;
	box.score= detection.score;
	return box;
}

void PoseDetector::detect(const cv::Mat& bgrFrame, std::vector<PersonDetection>& outDetections)
{
	outDetections.clear();
	if (!m_session.isValid() || bgrFrame.empty())
		return;

	const int frameWidth= bgrFrame.cols;
	const int frameHeight= bgrFrame.rows;

	// -- Preprocess (mp_persondet.py::_preprocess) --
	// Normalize FIRST ([0,255] BGR -> [-1,1] RGB), then aspect-preserving
	// resize + centered zero padding (zero == mid-gray after normalization)
	cv::cvtColor(bgrFrame, m_rgbMat, cv::COLOR_BGR2RGB);
	m_rgbMat.convertTo(m_normalizedMat, CV_32FC3, 2.0 / 255.0, -1.0);

	const double ratio= std::min(
		(double)kPersonInputSize / (double)frameHeight,
		(double)kPersonInputSize / (double)frameWidth);
	int padBiasX= 0;
	int padBiasY= 0;

	const cv::Mat* letterboxed= &m_normalizedMat;
	if (frameWidth != kPersonInputSize || frameHeight != kPersonInputSize)
	{
		const int ratioW= (int)((double)frameWidth * ratio);
		const int ratioH= (int)((double)frameHeight * ratio);
		cv::resize(m_normalizedMat, m_resizedMat, cv::Size(ratioW, ratioH));

		const int padW= kPersonInputSize - ratioW;
		const int padH= kPersonInputSize - ratioH;
		const int left= padW / 2;
		const int top= padH / 2;
		cv::copyMakeBorder(
			m_resizedMat, m_paddedMat,
			top, padH - top, left, padW - left,
			cv::BORDER_CONSTANT, cv::Scalar(0.f, 0.f, 0.f));
		letterboxed= &m_paddedMat;

		padBiasX= (int)((double)left / ratio);
		padBiasY= (int)((double)top / ratio);
	}

	// HWC -> CHW into the preallocated NCHW buffer
	cv::split(*letterboxed, m_channelPlanes.data());

	const std::array<int64_t, 4> inputShape= {1, 3, kPersonInputSize, kPersonInputSize};
	Ort::Value inputTensor= Ort::Value::CreateTensor<float>(
		OnnxSession::getCpuMemoryInfo(),
		m_inputBuffer.data(), m_inputBuffer.size(),
		inputShape.data(), inputShape.size());

	std::vector<Ort::Value> outputs= m_session.run(&inputTensor, 1);

	// -- Postprocess (mp_persondet.py::_postprocess) --
	const float* boxData= outputs[m_boxOutputIndex].GetTensorData<float>();
	const float* scoreData= outputs[m_scoreOutputIndex].GetTensorData<float>();
	const size_t anchorCount= std::min(
		m_anchors.size(),
		(size_t)outputs[m_scoreOutputIndex].GetTensorTypeAndShapeInfo().GetShape()[1]);

	const float scale= (float)std::max(frameWidth, frameHeight);
	const float invInputSize= 1.f / (float)kPersonInputSize;

	m_rawDetections.clear();
	for (size_t i= 0; i < anchorCount; ++i)
	{
		const float score= stableSigmoid(scoreData[i]);
		if (score < m_scoreThreshold)
			continue;

		const float* regressor= boxData + i * kPersonRegressorWidth;
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

		detection.keypoints.resize(kPersonKeypointCount);
		for (int k= 0; k < kPersonKeypointCount; ++k)
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
		PersonDetection detection;
		detection.boxMin= raw.boxMin;
		detection.boxMax= raw.boxMax;
		detection.score= raw.score;
		for (int k= 0; k < kPersonKeypointCount; ++k)
			detection.keypoints[k]= raw.keypoints[k];

		outDetections.push_back(detection);
	}
}
