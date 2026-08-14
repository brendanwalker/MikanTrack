#include "RtmPoseBodyModel.h"

#include <algorithm>
#include <cmath>

#include "opencv2/imgproc.hpp"

#include "Logger.h"

static constexpr int kInputWidth= 192;
static constexpr int kInputHeight= 256;
static constexpr int kSimccX= kInputWidth * 2;  // simcc_split_ratio 2.0
static constexpr int kSimccY= kInputHeight * 2;
static constexpr float kBoxPadding= 1.25f;

// ImageNet statistics, per the model's own deploy pipeline
static const cv::Scalar k_mean(123.675, 116.28, 103.53);
static const cv::Scalar k_std(58.395, 57.12, 57.375);

bool RtmPoseBodyModel::load(const std::string& modelPath, const std::string& preferredEp)
{
	if (!m_session.create(modelPath, preferredEp))
		return false;

	m_simccXOutputIndex= m_session.findOutputByLastDim(kSimccX);
	m_simccYOutputIndex= m_session.findOutputByLastDim(kSimccY);
	if (m_simccXOutputIndex < 0 || m_simccYOutputIndex < 0)
	{
		MIKAN_MT_LOG_ERROR("RtmPoseBodyModel::load") << "Unexpected output shapes in " << modelPath;
		return false;
	}

	m_inputBuffer.resize((size_t)kInputWidth * kInputHeight * 3);
	const size_t planeSize= (size_t)kInputWidth * kInputHeight;
	for (int c= 0; c < 3; ++c)
		m_channelPlanes[c]= cv::Mat(kInputHeight, kInputWidth, CV_32FC1, m_inputBuffer.data() + planeSize * c);

	return true;
}

void RtmPoseBodyModel::estimate(const cv::Mat& bgrFrame, const glm::vec2& boxMin, const glm::vec2& boxMax,
								RtmPoseResult& outResult)
{
	outResult= RtmPoseResult();
	if (!m_session.isValid() || bgrFrame.empty())
		return;

	// -- Box -> center/scale, aspect-corrected then padded (the model's
	// TopDownGetBboxCenterScale) --
	const glm::vec2 center= (boxMin + boxMax) * 0.5f;
	float boxWidth= boxMax.x - boxMin.x;
	float boxHeight= boxMax.y - boxMin.y;
	if (boxWidth < 2.f || boxHeight < 2.f)
		return;

	const float aspect= (float)kInputWidth / (float)kInputHeight;
	if (boxWidth > boxHeight * aspect)
		boxHeight= boxWidth / aspect;
	else
		boxWidth= boxHeight * aspect;
	boxWidth*= kBoxPadding;
	boxHeight*= kBoxPadding;

	// -- Affine to the 192x256 canvas (MMPose get_warp_matrix): the source
	// triangle is the box's center plus its "up" edge, so the crop keeps the
	// box upright and centered --
	const cv::Point2f srcDir(0.f, boxWidth * -0.5f);
	const cv::Point2f dstDir(0.f, (float)kInputWidth * -0.5f);
	cv::Point2f src[3];
	cv::Point2f dst[3];
	src[0]= cv::Point2f(center.x, center.y);
	src[1]= src[0] + srcDir;
	src[2]= src[1] + cv::Point2f(-srcDir.y, srcDir.x);
	dst[0]= cv::Point2f(kInputWidth * 0.5f, kInputHeight * 0.5f);
	dst[1]= dst[0] + dstDir;
	dst[2]= dst[1] + cv::Point2f(-dstDir.y, dstDir.x);

	const cv::Mat warpMatrix= cv::getAffineTransform(src, dst);
	cv::warpAffine(bgrFrame, m_cropMat, warpMatrix, cv::Size(kInputWidth, kInputHeight), cv::INTER_LINEAR);

	// -- Normalize into the preallocated NCHW buffer --
	cv::cvtColor(m_cropMat, m_rgbMat, cv::COLOR_BGR2RGB);
	m_rgbMat.convertTo(m_rgbMat, CV_32FC3);
	cv::subtract(m_rgbMat, k_mean, m_rgbMat);
	cv::divide(m_rgbMat, k_std, m_rgbMat);
	cv::split(m_rgbMat, m_channelPlanes.data());

	const std::array<int64_t, 4> inputShape= {1, 3, kInputHeight, kInputWidth};
	Ort::Value inputTensor= Ort::Value::CreateTensor<float>(
		OnnxSession::getCpuMemoryInfo(),
		m_inputBuffer.data(), m_inputBuffer.size(),
		inputShape.data(), inputShape.size());

	std::vector<Ort::Value> outputs= m_session.run(&inputTensor, 1);
	if (outputs.size() < 2)
		return;

	const float* simccX= outputs[m_simccXOutputIndex].GetTensorData<float>();
	const float* simccY= outputs[m_simccYOutputIndex].GetTensorData<float>();

	// -- Decode: per-axis argmax / split ratio, then back through the crop --
	cv::Mat inverseWarp;
	cv::invertAffineTransform(warpMatrix, inverseWarp);
	const double i00= inverseWarp.at<double>(0, 0), i01= inverseWarp.at<double>(0, 1),
				 i02= inverseWarp.at<double>(0, 2);
	const double i10= inverseWarp.at<double>(1, 0), i11= inverseWarp.at<double>(1, 1),
				 i12= inverseWarp.at<double>(1, 2);

	for (int keypoint= 0; keypoint < COCO_KEYPOINT_COUNT; ++keypoint)
	{
		const float* rowX= simccX + (size_t)keypoint * kSimccX;
		const float* rowY= simccY + (size_t)keypoint * kSimccY;

		int bestX= 0;
		float bestXValue= rowX[0];
		for (int i= 1; i < kSimccX; ++i)
		{
			if (rowX[i] > bestXValue)
			{
				bestXValue= rowX[i];
				bestX= i;
			}
		}
		int bestY= 0;
		float bestYValue= rowY[0];
		for (int i= 1; i < kSimccY; ++i)
		{
			if (rowY[i] > bestYValue)
			{
				bestYValue= rowY[i];
				bestY= i;
			}
		}

		const double cropX= bestX * 0.5;
		const double cropY= bestY * 0.5;
		outResult.points[keypoint]= glm::vec2(
			(float)(cropX * i00 + cropY * i01 + i02),
			(float)(cropX * i10 + cropY * i11 + i12));
		// The weaker axis governs: a keypoint is only as located as its worst
		// coordinate
		outResult.scores[keypoint]= std::min(bestXValue, bestYValue);
	}

	outResult.valid= true;
}
