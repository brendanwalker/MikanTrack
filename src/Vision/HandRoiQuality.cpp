#include "HandRoiQuality.h"

#include <algorithm>
#include <cmath>

#include "opencv2/imgproc.hpp"

namespace
{
// Clamps a rect to the frame and returns false if nothing usable remains
bool clampRectToFrame(cv::Rect& rect, const cv::Size& frameSize, int minDim)
{
	rect&= cv::Rect(0, 0, frameSize.width, frameSize.height);
	return rect.width >= minDim && rect.height >= minDim;
}

cv::Rect expandAboutCenter(const cv::Rect2f& box, float factor)
{
	const float centerX= box.x + box.width * 0.5f;
	const float centerY= box.y + box.height * 0.5f;
	const float halfWidth= box.width * 0.5f * factor;
	const float halfHeight= box.height * 0.5f * factor;
	return cv::Rect((int)std::floor(centerX - halfWidth), (int)std::floor(centerY - halfHeight),
					(int)std::ceil(halfWidth * 2.f), (int)std::ceil(halfHeight * 2.f));
}
} // namespace

void HandRoiQuality::analyzeHand(const cv::Mat& bgrFrame, TrackedHand& hand)
{
	hand.imageQuality= HandImageQuality();
	if (!hand.tracked || bgrFrame.empty())
		return;

	// Landmark bounding box in frame pixels
	glm::vec2 boxMin(bgrFrame.cols, bgrFrame.rows);
	glm::vec2 boxMax(0.f, 0.f);
	for (const glm::vec3& point : hand.imagePoints)
	{
		boxMin= glm::min(boxMin, glm::vec2(point.x, point.y));
		boxMax= glm::max(boxMax, glm::vec2(point.x, point.y));
	}
	const cv::Rect2f landmarkBox(boxMin.x, boxMin.y, boxMax.x - boxMin.x, boxMax.y - boxMin.y);
	if (landmarkBox.width <= 0.f || landmarkBox.height <= 0.f)
		return;

	cv::Rect innerRect= expandAboutCenter(landmarkBox, kRoiExpand);
	cv::Rect outerRect= expandAboutCenter(landmarkBox, kRoiExpand * kRingExpand);
	if (!clampRectToFrame(innerRect, bgrFrame.size(), 8) ||
		!clampRectToFrame(outerRect, bgrFrame.size(), 8))
		return;

	// Grayscale copy of the outer crop (its own buffer, so the filtered ops
	// below never reach outside it)
	cv::Mat grayOuter;
	cv::cvtColor(bgrFrame(outerRect), grayOuter, cv::COLOR_BGR2GRAY);

	// Decimate so the INNER ROI lands at kAnalysisMaxDim. Nearest-neighbor
	// SAMPLES pixels rather than averaging them, which keeps the clipping and
	// noise statistics intact.
	const int maxInnerDim= std::max(innerRect.width, innerRect.height);
	if (maxInnerDim > kAnalysisMaxDim)
	{
		const double scale= (double)kAnalysisMaxDim / (double)maxInnerDim;
		cv::resize(grayOuter, grayOuter, cv::Size(), scale, scale, cv::INTER_NEAREST);
		innerRect= cv::Rect((int)((innerRect.x - outerRect.x) * scale), (int)((innerRect.y - outerRect.y) * scale),
							(int)(innerRect.width * scale), (int)(innerRect.height * scale));
	}
	else
	{
		innerRect-= outerRect.tl();
	}
	if (!clampRectToFrame(innerRect, grayOuter.size(), 4))
		return;
	const cv::Mat inner= grayOuter(innerRect);

	// One histogram pass covers luminance, clipping and contrast
	int histogram[256]= {};
	for (int row= 0; row < inner.rows; ++row)
	{
		const uint8_t* pixel= inner.ptr<uint8_t>(row);
		for (int col= 0; col < inner.cols; ++col)
			++histogram[pixel[col]];
	}
	const double innerArea= (double)inner.rows * inner.cols;
	double innerSum= 0.0;
	double innerSumSq= 0.0;
	int shadowCount= 0;
	int highlightCount= 0;
	for (int value= 0; value < 256; ++value)
	{
		innerSum+= (double)value * histogram[value];
		innerSumSq+= (double)value * value * histogram[value];
		if (value <= 2)
			shadowCount+= histogram[value];
		if (value >= 253)
			highlightCount+= histogram[value];
	}
	const double innerMean= innerSum / innerArea;
	const double innerVariance= std::max(0.0, innerSumSq / innerArea - innerMean * innerMean);

	HandImageQuality& quality= hand.imageQuality;
	quality.meanLuma= (float)innerMean;
	quality.shadowClipRatio= (float)(shadowCount / innerArea);
	quality.highlightClipRatio= (float)(highlightCount / innerArea);
	quality.contrast= (float)std::sqrt(innerVariance);

	// Ring mean from the outer/inner sums (the band between the two boxes).
	// Internal contrast measures what the landmark regressor sees; separation
	// measures what the palm DETECTOR needs - a hand blending into the
	// background fails detection with perfectly good internal texture.
	const double outerArea= (double)grayOuter.rows * grayOuter.cols;
	const double ringArea= outerArea - innerArea;
	if (ringArea > 1.0)
	{
		const double ringMean= (cv::sum(grayOuter)[0] - innerSum) / ringArea;
		quality.backgroundSeparation= (float)std::abs(innerMean - ringMean);
	}

	// Noise = residual against a 3x3 median; sharpness = Laplacian variance of
	// the MEDIAN-FILTERED image. Splitting them matters: raw Laplacian
	// variance rewards sensor noise, scoring a noisy high-gain frame as
	// "sharp" in exactly the low-light case being diagnosed.
	cv::Mat median;
	cv::medianBlur(inner.clone(), median, 3);
	quality.noise= (float)cv::mean(cv::abs(inner - median))[0];

	cv::Mat laplacian;
	cv::Laplacian(median, laplacian, CV_16S, 3);
	cv::Scalar lapMean, lapStddev;
	cv::meanStdDev(laplacian, lapMean, lapStddev);
	quality.sharpness= (float)(lapStddev[0] * lapStddev[0]);

	quality.valid= true;
}

void LumaFlickerTracker::addFrame(const cv::Mat& bgrFrame, double timestampMs)
{
	if (bgrFrame.empty())
		return;

	// Reject stale/duplicate timestamps (they would corrupt the sample spacing)
	if (m_sampleCount > 0)
	{
		const int lastIndex= (m_writeIndex + kCapacity - 1) % kCapacity;
		if (timestampMs <= m_samples[lastIndex].timestampMs)
			return;
	}

	// Decimated frame mean: background-dominated, ~0.02 ms
	cv::Mat decimated;
	const double scale= std::min(1.0, 160.0 / (double)std::max(bgrFrame.cols, bgrFrame.rows));
	cv::resize(bgrFrame, decimated, cv::Size(), scale, scale, cv::INTER_NEAREST);
	const cv::Scalar channelMeans= cv::mean(decimated);
	const float luma= (float)((channelMeans[0] + channelMeans[1] + channelMeans[2]) / 3.0);

	m_samples[m_writeIndex]= {timestampMs, luma};
	m_writeIndex= (m_writeIndex + 1) % kCapacity;
	m_sampleCount= std::min(m_sampleCount + 1, kCapacity);

	if (timestampMs - m_lastAnalysisMs >= 500.0)
	{
		m_lastAnalysisMs= timestampMs;
		analyze();
	}
}

void LumaFlickerTracker::reset()
{
	m_sampleCount= 0;
	m_writeIndex= 0;
	m_lastAnalysisMs= 0.0;
	m_instability= 0.f;
	m_dominantHz= 0.f;
}

void LumaFlickerTracker::analyze()
{
	// Need at least ~1s of history before the numbers mean anything
	if (m_sampleCount < 64)
		return;

	// Chronological copy
	std::vector<Sample> ordered(m_sampleCount);
	const int oldestIndex= (m_writeIndex + kCapacity - m_sampleCount) % kCapacity;
	for (int i= 0; i < m_sampleCount; ++i)
		ordered[i]= m_samples[(oldestIndex + i) % kCapacity];

	const double spanSeconds= (ordered.back().timestampMs - ordered.front().timestampMs) / 1000.0;
	if (spanSeconds < 1.0)
		return;
	const double dtSeconds= spanSeconds / (m_sampleCount - 1);

	double meanLevel= 0.0;
	for (const Sample& sample : ordered)
		meanLevel+= sample.luma;
	meanLevel/= m_sampleCount;
	if (meanLevel < 1.0)
	{
		m_instability= 0.f;
		m_dominantHz= 0.f;
		return;
	}

	// Detrend with a ~0.5s centered moving average, so slow scene/exposure
	// drift stays out and only oscillation remains
	int halfWindow= std::max(2, (int)std::lround(0.25 / dtSeconds));
	halfWindow= std::min(halfWindow, (m_sampleCount - 1) / 2);
	std::vector<double> residual(m_sampleCount, 0.0);
	double acPower= 0.0;
	const int interiorCount= m_sampleCount - 2 * halfWindow;
	for (int i= halfWindow; i < m_sampleCount - halfWindow; ++i)
	{
		double windowSum= 0.0;
		for (int j= i - halfWindow; j <= i + halfWindow; ++j)
			windowSum+= ordered[j].luma;
		residual[i]= ordered[i].luma - windowSum / (2 * halfWindow + 1);
		acPower+= residual[i] * residual[i];
	}
	if (interiorCount < 32)
		return;
	const double acVariance= acPower / interiorCount;
	m_instability= (float)(std::sqrt(acVariance) / meanLevel);

	// Dominant frequency: project the residual onto candidate sinusoids using
	// the real timestamps (tolerates mildly uneven frame pacing). A frequency
	// only counts as dominant when it explains a large share of the AC
	// variance - broadband motion/AE noise never concentrates like that.
	m_dominantHz= 0.f;
	if (acVariance <= 1e-6)
		return;

	const double nyquistHz= 0.5 / dtSeconds;
	const double t0Seconds= ordered.front().timestampMs / 1000.0;
	double bestExplainedRatio= 0.0;
	float bestHz= 0.f;
	for (double freqHz= 0.5; freqHz < std::min(20.0, nyquistHz * 0.9); freqHz+= 0.25)
	{
		double cosSum= 0.0;
		double sinSum= 0.0;
		const double omega= 2.0 * CV_PI * freqHz;
		for (int i= halfWindow; i < m_sampleCount - halfWindow; ++i)
		{
			const double t= ordered[i].timestampMs / 1000.0 - t0Seconds;
			cosSum+= residual[i] * std::cos(omega * t);
			sinSum+= residual[i] * std::sin(omega * t);
		}
		const double amplitude= 2.0 * std::sqrt(cosSum * cosSum + sinSum * sinSum) / interiorCount;
		const double explainedRatio= (amplitude * amplitude * 0.5) / acVariance;
		if (explainedRatio > bestExplainedRatio)
		{
			bestExplainedRatio= explainedRatio;
			bestHz= (float)freqHz;
		}
	}
	if (bestExplainedRatio > 0.4)
		m_dominantHz= bestHz;
}
