#include "LumaFlickerTracker.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "opencv2/imgproc.hpp"

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
