#pragma once

#include <array>

#include "opencv2/core/mat.hpp"

// Tracks the whole-frame mean luminance over the last few seconds and measures
// its oscillation. Mains/PWM light flicker beating against the shutter and
// auto-exposure hunting both live here - and both destabilize landmarks while
// every single-frame statistic looks fine. The frame mean is dominated by the
// static background, so hand motion contributes little.
//
// Runs on the vision thread against the exact frame the model consumed. The
// per-frame cost is one nearest-neighbor decimation and one mean (~0.02 ms);
// the analysis is a 78-frequency scan over the sample ring, twice a second.
class LumaFlickerTracker
{
public:
	// Feeds one processed frame (decimated mean) and refreshes the analysis
	// roughly twice a second
	void addFrame(const cv::Mat& bgrFrame, double timestampMs);

	// Detrended AC RMS of the frame mean luminance as a fraction of its mean
	// level (0.02 = 2% ripple)
	float getInstability() const { return m_instability; }

	// Dominant oscillation frequency, Hz. 0 until a single frequency stands
	// out from the broadband background: periodic = flicker beat, 0 with high
	// instability = aperiodic auto-exposure hunting.
	float getDominantHz() const { return m_dominantHz; }

	void reset();

private:
	// Refreshes m_instability and m_dominantHz from the sample ring:
	//
	// 1. Detrend. Subtract a ~0.5 s centered moving average from each sample,
	//    so slow scene and exposure drift drop out and only oscillation
	//    remains. Instability is the RMS of that residual over the mean level.
	// 2. Periodogram. Project the residual onto a sine and a cosine at each
	//    candidate frequency from 0.5 Hz to 20 Hz in 0.25 Hz steps (capped
	//    below the Nyquist rate of the measured frame spacing), using the real
	//    sample timestamps so mildly uneven frame pacing does not smear the
	//    estimate. The projection's magnitude is that frequency's fitted
	//    sinusoid amplitude.
	// 3. Dominance gate. The best frequency is reported only when its fitted
	//    sinusoid explains over 40 percent of the residual variance.
	//    Broadband noise (hand motion, auto-exposure hunting) spreads its
	//    power across the scan and never concentrates like that, so it reads
	//    as instability with no frequency.
	void analyze();

	struct Sample
	{
		double timestampMs= 0.0;
		float luma= 0.f;
	};

	// ~4s at 60fps; enough cycles of any beat below the Nyquist rate
	static constexpr int kCapacity= 256;
	std::array<Sample, kCapacity> m_samples;
	int m_sampleCount= 0;
	int m_writeIndex= 0;

	double m_lastAnalysisMs= 0.0;
	float m_instability= 0.f;
	float m_dominantHz= 0.f;
};
