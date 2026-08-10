#pragma once

#include <array>

#include "opencv2/core/mat.hpp"

#include "TrackingTypes.h"

// Image-quality diagnostics for hand tracking: per-hand ROI statistics plus a
// per-camera temporal luminance tracker. Runs on the vision thread against the
// exact frame the model consumed. All heavy statistics operate on a
// nearest-neighbor decimated crop (sampling preserves the clipping and noise
// statistics that area-averaging would destroy), keeping the per-frame cost
// around 0.2 ms per camera - far under the inference budget.

namespace HandRoiQuality
{
// Working size the decimated ROI is reduced to. Metric values (sharpness,
// noise) are defined at this scale, which is what makes them comparable
// across camera resolutions.
constexpr int kAnalysisMaxDim= 160;

// ROI = landmark bounding box expanded by this factor about its center
constexpr float kRoiExpand= 1.5f;
// Background ring = ROI expanded by this additional factor; ring statistics
// come from the band between the two boxes
constexpr float kRingExpand= 1.4f;

// Fills hand.imageQuality from the frame the model consumed. Leaves
// imageQuality.valid false when the hand is untracked or its ROI is too
// small/degenerate to analyze.
void analyzeHand(const cv::Mat& bgrFrame, TrackedHand& hand);
} // namespace HandRoiQuality

// Tracks the whole-frame mean luminance over the last few seconds and measures
// its oscillation. Mains/PWM light flicker beating against the shutter and
// auto-exposure hunting both live here - and both destabilize landmarks while
// every single-frame statistic looks fine. The frame mean is dominated by the
// static background, so hand motion contributes little.
class LumaFlickerTracker
{
public:
	// Feeds one processed frame (decimated mean, ~0.02 ms) and refreshes the
	// analysis roughly twice a second
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
