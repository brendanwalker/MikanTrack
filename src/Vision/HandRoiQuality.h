#pragma once

#include "opencv2/core/mat.hpp"

#include "TrackingTypes.h"

// Image-quality diagnostics for hand tracking: per-hand ROI statistics. Runs
// on the vision thread against the exact frame the model consumed. All heavy
// statistics operate on a nearest-neighbor decimated crop (sampling preserves
// the clipping and noise statistics that area-averaging would destroy),
// keeping the per-frame cost around 0.2 ms per camera - far under the
// inference budget.

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
