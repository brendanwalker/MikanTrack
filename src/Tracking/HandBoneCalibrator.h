#pragma once

#include <array>
#include <vector>

#include "HandPoseModel.h"
#include "TrackingTypes.h"

// Measures the user's OWN hand geometry from stereo-triangulated landmarks.
//
// The landmark model's metric hand is not a measurement of the hand in front
// of the cameras: measured against triangulation it holds proximal phalanges
// around half their true length while the palm reads about right. That shape
// error propagates twice - the PnP object model is rescaled on the single
// wrist->middle-MCP bone, so an object with too little spread has to sit too
// close to match the image, and the same points feed the skeleton streamed to
// clients. Triangulated landmarks, by contrast, reproduce a ruler.
//
// So the skeleton is measured once from stereo and reused: per-component
// medians of HandPoseModel::computeSkeleton over a sampling window. Medians
// rather than means because a bone lying along a camera's view ray
// triangulates poorly and produces outliers rather than noise.
//
// Bone lengths measured this way carry a small upward bias: independent error
// on each endpoint can only lengthen a segment, never shorten it. It is a
// percent or two for the shortest bones at the current triangulation noise
// floor, and --test-bonecalib pins the magnitude.
//
// THREAD AFFINITY: none of its own. Live it is owned by the vision thread;
// offline the same instance runs against a replayed recording.
class HandBoneCalibrator
{
public:
	// Per-bone agreement across the sampled frames, so a capture can be judged
	// instead of trusted
	struct Quality
	{
		int sampleCount= 0;
		// Median absolute deviation of each phalanx length, meters
		std::array<std::array<float, 3>, FINGER_COUNT> phalanxSpread{};
		// Largest of those, meters - the one number worth surfacing
		float worstPhalanxSpread= 0.f;
	};

	// Below this a capture is refused: a handful of frames of one hand pose
	// measures that pose's triangulation luck, not the hand
	static constexpr int k_minSamples= 30;

	void reset();

	// One frame of triangulated landmarks for one hand, in any right-handed
	// metric space. Non-finite input is dropped.
	void addSample(eHandSide side, const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points);

	int getSampleCount(eHandSide side) const
	{
		return (int)m_sides[(int)side].skeletons.size();
	}

	// Per-component medians over the samples. False below k_minSamples.
	bool finish(eHandSide side, HandSkeleton& outSkeleton, Quality& outQuality) const;

private:
	struct SideSamples
	{
		std::vector<HandSkeleton> skeletons;
		// Kept across samples so the palmar sign stays stable, exactly as the
		// live per-hand memories do
		HandPoseModel::PalmarSideMemory palmarMemory;
	};

	std::array<SideSamples, 2> m_sides;
};
