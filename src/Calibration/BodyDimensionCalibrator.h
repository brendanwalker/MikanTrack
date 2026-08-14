#pragma once

#include <vector>

#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"

#include "BodyPoseSolver.h" // BodyDimensions
#include "HandFusion.h"     // CameraFrameResult

// Measures the body widths the pose solve rests on, because they are NOT
// anatomical numbers: they are separations between the MODEL'S OWN landmarks,
// which sit inside the real joints by an amount that varies with the person
// and the model. Assuming an anatomical 0.40 m between the shoulder landmarks
// put a live shoulder 0.8 m too far away.
//
// WHERE THE SCALE COMES FROM: the fused wrists. They are the only metric,
// world-anchored points in the frame - stereo-triangulated by the hand
// pipeline, independent of this camera and of every length being measured.
// Everything here is that scale carried onto the body landmarks.
//
// ONLY WIDTHS ARE MEASURED. An earlier version also measured the upper arm,
// which needed the arm straight and square to the camera; that pose could not
// be hit reliably at a desk, and missing it under-measured the upper arm by
// 20% - which made the elbow solve pick the wrong bend. The upper arm is now
// derived from the shoulder width instead (see BodyConfig), so this only has
// to measure widths, and the pose collapses to "hold one hand up".
//
// THE POSE: one hand raised near the shoulder. That puts a metric wrist at
// roughly the depth of the torso, which is all the shoulders and ears need to
// be measured. The check is that the hand is genuinely near its own shoulder:
// a hand resting out on the desk sits well in front of the torso, and
// measuring the shoulders at that depth would scale them wrong.
//
// The nose needs its own step: facing the camera it sits on the ear midpoint
// and says nothing about how far it juts forward. Turning the head swings
// that offset into the image plane, where it can be measured.
class BodyDimensionCalibrator
{
public:
	struct Sample
	{
		// Either hand can carry a frame on its own, so a resting hand never
		// vetoes a raised one
		bool bAcceptedLeft= false;
		bool bAcceptedRight= false;

		float shoulderWidth= 0.f;
		float headWidth= 0.f;

		// How far each hand sits from its own shoulder IN THE IMAGE, as a
		// fraction of the shoulders' separation: the test that it is raised
		// beside the shoulder rather than resting out in front
		float handOffsetLeft= 0.f;
		float handOffsetRight= 0.f;
	};

	struct Result
	{
		bool bValid= false;
		int sampleCount= 0;
		float shoulderWidth= 0.f;
		float headWidth= 0.f;
		// Derived from shoulderWidth, not measured
		float upperArmLength= 0.f;
		float noseForward= 0.f;
		bool bHaveNoseForward= false;
		// Spread across samples, as a fraction of the median. High means the
		// pose moved during the capture.
		float shoulderWidthSpread= 0.f;
		float headWidthSpread= 0.f;
	};

	void reset();

	// Adds one raised-hand sample, per hand. Returns false (and records
	// nothing) only when NEITHER hand is usable: untracked, a landmark under
	// the visibility gate, or the hand too far from its shoulder to stand in
	// for the torso's depth. The accepted hand's own wrist sets the depth the
	// shoulders and ears are measured at, so one hand is enough.
	bool addFrontalSample(
		const CameraFrameResult& camera,
		const TrackingFrameResult& fused,
		const BodyDimensions& currentDimensions,
		Sample& outSample);

	// Adds one head-turn sample: the nose's offset from the ear midpoint,
	// measured at the head distance the raised-hand step established.
	bool addHeadTurnSample(const CameraFrameResult& camera, float& outNoseForward);

	int getFrontalSampleCount() const { return (int)m_frontalSamples.size(); }
	int getHeadTurnSampleCount() const { return (int)m_noseForwardSamples.size(); }

	// Medians over the collected samples
	Result solve(const BodyDimensions& currentDimensions) const;

	// A raised hand sits within this fraction of the shoulder separation from
	// its own shoulder in the image. Beyond it the hand is out in front of
	// the body, where its depth no longer stands in for the torso's.
	static constexpr float k_maxRaisedHandOffset= 0.80f;
	// Enough samples to have a median worth trusting
	static constexpr int k_minSamples= 20;

private:
	std::vector<Sample> m_frontalSamples;
	std::vector<float> m_noseForwardSamples;
	// Head distance from the raised-hand step, reused by the head-turn step:
	// the torso does not move between them, only the head rotates
	float m_headRangeMeters= 0.f;
};
