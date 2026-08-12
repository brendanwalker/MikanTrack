#include "HandBoneCalibrator.h"

#include <algorithm>
#include <cmath>

namespace
{
float medianOf(std::vector<float>& values)
{
	if (values.empty())
		return 0.f;
	const size_t middle= values.size() / 2;
	std::nth_element(values.begin(), values.begin() + middle, values.end());
	return values[middle];
}

// Median absolute deviation about the median: an outlier-tolerant spread, in
// the same units as the samples
float medianAbsoluteDeviation(const std::vector<float>& values, float median)
{
	if (values.empty())
		return 0.f;
	std::vector<float> deviations;
	deviations.reserve(values.size());
	for (float value : values)
		deviations.push_back(std::fabs(value - median));
	return medianOf(deviations);
}
} // namespace

void HandBoneCalibrator::reset()
{
	for (SideSamples& side : m_sides)
	{
		side.skeletons.clear();
		side.palmarMemory.reset();
	}
}

void HandBoneCalibrator::addSample(eHandSide side,
								   const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points)
{
	for (const glm::vec3& point : points)
	{
		if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
			return;
	}

	SideSamples& samples= m_sides[(int)side];

	HandSkeleton skeleton;
	HandPoseModel::computeSkeleton(points, side, skeleton, &samples.palmarMemory);

	// A degenerate palm frame (landmarks collapsed onto a point or a line)
	// yields a skeleton of zeros or NaNs; it must not reach the medians
	const float referenceBone= skeleton.baseInPalm[(int)eFinger::Middle].x * 2.f;
	if (!std::isfinite(referenceBone) || referenceBone < 1e-3f)
		return;

	samples.skeletons.push_back(skeleton);
}

bool HandBoneCalibrator::finish(eHandSide side, HandSkeleton& outSkeleton, Quality& outQuality) const
{
	const SideSamples& samples= m_sides[(int)side];

	outQuality= Quality();
	outQuality.sampleCount= (int)samples.skeletons.size();
	if (outQuality.sampleCount < k_minSamples)
		return false;

	outSkeleton= HandSkeleton();

	std::vector<float> component;
	component.reserve(samples.skeletons.size());
	auto medianOfComponent= [&](auto&& select) {
		component.clear();
		for (const HandSkeleton& skeleton : samples.skeletons)
			component.push_back(select(skeleton));
		return medianOf(component);
	};

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		outSkeleton.baseInPalm[finger]= glm::vec3(
			medianOfComponent([finger](const HandSkeleton& s) { return s.baseInPalm[finger].x; }),
			medianOfComponent([finger](const HandSkeleton& s) { return s.baseInPalm[finger].y; }),
			medianOfComponent([finger](const HandSkeleton& s) { return s.baseInPalm[finger].z; }));

		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const float length= medianOfComponent(
				[finger, phalanx](const HandSkeleton& s) { return s.phalanxLengths[finger][phalanx]; });
			outSkeleton.phalanxLengths[finger][phalanx]= length;

			// component still holds this phalanx's samples
			const float spread= medianAbsoluteDeviation(component, length);
			outQuality.phalanxSpread[finger][phalanx]= spread;
			outQuality.worstPhalanxSpread= std::max(outQuality.worstPhalanxSpread, spread);
		}
	}

	// Palm +X is defined as wrist -> middle MCP, so that base sits exactly on
	// the axis. Per-component medians land a hair off it, and a nonzero y
	// there silently rotates every palm frame rebuilt from this skeleton.
	glm::vec3& middleBase= outSkeleton.baseInPalm[(int)eFinger::Middle];
	middleBase.y= 0.f;
	middleBase.z= 0.f;

	outSkeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(outSkeleton);
	return true;
}
