#include "ImuMountingCalibrator.h"

#include <algorithm>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

// Angular-velocity scatter (mounting axis estimation)
static constexpr float k_scatterMinRateRadiansPerSecond= 0.35f; // ~20 deg/s: deliberate motion, not jiggle
static constexpr float k_scatterHalfLifeSeconds= 8.f;
// Recording cap, ~60 s per motion at 200 Hz. A window this long is already
// far more than any gate needs; the bound only stops a wizard left open
// overnight from growing without limit.
static constexpr size_t k_maxMotionRecordingSamples= 12000;
// The pose average needs a decent run of tracked frames behind it
static constexpr int k_poseMountingMinSamples= 60;
static constexpr float k_poseMountingMinConfidence= 0.4f;
// Forearm-axis consistency: only rotation past this carries axis information,
// and the EMA needs a bit of motion before it means anything
static constexpr float k_axisConsistencyMinRotation= 0.004f; // ~0.5 deg
static constexpr float k_axisConsistencyEmaAlpha= 0.05f;
static constexpr int k_axisConsistencyMinSamples= 30;
// Wrist axial residual: the palm estimate has to be worth trusting before it
// says anything, and the EMA needs a run of samples behind it
static constexpr float k_axialResidualMinConfidence= 0.5f;
static constexpr float k_axialResidualEmaAlpha= 0.02f;
static constexpr int k_axialResidualMinSamples= 60;

void ImuMountingCalibrator::addMotionSample(const glm::vec3& rate, const glm::vec3& acceleration,
											 float dtSeconds, eMountingMotion recording)
{
	// Only real rotation carries axis information, and weighting by |w|^2
	// lets deliberate twisting dominate incidental jiggle. Decays so a
	// calibration reflects RECENT motion.
	const float rateMagnitude= glm::length(rate);
	if (rateMagnitude > k_scatterMinRateRadiansPerSecond)
	{
		const float decay= expf(-dtSeconds / k_scatterHalfLifeSeconds);
		m_rotationScatter*= decay;
		m_rotationPathRadians*= decay;
		m_rotationNet*= decay;

		m_rotationScatter+= glm::outerProduct(rate, rate) * dtSeconds;
		m_rotationPathRadians+= rateMagnitude * dtSeconds;
		m_rotationNet+= rate * dtSeconds;
	}

	// The calibration recording keeps EVERY sample, including the slow ones
	// the scatter gate above skips: the centripetal fit needs the
	// near-stationary samples to anchor gravity, and the stroke split needs
	// the turnarounds to know where one stroke ends.
	if (recording != eMountingMotion::None)
	{
		std::vector<MotionSample>& recorded=
			recording == eMountingMotion::Twist ? m_twistRecording : m_curlRecording;
		if (recorded.size() < k_maxMotionRecordingSamples)
		{
			MotionSample sample;
			sample.rate= rate;
			sample.acceleration= acceleration;
			sample.dtSeconds= dtSeconds;
			recorded.push_back(sample);
		}
	}
}

void ImuMountingCalibrator::clearRecording(eMountingMotion motion)
{
	if (motion == eMountingMotion::Twist)
		m_twistRecording.clear();
	else if (motion == eMountingMotion::Curl)
		m_curlRecording.clear();
}

void ImuMountingCalibrator::resetLiveMotion()
{
	m_rotationScatter= glm::mat3(0.f);
	m_rotationPathRadians= 0.f;
	m_rotationNet= glm::vec3(0.f);
	m_poseMountingSum= glm::vec4(0.f);
	m_poseMountingSamples= 0;
	m_poseSpreadSumDegrees= 0.f;
}

void ImuMountingCalibrator::getLiveTwist(float& outDominance, float& outProgress, float& outReversal) const
{
	imuEvaluateTwist(m_rotationScatter, m_rotationPathRadians, m_rotationNet, outDominance, outProgress,
					 outReversal);
}

void ImuMountingCalibrator::accumulatePoseMounting(const glm::quat& sensorToWorld,
												   const glm::quat& palmOrientationWorld, float confidence)
{
	if (confidence < k_poseMountingMinConfidence)
		return;

	const glm::quat sample= glm::normalize(glm::inverse(sensorToWorld) * glm::normalize(palmOrientationWorld));

	// Quaternions double-cover rotations, so align each sample to the running
	// mean's hemisphere before summing - otherwise q and -q cancel and the
	// average collapses toward zero
	glm::quat aligned= sample;
	if (m_poseMountingSamples > 0)
	{
		const glm::quat mean= glm::normalize(glm::quat(m_poseMountingSum.w, m_poseMountingSum.x,
													   m_poseMountingSum.y, m_poseMountingSum.z));
		if (glm::dot(aligned, mean) < 0.f)
			aligned= -aligned;

		const glm::quat delta= glm::inverse(mean) * aligned;
		m_poseSpreadSumDegrees+= glm::degrees(2.f * asinf(std::min(
			1.f, glm::length(glm::vec3(delta.x, delta.y, delta.z)))));
	}

	m_poseMountingSum+= glm::vec4(aligned.x, aligned.y, aligned.z, aligned.w);
	m_poseMountingSamples++;
}

void ImuMountingCalibrator::capture(const glm::quat* previousMountingHint, MountingCaptureResult& outResult)
{
	// The pose average is no longer the geometry - a wrist that would not hold
	// still is exactly what broke that - but it is still the reference that
	// says which side of the hinge axis the palm is on, a decision with a 180
	// degree margin that it clears easily
	glm::quat palmarHint(1.f, 0.f, 0.f, 0.f);
	ePalmarSource hintSource= ePalmarSource::None;
	if (m_poseMountingSamples >= k_poseMountingMinSamples)
	{
		palmarHint= glm::normalize(glm::quat(m_poseMountingSum.w, m_poseMountingSum.x, m_poseMountingSum.y,
											 m_poseMountingSum.z));
		hintSource= ePalmarSource::Vision;
	}
	else if (previousMountingHint != nullptr)
	{
		// Recalibrating without the cameras seeing the hand: the side already
		// known to be on is a valid reference for a two-way choice, even
		// though it is useless as a mounting
		palmarHint= *previousMountingHint;
		hintSource= ePalmarSource::PreviousMounting;
	}

	imuSolveMountingFromMotions(m_twistRecording, m_curlRecording, &palmarHint, hintSource, outResult);

	outResult.poseSamples= m_poseMountingSamples;
	outResult.poseSpreadDegrees=
		m_poseMountingSamples > 0 ? m_poseSpreadSumDegrees / (float)m_poseMountingSamples : 0.f;

	// A recapture invalidates the old mounting's quality score, and the axial
	// residual measured against the OLD mounting says nothing about this one
	m_axisConsistencyEma= -1.f;
	m_axisConsistencySamples= 0;
	m_bHasLastPublishedForearm= false;
	m_twistResidualDegreesEma= 0.f;
	m_twistSamples= 0;
}

void ImuMountingCalibrator::notePublishedForearm(const glm::quat& forearmToWorld)
{
	if (m_bHasLastPublishedForearm)
	{
		const glm::quat delta= glm::inverse(m_lastPublishedForearm) * forearmToWorld;
		const glm::vec3 axisPart(delta.x, delta.y, delta.z);
		const float axisLength= glm::length(axisPart);
		if (axisLength > k_axisConsistencyMinRotation)
		{
			const glm::vec3 axis= axisPart / axisLength;
			const float alignment= fabsf(axis.x); // +X in the forearm's own frame
			m_axisConsistencyEma= m_axisConsistencyEma < 0.f
				? alignment
				: m_axisConsistencyEma * (1.f - k_axisConsistencyEmaAlpha) + alignment * k_axisConsistencyEmaAlpha;
			m_axisConsistencySamples++;
		}
	}
	m_lastPublishedForearm= forearmToWorld;
	m_bHasLastPublishedForearm= true;
}

float ImuMountingCalibrator::getForearmAxisConsistency() const
{
	return m_axisConsistencySamples >= k_axisConsistencyMinSamples ? m_axisConsistencyEma : -1.f;
}

void ImuMountingCalibrator::updateWristAxialResidual(const glm::quat& forearmToWorld,
													 const glm::quat& palmOrientationWorld, float confidence)
{
	if (confidence < k_axialResidualMinConfidence)
		return;

	const glm::quat wrist= glm::normalize(glm::inverse(forearmToWorld) * glm::normalize(palmOrientationWorld));

	// THE constraint: the wrist has no axial degree of freedom, so whatever
	// this reads is mounting roll error rather than anatomy
	const float axialDegrees= imuSignedComponentDegrees(wrist, glm::vec3(1.f, 0.f, 0.f));

	m_twistResidualDegreesEma= m_twistSamples == 0
		? axialDegrees
		: m_twistResidualDegreesEma * (1.f - k_axialResidualEmaAlpha) + axialDegrees * k_axialResidualEmaAlpha;
	m_twistSamples++;
}

float ImuMountingCalibrator::getWristAxialTwistDegrees() const
{
	return m_twistSamples >= k_axialResidualMinSamples ? m_twistResidualDegreesEma : -999.f;
}
