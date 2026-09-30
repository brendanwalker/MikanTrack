#include "ImuMountingMath.h"

#include <algorithm>
#include <cmath>

#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

// -- Twist quality gates (mounting calibration) -----
// Total rotation the forearm must travel before the measured axis is
// trusted. ~5 rad is roughly two full pronation/supination sweeps.
static constexpr float k_minTwistPathRadians= 5.f;
// The motion must substantially REVERSE. A steady turn (or an uncorrected
// rate offset) also produces a rank-1 scatter, but pointing somewhere that
// has nothing to do with the arm - a Joy-Con whose bias had run away scored
// 0.9999 dominance while sitting still.
static constexpr float k_minTwistReversal= 0.5f;
static constexpr float k_minTwistDominance= 0.7f;

// -- Curl quality gates (the second calibration motion) -----
// A curl sweeps the whole forearm rather than spinning it in place, so it
// carries more soft-tissue wobble than a twist and cannot hold a twist's
// dominance. Measured at 0.94 on a clean capture against the twist's 0.99.
static constexpr float k_minCurlDominance= 0.85f;
// The two motions must be independent for the frame to close. Anatomically
// this is near 90 degrees (measured 89.3); far below it means the shoulder
// was turning during the curl and roll is being extrapolated.
static constexpr float k_minInterAxisAngleDegrees= 60.f;
// Half-strokes whose individually measured hinge axes disagree by more than
// this were not made at one settled pronation, so roll is not repeatable.
// A clean capture holds ~8 degrees once the opening stroke is dropped.
static constexpr float k_maxHingeSpreadDegrees= 15.f;
// A half-stroke shorter than this is a direction change, not a stroke
static constexpr int k_minStrokeSamples= 40;
// Rate below which a sample is between strokes rather than in one
static constexpr float k_strokeDeadbandRadiansPerSecond= 0.5f;
// The centripetal fit's radius is an elbow-to-sensor distance, so it has to
// come out a forearm length. Outside this the model did not apply and the
// measurement is discarded rather than believed.
static constexpr float k_minForearmLengthMeters= 0.10f;
static constexpr float k_maxForearmLengthMeters= 0.40f;
static constexpr float k_minLengthFitCorrelation= 0.5f;

glm::vec3 imuDominantRotationAxis(const glm::mat3& scatter, float& outDominance)
{
	glm::vec3 axis(1.f, 0.f, 0.f);
	for (int iteration= 0; iteration < 64; ++iteration)
	{
		const glm::vec3 next= scatter * axis;
		const float length= glm::length(next);
		if (length < 1e-20f)
		{
			outDominance= 0.f;
			return glm::vec3(1.f, 0.f, 0.f);
		}
		axis= next / length;
	}

	const float eigenvalue= glm::dot(axis, scatter * axis);
	const float trace= scatter[0][0] + scatter[1][1] + scatter[2][2];
	outDominance= trace > 1e-20f ? eigenvalue / trace : 0.f;
	return axis;
}

namespace
{
struct ImuMotionStats
{
	glm::mat3 scatter{0.f};
	float pathRadians= 0.f;
	glm::vec3 net{0.f};
};

ImuMotionStats imuAccumulateMotionStats(const MotionSample* samples, size_t count)
{
	ImuMotionStats stats;
	for (size_t index= 0; index < count; ++index)
	{
		const glm::vec3& rate= samples[index].rate;
		const float dtSeconds= samples[index].dtSeconds;
		stats.scatter+= glm::outerProduct(rate, rate) * dtSeconds;
		stats.pathRadians+= glm::length(rate) * dtSeconds;
		stats.net+= rate * dtSeconds;
	}
	return stats;
}
} // namespace

bool imuFitCentripetalRadius(const std::vector<MotionSample>& curl, const glm::vec3& longAxis,
							 const glm::vec3& hingeAxis, float& outSignedRadius, float& outCorrelation)
{
	outSignedRadius= 0.f;
	outCorrelation= 0.f;

	constexpr size_t k_minFitSamples= 100;
	constexpr float k_gravity= 9.81f;
	if (curl.size() < k_minFitSamples)
		return false;

	// Seed the gravity direction from the quietest sample in the opening
	// stretch, where the accelerometer reads little but gravity
	size_t seedIndex= 0;
	float seedRate= glm::length(curl[0].rate);
	const size_t seedSearch= std::min<size_t>(curl.size(), k_minFitSamples);
	for (size_t index= 1; index < seedSearch; ++index)
	{
		const float rateMagnitude= glm::length(curl[index].rate);
		if (rateMagnitude < seedRate)
		{
			seedRate= rateMagnitude;
			seedIndex= index;
		}
	}

	glm::vec3 up= curl[seedIndex].acceleration;
	if (glm::length(up) < 1e-3f)
		return false;
	up= glm::normalize(up);

	double sumX= 0.0, sumY= 0.0, sumXX= 0.0, sumYY= 0.0, sumXY= 0.0;
	int count= 0;
	for (size_t index= seedIndex; index < curl.size(); ++index)
	{
		const MotionSample& sample= curl[index];

		// World up carried through the sensor's own rotation. Integrated here
		// rather than read from the filter so the fit stays a pure function of
		// the recording, testable with no filter and no device.
		up-= glm::cross(sample.rate, up) * sample.dtSeconds;
		const float upLength= glm::length(up);
		if (upLength < 1e-3f)
			return false;
		up/= upLength;

		// Re-anchor whenever a sample looks like near-pure gravity. Integration
		// alone drifts, and the whole fit rests on the residual being
		// centripetal rather than accumulated tilt error.
		const float accelerationMagnitude= glm::length(sample.acceleration);
		if (glm::length(sample.rate) < 0.5f && fabsf(accelerationMagnitude - k_gravity) < 0.5f)
			up= glm::normalize(glm::mix(up, sample.acceleration / accelerationMagnitude, 0.02f));

		const float hingeRate= glm::dot(sample.rate, hingeAxis);
		const double x= (double)(hingeRate * hingeRate);
		const double y= (double)glm::dot(sample.acceleration - up * k_gravity, longAxis);
		sumX+= x;
		sumY+= y;
		sumXX+= x * x;
		sumYY+= y * y;
		sumXY+= x * y;
		count++;
	}
	if ((size_t)count < k_minFitSamples)
		return false;

	const double n= (double)count;
	const double varianceX= sumXX - sumX * sumX / n;
	const double varianceY= sumYY - sumY * sumY / n;
	const double covariance= sumXY - sumX * sumY / n;
	if (varianceX < 1e-6 || varianceY < 1e-9)
		return false;

	// residual . longAxis = -omega^2 * radius when longAxis points at the hand
	outSignedRadius= (float)(-covariance / varianceX);
	outCorrelation= (float)(covariance / sqrt(varianceX * varianceY));
	return true;
}

void imuSolveMountingFromMotions(const std::vector<MotionSample>& twist,
								 const std::vector<MotionSample>& curl, const glm::quat* palmarHint,
								 ePalmarSource hintSource, MountingCaptureResult& outResult)
{
	outResult= MountingCaptureResult();

	// -- The twist fixes the forearm's long axis -----
	const ImuMotionStats twistStats= imuAccumulateMotionStats(twist.data(), twist.size());
	float twistDominance= -1.f;
	imuEvaluateTwist(twistStats.scatter, twistStats.pathRadians, twistStats.net, twistDominance,
					 outResult.twistProgress, outResult.twistReversal);
	outResult.axisDominance= std::max(0.f, twistDominance);
	glm::vec3 longAxis= imuDominantRotationAxis(twistStats.scatter, twistDominance);

	// -- The curl fixes the roll about it -----
	const ImuMotionStats curlStats= imuAccumulateMotionStats(curl.data(), curl.size());
	float curlDominance= -1.f;
	imuEvaluateTwist(curlStats.scatter, curlStats.pathRadians, curlStats.net, curlDominance,
					 outResult.curlProgress, outResult.curlReversal);
	const glm::vec3 provisionalHinge= imuDominantRotationAxis(curlStats.scatter, curlDominance);

	// Split into half-strokes at reversals about that hinge and DROP THE
	// FIRST. The elbow hinge belongs to the ulna while the sensor rides the
	// pronating distal forearm, so the axis it measures only means something
	// once the pronation has settled - and on the opening stroke it has not,
	// landing tens of degrees off the strokes that follow it.
	std::vector<glm::vec3> strokeAxes;
	glm::mat3 acceptedScatter(0.f);
	{
		auto flushStroke= [&](size_t begin, size_t end) {
			if (end <= begin || end - begin < (size_t)k_minStrokeSamples)
				return;
			const ImuMotionStats strokeStats= imuAccumulateMotionStats(curl.data() + begin, end - begin);
			float strokeDominance= 0.f;
			glm::vec3 axis= imuDominantRotationAxis(strokeStats.scatter, strokeDominance);
			if (glm::dot(axis, provisionalHinge) < 0.f)
				axis= -axis;
			strokeAxes.push_back(axis);
			if (strokeAxes.size() > 1)
				acceptedScatter+= strokeStats.scatter;
		};

		size_t strokeStart= 0;
		int strokeSign= 0;
		for (size_t index= 0; index < curl.size(); ++index)
		{
			const float hingeRate= glm::dot(curl[index].rate, provisionalHinge);
			const int sign= hingeRate > k_strokeDeadbandRadiansPerSecond
				? 1
				: (hingeRate < -k_strokeDeadbandRadiansPerSecond ? -1 : 0);
			if (sign == 0)
				continue;

			if (strokeSign == 0)
			{
				strokeSign= sign;
				strokeStart= index;
			}
			else if (sign != strokeSign)
			{
				flushStroke(strokeStart, index);
				strokeSign= sign;
				strokeStart= index;
			}
		}
		if (strokeSign != 0)
			flushStroke(strokeStart, curl.size());
	}
	outResult.curlStrokes= (int)strokeAxes.size();

	glm::vec3 hingeAxis= provisionalHinge;
	if (strokeAxes.size() >= 2)
	{
		float acceptedDominance= 0.f;
		hingeAxis= imuDominantRotationAxis(acceptedScatter, acceptedDominance);
		if (glm::dot(hingeAxis, provisionalHinge) < 0.f)
			hingeAxis= -hingeAxis;
		curlDominance= acceptedDominance;

		for (size_t index= 1; index < strokeAxes.size(); ++index)
		{
			const float alignment= glm::clamp(glm::dot(strokeAxes[index], hingeAxis), -1.f, 1.f);
			outResult.hingeSpreadDegrees=
				std::max(outResult.hingeSpreadDegrees, glm::degrees(acosf(alignment)));
		}
	}
	outResult.curlDominance= std::max(0.f, curlDominance);

	const float axisAlignment= glm::clamp(fabsf(glm::dot(longAxis, hingeAxis)), 0.f, 1.f);
	outResult.interAxisAngleDegrees= glm::degrees(acosf(axisAlignment));

	// +X is trusted over the hinge, so the carrying angle lands entirely in +Y
	glm::vec3 yAxis= hingeAxis - longAxis * glm::dot(longAxis, hingeAxis);
	const float yLength= glm::length(yAxis);
	if (yLength < 1e-4f)
		return; // parallel axes: nothing to orthogonalize, and the gate below refuses it
	yAxis/= yLength;

	// -- Which end of the long axis is the hand -----
	bool bLongAxisSignResolved= false;
	float signedRadius= 0.f;
	float fitCorrelation= 0.f;
	if (imuFitCentripetalRadius(curl, longAxis, hingeAxis, signedRadius, fitCorrelation))
	{
		if (signedRadius < 0.f)
		{
			// The eigenvector came out pointing at the elbow
			longAxis= -longAxis;
			yAxis= -yAxis; // keep the triad right-handed
			signedRadius= -signedRadius;
			fitCorrelation= -fitCorrelation;
		}
		outResult.forearmLengthMeters= signedRadius;
		outResult.lengthFitCorrelation= fabsf(fitCorrelation);
		bLongAxisSignResolved= outResult.lengthFitCorrelation >= k_minLengthFitCorrelation;
		outResult.bLengthMeasured= bLongAxisSignResolved &&
			signedRadius >= k_minForearmLengthMeters && signedRadius <= k_maxForearmLengthMeters;
	}

	// -- Which side of the hinge the palm is on -----
	//
	// The two candidates differ by a half turn about the long axis, so any
	// reference that is even roughly right picks correctly. That is the whole
	// reason a pose average far too noisy to BE the mounting is still a fine
	// discriminator here.
	const glm::vec3 zAxis= glm::cross(longAxis, yAxis);
	const glm::quat candidate= glm::normalize(glm::quat_cast(glm::mat3(longAxis, yAxis, zAxis)));
	const glm::quat flipped= glm::normalize(glm::quat_cast(glm::mat3(longAxis, -yAxis, -zAxis)));

	outResult.forearmToSensor= candidate;
	if (palmarHint != nullptr && hintSource != ePalmarSource::None)
	{
		const float alignment= fabsf(glm::dot(candidate, *palmarHint));
		const float flippedAlignment= fabsf(glm::dot(flipped, *palmarHint));
		outResult.forearmToSensor= alignment >= flippedAlignment ? candidate : flipped;
		outResult.palmarSource= hintSource;
	}

	outResult.bMotionUsable=
		imuIsTwistUsable(outResult.axisDominance, outResult.twistProgress, outResult.twistReversal) &&
		outResult.curlProgress >= 1.f && outResult.curlReversal >= k_minTwistReversal &&
		outResult.curlDominance >= k_minCurlDominance && outResult.curlStrokes >= 3 &&
		outResult.hingeSpreadDegrees <= k_maxHingeSpreadDegrees &&
		outResult.interAxisAngleDegrees >= k_minInterAxisAngleDegrees && bLongAxisSignResolved &&
		outResult.palmarSource != ePalmarSource::None;
}


float imuSignedComponentDegrees(const glm::quat& rotation, const glm::vec3& axis)
{
	glm::quat q= rotation;
	if (q.w < 0.f) // shortest arc, or the sign of the whole thing flips
		q= -q;

	const glm::vec3 axisPart(q.x, q.y, q.z);
	const float axisLength= glm::length(axisPart);
	if (axisLength < 1e-6f)
		return 0.f;

	const float angle= 2.f * asinf(std::min(1.f, axisLength));
	return glm::degrees(angle * glm::dot(axisPart / axisLength, axis));
}

void imuEvaluateTwist(const glm::mat3& scatter, float pathRadians, const glm::vec3& net,
					  float& outDominance, float& outProgress, float& outReversal)
{
	outProgress= std::min(1.f, pathRadians / k_minTwistPathRadians);
	outReversal= pathRadians > 1e-6f ? std::clamp(1.f - glm::length(net) / pathRadians, 0.f, 1.f) : 0.f;
	outDominance= -1.f;
	if (pathRadians > 1e-6f)
		imuDominantRotationAxis(scatter, outDominance);
}

bool imuIsTwistUsable(float dominance, float progress, float reversal)
{
	return progress >= 1.f && reversal >= k_minTwistReversal && dominance >= k_minTwistDominance;
}
