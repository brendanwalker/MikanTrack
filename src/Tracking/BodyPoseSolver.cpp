#include "BodyPoseSolver.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

#include "SpaceTransforms.h"

// Landmark visibility below this is treated as unseen (gate only; scores come
// from measured stability)
static constexpr float kMinVisibility= 0.5f;
// Apparent joint acceleration at which an estimate counts as half as
// trustworthy, m/s^2. Roughly 2.5g: measured on a live front-camera session,
// ordinary gesturing sits near 3-9 m/s^2 while the frames where the solve
// visibly breaks down run past 45. Equivalent in spirit to the hands' 15mm
// at 30Hz (~14 m/s^2), loosened because the pose model is coarser.
static constexpr float kAccelReference= 25.f;
static constexpr float kJitterEmaAlpha= 0.2f;
// A model gap longer than this restarts the jitter history and the smoothing
// filters (a constant-velocity residual across a tracking gap is meaningless,
// and a filter resuming across one would drag the estimate through the hole)
static constexpr double kMaxHoldMs= 500.0;

// Two rays closer to parallel than this cannot resolve a separation into a
// depth (the pair projects to nearly one point)
static constexpr float kMinRaySeparation= 1e-3f;

// A joint this far from the camera watching it is not on the person. Any
// solve that divides a known separation by an apparent one blows up when the
// model collapses that pair: measured on recording 2026-08-14_19-53-18, the
// head keypoints landed 13 px apart and stacked vertically, dividing out to
// an 8.4 m head that the smoothing then dragged back over 80 frames. The gap
// is not close - every good frame that session solved between 0.55 and 0.71 m
// and every bad one past 4.7 m - so a generous window costs nothing and a
// wild value never reaches the estimates that hang off it.
static constexpr float kMinBodyDepthMeters= 0.15f;
static constexpr float kMaxBodyDepthMeters= 2.0f;

// How far past a straight arm the wrist may sit before the shoulder is not
// believable. Bone lengths are calibrated proportions, so a little overshoot
// is ordinary and straightens the arm; a lot means the shoulder is wrong, and
// the elbow should not be dragged toward it.
static constexpr float kMaxReachOvershoot= 1.15f;

// How long an arm may be carried on the bone circle with its elbow landmark
// unseen. The circle keeps both bones exact however the wrist moves, so the
// only thing that decays is WHICH point on the circle - continuity cannot
// tell that the arm rotated about the shoulder-to-wrist axis while it was
// occluded. Measured occlusions on this rig (one hand crossing the other
// elbow) run 6-17 model frames, so a second is generous cover with the
// confidence already faded most of the way out by then.
static constexpr double kMaxDeadReckonMs= 1000.0;

// One-euro parameters for the body estimates. Slower than the hand palm
// (3.0 Hz): these are IK hints refreshed at a fraction of the camera rate, so
// a little latency buys a lot of steadiness. Beta keeps genuine arm motion
// from lagging.
static constexpr float kPositionMinCutoffHz= 1.2f;
static constexpr float kPositionBeta= 0.05f;
static constexpr float kDirectionMinCutoffHz= 1.5f;
static constexpr float kDirectionBeta= 0.08f;
static constexpr float kDerivativeCutoffHz= 1.0f;

float BodyPoseSolver::JitterTracker::update(const glm::vec3& position, float dtSeconds, float accelReference)
{
	if (sampleCount >= 2)
	{
		const float accel= glm::length(position - 2.f * p1 + p2) / std::max(dtSeconds * dtSeconds, 1e-6f);
		accelEma= sampleCount == 2 ? accel : glm::mix(accelEma, accel, kJitterEmaAlpha);
	}
	p2= p1;
	p1= position;
	sampleCount= std::min(sampleCount + 1, 1000);

	// No penalty before there is history to measure
	if (sampleCount < 3)
		return 1.f;
	return HandFusion::stabilityFactor(accelEma, accelReference);
}

float BodyPoseSolver::depthFromKnownSeparation(
	const glm::vec3& rayA, const glm::vec3& rayB, float separationMeters)
{
	const float raySpread= glm::length(rayA - rayB);
	if (raySpread < kMinRaySeparation || separationMeters <= 0.f)
		return 0.f;
	return separationMeters / raySpread;
}

bool BodyPoseSolver::intersectRaySphere(
	const glm::vec3& rayOrigin, const glm::vec3& rayDir,
	const glm::vec3& sphereCenter, float sphereRadius, RaySphereHit& outHit)
{
	outHit= RaySphereHit();

	// |O + t*d - C|^2 = r^2  ->  t^2 - 2*t*b + c = 0
	const glm::vec3 toCenter= sphereCenter - rayOrigin;
	const float b= glm::dot(rayDir, toCenter);
	const float c= glm::dot(toCenter, toCenter) - sphereRadius * sphereRadius;
	const float discriminant= b * b - c;

	if (discriminant >= 0.f)
	{
		const float sqrtDisc= std::sqrt(discriminant);
		const float tNear= b - sqrtDisc;
		const float tFar= b + sqrtDisc;
		outHit.nearPoint= rayOrigin + rayDir * tNear;
		outHit.farPoint= rayOrigin + rayDir * tFar;
		outHit.bNearValid= tNear > 0.f;
		outHit.bFarValid= tFar > 0.f;
		return outHit.bNearValid || outHit.bFarValid;
	}

	if (b <= 0.f)
		return false; // the sphere is behind the camera

	const glm::vec3 closestOnRay= rayOrigin + rayDir * b;
	const glm::vec3 fromCenter= closestOnRay - sphereCenter;
	if (glm::dot(fromCenter, fromCenter) < 1e-8f)
		return false; // the ray runs through the center: no direction to take

	outHit.nearPoint= sphereCenter + glm::normalize(fromCenter) * sphereRadius;
	outHit.farPoint= outHit.nearPoint;
	outHit.bNearValid= true;
	outHit.bFarValid= true;
	outHit.bClamped= true;
	return true;
}

bool BodyPoseSolver::solveElbowNearestTo(
	const glm::vec3& shoulder, const glm::vec3& wrist,
	float upperArmLength, float forearmLength,
	const glm::vec3& target, glm::vec3& outElbow)
{
	BoneCircleGeometry circle;
	if (!computeBoneCircle(shoulder, wrist, upperArmLength, forearmLength, circle))
		return false;

	// Fully extended (or folded): the circle collapsed to a point
	if (circle.bDegenerate)
	{
		outElbow= circle.center;
		return true;
	}

	// Closest point on a circle to a point: drop the target into the circle's
	// plane, then push it out to the radius
	glm::vec3 inPlane= target - circle.center;
	inPlane-= circle.axis * glm::dot(inPlane, circle.axis);
	if (glm::dot(inPlane, inPlane) < 1e-10f)
	{
		// The target sits on the circle's axis, so every point is equally
		// near. Any deterministic choice will do; take the plane's own X.
		outElbow= circle.center + circle.planeX * circle.radius;
		return true;
	}

	outElbow= circle.center + glm::normalize(inPlane) * circle.radius;
	return true;
}

bool BodyPoseSolver::computeBoneCircle(
	const glm::vec3& shoulder, const glm::vec3& wrist,
	float upperArmLength, float forearmLength, BoneCircleGeometry& outCircle)
{
	outCircle= BoneCircleGeometry();
	if (upperArmLength <= 0.f || forearmLength <= 0.f)
		return false;

	const glm::vec3 shoulderToWrist= wrist - shoulder;
	const float span= glm::length(shoulderToWrist);
	if (span < 1e-4f)
		return false;
	const glm::vec3 axis= shoulderToWrist / span;
	outCircle.axis= axis;

	// Modestly out of reach is a straight arm plus calibration error, and the
	// closest the two bones can come to it is exactly straight. WILDLY out of
	// reach is a broken shoulder, and straightening toward one of those would
	// fling the elbow: decline instead and let the caller fall back to the
	// wrist alone, which at least cannot move further than a forearm.
	if (span > (upperArmLength + forearmLength) * kMaxReachOvershoot)
		return false;

	if (span >= upperArmLength + forearmLength ||
		span <= fabsf(upperArmLength - forearmLength))
	{
		outCircle.center= shoulder + axis * upperArmLength;
		outCircle.bDegenerate= true;
		return true;
	}

	// Circle where the two spheres meet: centred on the shoulder-wrist axis
	const float alongAxis=
		(span * span + upperArmLength * upperArmLength - forearmLength * forearmLength) / (2.f * span);
	const float radiusSquared= upperArmLength * upperArmLength - alongAxis * alongAxis;
	outCircle.center= shoulder + axis * alongAxis;
	if (radiusSquared <= 1e-8f)
	{
		outCircle.bDegenerate= true;
		return true;
	}

	outCircle.radius= std::sqrt(radiusSquared);
	const glm::vec3 reference= fabsf(axis.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
	outCircle.planeX= glm::normalize(glm::cross(reference, axis));
	outCircle.planeY= glm::cross(axis, outCircle.planeX);
	return true;
}

bool BodyPoseSolver::solveElbowOnBoneCircle(
	const glm::vec3& shoulder, const glm::vec3& wrist,
	float upperArmLength, float forearmLength,
	const glm::vec3& rayOrigin, const glm::vec3& rayDir, BoneCircleHit& outHit)
{
	outHit= BoneCircleHit();
	if (upperArmLength <= 0.f || forearmLength <= 0.f)
		return false;

	const glm::vec3 shoulderToWrist= wrist - shoulder;
	const float span= glm::length(shoulderToWrist);
	if (span < 1e-4f)
		return false;
	const glm::vec3 axis= shoulderToWrist / span;

	// Modestly out of reach is a straight arm plus calibration error, and the
	// closest the two bones can come to it is exactly straight. WILDLY out of
	// reach is a broken shoulder, and straightening toward one of those would
	// fling the elbow: decline instead and let the caller fall back to the
	// wrist alone, which at least cannot move further than a forearm.
	if (span > (upperArmLength + forearmLength) * kMaxReachOvershoot)
		return false;

	if (span >= upperArmLength + forearmLength ||
		span <= fabsf(upperArmLength - forearmLength))
	{
		outHit.candidates[0]= shoulder + axis * upperArmLength;
		outHit.count= 1;
		outHit.bClamped= true;
		return true;
	}

	// Circle where the two spheres meet: centred on the shoulder-wrist axis
	const float alongAxis=
		(span * span + upperArmLength * upperArmLength - forearmLength * forearmLength) / (2.f * span);
	const float radiusSquared= upperArmLength * upperArmLength - alongAxis * alongAxis;
	if (radiusSquared <= 1e-8f)
	{
		outHit.candidates[0]= shoulder + axis * alongAxis;
		outHit.count= 1;
		outHit.bClamped= true;
		return true;
	}
	const float radius= std::sqrt(radiusSquared);
	const glm::vec3 center= shoulder + axis * alongAxis;

	const glm::vec3 reference= fabsf(axis.z) < 0.9f ? glm::vec3(0.f, 0.f, 1.f) : glm::vec3(1.f, 0.f, 0.f);
	const glm::vec3 planeX= glm::normalize(glm::cross(reference, axis));
	const glm::vec3 planeY= glm::cross(axis, planeX);

	auto pointAt= [&](float angle) {
		return center + (planeX * cosf(angle) + planeY * sinf(angle)) * radius;
	};
	// Distance to the ray's LINE. A candidate behind the camera is rejected by
	// the caller's reach checks, not here, so the metric stays smooth.
	auto rayDistanceSquared= [&](const glm::vec3& point) {
		const glm::vec3 fromOrigin= point - rayOrigin;
		const glm::vec3 perpendicular= fromOrigin - rayDir * glm::dot(fromOrigin, rayDir);
		return glm::dot(perpendicular, perpendicular);
	};

	// Distance to the ray around the circle is a degree-2 trigonometric
	// polynomial, so it has at most two minima. Sampling finds which arcs hold
	// them and a ternary search sharpens each: closed form here is a quartic,
	// and this stays readable, deterministic (replay depends on it) and exact
	// to well under a millimetre.
	constexpr int kCircleSamples= 64;
	constexpr int kRefineSteps= 24;
	float sampled[kCircleSamples];
	for (int i= 0; i < kCircleSamples; ++i)
		sampled[i]= rayDistanceSquared(pointAt(glm::two_pi<float>() * (float)i / (float)kCircleSamples));

	for (int i= 0; i < kCircleSamples; ++i)
	{
		const int previous= (i + kCircleSamples - 1) % kCircleSamples;
		const int next= (i + 1) % kCircleSamples;
		if (sampled[i] > sampled[previous] || sampled[i] > sampled[next])
			continue;

		const float step= glm::two_pi<float>() / (float)kCircleSamples;
		float low= step * (float)(i - 1);
		float high= step * (float)(i + 1);
		for (int refine= 0; refine < kRefineSteps; ++refine)
		{
			const float third= (high - low) / 3.f;
			const float a= low + third;
			const float b= high - third;
			if (rayDistanceSquared(pointAt(a)) < rayDistanceSquared(pointAt(b)))
				high= b;
			else
				low= a;
		}
		if (outHit.count < 2)
			outHit.candidates[outHit.count++]= pointAt(0.5f * (low + high));
	}

	if (outHit.count == 0)
		return false;
	if (outHit.count == 2 &&
		rayDistanceSquared(outHit.candidates[1]) < rayDistanceSquared(outHit.candidates[0]))
		std::swap(outHit.candidates[0], outHit.candidates[1]);
	return true;
}

void BodyPoseSolver::reset()
{
	for (JitterTracker& tracker : m_jitter)
		tracker= JitterTracker();
	m_arms= {};
	m_head= HeadEstimate();
	m_lastModelFrameIndex= -1;
	m_lastSolveTimestampMs= -1.0;
	m_lastModelTimestampMs= -1.0;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		m_forearmDirFilter[sideIndex].reset();
		m_shoulderFilter[sideIndex].reset();
	}
	m_headPositionFilter.reset();
	m_headLeftDirFilter.reset();
	m_headForwardFilter.reset();
}

void BodyPoseSolver::solve(
	const std::vector<const CameraFrameResult*>& cameras,
	const BodyDimensions& dimensions,
	TrackingFrameResult& fused)
{
	if (!m_bFiltersConfigured)
	{
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			m_forearmDirFilter[sideIndex].configure(kDirectionMinCutoffHz, kDirectionBeta, kDerivativeCutoffHz);
			m_shoulderFilter[sideIndex].configure(kPositionMinCutoffHz, kPositionBeta, kDerivativeCutoffHz);
		}
		m_headPositionFilter.configure(kPositionMinCutoffHz, kPositionBeta, kDerivativeCutoffHz);
		m_headLeftDirFilter.configure(kDirectionMinCutoffHz, kDirectionBeta, kDerivativeCutoffHz);
		m_headForwardFilter.configure(kDirectionMinCutoffHz, kDirectionBeta, kDerivativeCutoffHz);
		m_bFiltersConfigured= true;
	}

	// The best body observation across cameras (normally at most one camera
	// has the stage enabled)
	const CameraFrameResult* source= nullptr;
	for (const CameraFrameResult* camera : cameras)
	{
		if (camera == nullptr || !camera->valid || !camera->hasExtrinsics || !camera->hasIntrinsics)
			continue;
		if (!camera->result.body.valid)
			continue;
		if (source == nullptr || camera->result.body.confidence > source->result.body.confidence)
			source= camera;
	}
	if (source == nullptr)
		return;

	// A gap in model results (detection lost, camera stalled) invalidates the
	// held estimates rather than letting them ride an unbounded distance
	const double elapsedMs= m_lastSolveTimestampMs >= 0.0 ? fused.timestampMs - m_lastSolveTimestampMs : 0.0;
	if (elapsedMs < 0.0 || elapsedMs > kMaxHoldMs)
		reset();

	// Only a NEW model result carries new information. Re-solving a repeated
	// one against a moved wrist would slide the estimate along a stale ray.
	const int64_t modelFrameIndex= source->result.body.modelFrameIndex;
	if (modelFrameIndex != m_lastModelFrameIndex)
	{
		solveFromObservation(*source, dimensions, fused);
		m_lastModelFrameIndex= modelFrameIndex;
		m_lastModelTimestampMs= fused.timestampMs;
	}
	m_lastSolveTimestampMs= fused.timestampMs;

	applyEstimates(dimensions.forearmLengthMeters, fused);
}

void BodyPoseSolver::solveFromObservation(
	const CameraFrameResult& source,
	const BodyDimensions& dimensions,
	const TrackingFrameResult& fused)
{
	const BodyPoseObservation& body= source.result.body;
	const glm::vec3 cameraPos= cameraPositionWorld(source.markerFromCamera);

	// dt for the smoothing filters: the interval between MODEL results, which
	// is what the filtered series is actually sampled at
	const float dtSeconds= m_lastModelTimestampMs >= 0.0
		? std::max((float)(fused.timestampMs - m_lastModelTimestampMs) * 0.001f, 1e-3f)
		: 1.f / 30.f;

	auto pixelRay= [&](int landmarkIndex) -> glm::vec3 {
		return pixelRayDirWorld(
			source.markerFromCamera,
			source.fx, source.fy, source.cx, source.cy,
			glm::vec2(body.imagePoints[landmarkIndex]));
	};
	auto isVisible= [&](int landmarkIndex) { return body.visibility[landmarkIndex] >= kMinVisibility; };

	// -- Shoulders, from their own two rays plus the CALIBRATED separation --
	// Solved FIRST and independently of the arms. Chaining them off the elbow
	// instead made them useless as a check on it: the shoulder then agrees
	// with whichever elbow was picked, so a wrong elbow simply dragged the
	// shoulder along and produced an arm bent the wrong way. This needs the
	// separation to be MEASURED - the same construction with an anatomical
	// guess put shoulders 0.8 m too far away - which is what the body
	// measurement wizard is for.
	{
		const int leftIndex= (int)ePoseLandmark::LEFT_SHOULDER;
		const int rightIndex= (int)ePoseLandmark::RIGHT_SHOULDER;
		JitterTracker& leftTracker= m_jitter[JointShoulderLeft];
		JitterTracker& rightTracker= m_jitter[JointShoulderRight];

		const glm::vec3 leftRay= pixelRay(leftIndex);
		const glm::vec3 rightRay= pixelRay(rightIndex);
		const float depth= isVisible(leftIndex) && isVisible(rightIndex)
			? depthFromKnownSeparation(leftRay, rightRay, dimensions.shoulderWidthMeters)
			: 0.f;

		if (depth >= kMinBodyDepthMeters && depth <= kMaxBodyDepthMeters)
		{
			const glm::vec3 shoulders[2]= {cameraPos + leftRay * depth, cameraPos + rightRay * depth};
			for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
			{
				JitterTracker& tracker= sideIndex == (int)eHandSide::Left ? leftTracker : rightTracker;
				ArmEstimate& arm= m_arms[sideIndex];
				const float stability= tracker.update(shoulders[sideIndex], dtSeconds, kAccelReference);

				arm.bHasShoulder= true;
				arm.shoulderPositionWorld= m_shoulderFilter[sideIndex].filter(shoulders[sideIndex], dtSeconds);
				arm.shoulderConfidence= stability;
			}
		}
		else
		{
			// One shoulder unseen: the pair no longer fixes a depth. The held
			// shoulders stay, since they are torso-fixed.
			leftTracker.invalidate();
			rightTracker.invalidate();
		}
	}

	// -- Elbows -> forearm directions --
	// The shoulders above are now an INDEPENDENT arbiter for the elbow's
	// depth, which the 2D alone cannot resolve: the two candidates sit up to
	// half a metre apart, and only one of them is an upper arm from the
	// shoulder.
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const int elbowIndex= sideIndex == (int)eHandSide::Left
			? (int)ePoseLandmark::LEFT_ELBOW : (int)ePoseLandmark::RIGHT_ELBOW;
		const int shoulderIndex= sideIndex == (int)eHandSide::Left
			? (int)ePoseLandmark::LEFT_SHOULDER : (int)ePoseLandmark::RIGHT_SHOULDER;
		JitterTracker& tracker= m_jitter[sideIndex == (int)eHandSide::Left
			? JointElbowLeft : JointElbowRight];
		ArmEstimate& arm= m_arms[sideIndex];

		const HandPose& pose= fused.poses[sideIndex];
		if (!pose.tracked || !pose.hasWorldPose)
		{
			tracker.invalidate();
			arm.bHasForearm= false;
			arm.deadReckonMs= 0.0;
			m_forearmDirFilter[sideIndex].reset();
			continue;
		}

		const glm::vec3 wristWorld= pose.getWristPositionWorld();
		const float forearmLength= dimensions.forearmLengthMeters;

		// -- Elbow occluded: dead-reckon on the bone circle --
		// One hand passing in front of the other elbow is routine on this rig,
		// and dropping the forearm there is expensive out of proportion to the
		// missing landmark: a consumer that leaves unstreamed arm bones at the
		// avatar's rest pose (VMC) snaps the whole arm to a T-pose, and the
		// arm's entire rotation then surfaces at the WRIST. So the arm is kept
		// alive from the shoulder and the wrist - both still measured - with
		// the circle point picked by continuity. Both bone lengths hold by
		// construction, which holding the last elbow position could not do.
		if (!isVisible(elbowIndex))
		{
			arm.deadReckonMs+= (double)(dtSeconds * 1000.f);
			const bool bCanDeadReckon= arm.bHasForearm && arm.bHasShoulder &&
				arm.deadReckonMs <= kMaxDeadReckonMs;

			glm::vec3 elbowWorld(0.f);
			if (!bCanDeadReckon ||
				!solveElbowNearestTo(arm.shoulderPositionWorld, wristWorld,
									 dimensions.upperArmLengthMeters, forearmLength,
									 wristWorld + arm.forearmDirWorld * forearmLength, elbowWorld))
			{
				tracker.invalidate();
				arm.bHasForearm= false;
				arm.deadReckonMs= 0.0;
				m_forearmDirFilter[sideIndex].reset();
				continue;
			}

			// The direction is republished (not just held) so the elbow tracks
			// the wrist along the circle rather than rigidly trailing it
			const glm::vec3 rawDir= glm::normalize(elbowWorld - wristWorld);
			const glm::vec3 filteredDir= m_forearmDirFilter[sideIndex].filter(rawDir, dtSeconds);
			arm.forearmDirWorld= glm::dot(filteredDir, filteredDir) > 1e-8f
				? glm::normalize(filteredDir) : rawDir;

			// Confidence decays with the gap and deliberately does NOT include
			// the jitter tracker: a dead-reckoned elbow is smooth by
			// construction, and scoring smoothness here would report a
			// confident arm precisely when it is being inferred (the frozen
			// head that scored 1.00, in another form).
			const float decay= 1.f - (float)(arm.deadReckonMs / kMaxDeadReckonMs);
			tracker.invalidate();
			arm.forearmConfidence= pose.confidence * std::clamp(decay, 0.f, 1.f);
			continue;
		}
		arm.deadReckonMs= 0.0;

		const glm::vec3 elbowRay= pixelRay(elbowIndex);

		glm::vec3 candidates[2];
		int candidateCount= 0;

		if (arm.bHasShoulder)
		{
			// Both bones known: the elbow is on the circle where their spheres
			// meet, and the ray only picks a point on it. Measuring the ray
			// against the shoulder the other way round - elbow from the ray,
			// then check the upper arm reaches - is blind whenever the
			// shoulder sits at the wrist's range, because the two ray
			// solutions then straddle it almost symmetrically and score
			// alike. Measured over recording 2026-08-14_20-46-03, that was
			// every frame of the left arm (the two differed by 15 mm, and the
			// nearer, wrong one scored BETTER), against 115 mm on the right.
			BoneCircleHit circle;
			if (solveElbowOnBoneCircle(arm.shoulderPositionWorld, wristWorld,
									   dimensions.upperArmLengthMeters, forearmLength,
									   cameraPos, elbowRay, circle))
			{
				candidateCount= circle.count;
				for (int i= 0; i < circle.count; ++i)
					candidates[i]= circle.candidates[i];
			}
		}

		if (candidateCount == 0)
		{
			// No shoulder to close the chain: fall back to the elbow one
			// forearm from the wrist along the ray, which fixes the forearm
			// but leaves the upper arm unconstrained
			RaySphereHit hit;
			if (!intersectRaySphere(cameraPos, elbowRay, wristWorld, forearmLength, hit))
			{
				// Degenerate geometry (the ray behind the camera, or straight
				// through the wrist). With no model 3D to fall back on, this
				// side simply has no forearm this frame.
				tracker.invalidate();
				arm.bHasForearm= false;
				m_forearmDirFilter[sideIndex].reset();
				continue;
			}
			if (hit.bNearValid)
				candidates[candidateCount++]= hit.nearPoint;
			if (hit.bFarValid && !hit.bClamped)
				candidates[candidateCount++]= hit.farPoint;
			if (candidateCount == 0)
			{
				tracker.invalidate();
				arm.bHasForearm= false;
				m_forearmDirFilter[sideIndex].reset();
				continue;
			}
		}

		// Both candidates are anatomically valid now, so this only decides
		// WHICH valid arm: continuity when there is any, else the desk prior -
		// hands reach toward the camera, so the elbow trails behind the wrist.
		int chosen= 0;
		if (candidateCount > 1)
		{
			if (arm.bHasForearm)
			{
				const glm::vec3 predicted= wristWorld + arm.forearmDirWorld * forearmLength;
				chosen= glm::length(candidates[0] - predicted) <= glm::length(candidates[1] - predicted)
					? 0 : 1;
			}
			else
			{
				chosen= glm::length(candidates[0] - cameraPos) >= glm::length(candidates[1] - cameraPos)
					? 0 : 1;
			}
		}
		const glm::vec3 elbowWorld= candidates[chosen];

		const float stability= tracker.update(elbowWorld, dtSeconds, kAccelReference);

		// Store the DIRECTION, not the position: the elbow is one rigid bone
		// from the wrist, so holding the direction lets it ride a wrist that
		// keeps updating at the full camera rate between model results
		const glm::vec3 rawDir= glm::normalize(elbowWorld - wristWorld);
		const glm::vec3 filteredDir= m_forearmDirFilter[sideIndex].filter(rawDir, dtSeconds);
		arm.forearmDirWorld= glm::dot(filteredDir, filteredDir) > 1e-8f
			? glm::normalize(filteredDir) : rawDir;
		arm.bHasForearm= true;
		arm.forearmConfidence= pose.confidence * stability;
	}

	// -- Head, from the two ear rays plus the known head width --
	{
		const int noseIndex= (int)ePoseLandmark::NOSE;
		const int leftEarIndex= (int)ePoseLandmark::LEFT_EAR;
		const int rightEarIndex= (int)ePoseLandmark::RIGHT_EAR;
		JitterTracker& tracker= m_jitter[JointHead];

		if (!isVisible(noseIndex) || !isVisible(leftEarIndex) || !isVisible(rightEarIndex))
		{
			tracker.invalidate();
			return;
		}

		const glm::vec3 leftRay= pixelRay(leftEarIndex);
		const glm::vec3 rightRay= pixelRay(rightEarIndex);

		// Both ears at one range. Geometrically that should read a turned head
		// as a receding one, since yaw foreshortens the true ear separation -
		// but it measurably does NOT on this model, and correcting for it made
		// things worse. Over recording 2026-08-14_19-53-18 the solved depth
		// correlates with the yaw evidence at -0.01, and multiplying the
		// separation by cos(yaw) took that to -0.54 and widened the depth
		// swing from 149 to 168 mm. RTMPose does not place the ears like
		// rigid points under rotation; it carries its own head prior and keeps
		// their apparent separation. Do not reintroduce the correction without
		// re-measuring that correlation.
		const float depth= depthFromKnownSeparation(leftRay, rightRay, dimensions.headWidthMeters);
		if (depth < kMinBodyDepthMeters || depth > kMaxBodyDepthMeters)
		{
			tracker.invalidate();
			return;
		}

		const glm::vec3 leftEar= cameraPos + leftRay * depth;
		const glm::vec3 rightEar= cameraPos + rightRay * depth;
		const glm::vec3 earMid= 0.5f * (leftEar + rightEar);

		// The nose sits a known distance forward of the ear midpoint, so its
		// own ray against that sphere places it. Its depth is the whole
		// signal for head yaw and pitch - putting it at the EAR depth instead
		// would flatten the face into the ear plane and lose them.
		RaySphereHit noseHit;
		if (!intersectRaySphere(cameraPos, pixelRay(noseIndex), earMid, dimensions.noseForwardMeters, noseHit) ||
			!noseHit.bNearValid)
		{
			tracker.invalidate();
			return;
		}
		// The near intersection: a face points toward the camera that sees it
		const glm::vec3 nose= noseHit.nearPoint;

		const glm::vec3 leftDirRaw= leftEar - rightEar;
		const glm::vec3 forwardRaw= nose - earMid;
		const glm::vec3 upRaw= glm::cross(forwardRaw, leftDirRaw);
		if (glm::dot(leftDirRaw, leftDirRaw) < 1e-8f || glm::dot(upRaw, upRaw) < 1e-8f)
		{
			tracker.invalidate();
			return;
		}

		const float stability= tracker.update(earMid, dtSeconds, kAccelReference);

		// Stability alone says a head is trustworthy whenever it holds still,
		// which a wrong one does perfectly: the collapsed-head frames above
		// reported 1.00 while sitting 3.9 m from the truth. The landmark
		// scores are the part of that judgement measured jitter cannot see.
		const float landmarkQuality= std::min(
			body.visibility[noseIndex],
			std::min(body.visibility[leftEarIndex], body.visibility[rightEarIndex]));

		m_head.bValid= true;
		m_head.positionWorld= m_headPositionFilter.filter(earMid, dtSeconds);
		m_head.leftDirWorld= m_headLeftDirFilter.filter(glm::normalize(leftDirRaw), dtSeconds);
		m_head.forwardWorld= m_headForwardFilter.filter(glm::normalize(forwardRaw), dtSeconds);
		m_head.confidence= stability * landmarkQuality;
	}
}

void BodyPoseSolver::applyEstimates(float forearmLengthMeters, TrackingFrameResult& fused) const
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		HandPose& pose= fused.poses[sideIndex];
		const ArmEstimate& arm= m_arms[sideIndex];

		// The shoulder is torso-fixed and no longer needs a tracked hand
		if (arm.bHasShoulder)
		{
			pose.hasShoulder= true;
			pose.shoulderPositionWorld= arm.shoulderPositionWorld;
			pose.shoulderConfidence= arm.shoulderConfidence;
		}

		// IMU wins: a measured forearm direction beats a monocular one
		if (pose.hasForearmPose || !arm.bHasForearm)
			continue;
		if (!pose.tracked || !pose.hasWorldPose)
			continue;

		// Forearm frame: +X from elbow toward the hand (equals the palm frame
		// at a neutral wrist). The direction fixes only two of its degrees of
		// freedom, so roll comes from the palm. Because the frame is rebuilt
		// against the CURRENT palm every iteration, the elbow rides the fresh
		// wrist even while the underlying direction is held.
		const glm::vec3 xAxis= -arm.forearmDirWorld;
		glm::vec3 zAxis= pose.palmOrientationWorld * glm::vec3(0.f, 0.f, 1.f);
		zAxis-= xAxis * glm::dot(zAxis, xAxis);
		if (glm::dot(zAxis, zAxis) < 1e-6f)
		{
			zAxis= pose.palmOrientationWorld * glm::vec3(0.f, 1.f, 0.f);
			zAxis-= xAxis * glm::dot(zAxis, xAxis);
		}
		if (glm::dot(zAxis, zAxis) < 1e-6f)
			continue;
		zAxis= glm::normalize(zAxis);
		const glm::vec3 yAxis= glm::cross(zAxis, xAxis);

		pose.hasForearmPose= true;
		pose.forearmOrientationWorld= glm::quat_cast(glm::mat3(xAxis, yAxis, zAxis));
		pose.forearmConfidence= arm.forearmConfidence;
	}

	if (m_head.bValid)
	{
		// Re-orthonormalize here rather than at solve time: the two direction
		// series are filtered independently, so their orthogonality is only
		// restored once, on the way out.
		//
		// FORWARD is the axis that survives this, not left. Yaw lives entirely
		// in the nose's offset along the ear axis, so rebuilding forward
		// perpendicular to the ears is exactly the projection that deletes it -
		// measured over recording 2026-08-14_19-53-18, that left 10 degrees of
		// yaw ANTI-correlated with the image evidence (-0.45) against the 48
		// degrees actually performed. Deriving left from forward instead
		// tracks it at +0.98. What gets discarded this way is the ear axis's
		// forward component, which is the quantity the equal-range ear
		// placement was guessing at anyway.
		const glm::vec3 forwardUnit= glm::normalize(m_head.forwardWorld);
		const glm::vec3 up= glm::cross(forwardUnit, m_head.leftDirWorld);
		if (glm::dot(up, up) < 1e-8f)
			return;
		const glm::vec3 upUnit= glm::normalize(up);
		const glm::vec3 leftUnit= glm::cross(upUnit, forwardUnit);

		fused.head.valid= true;
		fused.head.positionWorld= m_head.positionWorld;
		fused.head.orientationWorld= glm::quat_cast(glm::mat3(forwardUnit, leftUnit, upUnit));
		fused.head.confidence= m_head.confidence;
	}
}
