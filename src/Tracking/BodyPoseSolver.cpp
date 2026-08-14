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

// Escape hatch for a wrong initial choice of sphere intersection. The elbow
// must also sit one upper-arm length from the shoulder, which is INDEPENDENT
// of the depth ambiguity that picks the intersection - but measured live it
// prefers the wrong root on 19-43% of individual frames, so it only overrides
// continuity after a sustained, clear disagreement. Measured on the same
// session: fires once per arm over 29 seconds.
static constexpr float kShoulderLengthMarginM= 0.06f;
static constexpr int kRootOverrideFrames= 5;

// Two rays closer to parallel than this cannot resolve a separation into a
// depth (the pair projects to nearly one point)
static constexpr float kMinRaySeparation= 1e-3f;

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

	// -- Elbows -> forearm directions --
	// Solved BEFORE the shoulders, which hang off them. The root choice reads
	// the PREVIOUS model frame's shoulder, which is fine because the torso
	// barely moves between them, and it keeps the two solves acyclic.
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
		if (!pose.tracked || !pose.hasWorldPose || !isVisible(elbowIndex))
		{
			tracker.invalidate();
			arm.bHasForearm= false;
			m_forearmDirFilter[sideIndex].reset();
			continue;
		}

		const glm::vec3 wristWorld= pose.getWristPositionWorld();
		const float forearmLength= dimensions.forearmLengthMeters;

		// The elbow lies one forearm from the fused wrist, along the camera
		// ray through its 2D landmark
		RaySphereHit hit;
		if (!intersectRaySphere(cameraPos, pixelRay(elbowIndex), wristWorld, forearmLength, hit))
		{
			// Degenerate geometry (the ray behind the camera, or straight
			// through the wrist). With no model 3D to fall back on, this side
			// simply has no forearm this frame.
			tracker.invalidate();
			arm.bHasForearm= false;
			m_forearmDirFilter[sideIndex].reset();
			continue;
		}

		// Two intersections: the elbow is either in front of or behind the
		// plane through the wrist perpendicular to the view ray. They sit
		// ~400mm apart on this rig, so choosing wrong teleports the elbow.
		const glm::vec3 candidates[2]= {hit.nearPoint, hit.farPoint};
		int chosen= 0;
		if (hit.bNearValid != hit.bFarValid)
		{
			chosen= hit.bNearValid ? 0 : 1;
		}
		else
		{
			// How far a candidate elbow is from being reachable from the
			// shoulder. Only the shoulder's RAY is used, never a shoulder
			// position: the shoulder is solved by chaining off this very
			// elbow, so its position agrees with whichever root was picked
			// and cannot arbitrate. The ray does not depend on the elbow at
			// all, and a candidate lying farther from it than one upper arm
			// is simply unreachable, whatever the shoulder's depth.
			const glm::vec3 shoulderRay= pixelRay(shoulderIndex);
			auto reachViolation= [&](int candidate) -> float {
				if (!isVisible(shoulderIndex))
					return 0.f;
				const glm::vec3 toCandidate= candidates[candidate] - cameraPos;
				const glm::vec3 alongRay= shoulderRay * glm::dot(toCandidate, shoulderRay);
				const float distanceToRay= glm::length(toCandidate - alongRay);
				return std::max(distanceToRay - dimensions.upperArmLengthMeters, 0.f);
			};

			if (arm.bHasForearm)
			{
				const glm::vec3 predicted= wristWorld + arm.forearmDirWorld * forearmLength;
				chosen= glm::length(candidates[0] - predicted) <= glm::length(candidates[1] - predicted)
					? 0 : 1;

				// Sustained unreachability escapes a wrong lock. Individually
				// this test is noisy (the 2D shoulder wanders), so it has to
				// hold for several consecutive model frames.
				const int other= 1 - chosen;
				if (reachViolation(other) + kShoulderLengthMarginM < reachViolation(chosen))
				{
					arm.rootDisagreeStreak++;
					if (arm.rootDisagreeStreak >= kRootOverrideFrames)
					{
						chosen= other;
						arm.rootDisagreeStreak= 0;
					}
				}
				else
				{
					arm.rootDisagreeStreak= 0;
				}
			}
			else
			{
				const float violation[2]= {reachViolation(0), reachViolation(1)};
				if (violation[0] != violation[1])
				{
					chosen= violation[0] < violation[1] ? 0 : 1;
				}
				else
				{
					// Both equally reachable. Behind the wrist is the better
					// prior at a desk: the hands reach toward the camera.
					chosen= 1;
				}
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

	// -- Shoulders --
	// Chained off the solved elbow rather than placed by the shoulder pair's
	// apparent width. The width route floats free of anything measured, and
	// the landmark shoulder points sit well inside the anatomical shoulder
	// joints - measured on a live session, assuming a 0.40m biacromial width
	// put the shoulders 0.8m too far away, giving a 0.77m upper arm. Chaining
	// anchors the shoulder to the fused wrist, the best-measured point in the
	// system, through a bone length that IS the thing being assumed.
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const int shoulderIndex= sideIndex == (int)eHandSide::Left
			? (int)ePoseLandmark::LEFT_SHOULDER : (int)ePoseLandmark::RIGHT_SHOULDER;
		JitterTracker& tracker= m_jitter[sideIndex == (int)eHandSide::Left
			? JointShoulderLeft : JointShoulderRight];
		ArmEstimate& arm= m_arms[sideIndex];

		if (!isVisible(shoulderIndex) || !arm.bHasForearm || !fused.poses[sideIndex].hasWorldPose)
		{
			// The held shoulder stays: it is torso-fixed, so the last solved
			// position remains the best estimate until an elbow returns
			tracker.invalidate();
			continue;
		}

		const glm::vec3 elbowWorld=
			fused.poses[sideIndex].getWristPositionWorld() +
			arm.forearmDirWorld * dimensions.forearmLengthMeters;

		RaySphereHit hit;
		if (!intersectRaySphere(cameraPos, pixelRay(shoulderIndex), elbowWorld,
								dimensions.upperArmLengthMeters, hit))
		{
			tracker.invalidate();
			continue;
		}
		// The shoulder is the root farther from the camera: an arm reaches
		// toward what it works on, so the elbow leads the shoulder
		const glm::vec3 shoulderWorld= hit.bFarValid ? hit.farPoint : hit.nearPoint;

		const float stability= tracker.update(shoulderWorld, dtSeconds, kAccelReference);
		arm.bHasShoulder= true;
		arm.shoulderPositionWorld= m_shoulderFilter[sideIndex].filter(shoulderWorld, dtSeconds);
		arm.shoulderConfidence= fused.poses[sideIndex].confidence * stability;
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
		const float depth= depthFromKnownSeparation(leftRay, rightRay, dimensions.headWidthMeters);
		if (depth <= 0.f)
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

		m_head.bValid= true;
		m_head.positionWorld= m_headPositionFilter.filter(earMid, dtSeconds);
		m_head.leftDirWorld= m_headLeftDirFilter.filter(glm::normalize(leftDirRaw), dtSeconds);
		m_head.forwardWorld= m_headForwardFilter.filter(glm::normalize(forwardRaw), dtSeconds);
		m_head.confidence= stability;
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
		// restored once, on the way out
		const glm::vec3 up= glm::cross(m_head.forwardWorld, m_head.leftDirWorld);
		if (glm::dot(up, up) < 1e-8f)
			return;
		const glm::vec3 upUnit= glm::normalize(up);
		const glm::vec3 leftUnit= glm::normalize(m_head.leftDirWorld);
		const glm::vec3 forwardUnit= glm::cross(leftUnit, upUnit);

		fused.head.valid= true;
		fused.head.positionWorld= m_head.positionWorld;
		fused.head.orientationWorld= glm::quat_cast(glm::mat3(forwardUnit, leftUnit, upUnit));
		fused.head.confidence= m_head.confidence;
	}
}
