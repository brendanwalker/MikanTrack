#include "HandStateEstimator.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"

namespace
{
// Numeric central-difference step per state block. Small enough to stay in
// the locally linear regime of FK + projection, large enough that float
// cancellation noise stays orders of magnitude below the derivative.
constexpr float kDiffStepPosM= 1e-4f;
constexpr float kDiffStepRad= 1e-3f;
// Constant Levenberg damping on the normal-equation diagonal: the prior rows
// already make the system full rank, this just keeps a near-degenerate
// mono-view step from overshooting. Fixed (not adaptive) for determinism.
constexpr double kDamping= 1e-3;
// Early exit when a step stops moving the state (deterministic: driven purely
// by the inputs, so replay takes the same exit)
constexpr float kStepEpsilon= 1e-6f;
// Loose-prior factor for a reseed fit: the seed came from the classic
// per-frame extraction, so the measurements - not the seed - should place
// the state; the prior only has to keep the solve from wandering
constexpr float kReseedPriorScale= 5.f;

// dt clamp for the temporal prior (matches the smoothing clamp's spirit: a
// scheduler hiccup must not open the leash arbitrarily wide)
constexpr double kMinDtMs= 1.0;
constexpr double kMaxDtMs= 250.0;

// Anatomical joint limits in RAW angle space (flat hand = 0), radians,
// [dof][lo/hi] per finger. The four fingers share one profile: fingers
// hyperextend only 10-20 degrees (the measured reason flat hands broke the
// curl-evidence palmar test) while flexing past 90. The thumb's chained
// CMC/MCP/IP angles ride the pronated hinge and its base genuinely roams, so
// its ranges are deliberately generous - a too-tight thumb limit would fight
// real opposition poses.
constexpr float kFingerLimits[4][2]= {
	{-0.45f, 0.45f}, // lateral splay
	{-0.35f, 1.92f}, // proximal (MCP): ~-20..110 deg
	{-0.17f, 2.00f}, // intermediate (PIP): ~-10..115 deg
	{-0.17f, 1.60f}, // distal (DIP): ~-10..92 deg
};
constexpr float kThumbLimits[4][2]= {
	{-1.00f, 1.00f}, // lateral about the palm normal
	{-0.70f, 1.20f}, // CMC-ish proximal
	{-0.35f, 1.20f}, // MCP flexion
	{-0.35f, 1.40f}, // IP flexion
};

float angleComponent(const FingerAngles& angles, int dof)
{
	switch (dof)
	{
	case 0: return angles.lateral;
	case 1: return angles.proximal;
	case 2: return angles.intermediate;
	default: return angles.distal;
	}
}

void addToAngleComponent(FingerAngles& angles, int dof, float delta)
{
	switch (dof)
	{
	case 0: angles.lateral+= delta; break;
	case 1: angles.proximal+= delta; break;
	case 2: angles.intermediate+= delta; break;
	default: angles.distal+= delta; break;
	}
}

// Body-frame rotation from `from` to `to` as a rotation vector (the manifold
// difference the error-state parameterizes)
glm::vec3 quatDeltaVector(const glm::quat& from, const glm::quat& to)
{
	glm::quat dq= glm::inverse(from) * to;
	if (dq.w < 0.f)
		dq= -dq; // hemisphere-align: the short way around

	const glm::vec3 v(dq.x, dq.y, dq.z);
	const float vLength= glm::length(v);
	if (vLength < 1e-8f)
		return glm::vec3(0.f);

	const float angle= 2.f * atan2f(vLength, dq.w);
	return v * (angle / vLength);
}

// Cholesky solve of A x = b for a symmetric positive-definite A (row-major
// n x n). In-place decomposition; false when A is not positive definite.
bool solveCholesky(std::vector<double>& A, std::array<double, 26>& b, int n)
{
	// LL^T decomposition
	for (int i= 0; i < n; ++i)
	{
		for (int j= 0; j <= i; ++j)
		{
			double sum= A[i * n + j];
			for (int k= 0; k < j; ++k)
				sum-= A[i * n + k] * A[j * n + k];

			if (i == j)
			{
				if (sum <= 0.0)
					return false;
				A[i * n + i]= sqrt(sum);
			}
			else
			{
				A[i * n + j]= sum / A[j * n + j];
			}
		}
	}

	// Forward substitution: L y = b
	for (int i= 0; i < n; ++i)
	{
		double sum= b[i];
		for (int k= 0; k < i; ++k)
			sum-= A[i * n + k] * b[k];
		b[i]= sum / A[i * n + i];
	}

	// Back substitution: L^T x = y
	for (int i= n - 1; i >= 0; --i)
	{
		double sum= b[i];
		for (int k= i + 1; k < n; ++k)
			sum-= A[k * n + i] * b[k];
		b[i]= sum / A[i * n + i];
	}
	return true;
}
} // namespace

void HandStateEstimator::configure(const HandStateEstimatorConfig& config)
{
	m_config= config;
}

void HandStateEstimator::angleLimits(eFinger finger, int dof, float& outLo, float& outHi)
{
	const float (*limits)[2]= finger == eFinger::Thumb ? kThumbLimits : kFingerLimits;
	outLo= limits[dof][0];
	outHi= limits[dof][1];
}

float HandStateEstimator::angleLimitViolation(eFinger finger, int dof, float angleRad)
{
	float lo= 0.f;
	float hi= 0.f;
	angleLimits(finger, dof, lo, hi);
	if (angleRad > hi)
		return angleRad - hi;
	if (angleRad < lo)
		return angleRad - lo;
	return 0.f;
}

void HandStateEstimator::resetSide(eHandSide side)
{
	m_sides[(int)side]= SideState();
}

void HandStateEstimator::resetAll()
{
	resetSide(eHandSide::Left);
	resetSide(eHandSide::Right);
}

void HandStateEstimator::predictWorldLandmarks(const Pose& pose, const HandSkeleton& skeleton,
											   std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints)
{
	glm::mat4 palmTransform= glm::mat4_cast(pose.palmOrientationWorld);
	palmTransform[3]= glm::vec4(pose.palmPositionWorld, 1.f);

	std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
	HandPoseModel::buildFingerJoints(palmTransform, skeleton, pose.rawAngles, joints);

	// Palm +X is wrist -> middle MCP about the palm center, so the wrist sits
	// opposite the middle base (same reconstruction as the calibrated PnP
	// object model)
	outPoints[(int)eHandLandmark::WRIST]= glm::vec3(
		palmTransform * glm::vec4(-skeleton.baseInPalm[(int)eFinger::Middle].x, 0.f, 0.f, 1.f));
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int joint= 0; joint < 4; ++joint)
			outPoints[FINGER_JOINTS[finger][joint]]= joints[finger][joint];
}

HandStateEstimator::Pose HandStateEstimator::applyDelta(const Pose& pose,
														const std::array<float, kStateDim>& delta)
{
	Pose result= pose;
	result.palmPositionWorld+= glm::vec3(delta[0], delta[1], delta[2]);

	// Right-multiplied body-frame error quaternion (MEKF convention). The
	// small-angle form is exact enough at solver step sizes; normalize to
	// keep the quaternion on the manifold across iterations.
	const glm::quat errorQuat(1.f, 0.5f * delta[3], 0.5f * delta[4], 0.5f * delta[5]);
	result.palmOrientationWorld= glm::normalize(pose.palmOrientationWorld * errorQuat);

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int dof= 0; dof < 4; ++dof)
			addToAngleComponent(result.rawAngles[finger], dof, delta[6 + finger * 4 + dof]);

	return result;
}

bool HandStateEstimator::evalPixelResiduals(const Pose& pose, const HandSkeleton& skeleton,
											const std::vector<FitView>& views,
											std::vector<glm::vec2>& outResiduals)
{
	std::array<glm::vec3, HAND_LANDMARK_COUNT> worldPoints;
	predictWorldLandmarks(pose, skeleton, worldPoints);

	outResiduals.resize(views.size() * HAND_LANDMARK_COUNT);
	for (size_t v= 0; v < views.size(); ++v)
	{
		const FitView& view= views[v];
		const Observation* observation= view.observation;
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			const glm::dvec4 camPoint=
				view.cameraFromWorld * glm::dvec4(glm::dvec3(worldPoints[i]), 1.0);
			if (camPoint.z < 1e-3)
				return false; // behind the camera: the fit geometry is broken

			const glm::vec2 projected(
				(float)(observation->fx * camPoint.x / camPoint.z + observation->cx),
				(float)(observation->fy * camPoint.y / camPoint.z + observation->cy));
			outResiduals[v * HAND_LANDMARK_COUNT + i]=
				projected - glm::vec2((*observation->imagePoints)[i]);
		}
	}
	return true;
}

float HandStateEstimator::meanResidualPx(const std::vector<glm::vec2>& residuals,
										 const std::vector<FitView>& views)
{
	if (residuals.empty() || views.empty())
		return 0.f;

	float sum= 0.f;
	float weightSum= 0.f;
	for (size_t row= 0; row < residuals.size(); ++row)
	{
		const float weight= views[row / HAND_LANDMARK_COUNT].confidenceWeight;
		sum+= weight * glm::length(residuals[row]);
		weightSum+= weight;
	}
	return weightSum > 1e-6f ? sum / weightSum : 0.f;
}

int HandStateEstimator::solve(const Pose& start, const Pose& anchor,
							  const std::array<float, kStateDim>& priorSigma,
							  const glm::vec3* monoRayDir,
							  const HandStateEstimatorConfig::AnglePrior& anglePrior,
							  const HandSkeleton& skeleton, const std::vector<FitView>& views,
							  Pose& outPose) const
{
	Pose pose= start;
	outPose= start;

	std::vector<glm::vec2> residuals;
	std::vector<glm::vec2> residualsPlus;
	std::vector<glm::vec2> residualsMinus;
	const int rowPairs= (int)views.size() * HAND_LANDMARK_COUNT;

	// Per-row-pair Jacobian: d(residual uv)/d(state), rebuilt each iteration
	std::vector<glm::vec2> jacobian((size_t)rowPairs * kStateDim);
	std::vector<double> normal((size_t)kStateDim * kStateDim);
	std::array<double, kStateDim> rhs;

	int iterations= 0;
	for (int iteration= 0; iteration < m_config.maxIterations; ++iteration)
	{
		if (!evalPixelResiduals(pose, skeleton, views, residuals))
			break; // keep the last valid iterate

		// IRLS weights at the current iterate: per-landmark Huber over the uv
		// residual magnitude, times the view's presence weight, over the pixel
		// sigma. Stored squared for the normal-equation accumulation.
		std::vector<double> rowWeight(rowPairs);
		for (int row= 0; row < rowPairs; ++row)
		{
			const float errorPx= glm::length(residuals[row]);
			const float huber= errorPx <= m_config.huberDeltaPx
				? 1.f
				: m_config.huberDeltaPx / errorPx;
			const float weight=
				views[row / HAND_LANDMARK_COUNT].confidenceWeight / std::max(m_config.pixelSigmaPx, 1e-3f);
			rowWeight[row]= (double)(weight * weight * huber);
		}

		// Numeric central-difference Jacobian of the pixel residuals
		bool bJacobianValid= true;
		for (int dim= 0; dim < kStateDim && bJacobianValid; ++dim)
		{
			const float step= dim < 3 ? kDiffStepPosM : kDiffStepRad;
			std::array<float, kStateDim> delta{};
			delta[dim]= step;
			const Pose posePlus= applyDelta(pose, delta);
			delta[dim]= -step;
			const Pose poseMinus= applyDelta(pose, delta);

			if (!evalPixelResiduals(posePlus, skeleton, views, residualsPlus) ||
				!evalPixelResiduals(poseMinus, skeleton, views, residualsMinus))
			{
				bJacobianValid= false;
				break;
			}

			const float invTwoStep= 0.5f / step;
			for (int row= 0; row < rowPairs; ++row)
				jacobian[(size_t)row * kStateDim + dim]=
					(residualsPlus[row] - residualsMinus[row]) * invTwoStep;
		}
		if (!bJacobianValid)
			break;

		// Normal equations J^T W J d = -J^T W r, prior rows folded in as an
		// identity-Jacobian block (small-angle: the manifold difference moves
		// one-to-one with the delta parameterization)
		std::fill(normal.begin(), normal.end(), 0.0);
		rhs.fill(0.0);

		for (int row= 0; row < rowPairs; ++row)
		{
			const double w= rowWeight[row];
			const glm::vec2* jRow= &jacobian[(size_t)row * kStateDim];
			const glm::vec2& r= residuals[row];
			for (int i= 0; i < kStateDim; ++i)
			{
				const glm::vec2& ji= jRow[i];
				rhs[i]-= w * (double)(ji.x * r.x + ji.y * r.y);
				for (int j= 0; j <= i; ++j)
				{
					const glm::vec2& jj= jRow[j];
					normal[i * kStateDim + j]+= w * (double)(ji.x * jj.x + ji.y * jj.y);
				}
			}
		}

		// Temporal prior: pull toward the anchor, weighted by the process
		// sigmas. Residual = (pose (-) anchor) / sigma per dimension.
		{
			std::array<float, kStateDim> priorResidual;
			const glm::vec3 dPos= pose.palmPositionWorld - anchor.palmPositionWorld;
			priorResidual[0]= dPos.x;
			priorResidual[1]= dPos.y;
			priorResidual[2]= dPos.z;
			const glm::vec3 dRot=
				quatDeltaVector(anchor.palmOrientationWorld, pose.palmOrientationWorld);
			priorResidual[3]= dRot.x;
			priorResidual[4]= dRot.y;
			priorResidual[5]= dRot.z;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				for (int dof= 0; dof < 4; ++dof)
					priorResidual[6 + finger * 4 + dof]=
						angleComponent(pose.rawAngles[finger], dof) -
						angleComponent(anchor.rawAngles[finger], dof);

			// Position block: anisotropic under a single-camera fit - the
			// along-ray inverse variance is much larger, so the depth the mono
			// scale signal suggests is nearly ignored while lateral motion
			// tracks freely (symmetric, accumulated on the lower triangle like
			// the measurement rows)
			{
				const double invLat= 1.0 / std::max(priorSigma[0], 1e-5f);
				const double invRay= monoRayDir != nullptr
					? invLat / std::max(m_config.monoDepthPriorFactor, 1e-3f)
					: invLat;
				const glm::vec3 ray= monoRayDir != nullptr ? *monoRayDir : glm::vec3(0.f);
				for (int i= 0; i < 3; ++i)
				{
					for (int j= 0; j <= i; ++j)
					{
						const double outer= (double)ray[i] * (double)ray[j];
						const double w= invLat * invLat * ((i == j ? 1.0 : 0.0) - outer) +
							invRay * invRay * outer;
						normal[i * kStateDim + j]+= w;
						rhs[i]-= w * (double)priorResidual[j];
						if (j != i)
							rhs[j]-= w * (double)priorResidual[i];
					}
				}
			}
			for (int i= 3; i < kStateDim; ++i)
			{
				const double invSigma= 1.0 / std::max(priorSigma[i], 1e-5f);
				const double w= invSigma * invSigma;
				normal[i * kStateDim + i]+= w;
				rhs[i]-= w * (double)priorResidual[i];
			}
		}

		// Joint-limit prior: one-sided quadratic beyond the anatomical range,
		// re-linearized each iteration (the active set follows the iterate).
		// Zero cost inside the range, so plausible poses are untouched.
		if (m_config.jointLimitsEnabled)
		{
			const double invSigma= 1.0 / std::max(m_config.jointLimitSigmaRad, 1e-3f);
			const double w= invSigma * invSigma;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				for (int dof= 0; dof < 4; ++dof)
				{
					const float violation= angleLimitViolation(
						(eFinger)finger, dof, angleComponent(pose.rawAngles[finger], dof));
					if (violation == 0.f)
						continue;
					const int dim= 6 + finger * 4 + dof;
					normal[dim * kStateDim + dim]+= w;
					rhs[dim]-= w * (double)violation;
				}
			}
		}

		if (anglePrior.present)
		{
			// Fitted angle prior: Mahalanobis pull toward the user's own pose
			// distribution. The full precision matrix is the point - it makes
			// a coordinated pose change cheap and an uncorrelated single-DoF
			// excursion expensive, which is the difference between real
			// articulation and a landmark snap.
			constexpr int kAngles= HandStateEstimatorConfig::AnglePrior::k_angleCount;
			std::array<double, kAngles> deviation;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				for (int dof= 0; dof < 4; ++dof)
				{
					const int k= finger * 4 + dof;
					deviation[k]= (double)angleComponent(pose.rawAngles[finger], dof) -
						(double)anglePrior.mean[k];
				}

			// Linear scale on the precision (weight 1 = trust the fitted
			// distribution as-is)
			const double w= (double)anglePrior.weight;
			for (int i= 0; i < kAngles; ++i)
			{
				const int dimI= 6 + i;
				double pull= 0.0;
				for (int j= 0; j < kAngles; ++j)
					pull+= (double)anglePrior.precision[i * kAngles + j] * deviation[j];
				rhs[dimI]-= w * pull;
				for (int j= 0; j <= i; ++j)
					normal[dimI * kStateDim + (6 + j)]+= w * (double)anglePrior.precision[i * kAngles + j];
			}
		}
		else if (m_config.couplingSigmaRad > 0.f)
		{
			// Fallback DIP-PIP coupling for the four fingers (not the thumb -
			// its IP rides a different tendon arrangement): distal tracks
			// 0.67 x intermediate, weakly. residual r = distal - 0.67*inter.
			constexpr double kCouplingRatio= 0.67;
			const double invSigma= 1.0 / std::max(m_config.couplingSigmaRad, 1e-3f);
			const double w= invSigma * invSigma;
			for (int finger= (int)eFinger::Index; finger < FINGER_COUNT; ++finger)
			{
				const int dimInter= 6 + finger * 4 + 2;
				const int dimDistal= 6 + finger * 4 + 3;
				const double residual= (double)pose.rawAngles[finger].distal -
					kCouplingRatio * (double)pose.rawAngles[finger].intermediate;
				normal[dimDistal * kStateDim + dimDistal]+= w;
				normal[dimInter * kStateDim + dimInter]+= w * kCouplingRatio * kCouplingRatio;
				normal[dimDistal * kStateDim + dimInter]+= -w * kCouplingRatio;
				rhs[dimDistal]-= w * residual;
				rhs[dimInter]+= w * kCouplingRatio * residual;
			}
		}

		// Mirror the lower triangle + fixed Levenberg damping
		for (int i= 0; i < kStateDim; ++i)
		{
			for (int j= i + 1; j < kStateDim; ++j)
				normal[i * kStateDim + j]= normal[j * kStateDim + i];
			normal[i * kStateDim + i]*= 1.0 + kDamping;
		}

		if (!solveCholesky(normal, rhs, kStateDim))
			break;

		std::array<float, kStateDim> step;
		float stepNormSq= 0.f;
		for (int i= 0; i < kStateDim; ++i)
		{
			step[i]= (float)rhs[i];
			stepNormSq+= step[i] * step[i];
		}

		pose= applyDelta(pose, step);
		outPose= pose;
		++iterations;

		if (stepNormSq < kStepEpsilon * kStepEpsilon)
			break;
	}

	return iterations;
}

HandStateEstimator::UpdateResult HandStateEstimator::update(eHandSide side,
															const std::vector<Observation>& observations,
															double nowTimestampMs,
															const Pose* seed,
															const HandSkeleton& skeleton,
															Pose& outPose)
{
	UpdateResult result;
	SideState& state= m_sides[(int)side];

	// A side unobserved past the gap is a reacquisition: the old state says
	// nothing about where the hand is now
	if (state.bValid && nowTimestampMs - state.lastUpdateTimestampMs > m_config.resetGapMs)
		resetSide(side);

	// Every observation inside the measurement window enters the fit,
	// down-weighted by age (see measurementWindowMs for why there is no
	// freshness dedupe). During fast motion an older camera's pixels describe
	// an older hand, so age decays their say rather than gating them.
	std::vector<FitView> views;
	for (const Observation& observation : observations)
	{
		if (observation.imagePoints == nullptr || observation.fx <= 0.f || observation.fy <= 0.f)
			continue;

		const double ageMs= nowTimestampMs - observation.timestampMs;
		if (ageMs > m_config.measurementWindowMs)
			continue;

		const float ageFactor=
			std::clamp(1.f - (float)(ageMs / std::max(m_config.measurementWindowMs, 1.0)), 0.2f, 1.f);

		// Floor rather than zero: a camera whose confidence was never
		// measured (rescued from the low-presence pool) or has momentarily
		// collapsed still carries real 2D geometry, and dropping its rows
		// outright would throw away the second view a stereo fit needs.
		const float confidence=
			std::clamp(observation.confidence, m_config.minObservationConfidence, 1.f);

		FitView view;
		view.observation= &observation;
		view.cameraFromWorld= glm::inverse(observation.markerFromCamera);
		view.confidenceWeight= sqrtf(confidence * ageFactor);
		views.push_back(view);
	}
	result.cameraCount= (int)views.size();

	// Cold state: adopt the seed (from the classic per-frame extraction), then
	// let the measurements refine it under a loose prior
	if (!state.bValid)
	{
		if (seed == nullptr)
			return result;

		state.pose= *seed;
		result.bReseeded= true;

		if (views.empty())
		{
			// Nothing to fit yet: the seed is the state
			state.bValid= true;
			state.lastUpdateTimestampMs= nowTimestampMs;
			outPose= state.pose;
			result.bUpdated= true;
			return result;
		}
	}

	// Holding: state is alive but nothing new arrived for it this fuse
	if (views.empty())
	{
		outPose= state.pose;
		result.bUpdated= true;
		return result;
	}

	// Constant-position prediction: the anchor is the previous state, the
	// process sigmas (scaled by elapsed time) say how far the fit may stray
	const double dtMs= result.bReseeded
		? kMaxDtMs
		: std::clamp(nowTimestampMs - state.lastUpdateTimestampMs, kMinDtMs, kMaxDtMs);
	const float dt= (float)(dtMs / 1000.0);
	const float priorScale= result.bReseeded ? kReseedPriorScale : 1.f;

	std::array<float, kStateDim> priorSigma;
	for (int i= 0; i < 3; ++i)
		priorSigma[i]= m_config.palmPosSigmaMPerS * dt * priorScale;
	for (int i= 3; i < 6; ++i)
		priorSigma[i]= m_config.palmRotSigmaRadPerS * dt * priorScale;
	for (int i= 6; i < kStateDim; ++i)
		priorSigma[i]= m_config.angleSigmaRadPerS * dt * priorScale;

	const Pose anchor= state.pose;

	// A bad fit on a live state holds the previous pose; a streak of them (or
	// the reset gap running out, since holds do not advance the update stamp)
	// drops the state for a reseed. A bad fit while RESEEDING has no previous
	// pose worth holding - the state is simply not adopted.
	auto holdOrDrop= [&]() -> UpdateResult {
		if (result.bReseeded)
		{
			resetSide(side);
			return result; // bUpdated stays false: classic pose stands
		}
		++state.badFitStreak;
		if (state.badFitStreak >= m_config.maxBadFitStreak)
		{
			resetSide(side);
			return result;
		}
		outPose= state.pose;
		result.bUpdated= true;
		result.bHeldBadFit= true;
		return result;
	};

	// Single-camera fit: the position prior goes anisotropic along that
	// camera's view ray through the current palm (see monoDepthPriorFactor)
	glm::vec3 monoRay(0.f);
	bool bMonoRay= false;
	if (views.size() == 1)
	{
		const glm::dvec4 cameraPos= views[0].observation->markerFromCamera[3];
		const glm::vec3 toPalm= anchor.palmPositionWorld - glm::vec3(glm::dvec3(cameraPos));
		const float toPalmLength= glm::length(toPalm);
		if (toPalmLength > 1e-4f)
		{
			monoRay= toPalm / toPalmLength;
			bMonoRay= true;
		}
	}

	std::vector<glm::vec2> residuals;
	if (!evalPixelResiduals(anchor, skeleton, views, residuals))
		return holdOrDrop(); // predicted geometry behind a camera
	result.residualBeforePx= meanResidualPx(residuals, views);

	Pose fitted;
	result.iterations= solve(anchor, anchor, priorSigma, bMonoRay ? &monoRay : nullptr,
							 m_config.anglePrior[(int)side], skeleton, views, fitted);

	if (!evalPixelResiduals(fitted, skeleton, views, residuals))
		return holdOrDrop();
	result.residualAfterPx= meanResidualPx(residuals, views);

	if (result.residualAfterPx > m_config.maxResidualPx)
		return holdOrDrop();

	// Innovation gate: reject a physically impossible palm step however well
	// it fits (dt-scaled, so held/slow fuses allow a bigger catch-up step)
	if (!result.bReseeded &&
		glm::length(fitted.palmPositionWorld - anchor.palmPositionWorld) >
			m_config.maxStepMetersPerS * dt)
		return holdOrDrop();

	state.badFitStreak= 0;
	state.pose= fitted;
	state.bValid= true;
	state.lastUpdateTimestampMs= nowTimestampMs;

	outPose= state.pose;
	result.bUpdated= true;
	return result;
}
