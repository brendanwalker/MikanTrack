#pragma once

#include <array>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"
#include "glm/ext/quaternion_float.hpp"

#include "TrackingTypes.h"

// Temporally continuous per-hand state estimator in joint-angle space.
//
// One state per side - palm position (3) + palm orientation (error-state
// quaternion, 3 DoF) + 20 RAW finger angles = 26 DoF - fit each fuse to ALL
// fresh cameras' 2D landmarks at once by minimizing the forward-kinematics
// reprojection residual, with a temporal prior holding the state near its
// prediction. This replaces the per-frame winner selections (tri vs mono,
// stereo pair choice, articulation source, palmar sign re-decision) whose
// discrete switches are what the user sees as pops:
//   - bone lengths hold BY CONSTRUCTION (the fit moves palm + angles through
//     FK; landmark noise cannot stretch a bone the way per-landmark
//     triangulation can)
//   - a camera dropping out just removes measurement rows; the prior holds
//     the directions the remaining cameras cannot observe (the general form
//     of the along-ray position hold)
//   - the palmar sign is carried by state continuity instead of being
//     re-derived from curl evidence every frame
//   - MediaPipe's view-dependent monocular articulation (modelPoints) is not
//     in the loop; the fit consumes 2D imagePoints only
//
// Solved as damped Gauss-Newton with a fixed iteration cap, numeric central-
// difference Jacobian, Huber-weighted pixel residuals and a hand-rolled
// Cholesky over the 26x26 normal equations. Everything is deterministic (no
// wall clock, no randomness, fixed iteration schedule), so replay reproduces
// live bit-exactly.
//
// Pure math + a little per-side memory; no threads, no config singletons.
// Owned by HandFusion, which handles clustering/side assignment upstream and
// provides the seed pose from its existing tri/mono extraction.
struct HandStateEstimatorConfig
{
	// Temporal prior: how far each state block may plausibly move per second
	// (constant-position process model). This is a soft leash, not a clamp:
	// strong measurements (stereo agreement) override it freely, so fast
	// motion does not lag - it only binds when the measurement is weak or
	// inconsistent, which is exactly when the classic pipeline pops.
	float palmPosSigmaMPerS= 0.3f;
	float palmRotSigmaRadPerS= 2.f;
	float angleSigmaRadPerS= 2.5f;

	// Pixel measurement noise scale (relative weight of measurements vs prior)
	float pixelSigmaPx= 2.f;
	// Single-camera fits hold the palm's ALONG-RAY position this much tighter
	// than lateral (fraction of palmPosSigma). A mono fit's only depth signal
	// is apparent scale, which carries the per-camera bias stereo exists to
	// remove - drifting with it and snapping back when a second camera
	// returns was a measured 106 mm pop (recording 2026-08-15_22-38-17 frame
	// 581). The continuous form of the classic tri position hold.
	float monoDepthPriorFactor= 0.2f;
	// Per-landmark Huber threshold: residuals beyond this get down-weighted
	// linearly, so a single slipped landmark cannot drag the whole hand
	float huberDeltaPx= 10.f;

	// Gauss-Newton iteration cap (fixed for determinism)
	int maxIterations= 4;

	// Divergence guard: post-fit mean pixel residual above this means the fit
	// does not explain what the cameras see (bad correspondence, occlusion
	// garbage, bad seed). One bad fit HOLDS the previous state rather than
	// streaming the compromise; only a sustained streak drops the state for a
	// reseed - an instant drop turned every marginal frame into a
	// drop/classic-snap/reseed pop cycle on real recordings.
	float maxResidualPx= 25.f;
	int maxBadFitStreak= 5;

	// Physical innovation gate: a fitted palm step implying more than this
	// speed is not hand motion, it is a correspondence error (a mislabeled or
	// phantom cluster whose pixels fit perfectly well - measured: a solo
	// wrong-side observation teleported the palm 0.7 m in one fuse with a
	// 15 px residual). Gated steps take the bad-fit hold path, so a genuine
	// relocation still lands within maxBadFitStreak fuses via the reseed -
	// the gate can delay truth briefly but never censor it for good.
	float maxStepMetersPerS= 4.f;

	// Anatomical joint-limit prior: one-sided quadratic penalties beyond the
	// per-DoF limits (see angleLimits), in RAW angle space where flat hand is
	// zero. Soft on purpose - the fit must never sit on a hard boundary
	// discontinuity - and only active outside the limits, so plausible poses
	// cost nothing. This is what absorbs the single-finger landmark snaps the
	// measurement rows would otherwise follow.
	bool jointLimitsEnabled= true;
	// Excursion beyond a limit at which the penalty reaches unit cost
	// (smaller = stiffer). The limit fights measurement rows, so stiffness is
	// relative to pixelSigmaPx.
	float jointLimitSigmaRad= 0.05f;

	// Inter-joint coupling: DIP flexion is tendon-linked to PIP at roughly
	// 2/3, so the four fingers get a weak residual pulling distal toward
	// 0.67 x intermediate. Deviation at which the pull reaches unit cost
	// (large = weak, deliberately: the link is approximate). Automatically
	// superseded by a fitted angle prior, which contains the same coupling
	// empirically. 0 disables.
	float couplingSigmaRad= 0.25f;

	// Fitted angle prior (per side): weak Mahalanobis pull toward the user's
	// own pose distribution, fit offline by --fit-angle-prior. This is the
	// term that tells a coordinated pose change (cheap under the fitted
	// correlations) from a single-DoF landmark snap (expensive). weight
	// scales the precision matrix; 1 = trust the fit as-is.
	struct AnglePrior
	{
		static constexpr int k_angleCount= FINGER_COUNT * 4;
		bool present= false;
		float weight= 0.3f;
		std::array<float, k_angleCount> mean{};
		// Row-major inverse covariance
		std::array<float, k_angleCount * k_angleCount> precision{};
	};
	AnglePrior anglePrior[2];

	// Age at which an observation's weight has fully decayed. EVERY
	// observation inside this window enters every fit (there is deliberately
	// no per-camera freshness dedupe): cameras deliver in turn, and a fit
	// against only the freshest camera's pixels ping-pongs the state between
	// the cameras' systematically disagreeing solutions at fuse cadence
	// (measured 9-10 mm median step on a quiet recording). Refitting an
	// unchanged row from the converged state just stays converged - this is a
	// per-fuse MAP, not a covariance-carrying filter, so nothing over-counts.
	double measurementWindowMs= 66.0;

	// A side unobserved for longer than this resets (mirrors the fusion
	// staleness handling: a reacquired hand may be anywhere)
	double resetGapMs= 264.0;
};

class HandStateEstimator
{
public:
	// One camera's view of the hand for this fuse. imagePoints must outlive
	// the update() call; z components are ignored (MediaPipe relative depth).
	struct Observation
	{
		int cameraIndex= -1;
		double timestampMs= 0.0;
		float presence= 1.f;
		glm::dmat4 markerFromCamera{1.0};
		float fx= 0.f;
		float fy= 0.f;
		float cx= 0.f;
		float cy= 0.f;
		const std::array<glm::vec3, HAND_LANDMARK_COUNT>* imagePoints= nullptr;
	};

	// Palm transform + RAW angles (pre rest-offset: FK-consistent, the space
	// the state lives in; the caller applies the streamed rest convention)
	struct Pose
	{
		glm::vec3 palmPositionWorld{0.f};
		glm::quat palmOrientationWorld{1.f, 0.f, 0.f, 0.f};
		std::array<FingerAngles, FINGER_COUNT> rawAngles{};
	};

	struct UpdateResult
	{
		// State is valid and outPose filled (false = no state and no usable
		// seed, or the divergence guard dropped the fit - caller falls back)
		bool bUpdated= false;
		// Cameras whose observations entered the fit (0 = the state was held:
		// nothing inside the measurement window to fit against)
		int cameraCount= 0;
		int iterations= 0;
		// Mean pixel residual over all landmark rows at the predicted state
		// (before fitting) and at the solution - the fit's own quality metric
		float residualBeforePx= 0.f;
		float residualAfterPx= 0.f;
		// The state was (re)seeded this call (cold start or divergence recovery)
		bool bReseeded= false;
		// The fit exceeded the residual guard and the PREVIOUS state was
		// streamed instead (bad-fit hold; a streak of these drops the state)
		bool bHeldBadFit= false;
	};

	void configure(const HandStateEstimatorConfig& config);

	// Drops a side's state and its per-camera consumption stamps
	void resetSide(eHandSide side);
	void resetAll();

	bool hasState(eHandSide side) const { return m_sides[(int)side].bValid; }

	// One fuse: predict (constant position), fit against every observation
	// inside the measurement window (age-weighted; see measurementWindowMs for
	// why there is no freshness dedupe), guard, output.
	//
	// seed may be null when the caller has nothing to seed from; it is only
	// consulted when there is no valid state. skeleton is the FK geometry to
	// fit with (the caller owns its stability policy).
	UpdateResult update(eHandSide side,
						const std::vector<Observation>& observations,
						double nowTimestampMs,
						const Pose* seed,
						const HandSkeleton& skeleton,
						Pose& outPose);

	// -- Exposed for the self test and the metrics tooling -----

	// Predicted 21 world landmarks for a pose: FK joints + the wrist
	// reconstructed at (-baseInPalm[Middle].x, 0, 0) in the palm frame
	static void predictWorldLandmarks(const Pose& pose, const HandSkeleton& skeleton,
									  std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints);

	// Anatomical range of one angle DoF in RAW space (flat hand = 0, positive
	// bend toward the palmar side; identical for both hands - the palm frame
	// carries the chirality). dof: 0 lateral, 1 proximal, 2 intermediate,
	// 3 distal, matching FingerAngles.
	static void angleLimits(eFinger finger, int dof, float& outLo, float& outHi);

	// Signed excursion of an angle beyond its limits (0 inside the range)
	static float angleLimitViolation(eFinger finger, int dof, float angleRad);

private:
	static constexpr int kStateDim= 26; // 3 pos + 3 rot + 20 angles

	struct SideState
	{
		bool bValid= false;
		double lastUpdateTimestampMs= 0.0;
		Pose pose;
		// Consecutive fits over the residual guard (see maxBadFitStreak)
		int badFitStreak= 0;
	};

	// A view whose rows enter the fit, with the world->camera transform and
	// per-row weight resolved once
	struct FitView
	{
		const Observation* observation= nullptr;
		glm::dmat4 cameraFromWorld{1.0};
		float presenceWeight= 1.f; // sqrt(presence): weight on the residual rows
	};

	// state (+) delta on the manifold: position adds, orientation right-
	// multiplies the small-angle error quaternion (MEKF convention, matching
	// ImuOrientationFilter), angles add
	static Pose applyDelta(const Pose& pose, const std::array<float, kStateDim>& delta);

	// All pixel residuals (predicted - observed, unweighted) for a pose, one
	// vec2 per (view, landmark). Returns false when a predicted point lands
	// behind a camera (fit rejected: geometry is broken).
	static bool evalPixelResiduals(const Pose& pose, const HandSkeleton& skeleton,
								   const std::vector<FitView>& views,
								   std::vector<glm::vec2>& outResiduals);

	// Mean |residual| px over all rows (the human-readable fit quality)
	static float meanResidualPx(const std::vector<glm::vec2>& residuals);

	// One damped Gauss-Newton solve from `pose` toward the measurements +
	// prior anchor. Returns iterations run; outPose holds the solution.
	// monoRayDir non-null makes the position prior anisotropic: tight along
	// that (world) direction, priorSigma laterally (single-camera fits).
	// anglePrior is the side's fitted prior (ignored when not present).
	int solve(const Pose& start, const Pose& anchor, const std::array<float, kStateDim>& priorSigma,
			  const glm::vec3* monoRayDir, const HandStateEstimatorConfig::AnglePrior& anglePrior,
			  const HandSkeleton& skeleton, const std::vector<FitView>& views, Pose& outPose) const;

	HandStateEstimatorConfig m_config;
	std::array<SideState, 2> m_sides;
};
