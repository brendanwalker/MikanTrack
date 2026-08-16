#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

#include "HandFusion.h" // CameraFrameResult
#include "OneEuroFilter.h"
#include "TrackingTypes.h"

// Body proportions the solve assumes. Only lengths are assumed; every
// direction is measured.
struct BodyDimensions
{
	float forearmLengthMeters= 0.25f;
	// Usually derived from the shoulder width rather than measured (see
	// BodyConfig): a length that needs a hard pose loses to a multiple of a
	// width that does not.
	float upperArmLengthMeters= 0.30f;
	float upperArmPerShoulderWidth= 1.05f;
	float shoulderWidthMeters= 0.40f;
	float headWidthMeters= 0.15f;
	// Ear-axis midpoint to nose tip. Fixes the nose's depth, which is what
	// carries head yaw and pitch: placing the nose at the EAR depth instead
	// collapses the forward direction (its whole signal is that it sits
	// closer to the camera than the ears do).
	float noseForwardMeters= 0.11f;
};

// Post-fusion world-space solve for the vision body pose: takes the fused
// hand poses plus the per-camera body-pose observations and fills the fused
// result's forearm (elbow), shoulder, and head outputs.
//
// EVERYTHING IS SOLVED FROM 2D RAYS PLUS KNOWN LENGTHS. The pose models'
// own metric 3D output is deliberately not consumed: measured live, its
// forearm length swung from 9.9cm to 26.3cm within a single session, and the
// top-down backend does not produce 3D at all.
//   - elbow: the camera ray through the 2D elbow landmark intersected with a
//     forearm-length sphere around the FUSED wrist, which is the best-measured
//     point in the system
//   - shoulders: both shoulder rays are known and the distance between the
//     shoulders is known, which fixes their depth without any model 3D
//   - head: the same trick on the two ear rays with a known head width
//
// CADENCE: the pose models run at a fraction of the camera rate (frame
// divider, plus detection failures), so most solve() calls see a REPEATED
// observation. Re-solving one against a moved wrist slides the estimate along
// a ray cast for an older frame, so a repeated observation instead holds:
//   - the forearm DIRECTION is held, and the elbow rides the fresh wrist
//     through it (the forearm is rigidly one bone from the wrist)
//   - the shoulder and head hold their last world position (they hang off the
//     torso, not the wrist)
//
// IMU precedence: a side whose forearm was already filled (wrist IMU) is
// left untouched - a measured direction beats a monocular inference.
//
// Confidence: landmark visibility gates, but never scores - the score is
// measured jitter stability (model confidences are not quality signals).
//
// DETERMINISM: no wall clock and no config reads; state advances on the
// observation's modelFrameIndex and the fused timestamp, both of which are
// recorded, so a replayed recording produces bit-identical output. reset() on
// recording start keeps replay-from-zero aligned with live.
class BodyPoseSolver
{
public:
	void reset();

	// cameras: the same candidate list fusion consumed (per-camera lastResult
	// mirrors); fused: the fusion output for this iteration, modified in place
	void solve(
		const std::vector<const CameraFrameResult*>& cameras,
		const BodyDimensions& dimensions,
		TrackingFrameResult& fused);

	// Depth of two symmetric landmarks whose real-world separation is known:
	// both rays leave the same optical center, so equal range t gives
	// |t*a - t*b| = separation. Returns 0 when the rays are too parallel to
	// resolve (the pair projects to nearly one point). Public for the test.
	//
	// Assumes the pair is equidistant from the camera, which is true facing a
	// camera and degrades gracefully with torso yaw: the pair's midpoint stays
	// right while its depth leans toward the nearer joint.
	static float depthFromKnownSeparation(
		const glm::vec3& rayA, const glm::vec3& rayB, float separationMeters);

	// Where a camera ray meets a sphere of known radius about a known point:
	// the two intersections, or - when the ray misses - the point on the
	// sphere nearest its closest approach, reported in both slots. Clamping
	// rather than failing keeps the result CONTINUOUS through tangency, where
	// switching estimators would jump and 2D noise around a grazing ray would
	// make it jump repeatedly. Returns false only for degenerate geometry
	// (behind the camera, or straight through the sphere's center).
	struct RaySphereHit
	{
		glm::vec3 nearPoint{0.f};
		glm::vec3 farPoint{0.f};
		bool bNearValid= false;
		bool bFarValid= false;
		bool bClamped= false;
	};
	static bool intersectRaySphere(
		const glm::vec3& rayOrigin, const glm::vec3& rayDir,
		const glm::vec3& sphereCenter, float sphereRadius, RaySphereHit& outHit);

	// Where the elbow can be given BOTH bones: the circle on which the upper
	// arm's sphere about the shoulder meets the forearm's sphere about the
	// wrist. Both lengths then hold by construction, and the 2D landmark is
	// demoted from defining the elbow to picking a point on that circle - a
	// far weaker demand on a noisy landmark, and one that cannot produce an
	// anatomically impossible arm however wrong the landmark is.
	//
	// Returns the (at most two) circle points nearest the camera ray, nearest
	// first. An arm that cannot reach - the wrist further than both bones laid
	// end to end, or nearer than their difference - has no circle; that
	// straightens to a single point rather than failing, keeping the estimate
	// CONTINUOUS through full extension the way the ray-sphere clamp does.
	struct BoneCircleHit
	{
		glm::vec3 candidates[2]{};
		int count= 0;
		bool bClamped= false;
	};
	static bool solveElbowOnBoneCircle(
		const glm::vec3& shoulder, const glm::vec3& wrist,
		float upperArmLength, float forearmLength,
		const glm::vec3& rayOrigin, const glm::vec3& rayDir, BoneCircleHit& outHit);

	// The same bone circle, picked WITHOUT a camera ray: the point nearest
	// `target` (closed form - the closest point on a circle to a point is its
	// radial projection onto the circle's plane). This is what keeps an arm
	// alive while the elbow landmark is occluded, which on this rig happens
	// whenever one hand passes in front of the other elbow.
	//
	// Both bone lengths hold BY CONSTRUCTION, which is the whole reason to
	// re-solve rather than hold the last elbow POSITION: the wrist keeps
	// moving while the elbow is unseen, and a held position would stretch the
	// forearm to whatever the gap became.
	static bool solveElbowNearestTo(
		const glm::vec3& shoulder, const glm::vec3& wrist,
		float upperArmLength, float forearmLength,
		const glm::vec3& target, glm::vec3& outElbow);

private:
	// The bone circle itself, shared by the ray-picked and target-picked
	// solves above. bDegenerate marks a straight (or folded) arm, where the
	// circle collapses to the single point in `center`.
	struct BoneCircleGeometry
	{
		glm::vec3 center{0.f};
		glm::vec3 axis{1.f, 0.f, 0.f};
		glm::vec3 planeX{0.f, 1.f, 0.f};
		glm::vec3 planeY{0.f, 0.f, 1.f};
		float radius= 0.f;
		bool bDegenerate= false;
	};
	static bool computeBoneCircle(
		const glm::vec3& shoulder, const glm::vec3& wrist,
		float upperArmLength, float forearmLength, BoneCircleGeometry& outCircle);

	// Constant-velocity residual EMA per solved joint (HandFusion's jitter
	// pattern), divided by dt^2 so it reads as an ACCELERATION. The hands'
	// plain-distance form assumes a fixed sample rate; the pose models run at
	// a fraction of the camera rate and at a variable interval (measured: 96ms
	// median, 212ms p90), where a distance residual mostly measures how long
	// the gap was. Real motion still registers - that is inherent to the
	// method - but the threshold now means the same thing at every cadence.
	struct JitterTracker
	{
		int sampleCount= 0;
		glm::vec3 p1{0.f}; // previous sample
		glm::vec3 p2{0.f}; // sample before that
		float accelEma= 0.f;

		// Call once per NEW model frame; returns stability [0,1]
		float update(const glm::vec3& position, float dtSeconds, float accelReference);
		void invalidate() { sampleCount= 0; }
	};

	// What survives between model frames, per side
	struct ArmEstimate
	{
		bool bHasForearm= false;
		glm::vec3 forearmDirWorld{1.f, 0.f, 0.f}; // unit, wrist -> elbow
		float forearmConfidence= 0.f;
		// Model-time since the elbow landmark was last actually SEEN. While
		// the elbow is occluded the arm is dead-reckoned on the bone circle,
		// and this both decays its confidence and bounds how long that may
		// continue (see kMaxDeadReckonMs).
		double deadReckonMs= 0.0;

		bool bHasShoulder= false;
		glm::vec3 shoulderPositionWorld{0.f};
		float shoulderConfidence= 0.f;
	};

	struct HeadEstimate
	{
		bool bValid= false;
		glm::vec3 positionWorld{0.f};
		glm::vec3 leftDirWorld{0.f, 1.f, 0.f};
		glm::vec3 forwardWorld{1.f, 0.f, 0.f};
		float confidence= 0.f;
	};

	void solveFromObservation(
		const CameraFrameResult& source,
		const BodyDimensions& dimensions,
		const TrackingFrameResult& fused);

	void applyEstimates(float forearmLengthMeters, TrackingFrameResult& fused) const;

	enum eJoint : int
	{
		JointElbowLeft= 0,
		JointElbowRight,
		JointShoulderLeft,
		JointShoulderRight,
		JointHead,
		JointCount,
	};

	std::array<JitterTracker, JointCount> m_jitter;
	std::array<ArmEstimate, 2> m_arms;
	HeadEstimate m_head;

	int64_t m_lastModelFrameIndex= -1;
	double m_lastSolveTimestampMs= -1.0;
	// Timestamp of the last NEW model result: the filtered series is sampled
	// at the model cadence, not at the fuse cadence
	double m_lastModelTimestampMs= -1.0;

	// Smoothing runs on the ESTIMATES, not the outputs: the forearm direction
	// and the torso-anchored positions are the noisy monocular quantities,
	// while the wrist they hang off is already fused and filtered. Filtering
	// the elbow position instead would lag it behind its own wrist.
	std::array<OneEuroFilterVec3, 2> m_forearmDirFilter;
	std::array<OneEuroFilterVec3, 2> m_shoulderFilter;
	OneEuroFilterVec3 m_headPositionFilter;
	OneEuroFilterVec3 m_headLeftDirFilter;
	OneEuroFilterVec3 m_headForwardFilter;
	bool m_bFiltersConfigured= false;
};
