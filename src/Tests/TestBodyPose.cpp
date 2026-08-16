#include "TestCommon.h"

// Synthetic tests for BodyPoseSolver. Everything it produces comes from 2D
// rays plus known body lengths - no pose model 3D - so every case here can be
// constructed exactly and checked to sub-millimetre.

namespace
{
BodyDimensions makeDimensions()
{
	BodyDimensions dimensions;
	dimensions.forearmLengthMeters= 0.25f;
	dimensions.upperArmLengthMeters= 0.30f;
	dimensions.shoulderWidthMeters= 0.40f;
	dimensions.headWidthMeters= 0.15f;
	dimensions.noseForwardMeters= 0.09f;
	return dimensions;
}

// Camera at the world origin, identity rotation, simple pinhole
constexpr float kFx= 600.f;
constexpr float kFy= 600.f;
constexpr float kCx= 320.f;
constexpr float kCy= 240.f;

glm::vec2 project(const glm::vec3& p)
{
	return glm::vec2(kFx * p.x / p.z + kCx, kFy * p.y / p.z + kCy);
}

// Two points separated by `separation`, both at `range` from the camera,
// symmetric about the +Z axis and offset vertically by `height`
void placeSymmetricPair(float separation, float range, float height,
						glm::vec3& outLeft, glm::vec3& outRight)
{
	const float half= separation * 0.5f;
	const float depth= std::sqrt(std::max(range * range - half * half - height * height, 0.01f));
	outLeft= glm::vec3(half, height, depth);
	outRight= glm::vec3(-half, height, depth);
}

struct TestRig
{
	CameraFrameResult camera;
	std::vector<const CameraFrameResult*> candidates;

	TestRig()
	{
		camera.cameraIndex= 0;
		camera.valid= true;
		camera.timestampMs= 1000.0;
		camera.hasExtrinsics= true;
		camera.markerFromCamera= glm::dmat4(1.0);
		camera.hasIntrinsics= true;
		camera.fx= kFx;
		camera.fy= kFy;
		camera.cx= kCx;
		camera.cy= kCy;
		camera.result.body.valid= true;
		camera.result.body.modelFrameIndex= 0;
		camera.result.body.confidence= 0.9f;
		camera.result.body.visibility.fill(1.f);
		candidates.push_back(&camera);
	}

	BodyPoseObservation& body() { return camera.result.body; }

	void setLandmark(ePoseLandmark landmark, const glm::vec3& worldPoint)
	{
		camera.result.body.imagePoints[(int)landmark]= glm::vec3(project(worldPoint), 0.f);
	}
	// Places both shoulders at a known separation so the solver can recover
	// their depth, which the elbow's root choice then leans on
	void setShoulders(float separation, float range)
	{
		glm::vec3 left, right;
		placeSymmetricPair(separation, range, -0.2f, left, right);
		setLandmark(ePoseLandmark::LEFT_SHOULDER, left);
		setLandmark(ePoseLandmark::RIGHT_SHOULDER, right);
	}
};

void setPoseWithWrist(HandPose& pose, int sideIndex, const glm::vec3& wristWorld, float confidence)
{
	pose= HandPose();
	pose.tracked= true;
	pose.side= (eHandSide)sideIndex;
	pose.confidence= confidence;
	pose.hasWorldPose= true;
	pose.palmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
	pose.skeleton.baseInPalm[(int)eFinger::Middle]= glm::vec3(0.04f, 0.f, 0.f);
	pose.palmPositionWorld= wristWorld + glm::vec3(0.04f, 0.f, 0.f);
}

bool nearlyEqual(const glm::vec3& a, const glm::vec3& b, float tolerance)
{
	return glm::length(a - b) <= tolerance;
}
} // namespace

static int runBodyPoseTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-bodypose") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-bodypose") << "FAIL " << name;
			failures++;
		}
	};

	const BodyDimensions dims= makeDimensions();
	const int leftElbow= (int)ePoseLandmark::LEFT_ELBOW;

	// (a) Known-separation depth: the core primitive behind the shoulders and
	// the head. Two rays plus a real-world separation fix the range.
	{
		glm::vec3 left, right;
		placeSymmetricPair(0.40f, 1.20f, 0.f, left, right);
		const glm::vec3 rayL= glm::normalize(left);
		const glm::vec3 rayR= glm::normalize(right);
		const float depth= BodyPoseSolver::depthFromKnownSeparation(rayL, rayR, 0.40f);
		check(fabsf(depth - 1.20f) < 1e-4f, "known-separation depth recovers range");

		// Parallel rays carry no separation information
		check(BodyPoseSolver::depthFromKnownSeparation(rayL, rayL, 0.40f) == 0.f,
			  "parallel rays report no depth");
	}

	// (b) Shoulders come from their own two rays plus the CALIBRATED
	// separation, independent of the arms and of any tracked hand. They have
	// to be independent: chained off the elbow they agreed with whichever
	// elbow candidate was chosen and could never contradict a wrong one.
	//
	// This construction is only sound because the separation is MEASURED. The
	// same code with an anatomical guess put shoulders 0.8 m too far away,
	// which is what the body measurement wizard exists to prevent.
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		glm::vec3 leftShoulder, rightShoulder;
		placeSymmetricPair(dims.shoulderWidthMeters, 1.10f, -0.2f, leftShoulder, rightShoulder);
		rig.setLandmark(ePoseLandmark::LEFT_SHOULDER, leftShoulder);
		rig.setLandmark(ePoseLandmark::RIGHT_SHOULDER, rightShoulder);

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		check(fused.poses[0].hasShoulder && fused.poses[1].hasShoulder,
			  "shoulders solved with no tracked hand");
		check(nearlyEqual(fused.poses[0].shoulderPositionWorld, leftShoulder, 1e-3f), "left shoulder exact");
		check(nearlyEqual(fused.poses[1].shoulderPositionWorld, rightShoulder, 1e-3f), "right shoulder exact");
	}

	// (c) Elbow: ray through the 2D landmark intersected with the
	// forearm-length sphere around the fused wrist, near root
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		const glm::vec3 wristWorld(0.10f, 0.05f, 1.00f);
		const glm::vec3 elbowTrue=
			wristWorld + glm::normalize(glm::vec3(0.3f, 0.2f, -0.8f)) * dims.forearmLengthMeters;
		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		// A shoulder one upper-arm length from the true elbow makes the cold
		// start unambiguous
		rig.body().visibility[(int)ePoseLandmark::RIGHT_SHOULDER]= 0.f;
		rig.body().visibility[(int)ePoseLandmark::LEFT_SHOULDER]= 0.f;

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		const glm::vec3 elbowSolved= fused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		check(fused.poses[0].hasForearmPose, "elbow solved");
		check(fabsf(glm::length(elbowSolved - wristWorld) - dims.forearmLengthMeters) < 1e-4f,
			  "elbow sits one forearm length from the wrist");
	}

	// (c2) With a shoulder, the elbow comes off the circle where the two bone
	// spheres meet, so BOTH lengths hold and the landmark only picks a point
	// on it. The bone lengths must survive a badly wrong landmark.
	{
		const glm::vec3 shoulder(-0.25f, 0.10f, 1.05f);
		const glm::vec3 wrist(0.10f, 0.05f, 1.00f);
		BodyPoseSolver::BoneCircleHit circle;
		const glm::vec3 cameraPos(0.f);

		// A ray through a plausible elbow first: the solve has to recover it
		const glm::vec3 elbowTrue= shoulder +
			glm::normalize(glm::vec3(0.55f, -0.5f, 0.2f)) * dims.upperArmLengthMeters;
		const bool bSolved= BodyPoseSolver::solveElbowOnBoneCircle(
			shoulder, wrist, glm::length(elbowTrue - shoulder), glm::length(elbowTrue - wrist),
			cameraPos, glm::normalize(elbowTrue - cameraPos), circle);
		check(bSolved && circle.count >= 1, "bone circle solved");
		check(bSolved && glm::length(circle.candidates[0] - elbowTrue) < 1e-3f,
			  "the circle point nearest the ray IS the true elbow");

		// Now a ray pointing somewhere else entirely. The answer will be the
		// wrong point on the circle, but it must still be a REAL arm.
		BodyPoseSolver::BoneCircleHit wrongRay;
		const bool bWrongSolved= BodyPoseSolver::solveElbowOnBoneCircle(
			shoulder, wrist, dims.upperArmLengthMeters, dims.forearmLengthMeters,
			cameraPos, glm::normalize(glm::vec3(-0.9f, 0.4f, 0.2f)), wrongRay);
		const bool bBonesHold= bWrongSolved && wrongRay.count >= 1 &&
			fabsf(glm::length(wrongRay.candidates[0] - shoulder) - dims.upperArmLengthMeters) < 1e-4f &&
			fabsf(glm::length(wrongRay.candidates[0] - wrist) - dims.forearmLengthMeters) < 1e-4f;
		check(bBonesHold, "both bone lengths hold even for a badly wrong landmark");

		// A little past reach is a straight arm plus calibration error
		const float reach= dims.upperArmLengthMeters + dims.forearmLengthMeters;
		BodyPoseSolver::BoneCircleHit stretched;
		const glm::vec3 farWrist= shoulder + glm::vec3(0.f, 0.f, 1.f) * (reach + 0.03f);
		const bool bStretched= BodyPoseSolver::solveElbowOnBoneCircle(
			shoulder, farWrist, dims.upperArmLengthMeters, dims.forearmLengthMeters,
			cameraPos, glm::vec3(0.f, 0.f, 1.f), stretched);
		check(bStretched && stretched.bClamped && stretched.count == 1 &&
				  fabsf(glm::length(stretched.candidates[0] - shoulder) - dims.upperArmLengthMeters) < 1e-4f,
			  "an arm a little past full reach straightens instead of failing");

		// Far past reach is a broken shoulder, not a straight arm: declining
		// sends the caller back to the wrist alone rather than flinging the
		// elbow toward wherever the shoulder landed
		BodyPoseSolver::BoneCircleHit impossible;
		check(!BodyPoseSolver::solveElbowOnBoneCircle(
				  shoulder, shoulder + glm::vec3(0.f, 0.f, 1.f) * (reach * 2.f),
				  dims.upperArmLengthMeters, dims.forearmLengthMeters,
				  cameraPos, glm::vec3(0.f, 0.f, 1.f), impossible),
			  "an unreachable wrist declines the shoulder rather than trusting it");
	}

	// (c4) Occluded elbow: the arm is carried on the same bone circle, picked
	// by continuity instead of by the (missing) landmark ray. One hand
	// crossing the other elbow is routine on this rig, and the alternative -
	// holding the last elbow POSITION - stretches the forearm as soon as the
	// wrist moves, which is the whole reason to re-solve.
	{
		const glm::vec3 shoulder(-0.25f, 0.10f, 1.05f);
		const glm::vec3 wrist(0.10f, 0.05f, 1.00f);
		const glm::vec3 elbowTrue= shoulder +
			glm::normalize(glm::vec3(0.55f, -0.5f, 0.2f)) * dims.upperArmLengthMeters;
		const float upperArm= glm::length(elbowTrue - shoulder);
		const float forearm= glm::length(elbowTrue - wrist);

		// Continuity target = where the elbow was: it must come back exactly
		glm::vec3 held(0.f);
		const bool bHeld= BodyPoseSolver::solveElbowNearestTo(
			shoulder, wrist, upperArm, forearm, elbowTrue, held);
		check(bHeld && glm::length(held - elbowTrue) < 1e-4f,
			  "with the elbow unseen, continuity recovers the same point on the circle");

		// Now the WRIST MOVES while the elbow is still unseen. Both bones must
		// still hold exactly - a held position could not do this - and the
		// elbow must slide, not jump.
		// Moved ALONG the forearm, the direction a held elbow cannot absorb
		const glm::vec3 movedWrist=
			wrist + glm::normalize(wrist - elbowTrue) * 0.05f + glm::vec3(0.f, -0.02f, 0.f);
		glm::vec3 slid(0.f);
		const bool bSlid= BodyPoseSolver::solveElbowNearestTo(
			shoulder, movedWrist, upperArm, forearm, held, slid);
		const bool bBonesHold= bSlid &&
			fabsf(glm::length(slid - shoulder) - upperArm) < 1e-4f &&
			fabsf(glm::length(slid - movedWrist) - forearm) < 1e-4f;
		check(bBonesHold, "both bone lengths hold exactly while the wrist moves unseen");
		check(bSlid && glm::length(slid - held) < 0.05f,
			  "the dead-reckoned elbow slides along the circle rather than jumping");

		// The held-position alternative, measured rather than asserted: it
		// leaves the forearm the wrong length, which is what breaks a rig
		const float heldPositionForearm= glm::length(held - movedWrist);
		MIKAN_LOG_INFO("test-bodypose")
			<< "(c4) holding the elbow POSITION would make the forearm "
			<< heldPositionForearm * 1000.f << " mm against a true " << forearm * 1000.f << " mm";
		check(fabsf(heldPositionForearm - forearm) > 0.03f,
			  "negative control: holding the position really does break the bone length");
	}

	// (c3) The regression this replaced: a shoulder at the WRIST'S RANGE.
	// Solving the elbow from the ray and checking the upper arm afterwards is
	// blind there - the two ray solutions straddle the wrist depth almost
	// symmetrically, so both look like plausible upper arms and the nearer,
	// wrong one can score better. Over recording 2026-08-14_20-46-03 that put
	// the left elbow 23 cm TOWARD the camera from the wrist, at shoulder
	// height, for every frame. Reaching forward, the elbow trails.
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;

		// Shoulder and wrist at the same range, wrist reaching toward the
		// camera and inboard; the true elbow hangs below and behind
		const glm::vec3 shoulderTrue(-0.28f, 0.16f, 0.95f);
		const glm::vec3 wristWorld(-0.02f, 0.26f, 0.92f);
		const glm::vec3 elbowTrue= shoulderTrue + glm::normalize(glm::vec3(0.25f, 0.30f, 0.55f)) *
			dims.upperArmLengthMeters;

		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setShoulders(dims.shoulderWidthMeters, glm::length(shoulderTrue));
		rig.setLandmark(ePoseLandmark::LEFT_SHOULDER, shoulderTrue);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		const glm::vec3 elbow= fused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		check(fused.poses[0].hasForearmPose, "reaching-forward elbow solved");
		check(glm::length(elbow) > glm::length(wristWorld) - 0.05f,
			  "a reaching arm keeps the elbow behind the wrist, not toward the camera");
		check(fabsf(glm::length(elbow - fused.poses[0].shoulderPositionWorld) -
					dims.upperArmLengthMeters) < 1e-3f,
			  "the solved elbow is exactly one upper arm from the solved shoulder");
	}

	// (d) The shoulder RAY breaks the cold-start tie: the two intersections
	// are ~40cm apart, and one of them can be too far from the shoulder ray
	// to be reachable by any upper arm - a test that needs no shoulder DEPTH,
	// which matters because the shoulder is solved by chaining off this very
	// elbow and so cannot arbitrate its own input.
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		const glm::vec3 wristWorld(0.f, 0.f, 1.00f);
		// The near candidate lies toward the camera; the far one away from it
		const glm::vec3 elbowTrue=
			wristWorld + glm::normalize(glm::vec3(0.35f, 0.15f, -0.9f)) * dims.forearmLengthMeters;
		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		// A shoulder ray passing right by the near candidate, so only that
		// candidate is reachable
		rig.setLandmark(ePoseLandmark::LEFT_SHOULDER,
						elbowTrue + glm::vec3(0.05f, -dims.upperArmLengthMeters, 0.f));

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		const glm::vec3 elbowSolved= fused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		const float trueSign= glm::dot(elbowTrue - wristWorld, glm::vec3(0.f, 0.f, 1.f));
		const float solvedSign= glm::dot(elbowSolved - wristWorld, glm::vec3(0.f, 0.f, 1.f));
		check(fused.poses[0].hasForearmPose && (trueSign * solvedSign) > 0.f,
			  "shoulder ray picks the reachable root");

		// With no shoulder in view there is nothing to break the tie, so the
		// desk prior (elbow behind the wrist) applies
		TestRig blindRig;
		TrackingFrameResult blindFused;
		blindFused.timestampMs= 1000.0;
		setPoseWithWrist(blindFused.poses[0], 0, wristWorld, 0.8f);
		blindRig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		blindRig.body().visibility[(int)ePoseLandmark::LEFT_SHOULDER]= 0.f;
		BodyPoseSolver blindSolver;
		blindSolver.solve(blindRig.candidates, dims, blindFused);
		const glm::vec3 blindElbow= blindFused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		check(glm::dot(blindElbow - wristWorld, glm::vec3(0.f, 0.f, 1.f)) > 0.f,
			  "no shoulder falls back to the behind-the-wrist prior");
	}

	// (e) Grazing ray clamps to the sphere and stays continuous through
	// tangency, rather than switching to a different estimator
	{
		auto solveWithMiss= [&](float missFactor) -> glm::vec3 {
			TestRig rig;
			TrackingFrameResult fused;
			fused.timestampMs= 1000.0;
			const glm::vec3 wristWorld(0.10f, 0.05f, 1.00f);
			setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
			const glm::vec3 closestPoint=
				wristWorld + glm::vec3(0.f, 1.f, 0.f) * (dims.forearmLengthMeters * missFactor);
			rig.setLandmark(ePoseLandmark::LEFT_ELBOW, closestPoint);
			rig.body().visibility[(int)ePoseLandmark::LEFT_SHOULDER]= 0.f;
			rig.body().visibility[(int)ePoseLandmark::RIGHT_SHOULDER]= 0.f;

			BodyPoseSolver solver;
			solver.solve(rig.candidates, dims, fused);
			check(fused.poses[0].hasForearmPose, "grazing-ray elbow solved");
			return fused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		};

		const glm::vec3 wristWorld(0.10f, 0.05f, 1.00f);
		const glm::vec3 missed= solveWithMiss(1.05f);
		check(fabsf(glm::length(missed - wristWorld) - dims.forearmLengthMeters) < 1e-4f,
			  "clamped elbow sits one forearm length from the wrist");
		check(nearlyEqual(solveWithMiss(1.0005f), solveWithMiss(0.9995f), 0.02f),
			  "elbow continuous across tangency");
	}

	// (f) A repeated observation HOLDS: the elbow rides the fresh wrist
	// through the held direction rather than re-intersecting a stale ray, and
	// the shoulder keeps its torso-fixed world position
	{
		TestRig rig;
		const glm::vec3 wristA(0.10f, 0.05f, 1.00f);
		const glm::vec3 elbowTrue=
			wristA + glm::normalize(glm::vec3(0.3f, 0.2f, -0.8f)) * dims.forearmLengthMeters;
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		rig.setShoulders(dims.shoulderWidthMeters, 1.10f);

		BodyPoseSolver solver;
		TrackingFrameResult first;
		first.timestampMs= 1000.0;
		setPoseWithWrist(first.poses[0], 0, wristA, 0.8f);
		solver.solve(rig.candidates, dims, first);
		const glm::vec3 elbowA= first.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);
		const glm::vec3 dirA= glm::normalize(elbowA - wristA);
		const glm::vec3 shoulderA= first.poses[0].shoulderPositionWorld;

		// Same observation (same modelFrameIndex), wrist moved 10 cm
		const glm::vec3 wristB= wristA + glm::vec3(0.10f, 0.f, 0.f);
		TrackingFrameResult second;
		second.timestampMs= 1033.0;
		setPoseWithWrist(second.poses[0], 0, wristB, 0.8f);
		solver.solve(rig.candidates, dims, second);

		check(second.poses[0].hasForearmPose, "held observation still publishes a forearm");
		check(nearlyEqual(second.poses[0].getElbowPositionWorld(dims.forearmLengthMeters),
						  wristB + dirA * dims.forearmLengthMeters, 1e-4f),
			  "held elbow rides the fresh wrist");
		check(nearlyEqual(second.poses[0].shoulderPositionWorld, shoulderA, 1e-6f),
			  "held shoulder stays torso-fixed");
	}

	// (g) Continuity governs the root once there is a previous estimate, so a
	// noisy shoulder cannot teleport the elbow on a single frame
	{
		TestRig rig;
		const glm::vec3 wristWorld(0.f, 0.f, 1.00f);
		const glm::vec3 elbowTrue=
			wristWorld + glm::normalize(glm::vec3(0.35f, 0.15f, -0.9f)) * dims.forearmLengthMeters;
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		const glm::vec3 shoulderTrue=
			elbowTrue + glm::normalize(glm::vec3(0.2f, -0.9f, 0.1f)) * dims.upperArmLengthMeters;
		rig.setLandmark(ePoseLandmark::LEFT_SHOULDER, shoulderTrue);
		rig.setLandmark(ePoseLandmark::RIGHT_SHOULDER,
						shoulderTrue - glm::vec3(dims.shoulderWidthMeters, 0.f, 0.f));

		BodyPoseSolver solver;
		TrackingFrameResult first;
		first.timestampMs= 1000.0;
		setPoseWithWrist(first.poses[0], 0, wristWorld, 0.8f);
		solver.solve(rig.candidates, dims, first);
		const glm::vec3 elbowA= first.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);

		// A new model frame whose shoulders jumped somewhere implausible
		rig.body().modelFrameIndex= 1;
		rig.setShoulders(dims.shoulderWidthMeters, 2.5f);
		TrackingFrameResult second;
		second.timestampMs= 1100.0;
		setPoseWithWrist(second.poses[0], 0, wristWorld, 0.8f);
		solver.solve(rig.candidates, dims, second);
		const glm::vec3 elbowB= second.poses[0].getElbowPositionWorld(dims.forearmLengthMeters);

		check(nearlyEqual(elbowB, elbowA, 0.05f), "one bad shoulder frame does not flip the elbow");
	}

	// (h) IMU precedence: a pre-filled forearm is left untouched
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		const glm::vec3 wristWorld(0.10f, 0.05f, 1.00f);
		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW,
						wristWorld + glm::vec3(0.f, -dims.forearmLengthMeters, 0.f));

		const glm::quat imuForearm= glm::normalize(glm::quat(0.8f, 0.1f, 0.5f, 0.2f));
		fused.poses[0].hasForearmPose= true;
		fused.poses[0].forearmOrientationWorld= imuForearm;
		fused.poses[0].forearmConfidence= 0.77f;

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		check(fused.poses[0].forearmOrientationWorld == imuForearm, "IMU forearm orientation untouched");
		check(fused.poses[0].forearmConfidence == 0.77f, "IMU forearm confidence untouched");
	}

	// (i) Head from the two ear rays plus the known head width, with an
	// orthonormal frame built from deliberately non-orthogonal inputs
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		glm::vec3 leftEar, rightEar;
		placeSymmetricPair(dims.headWidthMeters, 1.20f, 0.1f, leftEar, rightEar);
		const glm::vec3 earMid= 0.5f * (leftEar + rightEar);
		// Nose forward of the ear axis AND skewed along it, at the modelled
		// distance so the ray-sphere solve recovers it exactly
		const glm::vec3 nose=
			earMid + glm::normalize(glm::vec3(0.02f, 0.01f, -0.10f)) * dims.noseForwardMeters;
		rig.setLandmark(ePoseLandmark::LEFT_EAR, leftEar);
		rig.setLandmark(ePoseLandmark::RIGHT_EAR, rightEar);
		rig.setLandmark(ePoseLandmark::NOSE, nose);

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		check(fused.head.valid, "head solved");
		check(nearlyEqual(fused.head.positionWorld, earMid, 1e-3f), "head at the ear midpoint");

		const glm::mat3 frame= glm::mat3_cast(fused.head.orientationWorld);
		const bool bOrthonormal=
			fabsf(glm::dot(frame[0], frame[1])) < 1e-5f && fabsf(glm::dot(frame[0], frame[2])) < 1e-5f &&
			fabsf(glm::dot(frame[1], frame[2])) < 1e-5f && fabsf(glm::length(frame[0]) - 1.f) < 1e-5f &&
			fabsf(glm::length(frame[1]) - 1.f) < 1e-5f && fabsf(glm::length(frame[2]) - 1.f) < 1e-5f;
		check(bOrthonormal, "head frame orthonormal");
		// FORWARD is the axis that survives orthonormalization, exactly; the
		// ear axis gives up its forward component instead. The other way round
		// deletes yaw, which lives entirely in the nose's skew along the ears.
		check(glm::dot(frame[0], glm::normalize(nose - earMid)) > 0.9999f, "head +X exactly toward the nose");
		check(glm::dot(frame[1], glm::normalize(leftEar - rightEar)) > 0.9f, "head +Y near the ear axis");
	}

	// (i2) A yawed head must report its yaw. The ear axis alone cannot carry
	// it - both ears are placed at one range, so that axis is always square to
	// the view - which is why forward has to be the axis the frame is built
	// from. Deriving forward from the ears instead read 10 degrees of yaw
	// ANTI-correlated with the truth over recording 2026-08-14_19-53-18, while
	// 48 degrees were being performed.
	{
		const glm::vec3 up(0.f, -1.f, 0.f); // camera +Y is down, so up is -Y
		const glm::vec3 headCenter(0.f, 0.f, 1.10f);
		const glm::vec3 zeroYawForward(0.f, 0.f, -1.f); // facing the camera

		auto solveYaw= [&](float trueYawRadians) {
			const glm::quat rotation= glm::angleAxis(trueYawRadians, up);
			const glm::vec3 forward= rotation * zeroYawForward;
			const glm::vec3 left= glm::cross(up, forward);

			TestRig rig;
			TrackingFrameResult fused;
			fused.timestampMs= 1000.0;
			rig.setLandmark(ePoseLandmark::LEFT_EAR, headCenter + left * (dims.headWidthMeters * 0.5f));
			rig.setLandmark(ePoseLandmark::RIGHT_EAR, headCenter - left * (dims.headWidthMeters * 0.5f));
			rig.setLandmark(ePoseLandmark::NOSE, headCenter + forward * dims.noseForwardMeters);

			BodyPoseSolver solver;
			solver.solve(rig.candidates, dims, fused);
			if (!fused.head.valid)
				return 0.f;

			const glm::vec3 recovered= glm::mat3_cast(fused.head.orientationWorld)[0];
			return atan2f(glm::dot(glm::cross(zeroYawForward, recovered), up),
						  glm::dot(zeroYawForward, recovered));
		};

		const float turnedLeft= glm::degrees(solveYaw(glm::radians(25.f)));
		const float turnedRight= glm::degrees(solveYaw(glm::radians(-25.f)));
		MIKAN_LOG_INFO("test-bodypose")
			<< "  yaw recovery: +25 deg reads " << turnedLeft << ", -25 deg reads " << turnedRight;
		check(fabsf(turnedLeft - 25.f) < 6.f, "a head turned one way reports that yaw");
		check(fabsf(turnedRight + 25.f) < 6.f, "a head turned the other way reports that yaw");
	}

	// (i3) An implausible range is a collapsed detection, not a distant head
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		glm::vec3 leftEar, rightEar;
		placeSymmetricPair(dims.headWidthMeters, 3.00f, 0.f, leftEar, rightEar);
		const glm::vec3 earMid= 0.5f * (leftEar + rightEar);
		rig.setLandmark(ePoseLandmark::LEFT_EAR, leftEar);
		rig.setLandmark(ePoseLandmark::RIGHT_EAR, rightEar);
		rig.setLandmark(ePoseLandmark::NOSE, earMid + glm::vec3(0.f, 0.f, -dims.noseForwardMeters));

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);
		check(!fused.head.valid, "a head beyond the plausible range is withheld");
	}

	// (j) Gates: low visibility withholds each output
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		const glm::vec3 wristWorld(0.10f, 0.05f, 1.00f);
		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW,
						wristWorld + glm::vec3(0.f, -dims.forearmLengthMeters, 0.f));
		rig.setShoulders(dims.shoulderWidthMeters, 1.10f);
		rig.body().visibility[leftElbow]= 0.2f;
		rig.body().visibility[(int)ePoseLandmark::LEFT_EAR]= 0.2f;

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);
		check(!fused.poses[0].hasForearmPose, "low elbow visibility withholds the forearm");
		check(!fused.head.valid, "low ear visibility withholds the head");
		// The shoulders stand on their own, so an elbow gate cannot take them
		check(fused.poses[0].hasShoulder, "shoulders survive an unrelated gate");

		// One shoulder unseen removes the pair that fixes the depth
		TestRig soloRig;
		TrackingFrameResult soloFused;
		soloFused.timestampMs= 1000.0;
		soloRig.setShoulders(dims.shoulderWidthMeters, 1.10f);
		soloRig.body().visibility[(int)ePoseLandmark::RIGHT_SHOULDER]= 0.1f;
		BodyPoseSolver soloSolver;
		soloSolver.solve(soloRig.candidates, dims, soloFused);
		check(!soloFused.poses[0].hasShoulder, "a single visible shoulder fixes no depth");
	}

	// The provided mask has to survive a recording round trip, and it holds
	// 33 slots - one more than a 32-bit field can address. Untested until
	// now, which is how a shift-by-33 sat in the legacy read path.
	{
		BodyPoseObservation body;
		body.valid= true;
		body.modelFrameIndex= 7;
		body.confidence= 0.8f;
		for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
		{
			body.imagePoints[landmark]= glm::vec3((float)landmark, (float)(landmark * 2), 0.f);
			body.visibility[landmark]= 0.5f;
		}
		// Every slot, including the last one
		body.providedMask= (1ull << POSE_LANDMARK_COUNT) - 1ull;
		check(body.isProvided(POSE_LANDMARK_COUNT - 1), "the last slot is addressable");

		BodyPoseObservation restored;
		TrackingJson::bodyPoseFromJson(TrackingJson::bodyPoseToJson(body), restored);
		check(restored.providedMask == body.providedMask, "the full mask survives the round trip");

		// Recordings predating the field must read as all-provided, not as
		// the single slot a truncated shift would leave
		nlohmann::json legacy= TrackingJson::bodyPoseToJson(body);
		legacy.erase("providedMask");
		BodyPoseObservation legacyBody;
		TrackingJson::bodyPoseFromJson(legacy, legacyBody);
		int legacyProvided= 0;
		for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
			legacyProvided+= legacyBody.isProvided(landmark) ? 1 : 0;
		// The value too, so a serialization path that narrowed the field
		// would be caught even though every slot still read as provided
		check(legacyProvided == POSE_LANDMARK_COUNT && legacyBody.providedMask == body.providedMask,
			  "a mask-less recording reads as all 33 provided");
	}

	if (failures == 0)
		MIKAN_LOG_INFO("test-bodypose") << "All body-pose solver tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-bodypose", "Body-pose solver: rays + known lengths, root choice, hold, gates",
					eTestCategory::SelfTest, runBodyPoseTest);
