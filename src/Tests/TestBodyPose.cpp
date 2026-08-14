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

	// (b) Shoulder chained off the measured wrist: wrist -> elbow (one
	// forearm) -> shoulder (one upper arm), each step a ray against a sphere.
	// Anchoring to the fused wrist is what keeps the arm anatomically sized;
	// placing the shoulder by its landmark separation instead drifted it 0.8m
	// away on real data, because the landmark shoulders sit well inside the
	// anatomical joints.
	{
		TestRig rig;
		TrackingFrameResult fused;
		fused.timestampMs= 1000.0;
		const glm::vec3 wristWorld(0.10f, 0.05f, 0.90f);
		const glm::vec3 elbowTrue=
			wristWorld + glm::normalize(glm::vec3(0.2f, 0.1f, 0.9f)) * dims.forearmLengthMeters;
		const glm::vec3 shoulderTrue=
			elbowTrue + glm::normalize(glm::vec3(0.1f, -0.3f, 0.9f)) * dims.upperArmLengthMeters;
		setPoseWithWrist(fused.poses[0], 0, wristWorld, 0.8f);
		rig.setLandmark(ePoseLandmark::LEFT_ELBOW, elbowTrue);
		rig.setLandmark(ePoseLandmark::LEFT_SHOULDER, shoulderTrue);

		BodyPoseSolver solver;
		solver.solve(rig.candidates, dims, fused);

		check(fused.poses[0].hasShoulder, "shoulder solved");
		check(nearlyEqual(fused.poses[0].shoulderPositionWorld, shoulderTrue, 2e-3f), "shoulder exact");
		check(fabsf(glm::length(fused.poses[0].shoulderPositionWorld -
							   fused.poses[0].getElbowPositionWorld(dims.forearmLengthMeters)) -
					dims.upperArmLengthMeters) < 2e-3f,
			  "upper arm comes out the configured length");

		// The chain starts at the wrist, so no tracked hand means no shoulder
		TrackingFrameResult handless;
		handless.timestampMs= 1000.0;
		BodyPoseSolver handlessSolver;
		handlessSolver.solve(rig.candidates, dims, handless);
		check(!handless.poses[0].hasShoulder, "no tracked hand leaves the shoulder unsolved");
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
		// Nose forward of the ear axis AND skewed along it
		const glm::vec3 nose= earMid + glm::vec3(0.02f, 0.01f, -0.10f);
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
		check(glm::dot(frame[1], glm::normalize(leftEar - rightEar)) > 0.999f, "head +Y along the ear axis");
		check(glm::dot(frame[0], glm::normalize(nose - earMid)) > 0.7f, "head +X toward the nose");
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
		// The shoulder chains off the elbow, so losing the elbow loses it too
		check(!fused.poses[0].hasShoulder, "a gated elbow also withholds the shoulder it carries");
	}

	if (failures == 0)
		MIKAN_LOG_INFO("test-bodypose") << "All body-pose solver tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-bodypose", "Body-pose solver: rays + known lengths, root choice, hold, gates",
					eTestCategory::SelfTest, runBodyPoseTest);
