#include "TestCommon.h"

static int runHandPoseTest(const TestArgs& args)
{
	int result= 0;

	// Build a synthetic RIGHT-hand landmark set with known articulation
	// by forward kinematics over a hand-authored skeleton, then verify
	// angle extraction + FK round-trips.
	HandSkeleton skeleton;
	// INVARIANT: the middle finger's base lies exactly on palm +X - the
	// palm frame defines +X as wrist -> middle MCP, so computeSkeleton
	// always emits (d, 0, 0) for it. Authoring a nonzero y here would
	// make the recovered palm frame differ from the frame this skeleton
	// is expressed in, which silently rotates every extracted angle.
	const float baseY[FINGER_COUNT]= {0.045f, 0.03f, 0.f, -0.01f, -0.03f};
	const float baseX[FINGER_COUNT]= {-0.01f, 0.035f, 0.04f, 0.035f, 0.03f};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		skeleton.baseInPalm[finger]= glm::vec3(baseX[finger], baseY[finger], 0.f);
		skeleton.phalanxLengths[finger]= {0.045f, 0.027f, 0.022f};
	}
	skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
	// NOTE on chirality: computePalmFrame derives +Z from the landmark
	// layout; this skeleton (thumb/index at +Y) matches a RIGHT hand
	// viewed in its own palm frame.

	// Round-trip helper: FK with the given angles -> landmark set ->
	// re-extract angles -> max absolute error
	std::array<glm::vec3, HAND_LANDMARK_COUNT> points{};
	auto roundTrip= [&](const std::array<FingerAngles, FINGER_COUNT>& anglesIn, const char* label) {
		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
		HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesIn, joints);

		const glm::vec3 middleBase= skeleton.baseInPalm[(int)eFinger::Middle];
		points[(int)eHandLandmark::WRIST]= glm::vec3(-middleBase.x, 0.f, 0.f);
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int joint= 0; joint < 4; ++joint)
				points[FINGER_JOINTS[finger][joint]]= joints[finger][joint];

		std::array<FingerAngles, FINGER_COUNT> anglesOut{};
		HandPoseModel::computeFingerAngles(points, eHandSide::Right, skeleton.neutralDirInPalm, anglesOut);

		float maxError= 0.f;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			maxError= std::max(maxError, fabsf(anglesOut[finger].lateral - anglesIn[finger].lateral));
			maxError= std::max(maxError, fabsf(anglesOut[finger].proximal - anglesIn[finger].proximal));
			maxError=
				std::max(maxError, fabsf(anglesOut[finger].intermediate - anglesIn[finger].intermediate));
			maxError= std::max(maxError, fabsf(anglesOut[finger].distal - anglesIn[finger].distal));
		}
		MIKAN_LOG_INFO("test-handpose") << label << ": round-trip max error rad=" << maxError;
		return maxError;
	};

	// Moderate articulation
	std::array<FingerAngles, FINGER_COUNT> anglesModerate{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		anglesModerate[finger].lateral= 0.05f * (float)(finger - 2);
		anglesModerate[finger].proximal= 0.3f + 0.1f * (float)finger;
		anglesModerate[finger].intermediate= 0.4f;
		anglesModerate[finger].distal= 0.2f;
	}
	if (roundTrip(anglesModerate, "moderate curl") > 0.02f)
	{
		MIKAN_LOG_ERROR("test-handpose") << "FAILED: moderate-curl round-trip error too large";
		result= 1;
	}

	// Deep curl (fist): combined proximal+intermediate bend passes 90
	// degrees, where the old per-joint palmZ-cross sign disambiguation
	// degenerated and flipped the distal sign (Z-shaped fingers)
	std::array<FingerAngles, FINGER_COUNT> anglesFist{};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		anglesFist[finger].lateral= 0.02f * (float)(finger - 2);
		anglesFist[finger].proximal= 1.2f;
		anglesFist[finger].intermediate= 1.4f;
		anglesFist[finger].distal= 1.0f;
	}
	if (roundTrip(anglesFist, "deep curl (fist)") > 0.02f)
	{
		MIKAN_LOG_ERROR("test-handpose") << "FAILED: deep-curl round-trip error too large "
											"(distal sign degeneracy regression)";
		result= 1;
	}

	// Thumb opposition: large MCP/IP flexion must round-trip AND must
	// actually sweep the thumb tip ACROSS the palm toward the pinky
	// (the pronated-hinge behavior; on the finger-style hinge this
	// motion was nearly unobservable)
	{
		std::array<FingerAngles, FINGER_COUNT> anglesOpposition{};
		anglesOpposition[(int)eFinger::Thumb].lateral= -0.3f;
		anglesOpposition[(int)eFinger::Thumb].proximal= 0.4f;
		anglesOpposition[(int)eFinger::Thumb].intermediate= 1.1f;
		anglesOpposition[(int)eFinger::Thumb].distal= 0.5f;
		if (roundTrip(anglesOpposition, "thumb opposition") > 0.02f)
		{
			MIKAN_LOG_ERROR("test-handpose") << "FAILED: thumb-opposition round-trip error too large";
			result= 1;
		}

		// Directional check: flexing the thumb must move its tip toward
		// the pinky side (-Y in this right-hand skeleton), not stay in
		// the palmar bend plane
		std::array<FingerAngles, FINGER_COUNT> anglesStraight{};
		anglesStraight[(int)eFinger::Thumb].lateral= -0.3f;
		anglesStraight[(int)eFinger::Thumb].proximal= 0.4f;

		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> jointsFlexed, jointsStraight;
		HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesOpposition, jointsFlexed);
		HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesStraight, jointsStraight);

		const float tipShiftTowardPinky=
			jointsStraight[(int)eFinger::Thumb][3].y - jointsFlexed[(int)eFinger::Thumb][3].y;
		MIKAN_LOG_INFO("test-handpose")
			<< "thumb flexion tip shift toward pinky mm=" << tipShiftTowardPinky * 1000.f;
		if (tipShiftTowardPinky < 0.02f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: thumb flexion must sweep the tip across the palm toward the pinky";
			result= 1;
		}
	}

	// Wrong-label robustness: MediaPipe's handedness classifier flips
	// when the palm rotates away from the camera. The palm-frame
	// chirality is derived geometrically (curl + thumb evidence), so
	// extracting with the WRONG side label from a curled hand must
	// still produce the same angles (this was the palm-down mirroring
	// bug: label-based chirality inverted every lateral/proximal).
	{
		std::array<FingerAngles, FINGER_COUNT> anglesWrongLabel{};
		HandPoseModel::computeFingerAngles(points, eHandSide::Left, skeleton.neutralDirInPalm, anglesWrongLabel);
		std::array<FingerAngles, FINGER_COUNT> anglesRightLabel{};
		HandPoseModel::computeFingerAngles(points, eHandSide::Right, skeleton.neutralDirInPalm, anglesRightLabel);

		float maxLabelError= 0.f;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			maxLabelError=
				std::max(maxLabelError, fabsf(anglesWrongLabel[finger].lateral - anglesRightLabel[finger].lateral));
			maxLabelError= std::max(maxLabelError,
									fabsf(anglesWrongLabel[finger].proximal - anglesRightLabel[finger].proximal));
			maxLabelError= std::max(
				maxLabelError, fabsf(anglesWrongLabel[finger].intermediate - anglesRightLabel[finger].intermediate));
			maxLabelError=
				std::max(maxLabelError, fabsf(anglesWrongLabel[finger].distal - anglesRightLabel[finger].distal));
		}
		MIKAN_LOG_INFO("test-handpose") << "wrong-handedness-label max angle delta rad=" << maxLabelError;
		if (maxLabelError > 0.001f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: angles must be invariant to a mislabeled handedness (geometric chirality)";
			result= 1;
		}
	}

	// Skeleton round-trip: recompute from the (last) landmark set
	HandSkeleton skeletonOut;
	HandPoseModel::computeSkeleton(points, eHandSide::Right, skeletonOut);
	float maxLenError= 0.f;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			maxLenError= std::max(maxLenError, fabsf(skeletonOut.phalanxLengths[finger][phalanx] -
													 skeleton.phalanxLengths[finger][phalanx]));
	MIKAN_LOG_INFO("test-handpose") << "skeleton round-trip max length error mm=" << maxLenError * 1000.f;
	if (maxLenError > 0.001f)
	{
		MIKAN_LOG_ERROR("test-handpose") << "FAILED: phalanx length round-trip mismatch";
		result= 1;
	}

	// Palm frame sanity: +X toward fingers, origin midway wrist<->middleMCP
	const glm::mat4 palmFrame= HandPoseModel::computePalmFrame(points, eHandSide::Right);
	const glm::vec3 xAxis= glm::vec3(palmFrame[0]);
	const glm::vec3 towardFingers=
		glm::normalize(points[(int)eHandLandmark::MIDDLE_MCP] - points[(int)eHandLandmark::WRIST]);
	MIKAN_LOG_INFO("test-handpose") << "palm X . towardFingers=" << glm::dot(xAxis, towardFingers);
	if (glm::dot(xAxis, towardFingers) < 0.99f)
	{
		MIKAN_LOG_ERROR("test-handpose") << "FAILED: palm frame X axis mismatch";
		result= 1;
	}

	// Convention checks (the four the OSC schema promises). Work in the
	// palm frame directly: build FK from a single nonzero angle and
	// verify the joint moves the promised way.
	{
		auto fkWith= [&](int finger, float lat, float prox, float inter, float dist) {
			std::array<FingerAngles, FINGER_COUNT> a{};
			a[finger].lateral= lat;
			a[finger].proximal= prox;
			a[finger].intermediate= inter;
			a[finger].distal= dist;
			std::array<std::array<glm::vec3, 4>, FINGER_COUNT> j;
			HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, a, j);
			return j[finger];
		};

		const int indexFinger= (int)eFinger::Index;
		const std::array<glm::vec3, 4> neutral= fkWith(indexFinger, 0.f, 0.f, 0.f, 0.f);

		// (i) zero angles reproduce the neutral direction
		const glm::vec3 neutralBone= glm::normalize(neutral[1] - neutral[0]);
		const glm::vec3 expectedNeutral= glm::normalize(skeleton.neutralDirInPalm[indexFinger]);
		MIKAN_LOG_INFO("test-handpose")
			<< "convention: zero-angle bone . neutralDir=" << glm::dot(neutralBone, expectedNeutral);
		if (glm::dot(neutralBone, expectedNeutral) < 0.999f)
		{
			MIKAN_LOG_ERROR("test-handpose") << "FAILED: zero angles must reproduce the neutral direction";
			result= 1;
		}

		// (ii) positive proximal bends TOWARD the palm (+Z)
		const std::array<glm::vec3, 4> bent= fkWith(indexFinger, 0.f, 0.6f, 0.f, 0.f);
		const float bendTowardPalm= glm::normalize(bent[1] - bent[0]).z;
		MIKAN_LOG_INFO("test-handpose") << "convention: +proximal bone z=" << bendTowardPalm;
		if (bendTowardPalm < 0.1f)
		{
			MIKAN_LOG_ERROR("test-handpose") << "FAILED: +proximal must curl toward the palm (+Z)";
			result= 1;
		}

		// (iii) positive lateral is counter-clockwise about palm +Z,
		// i.e. toward palm +Y = cross(palmZ, palmX)
		const std::array<glm::vec3, 4> splayed= fkWith(indexFinger, 0.4f, 0.f, 0.f, 0.f);
		const glm::vec3 splayDelta= glm::normalize(splayed[1] - splayed[0]) - neutralBone;
		MIKAN_LOG_INFO("test-handpose") << "convention: +lateral delta y=" << splayDelta.y;
		if (splayDelta.y < 0.05f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: +lateral must rotate CCW about palm +Z (toward palm +Y)";
			result= 1;
		}

		// (iv) intermediate/distal are relative to their PARENT bone:
		// with a nonzero proximal, an intermediate of 0 must leave the
		// middle bone collinear with the proximal bone
		const glm::vec3 proximalDir= glm::normalize(bent[1] - bent[0]);
		const glm::vec3 intermediateDir= glm::normalize(bent[2] - bent[1]);
		MIKAN_LOG_INFO("test-handpose")
			<< "convention: zero-inter collinearity=" << glm::dot(proximalDir, intermediateDir);
		if (glm::dot(proximalDir, intermediateDir) < 0.999f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: intermediate/distal must be relative to the parent bone, not the palm";
			result= 1;
		}
	}

	// Rest-pose capture: feeding a pose back as its own neutral must
	// make that exact pose read all-zero angles
	{
		std::array<FingerAngles, FINGER_COUNT> restAngles{};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			restAngles[finger].lateral= 0.15f - 0.05f * finger;
			restAngles[finger].proximal= 0.5f - 0.08f * finger;
		}
		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> restJoints;
		HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, restAngles, restJoints);

		std::array<glm::vec3, HAND_LANDMARK_COUNT> restPoints{};
		const glm::vec3 middleBaseRest= skeleton.baseInPalm[(int)eFinger::Middle];
		restPoints[(int)eHandLandmark::WRIST]= glm::vec3(-middleBaseRest.x, 0.f, 0.f);
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int joint= 0; joint < 4; ++joint)
				restPoints[FINGER_JOINTS[finger][joint]]= restJoints[finger][joint];

		// The rest OFFSET convention (used by fusedRestAngles): the angles a
		// pose reads against the flat-hand neutral ARE its offset, so
		// subtracting them makes that exact pose read all-zero on ALL FOUR
		// degrees of freedom, intermediate and distal included
		HandSkeleton restSkeleton;
		HandPoseModel::computeSkeleton(restPoints, eHandSide::Right, restSkeleton);
		std::array<FingerAngles, FINGER_COUNT> capturedRest{};
		HandPoseModel::computeFingerAngles(restPoints, eHandSide::Right,
										   restSkeleton.neutralDirInPalm, capturedRest);

		// ...and a DIFFERENT pose must still read its true deviation,
		// i.e. the offset shifts zero without distorting the scale
		std::array<FingerAngles, FINGER_COUNT> movedAngles= restAngles;
		movedAngles[(int)eFinger::Index].proximal+= 0.3f;
		movedAngles[(int)eFinger::Index].intermediate+= 0.2f;
		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> movedJoints;
		HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, movedAngles, movedJoints);

		std::array<glm::vec3, HAND_LANDMARK_COUNT> movedPoints= restPoints;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int joint= 0; joint < 4; ++joint)
				movedPoints[FINGER_JOINTS[finger][joint]]= movedJoints[finger][joint];

		std::array<FingerAngles, FINGER_COUNT> movedMeasured{};
		HandPoseModel::computeFingerAngles(movedPoints, eHandSide::Right,
										   restSkeleton.neutralDirInPalm, movedMeasured);
		const float proximalDeviation= movedMeasured[(int)eFinger::Index].proximal -
			capturedRest[(int)eFinger::Index].proximal;
		const float intermediateDeviation= movedMeasured[(int)eFinger::Index].intermediate -
			capturedRest[(int)eFinger::Index].intermediate;
		MIKAN_LOG_INFO("test-handpose") << "rest capture: deviation prox=" << proximalDeviation
			<< " inter=" << intermediateDeviation << " (expected 0.3 / 0.2)";
		if (fabsf(proximalDeviation - 0.3f) > 1e-3f || fabsf(intermediateDeviation - 0.2f) > 1e-3f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: rest-relative angles must still report true deviation";
			result= 1;
		}
	}

	// FK reprojection: rebuild the hand from ONLY palm pose + skeleton +
	// angles, project into a synthetic camera, and compare against the
	// projection of the original 3D landmarks. End-to-end check that the
	// parameterization loses nothing a client needs to redraw the hand.
	{
		std::array<FingerAngles, FINGER_COUNT> poseAngles{};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			poseAngles[finger].lateral= 0.1f - 0.04f * finger;
			poseAngles[finger].proximal= 0.35f + 0.05f * finger;
			poseAngles[finger].intermediate= 0.4f;
			poseAngles[finger].distal= 0.25f;
		}

		// Place the hand in front of a camera (OpenCV convention)
		const glm::quat handRotation= glm::angleAxis(0.6f, glm::normalize(glm::vec3(0.3f, 1.f, 0.2f)));
		glm::mat4 handTransform= glm::mat4_cast(handRotation);
		handTransform[3]= glm::vec4(0.02f, -0.01f, 0.45f, 1.f);

		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> truthJoints;
		HandPoseModel::buildFingerJoints(handTransform, skeleton, poseAngles, truthJoints);

		std::array<glm::vec3, HAND_LANDMARK_COUNT> truthPoints{};
		const glm::vec3 middleBaseFk= skeleton.baseInPalm[(int)eFinger::Middle];
		truthPoints[(int)eHandLandmark::WRIST]=
			glm::vec3(handTransform * glm::vec4(-middleBaseFk.x, 0.f, 0.f, 1.f));
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int joint= 0; joint < 4; ++joint)
				truthPoints[FINGER_JOINTS[finger][joint]]= truthJoints[finger][joint];

		// Extract the parametric representation, exactly as the app does
		const glm::mat4 extractedPalm= HandPoseModel::computePalmFrame(truthPoints, eHandSide::Right);
		HandSkeleton extractedSkeleton;
		HandPoseModel::computeSkeleton(truthPoints, eHandSide::Right, extractedSkeleton);
		extractedSkeleton.neutralDirInPalm= skeleton.neutralDirInPalm;
		std::array<FingerAngles, FINGER_COUNT> extractedAngles{};
		HandPoseModel::computeFingerAngles(truthPoints, eHandSide::Right,
										   extractedSkeleton.neutralDirInPalm, extractedAngles);

		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> rebuiltJoints;
		HandPoseModel::buildFingerJoints(extractedPalm, extractedSkeleton, extractedAngles, rebuiltJoints);

		// Project both into a 1280x720 pinhole camera
		const float fxTest= 900.f, fyTest= 900.f, cxTest= 640.f, cyTest= 360.f;
		auto project= [&](const glm::vec3& p) {
			return glm::vec2(fxTest * p.x / p.z + cxTest, fyTest * p.y / p.z + cyTest);
		};

		float maxReprojection= 0.f;
		float sumReprojection= 0.f;
		int reprojectionCount= 0;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			for (int joint= 0; joint < 4; ++joint)
			{
				const float error= glm::length(project(rebuiltJoints[finger][joint]) -
											   project(truthJoints[finger][joint]));
				maxReprojection= std::max(maxReprojection, error);
				sumReprojection+= error;
				reprojectionCount++;
			}
		}
		const float meanReprojection= sumReprojection / (float)reprojectionCount;
		MIKAN_LOG_INFO("test-handpose")
			<< "FK reprojection: mean px=" << meanReprojection << " max px=" << maxReprojection;
		if (maxReprojection > 1.f)
		{
			MIKAN_LOG_ERROR("test-handpose")
				<< "FAILED: the FK hand must reproject onto the source landmarks";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-handpose") << "All hand-pose checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-handpose", "Palm frame, finger angles, FK round-trip, conventions", eTestCategory::SelfTest, runHandPoseTest);
