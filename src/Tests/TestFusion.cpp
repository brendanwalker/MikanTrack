#include "TestCommon.h"

static int runFusionTest(const TestArgs& args)
{
	int result= 0;

	// Synthetic parametric hand observation for one camera
	auto makeObservation= [](const glm::vec3& palmPos, const glm::quat& palmOrient, float presence,
							 eHandSide labeledSide, float handednessScore, float bendAngle) {
		TrackingFrameResult frame;
		HandPose& pose= frame.poses[(int)labeledSide];
		pose.tracked= true;
		pose.side= labeledSide;
		pose.presence= presence;
		pose.hasWorldPose= true;
		pose.palmPositionWorld= palmPos;
		pose.palmOrientationWorld= palmOrient;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			pose.fingers[finger].proximal= bendAngle;
			pose.skeleton.phalanxLengths[finger]= {0.04f, 0.025f, 0.02f};
		}
		TrackedHand& hand= frame.hands[(int)labeledSide];
		hand.tracked= true;
		hand.side= labeledSide;
		hand.presence= presence;
		hand.handednessScore= handednessScore;
		// tests treat the score as already flip-adjusted
		hand.rightProb= handednessScore;
		return frame;
	};

	auto makeCameraResult= [](int cameraIndex, const glm::vec3& cameraPosWorld, double timestampMs,
							  const TrackingFrameResult& frame) {
		CameraFrameResult camera;
		camera.cameraIndex= cameraIndex;
		camera.valid= true;
		camera.timestampMs= timestampMs;
		camera.hasExtrinsics= true;
		camera.markerFromCamera= glm::dmat4(1.0);
		camera.markerFromCamera[3]= glm::dvec4(cameraPosWorld, 1.0);
		camera.result= frame;
		return camera;
	};

	HandFusionConfig fusionConfig;
	fusionConfig.smoothingEnabled= false; // exactness for pass-through checks
	HandFusion fusion;
	fusion.configure(fusionConfig);

	const glm::vec3 palmTruth(0.10f, 0.05f, 0.10f);
	const glm::vec3 cam1Pos(0.f, 0.f, 0.8f);   // overhead
	const glm::vec3 cam2Pos(0.f, -0.6f, 0.6f); // 45 degrees
	const double now= 10'000.0;
	// Palm normal (+Z of palm frame) pointing up at the overhead camera
	const glm::quat faceUpToCam1= glm::quat(1.f, 0.f, 0.f, 0.f);
	// Palm rotated 90 deg about X: normal points along -Y toward cam2,
	// edge-on to the overhead camera
	const glm::quat faceCam2= glm::angleAxis(glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f));

	// (a) Both cameras face-on-ish: fused position beats both noisy
	// inputs; the bend angle comes from the SELECTED source camera (the
	// face-on overhead one), not a blend of the two
	{
		const glm::vec3 noiseA(0.004f, -0.002f, 0.005f);
		const glm::vec3 noiseB(-0.003f, 0.004f, -0.004f);
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(palmTruth + noiseA, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.5f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now - 5.0, makeObservation(palmTruth + noiseB, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.7f));

		TrackingFrameResult fused;
		fusion.fuse({&camA, &camB}, now, fused);

		const HandPose& pose= fused.poses[(int)eHandSide::Left];
		const float errA= glm::length(noiseA);
		const float errB= glm::length(noiseB);
		const float errFused= glm::length(pose.palmPositionWorld - palmTruth);
		const float bend= pose.fingers[0].proximal;
		MIKAN_LOG_INFO("test-fusion") << "(a) palm err mm: A=" << errA * 1000.f << " B=" << errB * 1000.f
			<< " fused=" << errFused * 1000.f << " bend=" << bend;
		if (!pose.tracked || errFused > std::min(errA, errB) * 1.05f || fabsf(bend - 0.5f) > 1e-6f)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(a) FAILED";
			result= 1;
		}
	}

	// (b) Palm edge-on to the overhead camera, facing camera 2:
	// camera 2 must dominate (fresh fusion state - this tests the
	// weighting, not the articulation-source hysteresis)
	{
		HandFusion freshFusion;
		freshFusion.configure(fusionConfig);
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(palmTruth + glm::vec3(0.01f, 0.f, 0.f), faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now, makeObservation(palmTruth, faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));

		TrackingFrameResult fused;
		freshFusion.fuse({&camA, &camB}, now, fused);

		MIKAN_LOG_INFO("test-fusion") << "(b) dominant camera=" << freshFusion.getDominantCamera(eHandSide::Left);
		if (freshFusion.getDominantCamera(eHandSide::Left) != 1)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(b) FAILED: edge-on view should lose to the face-on camera";
			result= 1;
		}
	}

	// (c) Staleness: camera 2's result is 200ms old -> exact passthrough of camera 1
	{
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.3f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now - 200.0,
			makeObservation(palmTruth + glm::vec3(0.3f, 0.f, 0.f), faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.9f));

		TrackingFrameResult fused;
		fusion.fuse({&camA, &camB}, now, fused);

		const HandPose& pose= fused.poses[(int)eHandSide::Left];
		const float dist= glm::length(pose.palmPositionWorld - palmTruth);
		MIKAN_LOG_INFO("test-fusion") << "(c) stale exclusion: distToFreshCam mm=" << dist * 1000.f
			<< " bend=" << pose.fingers[0].proximal;
		if (dist > 1e-6f || fusion.getDominantCamera(eHandSide::Left) != 0 ||
			fabsf(pose.fingers[0].proximal - 0.3f) > 1e-6f)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(c) FAILED: stale camera should be excluded";
			result= 1;
		}
	}

	// (d) Handedness-mislabel recovery: camera 1 sees only the LEFT hand
	// (decisively labeled), camera 2 sees only the RIGHT hand but its
	// classifier MISLABELS it Left (weakly). Fusion must output two
	// separate hands at their own positions, not collapse them.
	{
		const glm::vec3 rightPalmTruth= palmTruth + glm::vec3(0.3f, 0.f, 0.f);
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.05f, 0.2f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now, makeObservation(rightPalmTruth, faceUpToCam1, 0.7f, eHandSide::Left, 0.52f, 0.6f));

		TrackingFrameResult fused;
		fusion.fuse({&camA, &camB}, now, fused);

		const HandPose& left= fused.poses[(int)eHandSide::Left];
		const HandPose& right= fused.poses[(int)eHandSide::Right];
		const float leftDist= left.tracked ? glm::length(left.palmPositionWorld - palmTruth) : 1e9f;
		const float rightDist= right.tracked ? glm::length(right.palmPositionWorld - rightPalmTruth) : 1e9f;
		MIKAN_LOG_INFO("test-fusion") << "(d) mislabel recovery: L tracked=" << left.tracked
			<< " R tracked=" << right.tracked << " Lerr mm=" << leftDist * 1000.f
			<< " Rerr mm=" << rightDist * 1000.f;
		if (!left.tracked || !right.tracked || leftDist > 0.001f || rightDist > 0.001f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(d) FAILED: two physical hands must fuse to two sides despite a mislabel";
			result= 1;
		}
	}

	// (e) Stereo hand-scale: both cameras observe the same palm with a
	// 20% depth overestimate; triangulation must recover ~1/1.2
	{
		const float depthError= 1.2f;
		const glm::vec3 palmA= cam1Pos + depthError * (palmTruth - cam1Pos);
		const glm::vec3 palmB= cam2Pos + depthError * (palmTruth - cam2Pos);

		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(palmA, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now, makeObservation(palmB, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.f));

		TrackingFrameResult fused;
		fusion.fuse({&camA, &camB}, now, fused);

		float correction= 0.f;
		const bool bHasSample= fusion.getStereoScaleSample(correction);
		const float expected= 1.f / depthError;
		MIKAN_LOG_INFO("test-fusion") << "(e) stereo scale: correction=" << correction
			<< " (expected " << expected << ")";
		if (!bHasSample || fabsf(correction - expected) > 0.01f)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(e) FAILED: triangulated scale correction mismatch";
			result= 1;
		}
	}

	// (f) Ray-aware clustering: camera 1 sees ONLY the left hand, but
	// with a 45% depth overestimate along its view ray (bad hand scale)
	// AND mislabels it Right (weakly - a DECISIVE opposite score would
	// trigger the vote-coherence veto, which is test (i)'s subject).
	// Euclidean clustering would split it into a phantom "right hand"
	// fighting camera 2's real right hand; ray-aware clustering must
	// merge it into the left cluster.
	{
		HandFusion freshFusion; // no temporal prior from earlier tests
		freshFusion.configure(fusionConfig);

		const glm::vec3 rightPalmTruth= palmTruth + glm::vec3(0.3f, 0.f, 0.f);
		// displaced ALONG cam1's view ray: euclidean offset ~0.32m (> the
		// 0.25m wrist gate), perpendicular offset 0
		const glm::vec3 leftPalmDisplaced= cam1Pos + 1.45f * (palmTruth - cam1Pos);

		const auto camA= makeCameraResult(
			0, cam1Pos, now,
			makeObservation(leftPalmDisplaced, faceUpToCam1, 0.7f, eHandSide::Right, 0.7f, 0.f));
		TrackingFrameResult camBFrame=
			makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.05f, 0.f);
		{
			// add camera 2's decisively-labeled right hand to the same frame
			const TrackingFrameResult rightFrame=
				makeObservation(rightPalmTruth, faceUpToCam1, 0.9f, eHandSide::Right, 0.95f, 0.f);
			camBFrame.poses[(int)eHandSide::Right]= rightFrame.poses[(int)eHandSide::Right];
			camBFrame.hands[(int)eHandSide::Right]= rightFrame.hands[(int)eHandSide::Right];
		}
		const auto camB= makeCameraResult(1, cam2Pos, now, camBFrame);

		TrackingFrameResult fused;
		freshFusion.fuse({&camA, &camB}, now, fused);

		const HandPose& left= fused.poses[(int)eHandSide::Left];
		const HandPose& right= fused.poses[(int)eHandSide::Right];
		const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalmTruth) : 1e9f;
		const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - palmTruth) : 1e9f;
		MIKAN_LOG_INFO("test-fusion") << "(f) ray clustering: L tracked=" << left.tracked
			<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
			<< " Rerr mm=" << rightErr * 1000.f;
		// Right must be the exact passthrough of camera 2's right hand
		// (no phantom competing for it); left may be pulled along cam1's
		// ray by the blend but must stay near the truth
		if (!left.tracked || !right.tracked || rightErr > 0.001f || leftErr > 0.25f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(f) FAILED: depth-displaced observation must merge into the left cluster";
			result= 1;
		}
	}

	// (g) Spatial side prior: two hands, both (mis)labeled Left - the one on
	// the right side of the desk even decisively so. The prior is a FIXED
	// convention (right hand toward world -Y, set by the printed board), so
	// assignment must follow geometry rather than the votes.
	{
		HandFusion freshFusion;
		freshFusion.configure(fusionConfig);

		const glm::vec3 leftPalm(0.05f, 0.15f, 0.1f);
		const glm::vec3 rightPalm(0.05f, -0.15f, 0.1f);
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(leftPalm, faceUpToCam1, 0.9f, eHandSide::Left, 0.45f, 0.f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now, makeObservation(rightPalm, faceUpToCam1, 0.8f, eHandSide::Left, 0.05f, 0.f));

		TrackingFrameResult fused;
		freshFusion.fuse({&camA, &camB}, now, fused);

		const HandPose& left= fused.poses[(int)eHandSide::Left];
		const HandPose& right= fused.poses[(int)eHandSide::Right];
		const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - leftPalm) : 1e9f;
		const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalm) : 1e9f;
		MIKAN_LOG_INFO("test-fusion") << "(g) spatial prior: L tracked=" << left.tracked
			<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
			<< " Rerr mm=" << rightErr * 1000.f;
		if (!left.tracked || !right.tracked || leftErr > 0.001f || rightErr > 0.001f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(g) FAILED: spatial prior must overrule a decisive mislabel";
			result= 1;
		}
	}

	// (h) Joint cluster pairing - regression from the 2026-08-01 clap
	// dump (real numbers, reacquisition frame i116). Both cameras'
	// slot LABELS are physically reversed after the clap, and greedy
	// nearest-position clustering paired the wrong hands cross-camera
	// (each camera's depth error put the wrong hand nearest). The
	// joint assignment + score-based votes must pair the physical
	// hands correctly and assign the true sides.
	{
		HandFusion freshFusion;
		freshFusion.configure(fusionConfig);

		const glm::vec3 cam0Pos(0.038f, 0.355f, 0.646f);  // real extrinsics
		const glm::vec3 cam1PosReal(0.073f, -0.284f, 0.650f);

		// cam0: physical RIGHT hand sits in its "Left" slot (hijack),
		// physical LEFT in its "Right" slot - scores tell the truth
		TrackingFrameResult cam0Frame=
			makeObservation(glm::vec3(0.018f, -0.023f, 0.164f), faceUpToCam1, 0.99f, eHandSide::Left, 0.34f, 0.f);
		{
			const TrackingFrameResult other= makeObservation(
				glm::vec3(0.018f, 0.139f, 0.164f), faceUpToCam1, 0.98f, eHandSide::Right, 0.04f, 0.f);
			cam0Frame.poses[(int)eHandSide::Right]= other.poses[(int)eHandSide::Right];
			cam0Frame.hands[(int)eHandSide::Right]= other.hands[(int)eHandSide::Right];
		}
		TrackingFrameResult cam1Frame=
			makeObservation(glm::vec3(0.011f, -0.039f, 0.110f), faceUpToCam1, 0.98f, eHandSide::Left, 0.97f, 0.f);
		{
			const TrackingFrameResult other= makeObservation(
				glm::vec3(0.023f, 0.066f, 0.201f), faceUpToCam1, 0.99f, eHandSide::Right, 0.78f, 0.f);
			cam1Frame.poses[(int)eHandSide::Right]= other.poses[(int)eHandSide::Right];
			cam1Frame.hands[(int)eHandSide::Right]= other.hands[(int)eHandSide::Right];
		}

		const auto camA= makeCameraResult(0, cam0Pos, now, cam0Frame);
		const auto camB= makeCameraResult(1, cam1PosReal, now, cam1Frame);

		TrackingFrameResult fused;
		freshFusion.fuse({&camA, &camB}, now, fused);

		// Physical right hand lives at y ~ -0.03, physical left at y ~ +0.10
		const HandPose& left= fused.poses[(int)eHandSide::Left];
		const HandPose& right= fused.poses[(int)eHandSide::Right];
		MIKAN_LOG_INFO("test-fusion") << "(h) clap-dump regression: L tracked=" << left.tracked
			<< " y=" << (left.tracked ? left.palmPositionWorld.y : 0.f)
			<< " R tracked=" << right.tracked
			<< " y=" << (right.tracked ? right.palmPositionWorld.y : 0.f);
		if (!left.tracked || !right.tracked ||
			left.palmPositionWorld.y < 0.05f || right.palmPositionWorld.y > 0.f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(h) FAILED: joint pairing must untangle the post-clap label reversal";
			result= 1;
		}
	}

	// (i) Decisive-disagreement veto: two cameras each track a
	// DIFFERENT physical hand only 8cm apart (post-clap separation),
	// with decisively opposite classifier scores. Position-only
	// clustering would merge them into one mixed cluster (cancelling
	// both votes); the veto must keep them separate so each side is
	// assigned correctly.
	{
		HandFusion freshFusion;
		freshFusion.configure(fusionConfig);

		const glm::vec3 leftPalm(0.10f, 0.05f, 0.10f);
		const glm::vec3 rightPalm(0.10f, 0.13f, 0.10f);
		const auto camA= makeCameraResult(
			0, cam1Pos, now, makeObservation(leftPalm, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
		const auto camB= makeCameraResult(
			1, cam2Pos, now, makeObservation(rightPalm, faceUpToCam1, 0.95f, eHandSide::Right, 0.95f, 0.f));

		TrackingFrameResult fused;
		freshFusion.fuse({&camA, &camB}, now, fused);

		const HandPose& left= fused.poses[(int)eHandSide::Left];
		const HandPose& right= fused.poses[(int)eHandSide::Right];
		const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - leftPalm) : 1e9f;
		const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalm) : 1e9f;
		MIKAN_LOG_INFO("test-fusion") << "(i) vote veto: L tracked=" << left.tracked
			<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
			<< " Rerr mm=" << rightErr * 1000.f;
		if (!left.tracked || !right.tracked || leftErr > 0.001f || rightErr > 0.001f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(i) FAILED: decisively opposed observations must not merge";
			result= 1;
		}
	}

	// (j) Pipeline slot side-collision ordering - regression from the
	// 2026-08-01_13-51-16 dump. Both slots' classifiers scored
	// "right"; the decisive one (0.98) must claim Right and displace
	// the weak one (0.64) to Left, regardless of the noise-level
	// presence difference that used to decide it.
	{
		// slot A = physical LEFT hand, weakly (wrongly) scored right,
		// marginally HIGHER presence; slot B = physical RIGHT hand,
		// decisively scored right
		const int order= HandTrackingPipeline::preferredSlotOrder(0.644f, 0.991f, 0.981f, 0.978f);
		MIKAN_LOG_INFO("test-fusion") << "(j) slot collision: first claim = slot " << order
			<< " (expected 1, the decisive one)";
		if (order != 1)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(j) FAILED: the decisive classifier must win the contested side";
			result= 1;
		}

		// Equal decisiveness falls back to presence
		if (HandTrackingPipeline::preferredSlotOrder(0.8f, 0.9f, 0.2f, 0.95f) != 1)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(j) FAILED: presence must break decisiveness ties";
			result= 1;
		}
	}

	// (k) Stability weighting - regression from the 2026-08-01_14-33-43
	// dump. Camera 0 sees the right hand edge-on: presence stays high
	// (0.87) but its palm estimate jitters by ~41mm frame to frame,
	// while camera 1 is rock steady at ~7mm. Presence cannot separate
	// these; measured jitter must, so the fused pose has to converge
	// on the steady camera.
	{
		HandFusion freshFusion;
		HandFusionConfig jitterConfig= fusionConfig;
		jitterConfig.jitterReferenceM= 0.015f;
		freshFusion.configure(jitterConfig);

		// Deterministic pseudo-noise (no Math.random in tests)
		auto noiseAt= [](int step) {
			const float phase= (float)step;
			return glm::vec3(0.041f * sinf(phase * 2.3f), 0.041f * sinf(phase * 3.7f),
							 0.041f * sinf(phase * 5.1f));
		};

		TrackingFrameResult fused;
		float lastConfidence= 0.f;
		for (int step= 0; step < 40; ++step)
		{
			const double stepTime= now + step * 33.0;
			// jittery camera: high presence, noisy position
			const auto camA= makeCameraResult(
				0, cam1Pos, stepTime,
				makeObservation(palmTruth + noiseAt(step), faceUpToCam1, 0.87f, eHandSide::Left, 0.05f, 0.f));
			// steady camera: exactly on truth
			const auto camB= makeCameraResult(
				1, cam2Pos, stepTime,
				makeObservation(palmTruth, faceUpToCam1, 0.97f, eHandSide::Left, 0.05f, 0.f));

			freshFusion.fuse({&camA, &camB}, stepTime, fused);
			lastConfidence= fused.poses[(int)eHandSide::Left].confidence;
		}

		const float err= glm::length(fused.poses[(int)eHandSide::Left].palmPositionWorld - palmTruth);
		MIKAN_LOG_INFO("test-fusion") << "(k) stability weighting: fused err mm=" << err * 1000.f
			<< " confidence=" << lastConfidence;
		// Without stability weighting the jittery camera would drag the
		// blend tens of mm off truth
		if (err > 0.008f || lastConfidence < 0.5f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(k) FAILED: the steady camera must dominate and confidence stay high";
			result= 1;
		}
	}

	// (l) Hard confidence gate: with minCameraConfidence above what a
	// jittery camera can reach, its observation is dropped entirely
	{
		HandFusion freshFusion;
		HandFusionConfig gateConfig= fusionConfig;
		gateConfig.jitterReferenceM= 0.015f;
		gateConfig.minCameraConfidence= 0.5f;
		freshFusion.configure(gateConfig);

		TrackingFrameResult fused;
		for (int step= 0; step < 40; ++step)
		{
			const double stepTime= now + step * 33.0;
			// Alternating 8cm displacement = sustained large jitter
			const glm::vec3 noise= (step % 2 == 0) ? glm::vec3(0.08f, 0.f, 0.f) : glm::vec3(0.f);
			const auto camA= makeCameraResult(
				0, cam1Pos, stepTime,
				makeObservation(palmTruth + noise, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
			freshFusion.fuse({&camA}, stepTime, fused);
		}

		const HandPose& pose= fused.poses[(int)eHandSide::Left];
		MIKAN_LOG_INFO("test-fusion") << "(l) confidence gate: tracked=" << pose.tracked;
		if (pose.tracked)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(l) FAILED: an observation below minCameraConfidence must be dropped";
			result= 1;
		}

		// ...and a steady observation at the same presence survives
		HandFusion steadyFusion;
		steadyFusion.configure(gateConfig);
		TrackingFrameResult steadyFused;
		for (int step= 0; step < 40; ++step)
		{
			const double stepTime= now + step * 33.0;
			const auto camA= makeCameraResult(
				0, cam1Pos, stepTime,
				makeObservation(palmTruth, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
			steadyFusion.fuse({&camA}, stepTime, steadyFused);
		}
		if (!steadyFused.poses[(int)eHandSide::Left].tracked)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(l) FAILED: a steady observation must pass the gate";
			result= 1;
		}
	}

	// (m) Stereo landmark triangulation: two cameras with full projective
	// geometry observe one synthetic hand. The fused pose must come from
	// the triangulated landmarks (exact recovery), NOT from the
	// (deliberately corrupted) per-camera monocular poses. A second run
	// feeds one camera a DIFFERENT physical hand's pixels - the
	// reprojection residual must veto the pairing.
	{
		// Authored RIGHT-hand skeleton (same conventions as --test-handpose;
		// middle finger base exactly on palm +X)
		HandSkeleton skeleton;
		const float baseY[FINGER_COUNT]= {0.045f, 0.03f, 0.f, -0.01f, -0.03f};
		const float baseX[FINGER_COUNT]= {-0.01f, 0.035f, 0.04f, 0.035f, 0.03f};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			skeleton.baseInPalm[finger]= glm::vec3(baseX[finger], baseY[finger], 0.f);
			skeleton.phalanxLengths[finger]= {0.045f, 0.027f, 0.022f};
		}
		skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);

		std::array<FingerAngles, FINGER_COUNT> anglesTruth{};
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			anglesTruth[finger].lateral= 0.04f * (float)(finger - 2);
			anglesTruth[finger].proximal= 0.25f + 0.1f * (float)finger;
			anglesTruth[finger].intermediate= 0.35f;
			anglesTruth[finger].distal= 0.15f;
		}

		// World-space hand at palmTruth (identity orientation: palm +Z up,
		// facing the overhead camera)
		auto buildWorldHand= [&](const glm::vec3& palmCenter,
								 std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints) {
			glm::mat4 palmTransform(1.f);
			palmTransform[3]= glm::vec4(palmCenter, 1.f);
			std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
			HandPoseModel::buildFingerJoints(palmTransform, skeleton, anglesTruth, joints);
			const glm::vec3 middleBase= skeleton.baseInPalm[(int)eFinger::Middle];
			outPoints[(int)eHandLandmark::WRIST]= palmCenter + glm::vec3(-middleBase.x, 0.f, 0.f);
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				for (int joint= 0; joint < 4; ++joint)
					outPoints[FINGER_JOINTS[finger][joint]]= joints[finger][joint];
		};

		// OpenCV-convention camera looking at a target (markerFromCamera
		// columns = camera axes in world; +Z toward the scene)
		auto makeLookAtCamera= [](const glm::vec3& cameraPos, const glm::vec3& target) {
			const glm::vec3 z= glm::normalize(target - cameraPos);
			const glm::vec3 x= glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), z));
			const glm::vec3 y= glm::cross(z, x);
			glm::dmat4 markerFromCamera(1.0);
			markerFromCamera[0]= glm::dvec4(x, 0.0);
			markerFromCamera[1]= glm::dvec4(y, 0.0);
			markerFromCamera[2]= glm::dvec4(z, 0.0);
			markerFromCamera[3]= glm::dvec4(cameraPos, 1.0);
			return markerFromCamera;
		};

		const float fx= 600.f, fy= 600.f, cx= 640.f, cy= 360.f;
		auto projectTo= [&](const glm::dmat4& markerFromCamera, const glm::vec3& world) {
			const glm::dvec4 camPoint= glm::inverse(markerFromCamera) * glm::dvec4(glm::dvec3(world), 1.0);
			return glm::vec3((float)(fx * camPoint.x / camPoint.z + cx),
							 (float)(fy * camPoint.y / camPoint.z + cy), 0.f);
		};

		// A camera observation: real image points (projected from
		// worldHand), but a corrupted monocular pose - 3cm depth error
		// along the view ray and +0.2 rad on every proximal angle. If any
		// of that corruption reaches the fused output, the stereo path
		// didn't run.
		auto makeStereoResult= [&](int cameraIndex, const glm::vec3& cameraPos,
								   const std::array<glm::vec3, HAND_LANDMARK_COUNT>& imageHand,
								   const std::array<glm::vec3, HAND_LANDMARK_COUNT>& worldHand,
								   const glm::vec3& palmCenter) {
			const glm::dmat4 markerFromCamera= makeLookAtCamera(cameraPos, palmCenter);

			TrackingFrameResult frame;
			TrackedHand& hand= frame.hands[(int)eHandSide::Right];
			hand.tracked= true;
			hand.side= eHandSide::Right;
			hand.presence= 0.9f;
			hand.handednessScore= 0.9f;
			hand.rightProb= 0.9f;
			for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
				hand.imagePoints[i]= projectTo(markerFromCamera, imageHand[i]);
			hand.worldPoints= worldHand; // carries the assumed hand scale
			hand.hasWorldSpace= true;

			HandPose& pose= frame.poses[(int)eHandSide::Right];
			pose.tracked= true;
			pose.side= eHandSide::Right;
			pose.presence= 0.9f;
			pose.hasWorldPose= true;
			pose.palmPositionWorld= palmCenter + glm::normalize(palmCenter - cameraPos) * 0.03f;
			pose.palmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
			pose.fingers= anglesTruth;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				pose.fingers[finger].proximal+= 0.2f; // monocular articulation error
			pose.skeleton= skeleton;

			CameraFrameResult camera;
			camera.cameraIndex= cameraIndex;
			camera.valid= true;
			camera.timestampMs= now;
			camera.hasExtrinsics= true;
			camera.markerFromCamera= markerFromCamera;
			camera.hasIntrinsics= true;
			camera.fx= fx;
			camera.fy= fy;
			camera.cx= cx;
			camera.cy= cy;
			camera.result= frame;
			return camera;
		};

		std::array<glm::vec3, HAND_LANDMARK_COUNT> worldHand;
		buildWorldHand(palmTruth, worldHand);
		const glm::vec3 camAPos= palmTruth + glm::vec3(0.f, 0.f, 0.8f);
		const glm::vec3 camBPos= palmTruth + glm::vec3(0.f, -0.55f, 0.4f);

		HandFusion triFusion;
		triFusion.configure(fusionConfig); // triangulation on by default

		const auto camA= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
		const auto camB= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
		TrackingFrameResult fused;
		triFusion.fuse({&camA, &camB}, now, fused);

		const HandPose& pose= fused.poses[(int)eHandSide::Right];
		float maxAngleError= 0.f;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].lateral - anglesTruth[finger].lateral));
			maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].proximal - anglesTruth[finger].proximal));
			maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].intermediate - anglesTruth[finger].intermediate));
			maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].distal - anglesTruth[finger].distal));
		}
		const float palmError= glm::length(pose.palmPositionWorld - palmTruth);
		const FusionDiagnostics& diagnostics= triFusion.getLastDiagnostics();
		const bool bDiagTriangulated=
			!diagnostics.clusters.empty() && diagnostics.clusters[0].triangulated;
		MIKAN_LOG_INFO("test-fusion") << "(m) triangulation: stereoTriangulated=" << pose.stereoTriangulated
			<< " palm err mm=" << palmError * 1000.f << " max angle err rad=" << maxAngleError
			<< " residual px=" << (diagnostics.clusters.empty() ? -1.f : diagnostics.clusters[0].triResidualRmsPx);
		// The corrupted mono poses had 30mm palm error and +0.2 rad on the
		// proximals - exact recovery proves the stereo geometry won
		if (!pose.tracked || !pose.stereoTriangulated || !bDiagTriangulated ||
			palmError > 0.002f || maxAngleError > 0.02f)
		{
			MIKAN_LOG_ERROR("test-fusion") << "(m) FAILED: triangulated pose must recover the true hand";
			result= 1;
		}

		// Residual gating: deterministic ~10px 2D noise on camera B makes
		// the triangulation survive (well under the veto) but the
		// residual factor must visibly reduce the fused confidence.
		// (The midpoint solve splits one view's error across both views'
		// residuals, so the RMS lands well below the injected amplitude.)
		{
			auto camBNoisy= camB;
			TrackedHand& noisyHand= camBNoisy.result.hands[(int)eHandSide::Right];
			for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			{
				noisyHand.imagePoints[i].x+= (i % 2 == 0) ? 10.f : -10.f;
				noisyHand.imagePoints[i].y+= (float)((i % 3) - 1) * 10.f;
			}

			HandFusion noisyFusion;
			noisyFusion.configure(fusionConfig);
			TrackingFrameResult noisyFused;
			noisyFusion.fuse({&camA, &camBNoisy}, now, noisyFused);

			const HandPose& noisyPose= noisyFused.poses[(int)eHandSide::Right];
			const float cleanConfidence= pose.confidence;
			MIKAN_LOG_INFO("test-fusion") << "(m) residual gate: clean confidence=" << cleanConfidence
				<< " noisy confidence=" << noisyPose.confidence << " residual px="
				<< noisyFusion.getLastDiagnostics().clusters[0].triResidualRmsPx;
			if (!noisyPose.stereoTriangulated || cleanConfidence < 0.85f ||
				noisyPose.confidence > cleanConfidence - 0.05f || noisyPose.confidence < 0.1f)
			{
				MIKAN_LOG_ERROR("test-fusion")
					<< "(m) FAILED: 2D noise must reduce confidence via the residual factor";
				result= 1;
			}
		}

		// Mismatched pairing: camera B's pixels come from a DIFFERENT hand
		// 15cm away, while both monocular poses still cluster together.
		// The reprojection residual must veto, and the output falls back
		// to the best single observation (still tracked).
		std::array<glm::vec3, HAND_LANDMARK_COUNT> otherHand;
		buildWorldHand(palmTruth + glm::vec3(0.15f, 0.f, 0.f), otherHand);

		HandFusion vetoFusion;
		vetoFusion.configure(fusionConfig);
		const auto camBWrong= makeStereoResult(1, camBPos, otherHand, worldHand, palmTruth);
		TrackingFrameResult vetoFused;
		vetoFusion.fuse({&camA, &camBWrong}, now, vetoFused);

		const HandPose& vetoPose= vetoFused.poses[(int)eHandSide::Right];
		const FusionDiagnostics& vetoDiagnostics= vetoFusion.getLastDiagnostics();
		const bool bVetoed=
			!vetoDiagnostics.clusters.empty() && vetoDiagnostics.clusters[0].triVetoed;
		MIKAN_LOG_INFO("test-fusion") << "(m) veto: tracked=" << vetoPose.tracked
			<< " stereoTriangulated=" << vetoPose.stereoTriangulated << " vetoed=" << bVetoed
			<< " residual px="
			<< (vetoDiagnostics.clusters.empty() ? -1.f : vetoDiagnostics.clusters[0].triResidualRmsPx);
		if (!vetoPose.tracked || vetoPose.stereoTriangulated || !bVetoed)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(m) FAILED: a mismatched pairing must be vetoed by the reprojection residual";
			result= 1;
		}

		// (o) Solo-cluster rescue: camera B's MONO palm is 30cm off
		// laterally (as measured live during a pointing gesture), so
		// position clustering strands it - but its image points are
		// good, and the probe triangulation must pair it anyway
		{
			auto camBFar= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
			camBFar.result.poses[(int)eHandSide::Right].palmPositionWorld=
				palmTruth + glm::vec3(0.3f, 0.f, 0.f);

			HandFusion rescueFusion;
			rescueFusion.configure(fusionConfig);
			TrackingFrameResult rescueFused;
			rescueFusion.fuse({&camA, &camBFar}, now, rescueFused);
			const bool bClusterRescued= rescueFused.poses[(int)eHandSide::Right].stereoTriangulated;

			// ...and the same via the low-presence pool: camera B's
			// pose is too weak to be a clustering candidate at all
			auto camBWeak= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
			camBWeak.result.poses[(int)eHandSide::Right].presence= 0.3f;
			camBWeak.result.hands[(int)eHandSide::Right].presence= 0.3f;

			HandFusion poolFusion;
			poolFusion.configure(fusionConfig);
			TrackingFrameResult poolFused;
			poolFusion.fuse({&camA, &camBWeak}, now, poolFused);
			const bool bPoolRescued= poolFused.poses[(int)eHandSide::Right].stereoTriangulated;

			MIKAN_LOG_INFO("test-fusion") << "(o) rescue: displaced-mono=" << bClusterRescued
				<< " low-presence-pool=" << bPoolRescued;
			if (!bClusterRescued || !bPoolRescued)
			{
				MIKAN_LOG_ERROR("test-fusion")
					<< "(o) FAILED: solo clusters must pair via probe triangulation";
				result= 1;
			}
		}

		// (p) Triangulated-angle hold: after a stereo fuse, a brief
		// single-camera fallback must keep the triangulated angles
		// (the mono articulation carries pose-dependent bias, +0.2 rad
		// here); a sustained fallback adopts the mono angles
		{
			HandFusion holdFusion;
			holdFusion.configure(fusionConfig);

			TrackingFrameResult holdFused;
			holdFusion.fuse({&camA, &camB}, now, holdFused);
			const float triProx= holdFused.poses[(int)eHandSide::Right].fingers[1].proximal;

			auto camASolo= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
			camASolo.timestampMs= now + 100.0;
			holdFusion.fuse({&camASolo}, now + 100.0, holdFused);
			const HandPose& heldPose= holdFused.poses[(int)eHandSide::Right];
			const float heldProx= heldPose.fingers[1].proximal;

			auto camALate= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
			camALate.timestampMs= now + 500.0;
			holdFusion.fuse({&camALate}, now + 500.0, holdFused);
			const float lateProx= holdFused.poses[(int)eHandSide::Right].fingers[1].proximal;

			const float monoProx= anglesTruth[1].proximal + 0.2f;
			MIKAN_LOG_INFO("test-fusion") << "(p) angle hold: tri=" << triProx << " held=" << heldProx
				<< " late=" << lateProx << " (mono=" << monoProx << ")";
			if (heldPose.stereoTriangulated || fabsf(heldProx - triProx) > 1e-4f ||
				fabsf(lateProx - monoProx) > 1e-4f)
			{
				MIKAN_LOG_ERROR("test-fusion")
					<< "(p) FAILED: brief fallback must hold tri angles; sustained fallback goes mono";
				result= 1;
			}
		}

		// (q) Triangulated palm-depth hold: the mono pose carries a
		// 30mm along-ray depth error; a brief single-camera fallback
		// must pin the along-ray component to the last triangulated
		// palm (the mono lateral is exact here, so the held position
		// is exact), and a sustained fallback adopts the mono depth
		{
			HandFusion holdFusion;
			holdFusion.configure(fusionConfig);

			TrackingFrameResult holdFused;
			holdFusion.fuse({&camA, &camB}, now, holdFused);
			const glm::vec3 triPalm= holdFused.poses[(int)eHandSide::Right].palmPositionWorld;

			auto camASolo= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
			camASolo.timestampMs= now + 100.0;
			holdFusion.fuse({&camASolo}, now + 100.0, holdFused);
			const glm::vec3 heldPalm= holdFused.poses[(int)eHandSide::Right].palmPositionWorld;
			const float heldErrMm= glm::length(heldPalm - triPalm) * 1000.f;

			auto camALate= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
			camALate.timestampMs= now + 500.0;
			holdFusion.fuse({&camALate}, now + 500.0, holdFused);
			const glm::vec3 latePalm= holdFused.poses[(int)eHandSide::Right].palmPositionWorld;
			const glm::vec3 monoPalm= camALate.result.poses[(int)eHandSide::Right].palmPositionWorld;
			const float lateErrMm= glm::length(latePalm - monoPalm) * 1000.f;

			MIKAN_LOG_INFO("test-fusion") << "(q) palm hold: held err mm=" << heldErrMm
				<< " (mono depth error was 30), late-vs-mono mm=" << lateErrMm;
			if (heldErrMm > 1.f || lateErrMm > 0.01f)
			{
				MIKAN_LOG_ERROR("test-fusion")
					<< "(q) FAILED: brief fallback must hold the tri palm depth; sustained goes mono";
				result= 1;
			}
		}
	}

	// (n) Articulation-source hysteresis (non-triangulated path): the
	// incumbent camera keeps supplying angles until a challenger beats
	// its weight decisively for several consecutive fuses - weight noise
	// alone must not flip the source
	{
		HandFusion selFusion;
		selFusion.configure(fusionConfig);

		auto fuseStep= [&](int step, float presenceA, float presenceB, TrackingFrameResult& outFused) {
			const double stepTime= now + step * 33.0;
			const auto camA= makeCameraResult(
				0, cam1Pos, stepTime,
				makeObservation(palmTruth, faceUpToCam1, presenceA, eHandSide::Left, 0.1f, 0.5f));
			const auto camB= makeCameraResult(
				1, cam2Pos, stepTime,
				makeObservation(palmTruth, faceUpToCam1, presenceB, eHandSide::Left, 0.1f, 0.9f));
			selFusion.fuse({&camA, &camB}, stepTime, outFused);
		};

		TrackingFrameResult fused;
		// Establish camera A as the incumbent
		for (int step= 0; step < 3; ++step)
			fuseStep(step, 0.9f, 0.7f, fused);
		const float bendIncumbent= fused.poses[(int)eHandSide::Left].fingers[0].proximal;

		// Challenger decisively ahead (incumbent presence stays at the
		// candidate threshold so it isn't dropped outright): the
		// incumbent must survive the first kArticulationSwitchFrames-1
		// fuses...
		float bendHolding= -1.f;
		for (int step= 3; step < 3 + 4; ++step)
		{
			fuseStep(step, 0.5f, 0.99f, fused);
			bendHolding= fused.poses[(int)eHandSide::Left].fingers[0].proximal;
		}
		// ...and lose the job on the 5th
		fuseStep(7, 0.5f, 0.99f, fused);
		const float bendSwitched= fused.poses[(int)eHandSide::Left].fingers[0].proximal;

		MIKAN_LOG_INFO("test-fusion") << "(n) articulation selection: incumbent bend=" << bendIncumbent
			<< " holding bend=" << bendHolding << " switched bend=" << bendSwitched;
		if (fabsf(bendIncumbent - 0.5f) > 1e-6f || fabsf(bendHolding - 0.5f) > 1e-6f ||
			fabsf(bendSwitched - 0.9f) > 1e-6f)
		{
			MIKAN_LOG_ERROR("test-fusion")
				<< "(n) FAILED: source must hold through the hysteresis window, then switch";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-fusion") << "All fusion checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-fusion", "Multi-camera fusion, clustering, triangulation, holds", eTestCategory::SelfTest, runFusionTest);
