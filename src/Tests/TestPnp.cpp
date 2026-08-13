#include "TestCommon.h"

// Real collapsed frame captured live 2026-08-12 (recording 2026-08-12_21-18-48
// record 627, camera 1, right hand at presence 0.98): a flat waving hand is a
// NEAR-planar PnP object model, and ITERATIVE's cold-start DLT is
// ill-conditioned near planarity - it collapsed this frame to a degenerate
// depth, the sanity gate rejected it, and with no warm start left the collapse
// repeated frame after frame, starving fusion of its best right-hand view for
// 1.7s. The cold path uses SQPnP, which has no planar degeneracy.
// fx=709.8433 fy=717.0490 cx=639.5 cy=359.5 refLen=0.0802821
static const float kCollapsedImagePoints[21][2]= {
	{594.0136f, 181.1727f}, {633.3922f, 179.0412f}, {675.0124f, 192.5410f},
	{704.4434f, 208.0729f}, {729.5499f, 216.6069f}, {671.3357f, 190.6852f},
	{712.4000f, 215.3370f}, {737.6059f, 235.6000f}, {758.4676f, 252.8480f},
	{653.7726f, 205.1958f}, {694.2023f, 237.3131f}, {719.1755f, 260.5496f},
	{738.2517f, 278.9870f}, {632.0893f, 221.4022f}, {665.1223f, 252.7435f},
	{685.2706f, 274.0037f}, {701.8394f, 289.1155f}, {607.6987f, 235.8633f},
	{625.8817f, 263.8148f}, {636.2778f, 281.2594f}, {645.2819f, 294.7353f},
};
static const float kCollapsedModelPoints[21][3]= {
	{-0.048175905f, -0.020280935f, 0.073803604f},
	{-0.012090249f, -0.015172486f, 0.064311564f},
	{0.011748169f, -0.003399489f, 0.055308938f},
	{0.033671662f, 0.002452330f, 0.030198976f},
	{0.051880442f, 0.008294753f, 0.010662496f},
	{0.017131833f, -0.005023675f, 0.002877742f},
	{0.043220822f, 0.009358629f, -0.003933258f},
	{0.058501761f, 0.019624775f, 0.002557978f},
	{0.071209125f, 0.032210153f, 0.012792252f},
	{0.004068288f, -0.001713665f, -0.005302221f},
	{0.027112709f, 0.020408731f, -0.005995214f},
	{0.047325779f, 0.041690100f, 0.003904015f},
	{0.063700184f, 0.054588012f, 0.017173350f},
	{-0.011084175f, 0.005361823f, -0.002454299f},
	{0.011556239f, 0.026930016f, -0.001112856f},
	{0.026443787f, 0.042801294f, 0.005607322f},
	{0.040008407f, 0.058769338f, 0.018189251f},
	{-0.028603898f, 0.012428292f, 0.007194787f},
	{-0.016306015f, 0.031820092f, 0.005162418f},
	{-0.006933329f, 0.049818959f, 0.008532971f},
	{-0.005045247f, 0.061089151f, 0.015516363f},
};
// The session's calibrated right-hand skeleton: 5 finger bases (xyz) then 5
// phalanx-length triples, the config serialization layout
static const float kCollapsedSkeletonRight[30]= {
	-0.022142902f, -0.032456264f, 0.012881516f,
	0.039979830f, -0.023224011f, 0.003556268f,
	0.040228657f, 0.000000000f, 0.000000000f,
	0.034741379f, 0.020518646f, 0.000301838f,
	0.025826164f, 0.038612425f, 0.003011666f,
	0.036547299f, 0.029250180f, 0.024595857f,
	0.040840302f, 0.024664853f, 0.019590959f,
	0.043487467f, 0.026541002f, 0.020960653f,
	0.039015729f, 0.024508586f, 0.019830758f,
	0.030430714f, 0.018876871f, 0.016402410f,
};

static int runPnpTest(const TestArgs& args)
{
	int result= 0;

	// Pinhole camera
	const float fx= 800.f, fy= 800.f, cx= 640.f, cy= 360.f;
	MikanMonoIntrinsics intrinsics;
	intrinsics.pixel_width= 1280;
	intrinsics.pixel_height= 720;
	intrinsics.undistorted_camera_matrix.x0= fx;
	intrinsics.undistorted_camera_matrix.y1= fy;
	intrinsics.undistorted_camera_matrix.z0= cx;
	intrinsics.undistorted_camera_matrix.z1= cy;

	// Canonical hand model in hand-local meters, wrist at the origin,
	// slightly non-planar (finger curl) so the pose is well-determined
	std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints;
	for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
	{
		const float along= 0.02f + 0.15f * (float)(lm % 4) / 4.f * (lm >= 1 ? 1.f : 0.f);
		const float spread= ((float)(lm / 4) - 2.f) * 0.02f;
		const float curl= 0.012f * (float)(lm % 4);
		modelPoints[lm]= glm::vec3(spread, along, curl);
	}
	modelPoints[0]= glm::vec3(0.f);
	const float boneLength= glm::length(modelPoints[(int)eHandLandmark::MIDDLE_MCP]);

	// Ground-truth rigid pose (OpenCV camera convention)
	const glm::mat3 rotationTruth=
		glm::mat3(glm::rotate(glm::mat4(1.f), 0.4f, glm::vec3(0, 1, 0)) *
				  glm::rotate(glm::mat4(1.f), -0.3f, glm::vec3(1, 0, 0)));
	const glm::vec3 translationTruth(0.05f, -0.03f, 0.7f);

	auto buildHand= [&](std::array<glm::vec3, HAND_LANDMARK_COUNT>& outTruthCamera) {
		TrackedHand hand;
		hand.tracked= true;
		hand.side= eHandSide::Left;
		hand.presence= 0.9f;
		hand.modelPoints= modelPoints;
		for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
		{
			const glm::vec3 cameraPoint= rotationTruth * modelPoints[lm] + translationTruth;
			outTruthCamera[lm]= cameraPoint;
			hand.imagePoints[lm]= glm::vec3(
				fx * cameraPoint.x / cameraPoint.z + cx,
				fy * cameraPoint.y / cameraPoint.z + cy,
				0.f);
		}
		return hand;
	};

	auto runEstimator= [&](const char* label) {
		LandmarkTo3D landmarkTo3D;
		landmarkTo3D.configure(intrinsics, boneLength);

		std::array<glm::vec3, HAND_LANDMARK_COUNT> truthCamera;
		TrackingFrameResult frame;
		frame.timestampMs= 1000.0;
		frame.hands[(int)eHandSide::Left]= buildHand(truthCamera);
		landmarkTo3D.process(frame);

		const TrackedHand& hand= frame.hands[(int)eHandSide::Left];
		float sum= 0.f;
		for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
			sum+= glm::dot(hand.cameraPoints[lm] - truthCamera[lm], hand.cameraPoints[lm] - truthCamera[lm]);
		const float rms= sqrtf(sum / (float)HAND_LANDMARK_COUNT);

		MIKAN_LOG_INFO("test-pnp") << label << ": tracked=" << hand.hasCameraSpace
			<< " rms error mm=" << rms * 1000.f;
		return hand.hasCameraSpace ? rms : 1e9f;
	};

	// PnP must recover the exact synthetic pose (sub-mm)
	const float rmsPnp= runEstimator("PnP all-21");
	if (rmsPnp > 0.001f)
	{
		MIKAN_LOG_ERROR("test-pnp") << "FAILED: PnP must recover the exact synthetic pose";
		result= 1;
	}

	// The real collapsed frame must solve to a plausible depth from a cold
	// start (the live hand was under a meter away; ITERATIVE read it at 2-4cm
	// or behind the camera and the sanity gate rejected every frame)
	{
		MikanMonoIntrinsics liveIntrinsics;
		liveIntrinsics.pixel_width= 1280;
		liveIntrinsics.pixel_height= 720;
		liveIntrinsics.undistorted_camera_matrix.x0= 709.8433f;
		liveIntrinsics.undistorted_camera_matrix.y1= 717.0490f;
		liveIntrinsics.undistorted_camera_matrix.z0= 639.5f;
		liveIntrinsics.undistorted_camera_matrix.z1= 359.5f;

		HandSkeleton skeleton;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			skeleton.baseInPalm[finger]= glm::vec3(kCollapsedSkeletonRight[finger * 3 + 0],
												   kCollapsedSkeletonRight[finger * 3 + 1],
												   kCollapsedSkeletonRight[finger * 3 + 2]);
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				skeleton.phalanxLengths[finger][phalanx]=
					kCollapsedSkeletonRight[FINGER_COUNT * 3 + finger * 3 + phalanx];
		}
		skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);

		LandmarkTo3D landmarkTo3D;
		landmarkTo3D.configure(liveIntrinsics, 0.0802821);
		landmarkTo3D.setCalibratedSkeleton(eHandSide::Right, skeleton);

		TrackedHand hand;
		hand.tracked= true;
		hand.side= eHandSide::Right;
		hand.presence= 0.98f;
		for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
		{
			hand.imagePoints[lm]= glm::vec3(kCollapsedImagePoints[lm][0], kCollapsedImagePoints[lm][1], 0.f);
			hand.modelPoints[lm]= glm::vec3(kCollapsedModelPoints[lm][0], kCollapsedModelPoints[lm][1],
											kCollapsedModelPoints[lm][2]);
		}

		TrackingFrameResult frame;
		frame.timestampMs= 1000.0;
		frame.hands[(int)eHandSide::Right]= hand;
		landmarkTo3D.process(frame);

		const TrackedHand& solved= frame.hands[(int)eHandSide::Right];
		const float wristDepth= solved.cameraPoints[(int)eHandLandmark::WRIST].z;
		MIKAN_LOG_INFO("test-pnp") << "PnP live collapsed frame: tracked=" << solved.hasCameraSpace
			<< " wrist depth m=" << wristDepth;
		if (!solved.hasCameraSpace || wristDepth < 0.4f || wristDepth > 1.6f)
		{
			MIKAN_LOG_ERROR("test-pnp")
				<< "FAILED: live collapsed frame must solve to a plausible depth";
			result= 1;
		}
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-pnp") << "All PnP checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-pnp", "Monocular PnP recovers a synthetic hand pose", eTestCategory::SelfTest, runPnpTest);
