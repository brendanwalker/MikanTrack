#include "TestCommon.h"

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

	if (result == 0)
		MIKAN_LOG_INFO("test-pnp") << "All PnP checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-pnp", "Monocular PnP recovers a synthetic hand pose", eTestCategory::SelfTest, runPnpTest);
