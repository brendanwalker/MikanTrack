#include "TestCommon.h"

#include "BodyDimensionCalibrator.h"

// Builds a synthetic body of KNOWN landmark separations, projects it into a
// camera, and checks the calibrator measures those separations back. The point
// of the calibrator is that the model's landmark separations are NOT
// anatomical, so the test asserts recovery of whatever separations it was
// given - deliberately using values well away from the anatomical defaults.

namespace
{
constexpr float kFx= 700.f;
constexpr float kFy= 700.f;
constexpr float kCx= 640.f;
constexpr float kCy= 360.f;

// The truth this test measures back
constexpr float kTrueShoulderWidth= 0.29f; // narrower than the 0.40 default
constexpr float kTrueHeadWidth= 0.128f;
constexpr float kTrueNoseForward= 0.095f;
constexpr float kPlaneRange= 1.05f; // how far the torso sits from the camera

glm::vec2 project(const glm::vec3& p)
{
	return glm::vec2(kFx * p.x / p.z + kCx, kFy * p.y / p.z + kCy);
}

// A camera at the world origin looking down +Z, with a body facing it
struct SyntheticBody
{
	CameraFrameResult camera;
	TrackingFrameResult fused;

	void setLandmark(ePoseLandmark landmark, const glm::vec3& p)
	{
		const int index= (int)landmark;
		camera.result.body.imagePoints[index]= glm::vec3(project(p), 0.f);
		camera.result.body.visibility[index]= 1.f;
		camera.result.body.providedMask|= 1ull << index;
	}

	void setWrist(int sideIndex, const glm::vec3& wristWorld)
	{
		HandPose& pose= fused.poses[sideIndex];
		pose= HandPose();
		pose.tracked= true;
		pose.side= (eHandSide)sideIndex;
		pose.confidence= 0.9f;
		pose.hasWorldPose= true;
		pose.palmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
		pose.skeleton.baseInPalm[(int)eFinger::Middle]= glm::vec3(0.04f, 0.f, 0.f);
		pose.palmPositionWorld= wristWorld + glm::vec3(0.04f, 0.f, 0.f);
	}

	glm::vec3 shoulderPosition(int sideIndex, float yOffset= 0.f) const
	{
		const float x= (sideIndex == 0 ? +1.f : -1.f) * kTrueShoulderWidth * 0.5f;
		return glm::vec3(x, yOffset, kPlaneRange);
	}

	// yOffset walks the body slightly between samples, so the medians are not
	// taken over one repeated frame
	void buildTorso(float yOffset)
	{
		camera= CameraFrameResult();
		camera.valid= true;
		camera.hasExtrinsics= true;
		camera.markerFromCamera= glm::dmat4(1.0);
		camera.hasIntrinsics= true;
		camera.fx= kFx;
		camera.fy= kFy;
		camera.cx= kCx;
		camera.cy= kCy;
		camera.result.body.valid= true;
		camera.result.body.confidence= 0.9f;
		fused= TrackingFrameResult();

		setLandmark(ePoseLandmark::LEFT_SHOULDER, shoulderPosition(0, yOffset));
		setLandmark(ePoseLandmark::RIGHT_SHOULDER, shoulderPosition(1, yOffset));
		setLandmark(ePoseLandmark::LEFT_EAR,
					glm::vec3(+kTrueHeadWidth * 0.5f, yOffset - 0.25f, kPlaneRange));
		setLandmark(ePoseLandmark::RIGHT_EAR,
					glm::vec3(-kTrueHeadWidth * 0.5f, yOffset - 0.25f, kPlaneRange));
	}

	// A hand raised beside its own shoulder: the whole pose the wizard asks for
	void raiseHand(int sideIndex, float yOffset= 0.f)
	{
		setWrist(sideIndex, shoulderPosition(sideIndex, yOffset) + glm::vec3(0.f, -0.12f, 0.f));
	}

	// The head turned 90 degrees: the nose's forward offset now lies across
	// the image instead of along the view ray
	void buildHeadTurnPose()
	{
		const float z= kPlaneRange;
		const float headY= -0.25f;
		setLandmark(ePoseLandmark::LEFT_EAR, glm::vec3(0.f, headY, z + kTrueHeadWidth * 0.5f));
		setLandmark(ePoseLandmark::RIGHT_EAR, glm::vec3(0.f, headY, z - kTrueHeadWidth * 0.5f));
		setLandmark(ePoseLandmark::NOSE, glm::vec3(kTrueNoseForward, headY, z));
	}
};
} // namespace

static int runBodyCalibrationTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-bodycalib") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-bodycalib") << "FAIL " << name;
			failures++;
		}
	};

	BodyDimensions dimensions; // defaults, deliberately wrong for this body

	// (a) A raised hand recovers the widths it was built from
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		for (int i= 0; i < 60; ++i)
		{
			const float yOffset= 0.002f * (float)(i % 5);
			body.buildTorso(yOffset);
			body.raiseHand(0, yOffset);
			BodyDimensionCalibrator::Sample sample;
			calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample);
		}
		check(calibrator.getFrontalSampleCount() == 60, "every raised-hand frame accepted");

		const BodyDimensionCalibrator::Result result= calibrator.solve(dimensions);
		check(result.bValid, "solve reports a result");
		MIKAN_LOG_INFO("test-bodycalib")
			<< "shoulders " << result.shoulderWidth << " (true " << kTrueShoulderWidth << "), head "
			<< result.headWidth << " (true " << kTrueHeadWidth << ")";
		check(fabsf(result.shoulderWidth - kTrueShoulderWidth) < 0.005f, "shoulder width recovered");
		check(fabsf(result.headWidth - kTrueHeadWidth) < 0.005f, "head width recovered");
		check(result.shoulderWidthSpread < 0.05f, "spread reported small for a still pose");

		// The upper arm is DERIVED, never measured: measuring it needed the arm
		// straight and square to the camera, a pose too hard to hold at a desk,
		// and a missed one read it 20% short - enough to bend the elbow wrongly
		check(fabsf(result.upperArmLength - result.shoulderWidth * dimensions.upperArmPerShoulderWidth) <
				  1e-4f,
			  "upper arm derived from the shoulder width");
	}

	// (b) A hand resting out in front is rejected rather than allowed to set
	// the depth: at a desk it sits well ahead of the torso, and measuring the
	// shoulders at THAT depth would scale them wrong
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		body.buildTorso(0.f);
		body.setWrist(0, glm::vec3(+0.20f, 0.25f, kPlaneRange - 0.35f));
		body.setWrist(1, glm::vec3(-0.20f, 0.25f, kPlaneRange - 0.35f));

		BodyDimensionCalibrator::Sample sample;
		check(!calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample),
			  "hands out on the desk rejected");
		check(calibrator.getFrontalSampleCount() == 0, "rejected sample not recorded");
	}

	// (c) ONE raised hand carries the frame while the other rests, which is
	// the normal state of affairs at a desk
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		for (int i= 0; i < 60; ++i)
		{
			body.buildTorso(0.f);
			body.raiseHand(0);
			body.setWrist(1, glm::vec3(-0.20f, 0.25f, kPlaneRange - 0.35f));

			BodyDimensionCalibrator::Sample sample;
			const bool bAny= calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample);
			if (i == 0)
			{
				check(bAny, "frame accepted on the strength of one raised hand");
				check(sample.bAcceptedLeft, "the raised hand counts");
				check(!sample.bAcceptedRight, "the hand out on the desk does not");
			}
		}

		const BodyDimensionCalibrator::Result result= calibrator.solve(dimensions);
		check(result.bValid, "one raised hand produces a result");
		check(fabsf(result.shoulderWidth - kTrueShoulderWidth) < 0.005f,
			  "shoulder width recovered from one raised hand");
		check(fabsf(result.headWidth - kTrueHeadWidth) < 0.005f, "head width recovered from one raised hand");
	}

	// (d) Too few samples is reported rather than averaged into a number
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		for (int i= 0; i < 3; ++i)
		{
			body.buildTorso(0.f);
			body.raiseHand(0);
			BodyDimensionCalibrator::Sample sample;
			calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample);
		}
		check(!calibrator.solve(dimensions).bValid, "too few samples reports invalid");
	}

	// (e) The head turn recovers the nose's forward offset
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		for (int i= 0; i < 40; ++i)
		{
			body.buildTorso(0.f);
			body.raiseHand(0);
			BodyDimensionCalibrator::Sample sample;
			calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample);
		}
		body.buildHeadTurnPose();
		for (int i= 0; i < 30; ++i)
		{
			float noseForward= 0.f;
			calibrator.addHeadTurnSample(body.camera, noseForward);
		}
		const BodyDimensionCalibrator::Result result= calibrator.solve(dimensions);
		check(result.bHaveNoseForward, "nose forward measured");
		MIKAN_LOG_INFO("test-bodycalib")
			<< "nose forward " << result.noseForward << " (true " << kTrueNoseForward << ")";
		check(fabsf(result.noseForward - kTrueNoseForward) < 0.01f, "nose forward recovered");
	}

	// (f) A face-on head measures nothing: the offset it would report is the
	// ear separation, not the nose's reach
	{
		BodyDimensionCalibrator calibrator;
		SyntheticBody body;
		for (int i= 0; i < 40; ++i)
		{
			body.buildTorso(0.f);
			body.raiseHand(0);
			BodyDimensionCalibrator::Sample sample;
			calibrator.addFrontalSample(body.camera, body.fused, dimensions, sample);
		}
		body.setLandmark(ePoseLandmark::NOSE, glm::vec3(0.f, -0.25f, kPlaneRange));
		float noseForward= 0.f;
		check(!calibrator.addHeadTurnSample(body.camera, noseForward), "face-on head rejected");
	}

	if (failures == 0)
		MIKAN_LOG_INFO("test-bodycalib") << "All body calibration tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-bodycalib", "Body dimension calibration recovers known landmark separations",
					eTestCategory::SelfTest, runBodyCalibrationTest);
