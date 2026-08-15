#include "TestCommon.h"

static int runDepthViewTest(const TestArgs& args)
{
	int result= 0;

	// Synthetic RealSense frame: flat plane at 0.6m, pinhole streams,
	// depth camera offset 15mm to the left of color (D455-ish layout)
	constexpr int kW= 848, kH= 480;
	std::vector<uint16_t> depthImage((size_t)kW * kH, 600); // 600 * 1mm = 0.6m

	DepthFrameView view;
	view.valid= true;
	view.depthData= depthImage.data();
	view.depthWidth= kW;
	view.depthHeight= kH;
	view.depthUnitsMeters= 0.001f;
	view.colorIntrinsics= {1280, 720, 900.f, 900.f, 640.f, 360.f, 0, {}};
	view.depthIntrinsics= {kW, kH, 420.f, 420.f, 424.f, 240.f, 0, {}};
	view.depthToColorTranslation[0]= 0.015f;

	// (a) center of the color image -> straight-ahead point at 0.6m
	glm::vec3 point(0.f);
	bool bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
	MIKAN_LOG_INFO("test-depthview") << "(a) center: ok=" << bOk << " p=(" << point.x << "," << point.y
		<< "," << point.z << ")";
	if (!bOk || fabsf(point.x) > 0.003f || fabsf(point.y) > 0.003f || fabsf(point.z - 0.6f) > 0.005f)
	{
		MIKAN_LOG_ERROR("test-depthview") << "(a) FAILED: expected ~(0,0,0.6)";
		result= 1;
	}

	// (b) off-center pixel: the deprojected ray must hit the plane at
	// the right lateral offset (100px right of center at fx=900, 0.6m
	// -> x = 100/900*0.6 = 66.7mm)
	bOk= view.sampleCameraPointAtColorPixel(740.f, 360.f, 0.15f, 1.5f, point);
	MIKAN_LOG_INFO("test-depthview") << "(b) offset: ok=" << bOk << " x=" << point.x;
	if (!bOk || fabsf(point.x - 0.0667f) > 0.004f || fabsf(point.z - 0.6f) > 0.005f)
	{
		MIKAN_LOG_ERROR("test-depthview") << "(b) FAILED: lateral offset wrong";
		result= 1;
	}

	// (c) hole rejection: zero out a patch -> sampling inside must fail
	for (int y= 200; y < 280; ++y)
		for (int x= 380; x < 470; ++x)
			depthImage[(size_t)y * kW + x]= 0;
	bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
	MIKAN_LOG_INFO("test-depthview") << "(c) hole: ok=" << bOk << " (expected 0)";
	if (bOk)
	{
		MIKAN_LOG_ERROR("test-depthview") << "(c) FAILED: hole must not resolve";
		result= 1;
	}

	// (d) out-of-range rejection: plane beyond maxDepth reads as no hand
	std::fill(depthImage.begin(), depthImage.end(), (uint16_t)2000); // 2m
	bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
	MIKAN_LOG_INFO("test-depthview") << "(d) far plane: ok=" << bOk << " (expected 0)";
	if (bOk)
	{
		MIKAN_LOG_ERROR("test-depthview") << "(d) FAILED: 2m surface must be rejected as non-hand";
		result= 1;
	}

	if (result == 0)
		MIKAN_LOG_INFO("test-depthview") << "All depth-view checks passed";

	return result;
}

MIKAN_REGISTER_TEST("--test-depthview", "RealSense depth-to-camera-point mapping", eTestCategory::SelfTest, runDepthViewTest);
