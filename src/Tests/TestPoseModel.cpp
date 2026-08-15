#include "TestCommon.h"

#include "PoseDetector.h"
#include "RtmPoseBodyModel.h"

// Ground truth from the Python reference implementation of the same pipeline
// (rtm_demo.jpg with the whole image as the person box), which is how the
// opencv_zoo ports were validated too: a numeric cross-check up front means
// every later tracking bug is OUR logic, never the inference stack.
namespace
{
struct ReferenceKeypoint
{
	const char* name;
	float x;
	float y;
};
const ReferenceKeypoint k_rtmReference[COCO_KEYPOINT_COUNT]= {
	{"nose", 87.0f, 35.3f},    {"eyeL", 93.8f, 28.6f},    {"eyeR", 81.1f, 29.4f},
	{"earL", 107.3f, 31.9f},   {"earR", 76.1f, 35.3f},    {"shoL", 126.7f, 69.1f},
	{"shoR", 82.8f, 82.6f},    {"elbL", 157.1f, 103.7f},  {"elbR", 60.0f, 117.2f},
	{"wriL", 167.3f, 118.1f},  {"wriR", 27.1f, 133.3f},   {"hipL", 151.2f, 167.1f},
	{"hipR", 120.8f, 173.8f},  {"kneeL", 152.1f, 244.0f}, {"kneeR", 90.4f, 228.8f},
	{"ankL", 194.3f, 315.8f},  {"ankR", 125.0f, 297.2f},
};
} // namespace

// Smoke test for the body-pose models: both load, both run, and RTMPose's
// decode matches the reference implementation. Hardware category (needs the
// models downloaded by InitialSetup_x64.bat and an ONNX runtime EP); skips
// with a warning when the models are absent.
static int runPoseModelTest(const TestArgs&)
{
	if (!std::filesystem::exists("models/person_detection.onnx") ||
		!std::filesystem::exists("models/rtmpose_body.onnx"))
	{
		MIKAN_LOG_WARNING("test-posemodel")
			<< "models/person_detection.onnx or models/rtmpose_body.onnx missing - "
			   "run InitialSetup_x64.bat; SKIPPING";
		return 0;
	}

	// The person detector supplies RTMPose's box. Only its KEYPOINTS are used:
	// its own box is a face box, and feeding that to a top-down model cropped
	// the arms away.
	PoseDetector detector;
	if (!detector.load("models/person_detection.onnx", "directml"))
	{
		MIKAN_LOG_ERROR("test-posemodel") << "person detector failed to load";
		return 1;
	}
	MIKAN_LOG_INFO("test-posemodel") << "person detector loaded (ep=" << detector.activeEp() << ")";

	RtmPoseBodyModel rtmPose;
	if (!rtmPose.load("models/rtmpose_body.onnx", "directml"))
	{
		MIKAN_LOG_ERROR("test-posemodel") << "RTMPose model failed to load";
		return 1;
	}
	MIKAN_LOG_INFO("test-posemodel") << "RTMPose model loaded (ep=" << rtmPose.activeEp() << ")";

	// A gradient frame with a bright vertical bar: deterministic, cheap, and
	// enough structure to exercise the full pre/postprocess path
	cv::Mat frame(720, 1280, CV_8UC3);
	for (int y= 0; y < frame.rows; ++y)
		for (int x= 0; x < frame.cols; ++x)
			frame.at<cv::Vec3b>(y, x)= cv::Vec3b((uint8_t)(x % 256), (uint8_t)(y % 256), 128);
	cv::rectangle(frame, cv::Rect(560, 100, 160, 520), cv::Scalar(230, 220, 210), cv::FILLED);

	std::vector<PersonDetection> detections;
	detector.detect(frame, detections);
	MIKAN_LOG_INFO("test-posemodel") << "detector ran: " << detections.size() << " detections on synthetic frame";

	const cv::Mat referenceImage= cv::imread("rtm_demo.jpg", cv::IMREAD_COLOR);
	if (referenceImage.empty())
	{
		MIKAN_LOG_WARNING("test-posemodel")
			<< "rtm_demo.jpg missing - SKIPPING the numeric cross-check (re-run InitialSetup_x64.bat)";
		MIKAN_LOG_INFO("test-posemodel") << "Pose model smoke test passed";
		return 0;
	}

	RtmPoseResult rtmResult;
	rtmPose.estimate(referenceImage, glm::vec2(0.f, 0.f),
					 glm::vec2((float)referenceImage.cols, (float)referenceImage.rows), rtmResult);
	if (!rtmResult.valid)
	{
		MIKAN_LOG_ERROR("test-posemodel") << "RTMPose produced no result on the reference image";
		return 1;
	}

	// Half a pixel: the reference decodes on the same 0.5px SimCC grid, so
	// anything larger means the crop or the decode disagrees
	float worstError= 0.f;
	const char* worstName= "";
	for (int keypoint= 0; keypoint < COCO_KEYPOINT_COUNT; ++keypoint)
	{
		const ReferenceKeypoint& expected= k_rtmReference[keypoint];
		const float error= glm::length(rtmResult.points[keypoint] - glm::vec2(expected.x, expected.y));
		if (error > worstError)
		{
			worstError= error;
			worstName= expected.name;
		}
	}
	MIKAN_LOG_INFO("test-posemodel")
		<< "RTMPose vs Python reference: worst keypoint error " << worstError << " px (" << worstName << ")";
	if (worstError > 0.5f)
	{
		MIKAN_LOG_ERROR("test-posemodel") << "RTMPose does not match the reference implementation";
		return 1;
	}

	// What the stage costs per model frame. Logged rather than asserted - it
	// is a machine-dependent number, and the reason to have it is to settle
	// "is the pose stage what slowed the capture down".
	{
		const int kIterations= 30;
		auto timeCalls= [&](auto&& call) -> double {
			call(); // warm up
			const auto start= std::chrono::steady_clock::now();
			for (int i= 0; i < kIterations; ++i)
				call();
			return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
				kIterations;
		};

		const double detectMs= timeCalls([&]() { detector.detect(frame, detections); });
		const double rtmMs= timeCalls([&]() {
			rtmPose.estimate(referenceImage, glm::vec2(0.f, 0.f),
							 glm::vec2((float)referenceImage.cols, (float)referenceImage.rows), rtmResult);
		});
		MIKAN_LOG_INFO("test-posemodel")
			<< "person detector " << detectMs << " ms, RTMPose " << rtmMs << " ms";
	}

	MIKAN_LOG_INFO("test-posemodel") << "Pose model smoke test passed";
	return 0;
}

MIKAN_REGISTER_TEST("--test-posemodel", "Body pose ONNX models load + match the reference decode",
					eTestCategory::Hardware, runPoseModelTest);
