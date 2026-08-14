#include "TestCommon.h"

#include "PoseDetector.h"
#include "PoseLandmarkModel.h"
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

// Smoke test for the restored BlazePose models: loads both ONNX files and
// pushes a synthetic frame through the full preprocess/inference/decode path.
// Hardware category (needs the models downloaded by InitialSetup_x64.bat and
// an ONNX runtime EP); skips with a warning when the models are absent. The
// models are not expected to FIND anything in synthetic noise - the assertion
// is that the tensor contracts hold end to end without throwing.

static int runPoseModelTest(const TestArgs&)
{
	if (!std::filesystem::exists("models/person_detection.onnx") ||
		!std::filesystem::exists("models/pose_landmark.onnx"))
	{
		MIKAN_LOG_WARNING("test-posemodel")
			<< "models/person_detection.onnx or models/pose_landmark.onnx missing - "
			   "run InitialSetup_x64.bat; SKIPPING";
		return 0;
	}

	PoseDetector detector;
	if (!detector.load("models/person_detection.onnx", "directml"))
	{
		MIKAN_LOG_ERROR("test-posemodel") << "person detector failed to load";
		return 1;
	}
	MIKAN_LOG_INFO("test-posemodel") << "person detector loaded (ep=" << detector.activeEp() << ")";

	PoseLandmarkModel landmarkModel;
	if (!landmarkModel.load("models/pose_landmark.onnx", "directml"))
	{
		MIKAN_LOG_ERROR("test-posemodel") << "pose landmark model failed to load";
		return 1;
	}
	MIKAN_LOG_INFO("test-posemodel") << "pose landmark model loaded (ep=" << landmarkModel.activeEp() << ")";

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

	// Landmark model on a fixed central ROI (whatever the detector thought)
	PoseRoi roi;
	roi.hipCenter= glm::vec2(640.f, 400.f);
	roi.fullBodyPoint= glm::vec2(640.f, 120.f);
	PoseLandmarkResult result;
	landmarkModel.estimate(frame, roi, result);
	MIKAN_LOG_INFO("test-posemodel")
		<< "landmark model ran: valid=" << result.valid << " confidence=" << result.confidence;

	// Decode sanity when the model produced output: image points inside a
	// sane pixel envelope, visibilities in [0,1]
	if (result.valid)
	{
		for (int landmark= 0; landmark < POSE_LANDMARK_COUNT; ++landmark)
		{
			const glm::vec3& p= result.imagePoints[landmark];
			if (fabsf(p.x) > 10000.f || fabsf(p.y) > 10000.f ||
				result.visibility[landmark] < 0.f || result.visibility[landmark] > 1.f)
			{
				MIKAN_LOG_ERROR("test-posemodel") << "decoded landmark " << landmark << " out of range";
				return 1;
			}
		}
	}

	// What the stage costs per model frame. Both models together set the
	// floor for a body-pose camera: the detector runs whenever ROI tracking
	// is lost, the landmark model on every model frame. Logged rather than
	// asserted - it is a machine-dependent number, and the reason to have it
	// is to settle "is the pose stage what slowed the capture down".
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
		const double landmarkMs= timeCalls([&]() { landmarkModel.estimate(frame, roi, result); });
		MIKAN_LOG_INFO("test-posemodel")
			<< "person detector " << detectMs << " ms, landmark model " << landmarkMs << " ms";
	}

	// -- RTMPose: numeric agreement with the Python reference --
	if (!std::filesystem::exists("models/rtmpose_body.onnx"))
	{
		MIKAN_LOG_WARNING("test-posemodel") << "models/rtmpose_body.onnx missing - SKIPPING the RTMPose check";
		MIKAN_LOG_INFO("test-posemodel") << "Pose model smoke test passed";
		return 0;
	}

	RtmPoseBodyModel rtmPose;
	if (!rtmPose.load("models/rtmpose_body.onnx", "directml"))
	{
		MIKAN_LOG_ERROR("test-posemodel") << "RTMPose model failed to load";
		return 1;
	}
	MIKAN_LOG_INFO("test-posemodel") << "RTMPose model loaded (ep=" << rtmPose.activeEp() << ")";

	const cv::Mat referenceImage= cv::imread("rtm_demo.jpg", cv::IMREAD_COLOR);
	if (referenceImage.empty())
	{
		MIKAN_LOG_WARNING("test-posemodel")
			<< "rtm_demo.jpg missing - SKIPPING the numeric cross-check (re-run InitialSetup_x64.bat)";
	}
	else
	{
		RtmPoseResult rtmResult;
		rtmPose.estimate(referenceImage, glm::vec2(0.f, 0.f),
						 glm::vec2((float)referenceImage.cols, (float)referenceImage.rows), rtmResult);
		if (!rtmResult.valid)
		{
			MIKAN_LOG_ERROR("test-posemodel") << "RTMPose produced no result on the reference image";
			return 1;
		}

		// Half a pixel: the reference decodes on the same 0.5px SimCC grid,
		// so anything larger means the crop or the decode disagrees
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

		const double rtmMs= [&]() {
			const int kIterations= 30;
			rtmPose.estimate(referenceImage, glm::vec2(0.f), glm::vec2(1.f, 1.f), rtmResult); // warm up
			const auto start= std::chrono::steady_clock::now();
			for (int i= 0; i < kIterations; ++i)
			{
				rtmPose.estimate(referenceImage, glm::vec2(0.f, 0.f),
								 glm::vec2((float)referenceImage.cols, (float)referenceImage.rows), rtmResult);
			}
			return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() /
				kIterations;
		}();
		MIKAN_LOG_INFO("test-posemodel") << "RTMPose inference " << rtmMs << " ms";
	}

	MIKAN_LOG_INFO("test-posemodel") << "Pose model smoke test passed";
	return 0;
}

MIKAN_REGISTER_TEST("--test-posemodel", "BlazePose ONNX models load + synthetic-frame inference",
					eTestCategory::Hardware, runPoseModelTest);
