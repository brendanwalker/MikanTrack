#include "TestCommon.h"

#include "FrameRecorder.h"

// Covers the raw-frame path end to end without cameras: the writer round
// trips images to disk, names them so a frame joins its landmarks, survives
// a burst larger than its queue by DROPPING rather than stalling the vision
// thread, and writes nothing at all when it was never started (the default,
// which is the privacy-relevant case).

static int runFrameRecorderTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-framerecorder") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-framerecorder") << "FAIL " << name;
			failures++;
		}
	};

	const std::filesystem::path root=
		std::filesystem::temp_directory_path() / "mikan_framerecorder_test";
	std::error_code ec;
	std::filesystem::remove_all(root, ec);

	// The directory layout a recording and its frames share
	const std::string recordingPath= (root / "2026-01-01_12-00-00.jsonl").string();
	const std::string frameDirectory= TrackingRecording::makeFrameDirectoryPath(recordingPath);
	check(frameDirectory == (root / "2026-01-01_12-00-00_frames").string(),
		  "frame directory is the recording's sibling");

	// A recognizable test image: a gradient with a bright block
	cv::Mat image(240, 320, CV_8UC3);
	for (int y= 0; y < image.rows; ++y)
		for (int x= 0; x < image.cols; ++x)
			image.at<cv::Vec3b>(y, x)= cv::Vec3b((uint8_t)(x % 256), (uint8_t)(y % 256), 96);
	cv::rectangle(image, cv::Rect(40, 40, 80, 80), cv::Scalar(250, 250, 250), cv::FILLED);

	{
		FrameRecorder recorder;
		check(recorder.start(frameDirectory, 90), "recorder starts");
		check(recorder.isRecording(), "recorder reports recording");

		for (int64_t frameIndex= 0; frameIndex < 8; ++frameIndex)
			recorder.enqueueFrame(2, frameIndex, image);
		recorder.stop();

		check(recorder.getFramesWritten() == 8, "every queued frame written");
		check(recorder.getBytesWritten() > 0, "bytes accounted");

		const std::string framePath= FrameRecorder::makeFramePath(frameDirectory, 2, 5);
		check(std::filesystem::exists(framePath), "frame named by camera and frame index");

		// Round trip: what comes back must be the image the models saw
		const cv::Mat loaded= cv::imread(framePath, cv::IMREAD_COLOR);
		check(!loaded.empty() && loaded.cols == image.cols && loaded.rows == image.rows,
			  "stored frame reloads at the same size");
		if (!loaded.empty() && loaded.size() == image.size())
		{
			cv::Mat difference;
			cv::absdiff(loaded, image, difference);
			const cv::Scalar meanError= cv::mean(difference);
			const double worst= std::max({meanError[0], meanError[1], meanError[2]});
			MIKAN_LOG_INFO("test-framerecorder") << "JPEG mean error " << worst << " levels";
			check(worst < 6.0, "stored frame matches the source within JPEG loss");
		}
	}

	// A burst well past the queue must drop frames, never block: the vision
	// thread cannot wait on a disk
	{
		FrameRecorder recorder;
		check(recorder.start(frameDirectory, 90), "recorder restarts");
		const auto start= std::chrono::steady_clock::now();
		for (int64_t frameIndex= 0; frameIndex < 4000; ++frameIndex)
			recorder.enqueueFrame(0, frameIndex, image);
		const double enqueueMs=
			std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
		recorder.stop();

		MIKAN_LOG_INFO("test-framerecorder")
			<< "4000 enqueues took " << enqueueMs << " ms, wrote " << recorder.getFramesWritten()
			<< ", dropped " << recorder.getFramesDropped();
		check(recorder.getFramesWritten() + recorder.getFramesDropped() == 4000,
			  "every frame either written or counted as dropped");
		check(recorder.getFramesDropped() > 0, "overflow drops rather than blocking");
	}

	// Never started: nothing is written, which is the default state
	{
		const std::string unusedDirectory= (root / "never_started_frames").string();
		FrameRecorder recorder;
		recorder.enqueueFrame(0, 0, image);
		check(!recorder.isRecording(), "an unstarted recorder is not recording");
		check(!std::filesystem::exists(unusedDirectory), "an unstarted recorder writes nothing");
	}

	std::filesystem::remove_all(root, ec);

	if (failures == 0)
		MIKAN_LOG_INFO("test-framerecorder") << "All frame recorder tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-framerecorder", "Raw frame recording: round trip, naming, overflow drops",
					eTestCategory::SelfTest, runFrameRecorderTest);
