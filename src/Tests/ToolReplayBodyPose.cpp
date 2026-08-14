#include "TestCommon.h"

#include "BodyPoseTracker.h"
#include "FrameRecorder.h"

// Re-runs a body-pose backend over the RAW FRAMES stored beside a recording,
// so "which pose model is better on my rig" is a measurement rather than an
// impression. The landmark recording alone cannot answer it: it replays every
// stage after inference, but the model's input is gone.
//
// Frames are only present when raw frame recording was turned on for that
// session (opt-in, off by default).
//
// usage: --replay-bodypose <recording.jsonl> [blazepose|rtmpose] [camera]

namespace
{
struct JointStats
{
	const char* name;
	ePoseLandmark landmark;
	std::vector<float> scores;
	std::vector<float> steps; // px between consecutive model results
	glm::vec2 lastPoint{0.f};
	bool bHaveLast= false;
};

float median(std::vector<float>& values)
{
	if (values.empty())
		return 0.f;
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

float percentile(std::vector<float>& values, float fraction)
{
	if (values.empty())
		return 0.f;
	std::sort(values.begin(), values.end());
	return values[std::min((size_t)(values.size() * fraction), values.size() - 1)];
}
} // namespace

static int runReplayBodyPose(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("replay-bodypose")
			<< "usage: --replay-bodypose <recording.jsonl> [blazepose|rtmpose] [cameraIndex]";
		return 1;
	}

	const std::string recordingPath= args[0];
	eBodyPoseBackend backend= eBodyPoseBackend::RtmPose;
	if (args.size() > 1)
	{
		std::string name= args[1];
		std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return (char)tolower(c); });
		if (name == "blazepose")
			backend= eBodyPoseBackend::BlazePose;
		else if (name != "rtmpose")
		{
			MIKAN_LOG_ERROR("replay-bodypose") << "Unknown backend '" << args[1] << "'";
			return 1;
		}
	}
	const int requestedCamera= args.size() > 2 ? atoi(args[2].c_str()) : -1;

	TrackingReplay replay;
	std::string error;
	if (!replay.load(recordingPath, error))
	{
		MIKAN_LOG_ERROR("replay-bodypose") << "Load failed: " << error;
		return 1;
	}

	const std::string frameDirectory= TrackingRecording::makeFrameDirectoryPath(recordingPath);
	if (!std::filesystem::exists(frameDirectory))
	{
		MIKAN_LOG_ERROR("replay-bodypose")
			<< "No frames beside this recording (" << frameDirectory
			<< "). Enable 'Also record raw camera frames' in the Timeline panel and record again.";
		return 1;
	}

	// Default to whichever camera actually ran body pose in that session
	int cameraIndex= requestedCamera;
	if (cameraIndex < 0)
	{
		for (int i= 0; i < (int)replay.getRecordedConfig().cameraCount(); ++i)
		{
			if (replay.getRecordedConfig().camera(i).bodyPose.enabled)
			{
				cameraIndex= i;
				break;
			}
		}
	}
	if (cameraIndex < 0)
	{
		MIKAN_LOG_ERROR("replay-bodypose") << "No camera had body pose enabled; pass a camera index";
		return 1;
	}

	BodyPoseTrackerConfig trackerConfig;
	trackerConfig.backend= backend;
	// Every stored frame is fed through: the recorded divider already decided
	// which frames exist, and re-dividing here would just sample them again
	trackerConfig.frameDivider= 1;
	trackerConfig.detectorIntervalFrames=
		replay.getRecordedConfig().camera(cameraIndex).bodyPose.detectorIntervalFrames;

	BodyPoseTracker tracker;
	if (!tracker.load("models", "directml", trackerConfig))
	{
		MIKAN_LOG_ERROR("replay-bodypose") << "Failed to load models for " << bodyPoseBackendName(backend);
		return 1;
	}

	MIKAN_LOG_INFO("replay-bodypose")
		<< "Backend " << bodyPoseBackendName(backend) << " (ep=" << tracker.activeEp() << "), camera "
		<< cameraIndex << ", frames from " << frameDirectory;

	std::vector<JointStats> joints= {
		{"shoulderL", ePoseLandmark::LEFT_SHOULDER}, {"shoulderR", ePoseLandmark::RIGHT_SHOULDER},
		{"elbowL", ePoseLandmark::LEFT_ELBOW},       {"elbowR", ePoseLandmark::RIGHT_ELBOW},
		{"wristL", ePoseLandmark::LEFT_WRIST},       {"wristR", ePoseLandmark::RIGHT_WRIST},
		{"nose", ePoseLandmark::NOSE},
	};

	int framesFound= 0;
	int framesMissing= 0;
	int observationsValid= 0;
	int boxSourceCounts[4]= {0, 0, 0, 0};
	std::vector<float> inferenceMs;

	for (int frameIndex= 0; frameIndex < replay.getFrameCount(); ++frameIndex)
	{
		const RecordedFrame& recorded= *replay.getFrame(frameIndex).recorded;
		for (const RecordedCameraInput& fresh : recorded.freshCameras)
		{
			if (fresh.cameraIndex != cameraIndex)
				continue;

			const std::string framePath=
				FrameRecorder::makeFramePath(frameDirectory, cameraIndex, fresh.frameIndex);
			const cv::Mat image= cv::imread(framePath, cv::IMREAD_COLOR);
			if (image.empty())
			{
				framesMissing++;
				continue;
			}
			framesFound++;

			BodyPoseObservation observation;
			const auto start= std::chrono::steady_clock::now();
			tracker.process(image, observation);
			inferenceMs.push_back(
				(float)std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
					.count());

			boxSourceCounts[(int)observation.boxSource]++;
			if (!observation.valid)
				continue;
			observationsValid++;

			for (JointStats& joint : joints)
			{
				const int index= (int)joint.landmark;
				if (!observation.isProvided(index))
					continue;
				joint.scores.push_back(observation.visibility[index]);

				const glm::vec2 point(observation.imagePoints[index]);
				if (joint.bHaveLast)
					joint.steps.push_back(glm::length(point - joint.lastPoint));
				joint.lastPoint= point;
				joint.bHaveLast= true;
			}
		}
	}

	if (framesFound == 0)
	{
		MIKAN_LOG_ERROR("replay-bodypose") << "No frames matched camera " << cameraIndex;
		return 1;
	}

	MIKAN_LOG_INFO("replay-bodypose")
		<< "frames " << framesFound << " (missing " << framesMissing << "), observation valid on "
		<< observationsValid << " (" << (100 * observationsValid / std::max(framesFound, 1)) << "%)";
	MIKAN_LOG_INFO("replay-bodypose")
		<< "box source: detector " << boxSourceCounts[(int)eBodyBoxSource::Detector] << ", tracked "
		<< boxSourceCounts[(int)eBodyBoxSource::Tracked] << ", full frame "
		<< boxSourceCounts[(int)eBodyBoxSource::FullFrame] << ", none "
		<< boxSourceCounts[(int)eBodyBoxSource::None];
	MIKAN_LOG_INFO("replay-bodypose") << "inference median " << median(inferenceMs) << " ms";

	MIKAN_LOG_INFO("replay-bodypose") << "per-joint score and 2D step between model results:";
	for (JointStats& joint : joints)
	{
		if (joint.scores.empty())
		{
			MIKAN_LOG_INFO("replay-bodypose") << "  " << joint.name << ": not emitted by this backend";
			continue;
		}
		const float scoreMedian= median(joint.scores);
		const int aboveGate= (int)std::count_if(joint.scores.begin(), joint.scores.end(),
												[](float s) { return s >= 0.5f; });
		MIKAN_LOG_INFO("replay-bodypose")
			<< "  " << joint.name << ": score med " << scoreMedian << ", >=0.5 on "
			<< (100 * aboveGate / (int)joint.scores.size()) << "%, step med " << median(joint.steps)
			<< " px p90 " << percentile(joint.steps, 0.9f) << " px";
	}

	return 0;
}

MIKAN_REGISTER_TEST("--replay-bodypose", "Re-run a body-pose backend over a recording's raw frames",
					eTestCategory::Tool, runReplayBodyPose);
