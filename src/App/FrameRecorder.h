#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

#include "opencv2/core/mat.hpp"

// Asynchronous JPEG writer for the raw camera frames behind a tracking
// recording. OFF BY DEFAULT AND OPT-IN: a landmark recording is abstract
// enough to share, whereas frames are video of a room, so this is a local
// choice rather than something a shared build does silently.
//
// Frames are what make a MODEL comparison replayable. The landmark recording
// replays every stage after inference, which is why it can be checksummed;
// it cannot answer "would a different pose model have done better", because
// the model's input is gone. With frames on disk, that becomes an offline
// measurement instead of a live impression.
//
// The vision thread hands over a cloned frame and the writer thread encodes
// it. Unlike TrackingRecorder, a queue overflow DROPS frames rather than
// aborting: frames are not inputs to the replayed pipeline, so a gap costs
// coverage rather than correctness. Drops are counted and logged.
//
// THREAD AFFINITY: start/enqueueFrame/stop are called from the vision thread;
// the getters are safe from any thread (atomics).
class FrameRecorder
{
public:
	~FrameRecorder() { stop(); }

	// Creates the frame directory and spawns the writer thread. Returns false
	// (and records nothing) if the directory cannot be created.
	bool start(const std::string& frameDirectory, int jpegQuality);

	// Queues one camera frame. The frame is cloned here, on the caller's
	// thread, because the source buffer is reused for the next capture.
	void enqueueFrame(int cameraIndex, int64_t frameIndex, const cv::Mat& bgrFrame);

	void stop();

	bool isRecording() const { return m_bRecording; }
	int64_t getFramesWritten() const { return m_framesWritten; }
	int64_t getFramesDropped() const { return m_framesDropped; }
	uint64_t getBytesWritten() const { return m_bytesWritten; }
	const std::string& getDirectory() const { return m_directory; }

	// <frameDirectory>/cam<N>_<frameIndex>.jpg - the same frameIndex the
	// JSONL records per camera, so a frame and its landmarks join up
	static std::string makeFramePath(const std::string& frameDirectory, int cameraIndex, int64_t frameIndex);

private:
	struct QueuedFrame
	{
		int cameraIndex= -1;
		int64_t frameIndex= -1;
		cv::Mat image;
	};

	void writerLoop();

	// Encoding a 720p JPEG costs a few ms, so the queue only has to absorb
	// bursts; past this the recording keeps running and loses frames
	static constexpr size_t k_maxQueuedFrames= 64;

	std::string m_directory;
	int m_jpegQuality= 85;

	std::thread m_writerThread;
	std::mutex m_queueMutex;
	std::condition_variable m_queueCondition;
	std::deque<QueuedFrame> m_queue;
	bool m_bWriterShouldExit= false;

	std::atomic_bool m_bRecording{false};
	std::atomic<int64_t> m_framesWritten{0};
	std::atomic<int64_t> m_framesDropped{0};
	std::atomic<uint64_t> m_bytesWritten{0};
};
