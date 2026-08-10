#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "nlohmann/json.hpp"

#include "TrackingRecording.h"

// Asynchronous JSONL writer for tracking recordings. The vision thread does a
// POD move into a bounded queue per fused frame (~10 KB, no serialization);
// the writer thread turns frames into JSON lines and writes them.
//
// Queue overflow ABORTS the recording rather than dropping frames: a silent
// gap would break replay determinism for every subsequent frame, which is
// strictly worse than an honest truncated file.
//
// THREAD AFFINITY: start/enqueueFrame/stop are called from the vision thread;
// the getters are safe from any thread (atomics).
class TrackingRecorder
{
public:
	~TrackingRecorder() { stop(); }

	// Opens the file, writes the header line, spawns the writer thread.
	// Returns false (and records nothing) when the file cannot be opened.
	bool start(const std::string& filePath, const nlohmann::json& headerJson);

	// Queues one frame for writing. On queue overflow the recording aborts
	// (isRecording() flips false; the footer says aborted).
	void enqueueFrame(RecordedFrame&& frame);

	// Flushes the queue, writes the footer, joins the writer thread. Safe to
	// call when not recording.
	void stop(bool bAborted= false, const std::string& reason= "");

	bool isRecording() const { return m_bRecording; }
	int64_t getFrameCount() const { return m_frameCount; }
	uint64_t getBytesWritten() const { return m_bytesWritten; }
	bool wasAborted() const { return m_bAborted; }
	const std::string& getFilePath() const { return m_filePath; }

private:
	void writerLoop();

	// ~20s at 120 iterations/s; the writer only falls behind on disk stalls
	static constexpr size_t k_maxQueuedFrames= 2048;

	std::string m_filePath;
	std::ofstream m_file;

	std::thread m_writerThread;
	std::mutex m_queueMutex;
	std::condition_variable m_queueCondition;
	std::deque<RecordedFrame> m_queue;
	bool m_bWriterShouldExit= false;

	std::atomic_bool m_bRecording{false};
	std::atomic<int64_t> m_frameCount{0};
	std::atomic<uint64_t> m_bytesWritten{0};
	std::atomic_bool m_bAborted{false};
	std::atomic_bool m_bOverflowed{false};
};
