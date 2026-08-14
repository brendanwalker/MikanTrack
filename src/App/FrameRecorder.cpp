#include "FrameRecorder.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include "opencv2/imgcodecs.hpp"

#include "Logger.h"

std::string FrameRecorder::makeFramePath(const std::string& frameDirectory, int cameraIndex, int64_t frameIndex)
{
	char name[64];
	snprintf(name, sizeof(name), "cam%d_%lld.jpg", cameraIndex, (long long)frameIndex);
	return frameDirectory + "/" + name;
}

bool FrameRecorder::start(const std::string& frameDirectory, int jpegQuality)
{
	stop();

	std::error_code ec;
	std::filesystem::create_directories(frameDirectory, ec);
	if (ec)
	{
		MIKAN_MT_LOG_ERROR("FrameRecorder::start")
			<< "Failed to create " << frameDirectory << ": " << ec.message();
		return false;
	}

	m_directory= frameDirectory;
	m_jpegQuality= std::clamp(jpegQuality, 40, 100);
	m_framesWritten= 0;
	m_framesDropped= 0;
	m_bytesWritten= 0;
	m_bWriterShouldExit= false;
	m_bRecording= true;

	m_writerThread= std::thread(&FrameRecorder::writerLoop, this);

	MIKAN_MT_LOG_INFO("FrameRecorder::start")
		<< "Recording raw frames to " << frameDirectory << " (JPEG quality " << m_jpegQuality << ")";
	return true;
}

void FrameRecorder::enqueueFrame(int cameraIndex, int64_t frameIndex, const cv::Mat& bgrFrame)
{
	if (!m_bRecording || bgrFrame.empty())
		return;

	{
		std::lock_guard<std::mutex> lock(m_queueMutex);
		if (m_queue.size() >= k_maxQueuedFrames)
		{
			// Frames are evidence, not pipeline inputs: losing one costs
			// coverage of a moment, never the correctness of a replay
			m_framesDropped++;
			return;
		}

		QueuedFrame queued;
		queued.cameraIndex= cameraIndex;
		queued.frameIndex= frameIndex;
		// Cloned on this thread: the capture buffer is reused immediately
		queued.image= bgrFrame.clone();
		m_queue.push_back(std::move(queued));
	}
	m_queueCondition.notify_one();
}

void FrameRecorder::stop()
{
	if (!m_writerThread.joinable())
	{
		m_bRecording= false;
		return;
	}

	{
		std::lock_guard<std::mutex> lock(m_queueMutex);
		m_bWriterShouldExit= true;
	}
	m_queueCondition.notify_one();
	m_writerThread.join();
	m_bRecording= false;

	MIKAN_MT_LOG_INFO("FrameRecorder::stop")
		<< "Wrote " << m_framesWritten.load() << " frames (" << (m_bytesWritten.load() / (1024 * 1024))
		<< " MB) to " << m_directory
		<< (m_framesDropped.load() > 0
				? ", DROPPED " + std::to_string(m_framesDropped.load()) + " (encoder behind)"
				: "");
}

void FrameRecorder::writerLoop()
{
	const std::vector<int> encodeParams= {cv::IMWRITE_JPEG_QUALITY, m_jpegQuality};
	std::vector<uint8_t> encoded;

	for (;;)
	{
		QueuedFrame queued;
		{
			std::unique_lock<std::mutex> lock(m_queueMutex);
			m_queueCondition.wait(lock, [this] { return !m_queue.empty() || m_bWriterShouldExit; });
			if (m_queue.empty())
			{
				if (m_bWriterShouldExit)
					return;
				continue;
			}
			queued= std::move(m_queue.front());
			m_queue.pop_front();
		}

		encoded.clear();
		if (!cv::imencode(".jpg", queued.image, encoded, encodeParams))
		{
			m_framesDropped++;
			continue;
		}

		const std::string path= makeFramePath(m_directory, queued.cameraIndex, queued.frameIndex);
		std::ofstream file(path, std::ios::binary | std::ios::trunc);
		if (!file.is_open())
		{
			m_framesDropped++;
			continue;
		}
		file.write((const char*)encoded.data(), (std::streamsize)encoded.size());
		file.close();

		m_framesWritten++;
		m_bytesWritten+= encoded.size();
	}
}
