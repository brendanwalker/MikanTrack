#include "TrackingRecorder.h"

#include "Logger.h"

using json= nlohmann::json;

bool TrackingRecorder::start(const std::string& filePath, const json& headerJson)
{
	stop();

	m_file.open(filePath, std::ios::out | std::ios::trunc);
	if (!m_file.is_open())
	{
		MIKAN_MT_LOG_ERROR("TrackingRecorder") << "Failed to open " << filePath;
		return false;
	}

	m_filePath= filePath;
	m_frameCount= 0;
	m_bytesWritten= 0;
	m_bAborted= false;
	m_bOverflowed= false;
	m_bWriterShouldExit= false;

	const std::string headerLine= headerJson.dump();
	m_file << headerLine << '\n';
	m_bytesWritten= headerLine.size() + 1;

	m_bRecording= true;
	m_writerThread= std::thread([this]() { writerLoop(); });

	MIKAN_MT_LOG_INFO("TrackingRecorder") << "Recording to " << filePath;
	return true;
}

void TrackingRecorder::enqueueFrame(RecordedFrame&& frame)
{
	if (!m_bRecording)
		return;

	{
		std::lock_guard<std::mutex> lock(m_queueMutex);
		if (m_queue.size() >= k_maxQueuedFrames)
		{
			// A dropped frame silently breaks determinism for every later
			// frame - abort instead and say so
			m_bOverflowed= true;
			m_bRecording= false;
		}
		else
		{
			m_queue.push_back(std::move(frame));
		}
	}
	m_queueCondition.notify_one();

	if (m_bOverflowed)
		stop(true, "writer queue overflow");
}

void TrackingRecorder::stop(bool bAborted, const std::string& reason)
{
	if (!m_writerThread.joinable())
		return;

	m_bRecording= false;
	{
		std::lock_guard<std::mutex> lock(m_queueMutex);
		m_bWriterShouldExit= true;
	}
	m_queueCondition.notify_one();
	m_writerThread.join();

	// Writer has drained the queue; finish the file on this thread
	if (m_file.is_open())
	{
		RecordingFooter footer;
		footer.frameCount= m_frameCount;
		footer.bAborted= bAborted || m_bOverflowed;
		footer.reason= m_bOverflowed ? "writer queue overflow" : reason;

		const std::string footerLine= TrackingRecording::footerToJson(footer).dump();
		m_file << footerLine << '\n';
		m_bytesWritten+= footerLine.size() + 1;
		m_file.close();

		m_bAborted= footer.bAborted;
		MIKAN_MT_LOG_INFO("TrackingRecorder")
			<< "Finalized " << m_filePath << ": " << m_frameCount << " frames, "
			<< (m_bytesWritten / 1024) << " KB" << (footer.bAborted ? " (ABORTED)" : "");
	}
}

void TrackingRecorder::writerLoop()
{
	while (true)
	{
		RecordedFrame frame;
		{
			std::unique_lock<std::mutex> lock(m_queueMutex);
			m_queueCondition.wait(lock, [this]() { return !m_queue.empty() || m_bWriterShouldExit; });
			if (m_queue.empty())
			{
				if (m_bWriterShouldExit)
					return;
				continue;
			}
			frame= std::move(m_queue.front());
			m_queue.pop_front();
		}

		const std::string line= TrackingRecording::frameToJson(frame).dump();
		m_file << line << '\n';
		m_bytesWritten+= line.size() + 1;
		++m_frameCount;
	}
}
