#include "WorkerThread.h"
#include "ThreadUtils.h"
#include "Logger.h"

class WorkerThreadImpl
{
public:
	// Multithreaded state
	const std::string threadName;
	std::atomic_bool exitSignaled;
	std::atomic_bool threadEnded;

	// Main Thread State
	bool threadStarted;
	std::thread workerThread;

	WorkerThreadImpl(const std::string thread_name)
		: threadName(thread_name)
		, exitSignaled({false})
		, threadEnded({false})
		, threadStarted(false)
		, workerThread()
	{
	}
};

WorkerThread::WorkerThread(const std::string thread_name)
	: m_impl(new WorkerThreadImpl(thread_name))
{
}

WorkerThread::~WorkerThread() { delete m_impl; }

bool WorkerThread::hasThreadStarted() const { return m_impl->threadStarted; }

bool WorkerThread::hasThreadEnded() const { return m_impl->threadEnded.load(); }

void WorkerThread::startThread()
{
	if (!m_impl->threadStarted)
	{
		m_impl->threadEnded.store(false);
		m_impl->exitSignaled= false;

		MIKAN_LOG_INFO("WorkerThread::start") << "Starting worker thread: " << m_impl->threadName;
		onThreadStarted();

		m_impl->workerThread= std::thread(&WorkerThread::threadFunc, this);
		m_impl->threadStarted= true;
	}
}

void WorkerThread::stopThread()
{
	if (m_impl->threadStarted)
	{
		if (!m_impl->exitSignaled)
		{
			MIKAN_LOG_INFO("WorkerThread::stop") << "Stopping worker thread: " << m_impl->threadName;
			// Set the atomic exit flag
			m_impl->exitSignaled.store(true);

			// Give the thread a chance to set any state in response to the exit flag getting set
			onThreadHaltBegin();

			// Block until the worker thread exists
			m_impl->workerThread.join();

			MIKAN_LOG_INFO("WorkerThread::stop") << "Worker thread stopped: " << m_impl->threadName;
			onThreadHaltComplete();
		}
		else
		{
			MIKAN_LOG_INFO("WorkerThread::stop") << "Worker thread already stopped: " << m_impl->threadName;
		}

		m_impl->threadStarted= false;
		m_impl->exitSignaled= false;
	}
}

void WorkerThread::threadFunc()
{
	ThreadUtils::setCurrentThreadName(m_impl->threadName.c_str());

	// Stay in the poll loop until asked to exit by the main thread
	while (!m_impl->exitSignaled)
	{
		if (!doWork())
		{
			break;
		}
	}

	m_impl->threadEnded.store(true);
}