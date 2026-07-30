#pragma once

#include <MikanCoreAppExport.h>

#include <atomic>
#include <string>
#include <thread>

class MIKAN_COREAPP_CLASS WorkerThread
{
public:
	WorkerThread(const std::string thread_name);
	virtual ~WorkerThread();

	bool hasThreadStarted() const;
	bool hasThreadEnded() const;
	void startThread();
	void stopThread();

protected:
	virtual void onThreadStarted() {}
	virtual void onThreadHaltBegin() {}
	virtual void onThreadHaltComplete() {}
	virtual bool doWork()= 0;

private:
	void threadFunc();

protected:
	class WorkerThreadImpl* m_impl;
};