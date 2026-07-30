#pragma once

#include "MikanCoreAppExport.h"

//-- utility methods -----
namespace ThreadUtils
{
// Store the ID of the main thread
MIKAN_COREAPP_FUNC(void) initMainThreadId();

/// Returns true if the current thread is the main thread
MIKAN_COREAPP_FUNC(bool) isRunningInMainThread();

/// Sets the name of the current thread
MIKAN_COREAPP_FUNC(void) setCurrentThreadName(const char* thread_name);

/// Sleeps the current thread for the given number of milliseconds
MIKAN_COREAPP_FUNC(void) sleepMilliseconds(int milliseconds);
}; // namespace ThreadUtils