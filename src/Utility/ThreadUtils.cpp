// -- includes -----
#include "ThreadUtils.h"

#include <thread>

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h>

#else
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#if defined __MACH__ && defined __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#endif
#define MILLISECONDS_TO_NANOSECONDS 1000000
#endif

// -- public methods -----
namespace ThreadUtils
{
std::thread::id MAIN_THREAD_ID;

void initMainThreadId() { MAIN_THREAD_ID= std::this_thread::get_id(); }

bool isRunningInMainThread() { return MAIN_THREAD_ID == std::this_thread::get_id(); }

#if defined WIN32 || defined _WIN32 || defined WINCE
const DWORD MS_VC_EXCEPTION= 0x406D1388;

#pragma pack(push, 8)
typedef struct tagTHREADNAME_INFO
{
	DWORD dwType;     // Must be 0x1000.
	LPCSTR szName;    // Pointer to name (in user addr space).
	DWORD dwThreadID; // Thread ID (-1=caller thread).
	DWORD dwFlags;    // Reserved for future use, must be zero.
} THREADNAME_INFO;
#pragma pack(pop)

void setCurrentThreadName(const char* thread_name)
{
	THREADNAME_INFO info;
	info.dwType= 0x1000;
	info.szName= thread_name;
	info.dwThreadID= GetCurrentThreadId();
	info.dwFlags= 0;

	__try
	{
		RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(ULONG_PTR), (ULONG_PTR*)&info);
	}
	__except (EXCEPTION_EXECUTE_HANDLER)
	{
	}
}
#else
void setCurrentThreadName(const char* threadName)
{
	// Not sure how to implement this on linux/osx, so left empty...
}
#endif

void sleepMilliseconds(int milliseconds)
{
#ifdef _MSC_VER
	Sleep(milliseconds);
#else
	struct timespec req= {0};
	req.tv_sec= 0;
	req.tv_nsec= milliseconds * MILLISECONDS_TO_NANOSECONDS;
	nanosleep(&req, (struct timespec*)NULL);
#endif
}
}; // namespace ThreadUtils