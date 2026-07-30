//-- includes -----
#include "Logger.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <ostream>

#ifdef _MSC_VER
#pragma warning(disable : 4996) // 'This function or variable may be unsafe': localtime
#endif

#ifdef WIN32
#include <windows.h>
#endif

//-- globals -----
bool g_is_initialized= false;
LogSeverityLevel g_min_log_level= LogSeverityLevel::info;
bool g_owns_win32_console= false;
std::ostream* g_file_stream= nullptr;
std::mutex* g_logger_mutex= nullptr;
t_logCallback g_logger_callback= nullptr;

void log_default_callback(int log_level, const char* line)
{
	// Always emit to the standard streams. For console apps (e.g. MikanCmd)
	// these are the process's existing stdout/stderr; for a Win32 GUI app
	// that opted into a console via enable_console, RedirectIOToConsole() has
	// already pointed them at the allocated console. When a GUI app runs
	// without a console the writes are simply discarded.
	if (log_level >= (int)LogSeverityLevel::error)
		std::cerr << line << std::endl;
	else
		std::cout << line << std::endl;

	if (g_file_stream != nullptr)
	{
		*g_file_stream << line << std::endl;
	}
}

//-- public implementation -----
#ifdef WIN32
bool RedirectIOToConsole()
{
	if (GetConsoleWindow() == NULL)
	{
		if (!AllocConsole())
		{
			return false;
		}
	}

	FILE* fDummy;
	freopen_s(&fDummy, "CONOUT$", "w", stdout);
	freopen_s(&fDummy, "CONOUT$", "w", stderr);
	freopen_s(&fDummy, "CONIN$", "r", stdin);
	std::cout.clear();
	std::clog.clear();
	std::cerr.clear();
	std::cin.clear();

	return true;
}
#endif // WIN32

void log_init(const LoggerSettings& settings)
{
	if (!g_is_initialized)
	{
		g_min_log_level= settings.min_log_level;

		// enable_console controls whether a Win32 GUI process (which has no
		// attached console) allocates one and redirects stdout/stderr to it.
		// Console apps and non-Windows platforms already have valid standard
		// streams, so this is a no-op there; logging to the standard streams
		// happens regardless in log_default_callback.
		if (settings.enable_console)
		{
#ifdef WIN32
			if (RedirectIOToConsole())
			{
				g_owns_win32_console= true;
			}
#endif
		}

		if (settings.log_filename.length() > 0)
		{
			g_file_stream= new std::ofstream(settings.log_filename, std::ofstream::out);
		}

		if (settings.log_callback != nullptr)
		{
			g_logger_callback= settings.log_callback;
		}
		else
		{
			g_logger_callback= log_default_callback;
		}

		g_logger_mutex= new std::mutex();

		g_is_initialized= true;
	}
}

void log_dispose()
{
	if (g_file_stream != nullptr)
	{
		delete g_file_stream;
		g_file_stream= nullptr;
	}

	if (g_logger_mutex != nullptr)
	{
		delete g_logger_mutex;
		g_logger_mutex= nullptr;
	}

	if (g_owns_win32_console)
	{
#ifdef WIN32
		FreeConsole();
#endif
		g_owns_win32_console= false;
	}

	g_is_initialized= false;
}

bool log_can_emit_level(LogSeverityLevel level) { return (level >= g_min_log_level); }

std::string log_get_timestamp_prefix()
{
	auto now= std::chrono::system_clock::now();
	auto seconds= std::chrono::time_point_cast<std::chrono::seconds>(now);
	auto milliseconds= std::chrono::duration_cast<std::chrono::milliseconds>(now - seconds);
	time_t in_time_t= std::chrono::system_clock::to_time_t(now);

	std::stringstream ss;
	ss << "[" << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") << "." << milliseconds.count() << "]: ";

	return ss.str();
}

//-- member functions -----
class LoggerStreamImpl
{
private:
	std::ostringstream m_lineBuffer;
	LogSeverityLevel m_level;
	bool m_hasWrittenLog;

public:
	LoggerStreamImpl(LogSeverityLevel level)
		: m_lineBuffer()
		, m_level(level)
		, m_hasWrittenLog(false)
	{
	}

	template <class T>
	void operator<<(const T& x)
	{
		if (log_can_emit_level(m_level))
		{
			m_lineBuffer << x;
			m_hasWrittenLog= true;
		}
	}

	void write_line()
	{
		if (g_is_initialized && g_logger_callback != nullptr && m_hasWrittenLog && log_can_emit_level(m_level))
		{
			const std::string line= m_lineBuffer.str();

			(*g_logger_callback)((int)m_level, line.c_str());
		}
	}
};

LoggerStream::LoggerStream(LogSeverityLevel level)
	: m_impl(new LoggerStreamImpl(level))
{
}

LoggerStream::~LoggerStream()
{
	write_line();
	delete m_impl;
}

void LoggerStream::write_line() { m_impl->write_line(); }

// Wrapper for forwarding value
LoggerStream& LoggerStream::operator<<(bool value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(char value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(short value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(unsigned short value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(int value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(unsigned int value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(long value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(unsigned long value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(long long value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(unsigned long long value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(float value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(double value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(long double value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(const void* value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(const char* value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(const std::string& value)
{
	*m_impl << value;
	return *this;
}
LoggerStream& LoggerStream::operator<<(const std::filesystem::path& value)
{
	*m_impl << value;
	return *this;
}

ThreadSafeLoggerStream::ThreadSafeLoggerStream(LogSeverityLevel level)
	: LoggerStream(level)
{
}

void ThreadSafeLoggerStream::write_line()
{
	if (g_is_initialized)
	{
		std::lock_guard<std::mutex> lock(*g_logger_mutex);

		LoggerStream::write_line();
	}
}