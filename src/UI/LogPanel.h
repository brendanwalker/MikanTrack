#pragma once

#include <deque>
#include <mutex>
#include <string>
#include <vector>

// Thread-safe ring buffer of log lines fed from the Logger callback,
// rendered as a collapsible ImGui panel.
class LogPanel
{
public:
	static LogPanel& getInstance();

	// Registered as the Logger's t_logCallback (may be called from any thread)
	static void logCallback(int level, const char* line);

	void draw(bool* pOpen);

private:
	struct LogLine
	{
		int level;
		std::string text;
	};

	void addLine(int level, const char* line);

	std::mutex m_mutex;
	std::deque<LogLine> m_lines;
	bool m_bAutoScroll= true;

	static constexpr size_t k_maxLines= 2000;
};
