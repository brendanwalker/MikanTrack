#include "LogPanel.h"

#include "imgui.h"

#include "LocText.h"

#include "Logger.h"

LogPanel& LogPanel::getInstance()
{
	static LogPanel instance;
	return instance;
}

void LogPanel::logCallback(int level, const char* line)
{
	getInstance().addLine(level, line);
}

void LogPanel::addLine(int level, const char* line)
{
	std::lock_guard<std::mutex> lock(m_mutex);
	m_lines.push_back({level, std::string(line)});
	while (m_lines.size() > k_maxLines)
		m_lines.pop_front();
}

void LogPanel::draw(bool* pOpen)
{
	if (!ImGui::Begin(locWindowTitle("windows.log"), pOpen))
	{
		ImGui::End();
		return;
	}

	if (ImGui::Button(locLabel("logPanel.clearButton")))
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_lines.clear();
	}
	ImGui::SameLine();
	ImGui::Checkbox(locLabel("logPanel.autoScrollCheckbox"), &m_bAutoScroll);

	ImGui::Separator();
	ImGui::BeginChild("LogScroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

	{
		std::lock_guard<std::mutex> lock(m_mutex);
		for (const LogLine& line : m_lines)
		{
			ImVec4 color(0.8f, 0.8f, 0.8f, 1.f);
			switch ((LogSeverityLevel)line.level)
			{
				case LogSeverityLevel::warning:
					color= ImVec4(1.f, 0.85f, 0.3f, 1.f);
					break;
				case LogSeverityLevel::error:
				case LogSeverityLevel::fatal:
					color= ImVec4(1.f, 0.4f, 0.4f, 1.f);
					break;
				case LogSeverityLevel::debug:
				case LogSeverityLevel::trace:
					color= ImVec4(0.6f, 0.6f, 0.6f, 1.f);
					break;
				default:
					break;
			}
			ImGui::PushStyleColor(ImGuiCol_Text, color);
			ImGui::TextUnformatted(line.text.c_str());
			ImGui::PopStyleColor();
		}
	}

	if (m_bAutoScroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.f)
		ImGui::SetScrollHereY(1.0f);

	ImGui::EndChild();
	ImGui::End();
}
