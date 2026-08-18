#pragma once

#include <filesystem>
#include <string>

// Fullscreen startup menu: Resume Project / New Project / Load Project / Exit.
// Drawn every frame while the app is in the MainMenu state; reports the action
// the user picked and MainWindow dispatches it to App. Runs the project-name
// modal for New Project and the native open-file dialog for Load Project.
class MainMenuScreen
{
public:
	struct Action
	{
		enum class Type
		{
			None,
			Resume,
			NewProject,
			LoadProject,
			Exit,
		};

		Type type= Type::None;
		std::string projectName;           // NewProject
		std::filesystem::path projectFile; // LoadProject
	};

	Action draw(bool bHasLastProject, const std::string& lastProjectName);

	void setStatusMessage(const std::string& message) { m_statusMessage= message; }

private:
	char m_nameBuffer[64]= {};
	std::string m_statusMessage;
	bool m_bNameModalRequested= false;
};
