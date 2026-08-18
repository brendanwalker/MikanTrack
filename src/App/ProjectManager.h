#pragma once

#include <filesystem>
#include <string>

class AppConfig;
class GlobalSettings;

// Project lifecycle: creating project folders under
// %USERPROFILE%/Documents/MikanTrack, loading a project.json into the app's
// AppConfig in place (the UI and the vision thread hold raw pointers to that
// object), and migrating the legacy whole-app config into a project on first
// run. Callers stop the vision thread before any load: the thread reads the
// AppConfig without synchronization.
class ProjectManager
{
public:
	ProjectManager(GlobalSettings* globalSettings, AppConfig* config);

	static std::filesystem::path getProjectsRootDirectory();
	static bool isValidProjectName(const std::string& name, std::string& outError);
	// First folder name of "base", "base 2", "base 3", ... not taken under
	// the projects root
	static std::string findAvailableProjectName(const std::string& baseName);
	// Loads the last-used project's config for headless tools: settings file
	// to lastProjectPath to config. Returns false (config keeps defaults)
	// when any step is missing.
	static bool loadActiveProjectConfig(AppConfig& config);

	// Creates <root>/<name>/{project.json, recordings/, dumps/} with the
	// config's current state as the initial file contents. Does not load it.
	bool createProject(const std::string& name, std::filesystem::path& outProjectFile);
	// Resets the AppConfig to defaults in place, loads the file into it,
	// points the process project directory at its folder, and records it as
	// the last project
	bool loadProject(const std::filesystem::path& projectFile);
	// One-time pivot of the legacy %APPDATA% whole-app config into a project
	bool migrateLegacyConfigIfNeeded();

	// Removes a project folder from disk. Refuses anything that is not a
	// project.json directly under the projects root, so a stray path can
	// never turn into a recursive delete somewhere else.
	static bool deleteProject(const std::filesystem::path& projectFile);

	// Folder name of the loaded project (empty when none)
	std::string getActiveProjectName() const;

private:
	GlobalSettings* m_globalSettings= nullptr;
	AppConfig* m_config= nullptr;
};
