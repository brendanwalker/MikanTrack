#pragma once

#include <filesystem>
#include <string>

// App-global settings shared across projects, stored as JSON at
// %APPDATA%/MikanTrack/config.json. Per-project tracking settings live in each
// project's project.json (see AppConfig). This file previously held the whole
// AppConfig; ProjectManager migrates that legacy schema into a project on
// first run.
class GlobalSettings
{
public:
	std::filesystem::path lastProjectPath;

	bool hasLastProjectPath() const { return !lastProjectPath.empty(); }

	// Returns false (keeping defaults) when the file is missing, corrupt, or
	// still carries the legacy AppConfig schema
	bool load();
	bool save() const;

	static std::string getSettingsFilePath();
};
