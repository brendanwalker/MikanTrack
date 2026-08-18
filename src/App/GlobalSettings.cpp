#include "GlobalSettings.h"

#include <fstream>

#include "nlohmann/json.hpp"

#include "AppConfig.h"
#include "Logger.h"
#include "PathUtils.h"

using json= nlohmann::json;

static constexpr int k_appSettingsVersion= 1;

std::string GlobalSettings::getSettingsFilePath()
{
	// Same file the legacy whole-app config lived in; the schema pivoted to
	// app-global settings when projects arrived
	return AppConfig::getLegacyConfigFilePath();
}

bool GlobalSettings::load()
{
	const std::string path= getSettingsFilePath();
	std::ifstream file(path);
	if (!file.is_open())
		return false;

	try
	{
		json j;
		file >> j;

		// The legacy whole-app schema has no appSettingsVersion; migration
		// (ProjectManager) owns that case
		if (!j.contains("appSettingsVersion"))
			return false;

		lastProjectPath= PathUtils::utf8ToPath(j.value("lastProjectPath", std::string()));
	}
	catch (const std::exception& e)
	{
		MIKAN_LOG_ERROR("GlobalSettings::load") << "Failed to parse " << path << ": " << e.what();
		return false;
	}

	return true;
}

bool GlobalSettings::save() const
{
	const std::string path= getSettingsFilePath();
	std::ofstream file(path);
	if (!file.is_open())
	{
		MIKAN_LOG_ERROR("GlobalSettings::save") << "Failed to open " << path << " for writing";
		return false;
	}

	const json j= {
		{"appSettingsVersion", k_appSettingsVersion},
		{"lastProjectPath", PathUtils::pathToUtf8(lastProjectPath)},
	};
	file << j.dump(2);
	return true;
}
