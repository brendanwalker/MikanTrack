#include "ProjectManager.h"

#include <fstream>

#include "nlohmann/json.hpp"

#include "AppConfig.h"
#include "GlobalSettings.h"
#include "Logger.h"
#include "PathUtils.h"

using json= nlohmann::json;

ProjectManager::ProjectManager(GlobalSettings* globalSettings, AppConfig* config)
	: m_globalSettings(globalSettings)
	, m_config(config)
{
}

std::filesystem::path ProjectManager::getProjectsRootDirectory()
{
	return PathUtils::getProjectsRootDirectory();
}

bool ProjectManager::isValidProjectName(const std::string& name, std::string& outError)
{
	if (name.empty())
	{
		outError= "Project name is empty";
		return false;
	}
	if (name == "." || name == "..")
	{
		outError= "Project name is reserved";
		return false;
	}
	if (name.size() > 64)
	{
		outError= "Project name is longer than 64 characters";
		return false;
	}

	const std::string invalidChars= "<>:\"/\\|?*";
	for (const char c : name)
	{
		if (invalidChars.find(c) != std::string::npos || (unsigned char)c < 0x20)
		{
			outError= "Project name contains a character not allowed in folder names";
			return false;
		}
	}
	if (name.front() == ' ' || name.back() == ' ' || name.back() == '.')
	{
		outError= "Project name cannot begin or end with a space, or end with a period";
		return false;
	}

	outError.clear();
	return true;
}

std::string ProjectManager::findAvailableProjectName(const std::string& baseName)
{
	const std::filesystem::path root= getProjectsRootDirectory();
	std::string name= baseName;
	for (int suffix= 2; std::filesystem::exists(root / name); ++suffix)
		name= baseName + " " + std::to_string(suffix);
	return name;
}

bool ProjectManager::loadActiveProjectConfig(AppConfig& config)
{
	GlobalSettings settings;
	if (!settings.load() || !settings.hasLastProjectPath())
		return false;
	return config.load(settings.lastProjectPath);
}

bool ProjectManager::createProject(const std::string& name, std::filesystem::path& outProjectFile)
{
	std::string error;
	if (!isValidProjectName(name, error))
	{
		MIKAN_LOG_ERROR("ProjectManager::createProject")
			<< "Invalid project name '" << name << "': " << error;
		return false;
	}

	const std::filesystem::path projectDir= getProjectsRootDirectory() / name;
	if (std::filesystem::exists(projectDir))
	{
		MIKAN_LOG_ERROR("ProjectManager::createProject")
			<< "Project folder already exists: " << projectDir;
		return false;
	}

	std::error_code ec;
	std::filesystem::create_directories(projectDir / "recordings", ec);
	std::filesystem::create_directories(projectDir / "dumps", ec);
	if (!std::filesystem::is_directory(projectDir))
	{
		MIKAN_LOG_ERROR("ProjectManager::createProject")
			<< "Failed to create project folder: " << projectDir;
		return false;
	}

	outProjectFile= projectDir / "project.json";
	std::ofstream file(outProjectFile);
	if (!file.is_open())
	{
		MIKAN_LOG_ERROR("ProjectManager::createProject") << "Failed to write " << outProjectFile;
		return false;
	}
	file << m_config->toJsonString();

	MIKAN_LOG_INFO("ProjectManager::createProject") << "Created project " << outProjectFile;
	return true;
}

bool ProjectManager::loadProject(const std::filesystem::path& projectFile)
{
	if (!std::filesystem::is_regular_file(projectFile))
	{
		MIKAN_LOG_ERROR("ProjectManager::loadProject") << "No project file at " << projectFile;
		return false;
	}

	// Reset to defaults IN PLACE (the UI and vision thread hold raw pointers
	// to this object) so keys missing from the file cannot leak the previous
	// project's values
	*m_config= AppConfig();

	if (!m_config->load(projectFile))
		return false;

	PathUtils::setProjectDirectory(projectFile.parent_path());
	m_globalSettings->lastProjectPath= projectFile;
	m_globalSettings->save();

	MIKAN_LOG_INFO("ProjectManager::loadProject") << "Loaded project " << projectFile;
	return true;
}

bool ProjectManager::migrateLegacyConfigIfNeeded()
{
	const std::filesystem::path legacyPath= AppConfig::getLegacyConfigFilePath();

	json legacyJson;
	{
		std::ifstream file(legacyPath);
		if (!file.is_open())
			return false; // fresh machine, nothing to migrate

		try
		{
			file >> legacyJson;
		}
		catch (const std::exception&)
		{
			return false; // unreadable; leave it alone
		}
	}

	// The slim app-settings schema carries this key; the legacy whole-app
	// schema does not
	if (legacyJson.contains("appSettingsVersion"))
		return false;

	MIKAN_LOG_INFO("ProjectManager::migrateLegacyConfig")
		<< "Migrating legacy config at " << legacyPath << " into a project";

	const std::string name= findAvailableProjectName("Default");
	const std::filesystem::path projectDir= getProjectsRootDirectory() / name;
	std::error_code ec;
	std::filesystem::create_directories(projectDir, ec);
	if (!std::filesystem::is_directory(projectDir))
	{
		MIKAN_LOG_ERROR("ProjectManager::migrateLegacyConfig")
			<< "Failed to create " << projectDir << ", keeping the legacy config in place";
		return false;
	}

	const std::filesystem::path projectFile= projectDir / "project.json";
	std::filesystem::copy_file(legacyPath, projectFile, ec);
	if (ec)
	{
		MIKAN_LOG_ERROR("ProjectManager::migrateLegacyConfig")
			<< "Failed to copy the legacy config to " << projectFile << ": " << ec.message();
		return false;
	}

	// Bring the recordings/dumps along; on failure they just stay behind
	const std::filesystem::path legacyDir= legacyPath.parent_path();
	for (const char* folderName : {"recordings", "dumps"})
	{
		const std::filesystem::path from= legacyDir / folderName;
		const std::filesystem::path to= projectDir / folderName;
		if (std::filesystem::is_directory(from))
		{
			ec.clear();
			std::filesystem::rename(from, to, ec);
			if (ec)
			{
				MIKAN_LOG_WARNING("ProjectManager::migrateLegacyConfig")
					<< "Could not move " << from << " into the project: " << ec.message();
			}
		}
		if (!std::filesystem::is_directory(to))
			std::filesystem::create_directories(to, ec);
	}

	// Keep a backup of the original, then pivot the file to the slim schema
	ec.clear();
	std::filesystem::copy_file(legacyPath, legacyPath.string() + ".bak",
							   std::filesystem::copy_options::overwrite_existing, ec);

	m_globalSettings->lastProjectPath= projectFile;
	if (!m_globalSettings->save())
	{
		// The appdata file still holds the old schema; migration retries next
		// run against the already-created project folder name's successor
		return false;
	}

	MIKAN_LOG_INFO("ProjectManager::migrateLegacyConfig") << "Migrated into " << projectFile;
	return true;
}

bool ProjectManager::deleteProject(const std::filesystem::path& projectFile)
{
	if (projectFile.filename() != "project.json")
	{
		MIKAN_LOG_ERROR("ProjectManager::deleteProject")
			<< "Not a project file, refusing to delete: " << projectFile;
		return false;
	}

	const std::filesystem::path projectDir= projectFile.parent_path();
	std::error_code ec;
	if (!std::filesystem::equivalent(projectDir.parent_path(), getProjectsRootDirectory(), ec))
	{
		MIKAN_LOG_ERROR("ProjectManager::deleteProject")
			<< "Not under the projects root, refusing to delete: " << projectDir;
		return false;
	}

	std::filesystem::remove_all(projectDir, ec);
	if (ec)
	{
		MIKAN_LOG_WARNING("ProjectManager::deleteProject")
			<< "Failed to delete " << projectDir << ": " << ec.message();
		return false;
	}

	MIKAN_LOG_INFO("ProjectManager::deleteProject") << "Deleted project " << projectDir;
	return true;
}

std::string ProjectManager::getActiveProjectName() const
{
	const std::filesystem::path& projectFile= m_config->getProjectFilePath();
	if (projectFile.empty())
		return std::string();
	return projectFile.parent_path().filename().string();
}
