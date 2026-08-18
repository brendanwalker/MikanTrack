// -- includes -----
#include "PathUtils.h"
#include "StringUtils.h"

#include <assert.h>
#include <filesystem>

#if defined WIN32 || defined _WIN32 || defined WINCE
#include <windows.h>
#include <shellapi.h> // ShellExecuteA (WIN32_LEAN_AND_MEAN strips it from windows.h)
#else
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <dirent.h>
#include <time.h>
#include <unistd.h>
#endif

// -- public methods -----
namespace PathUtils
{
void addDllSearchDirectory(const std::filesystem::path& dllPath)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
	HMODULE hKernel32= GetModuleHandle("kernel32.dll");

	if (hKernel32 != NULL)
	{
		typedef DLL_DIRECTORY_COOKIE(WINAPI * AddDllDirectoryFunc)(PCWSTR);
		AddDllDirectoryFunc addDllDirectory= (AddDllDirectoryFunc)GetProcAddress(hKernel32, "AddDllDirectory");

		if (addDllDirectory != NULL)
		{
			const std::wstring wideDllPath= StringUtils::convertUTF8StringToWString(dllPath.string());

			addDllDirectory(wideDllPath.c_str());
		}
	}
#endif
}

bool openFileWithDefaultApplication(const std::filesystem::path& filePath)
{
#if defined WIN32 || defined _WIN32 || defined WINCE
	// ShellExecute returns a value greater than 32 on success
	HINSTANCE result= ShellExecuteA(NULL, "open", filePath.string().c_str(), NULL, NULL, SW_SHOWNORMAL);
	return reinterpret_cast<intptr_t>(result) > 32;
#elif defined(__APPLE__)
	const std::string command= "open \"" + filePath.string() + "\"";
	return system(command.c_str()) == 0;
#else
	const std::string command= "xdg-open \"" + filePath.string() + "\"";
	return system(command.c_str()) == 0;
#endif
}

std::filesystem::path getModulePath()
{
	std::filesystem::path moduleFilePath;

#if defined WIN32 || defined _WIN32 || defined WINCE
	char result[MAX_PATH];

	GetModuleFileName(NULL, result, MAX_PATH);
	moduleFilePath= std::filesystem::path(result);
#else
	moduleFilePath= std::filesystem::canonical("/proc/self/exe");
#endif

	return moduleFilePath.parent_path();
}

std::filesystem::path getResourceDirectory()
{
	std::filesystem::path resourceDir= std::filesystem::current_path();
	resourceDir/= std::string("resources");

	return resourceDir;
}

std::filesystem::path getFontPath(const std::string& fontName)
{
	const std::string fileFilename= fontName + ".ttf";

	return getResourceDirectory() / "font" / fileFilename;
}

std::filesystem::path makeAbsoluteResourceFilePath(const std::filesystem::path& relative_path)
{
	if (relative_path.is_absolute())
		return relative_path;

	std::filesystem::path full_path(getResourceDirectory());
	full_path/= relative_path;

	return full_path;
}

std::filesystem::path getHomeDirectory()
{
	std::filesystem::path home_dir;

#if defined WIN32 || defined _WIN32 || defined WINCE
	size_t homedir_buffer_req_size;
	char homedir_buffer[512];
	getenv_s(&homedir_buffer_req_size, homedir_buffer, "APPDATA");
	assert(homedir_buffer_req_size <= sizeof(homedir_buffer));
	home_dir= homedir_buffer;
#else
	// if run as root, use system-wide data directory
	if (geteuid() == 0)
	{
		home_dir= "/etc/";
	}
	else
	{
		homedir= getenv("HOME");
	}
#endif
	return home_dir;
}

// Static variable to track the current project directory
static std::filesystem::path g_projectDirectory;

std::filesystem::path getProjectDirectory() { return g_projectDirectory; }

void setProjectDirectory(const std::filesystem::path& projectDir) { g_projectDirectory= projectDir; }

std::filesystem::path getProjectsRootDirectory()
{
	std::filesystem::path projectsRoot;

#if defined WIN32 || defined _WIN32 || defined WINCE
	size_t homedir_buffer_req_size;
	char homedir_buffer[512];
	getenv_s(&homedir_buffer_req_size, homedir_buffer, "USERPROFILE");
	assert(homedir_buffer_req_size <= sizeof(homedir_buffer));
	projectsRoot= homedir_buffer;
	projectsRoot/= "Documents";
#else
	const char* homedir= getenv("HOME");
	if (homedir)
	{
		projectsRoot= homedir;
	}
	else
	{
		projectsRoot= "/";
	}
#endif
	projectsRoot/= "MikanTrack";

	// Create the directory if it doesn't exist
	if (!std::filesystem::exists(projectsRoot))
	{
		std::filesystem::create_directories(projectsRoot);
	}

	return projectsRoot;
}

std::filesystem::path resolveProjectResource(const std::filesystem::path& path)
{
	if (path.is_absolute())
	{
		if (std::filesystem::exists(path))
		{
			return path;
		}
	}
	else
	{
		// Check project folder first
		if (!g_projectDirectory.empty())
		{
			std::filesystem::path projectResourcePath= g_projectDirectory / path;
			if (std::filesystem::exists(projectResourcePath))
			{
				return projectResourcePath;
			}
		}

		// Fall	back to app resource folder
		std::filesystem::path appResourcePath= getResourceDirectory() / path;
		if (std::filesystem::exists(appResourcePath))
		{
			return appResourcePath;
		}
	}

	return std::filesystem::path();
}

std::vector<std::string> listFilenamesInDirectory(const std::filesystem::path& path,
												  const std::string& extension_filter)
{
	std::vector<std::string> filenames;

	for (auto const& dir_entry : std::filesystem::directory_iterator{path})
	{
		if (dir_entry.is_regular_file())
		{
			const std::filesystem::path& filename= dir_entry.path().filename();

			if (extension_filter.empty() || filename.extension() == extension_filter)
			{
				filenames.push_back(filename.string());
			}
		}
	}

	return filenames;
}

std::vector<std::string> listDirectoriesInDirectory(const std::filesystem::path& path)
{
	std::vector<std::string> dirnames;

	for (auto const& dir_entry : std::filesystem::directory_iterator{path})
	{
		if (dir_entry.is_directory())
		{
			const std::filesystem::path& filename= dir_entry.path().filename();

			dirnames.push_back(filename.string());
		}
	}

	return dirnames;
}

std::vector<std::string> listVolumes()
{
	std::vector<std::string> result;

#if defined WIN32 || defined _WIN32 || defined WINCE
	char szLogicalDrives[MAX_PATH]= {0};
	DWORD dwResult= GetLogicalDriveStringsA(MAX_PATH, szLogicalDrives);
	if (dwResult > 0 && dwResult <= MAX_PATH)
	{
		char* szSingleDrive= szLogicalDrives;
		while (*szSingleDrive)
		{
			result.push_back(szSingleDrive);

			// get the next drive
			szSingleDrive+= strlen(szSingleDrive) + 1;
		}
	}
#else
	// TODO: Just list the root directory for now unix systems
	result.push_back("/");
#endif

	return result;
}

std::string makeTimestampedFileName(const std::string& prefix, const std::string& suffix)
{
	time_t t= time(0);
	struct tm* now= localtime(&t);

	char dateTimeString[128];
	strftime(dateTimeString, 80, "%Y_%m_%d_%H_%M_%S", now);

	return StringUtils::stringify(prefix, "_", dateTimeString, suffix);
}

std::filesystem::path makeTimestampedFilePath(const std::filesystem::path& parentDir, const std::string& prefix,
											  const std::string& suffix)
{
	const std::string filename= makeTimestampedFileName(prefix, suffix);

	std::filesystem::path result= parentDir;
	result/= filename;

	return result;
}

std::string createTrimmedPathString(const std::filesystem::path& path, const size_t maxLength)
{
	const std::string pathString= path.string();

	if (pathString.length() > maxLength)
	{
		return "..." + pathString.substr(pathString.length() - maxLength);
	}
	else
	{
		return pathString;
	}
}
}; // namespace PathUtils