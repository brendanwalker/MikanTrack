#pragma once

#include <filesystem>
#include <memory>
#include <string>

struct SDL_Window;
typedef void* SDL_GLContext;

class AppConfig;
class GlobalSettings;
class ProjectManager;
class VideoCaptureSystem;
class VisionThread;
class MainWindow;

class App
{
public:
	// MainMenu shows only the startup menu (no project loaded, cameras closed,
	// vision thread stopped); Project is the full tracking UI
	enum class eAppState
	{
		MainMenu,
		Project,
	};

	App();
	~App();

	static App* getInstance() { return m_instance; }

	int exec(int argc, char** argv);
	void requestShutdown() { m_bShutdownRequested= true; }

	eAppState getAppState() const { return m_appState; }
	// Stops tracking, loads the project file into the config in place, and
	// enters the tracking UI
	bool activateProject(const std::filesystem::path& projectFile);
	// Creates a default-config project folder, loads it, and flags the setup
	// flow to start on entering the tracking UI
	bool activateNewProject(const std::string& projectName);
	// Saves and closes out the current project and returns to the main menu
	void returnToMainMenu();
	// Cancelled guided setup: deletes the just-created project from disk and
	// returns to the main menu, with Resume pointing back at whatever was
	// loaded before it
	void discardNewProjectAndReturnToMenu();
	// One-shot flag raised by activateNewProject, consumed by MainWindow to
	// start the guided setup flow
	bool consumeStartSetupFlowFlag();

	// Applies a config camera-count change: restarts the vision thread (its
	// context list is fixed while running) and resizes the capture slots
	void applyCameraCountChange();

	AppConfig* getConfig() { return m_config.get(); }
	GlobalSettings* getGlobalSettings() { return m_globalSettings.get(); }
	ProjectManager* getProjectManager() { return m_projectManager.get(); }
	VideoCaptureSystem* getVideoCapture() { return m_videoCapture.get(); }
	VisionThread* getVisionThread() { return m_visionThread.get(); }
	SDL_Window* getSdlWindow() { return m_sdlWindow; }

protected:
	bool startup();
	void shutdown();
	void tick(float deltaSeconds);

private:
	static App* m_instance;

	std::unique_ptr<AppConfig> m_config;
	std::unique_ptr<GlobalSettings> m_globalSettings;
	std::unique_ptr<ProjectManager> m_projectManager;
	std::unique_ptr<VideoCaptureSystem> m_videoCapture;
	std::unique_ptr<VisionThread> m_visionThread;
	std::unique_ptr<MainWindow> m_mainWindow;

	SDL_Window* m_sdlWindow= nullptr;
	SDL_GLContext m_glContext= nullptr;

	eAppState m_appState= eAppState::MainMenu;
	bool m_bStartSetupFlowOnEnter= false;
	// The last-project pointer as it was before activateNewProject overwrote
	// it, so a discarded new project can hand Resume back to its predecessor
	std::filesystem::path m_lastProjectPathBeforeNew;
	bool m_bShutdownRequested= false;
};
