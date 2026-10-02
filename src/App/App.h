#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

struct SDL_Window;
typedef void* SDL_GLContext;

class AppConfig;
struct AvatarModel;
class AvatarSkeleton;
class GlobalSettings;
class ImuService;
class LocalizationManager;
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

	// Project actions requested from inside the tracking UI. They are applied
	// at the top of the next tick rather than when clicked: switching or
	// closing a project mid-frame would mutate the config under panels that
	// already sized their per-camera state. A later request in the same frame
	// replaces an earlier one.
	void requestLoadProject(const std::filesystem::path& projectFile);
	void requestCloseProject();
	void requestDiscardNewProject();

	// Applies a config camera-count change: restarts the vision thread (its
	// context list is fixed while running) and resizes the capture slots
	void applyCameraCountChange();

	// The loaded VRM avatar, if any. Loaded on the main thread (the renderer
	// builds GL resources from it) when a project activates and from the
	// Avatar panel; cleared with the project. The generation counter advances
	// on every change so the renderer knows when to re-upload.
	bool loadAvatar(const std::filesystem::path& path);
	void clearAvatar();
	std::shared_ptr<const AvatarModel> getAvatarModel() const { return m_avatarModel; }
	std::shared_ptr<const AvatarSkeleton> getAvatarSkeleton() const { return m_avatarSkeleton; }
	uint32_t getAvatarGeneration() const { return m_avatarGeneration; }
	const std::string& getAvatarLoadError() const { return m_avatarLoadError; }
	// Resolves AvatarConfig::modelPath: absolute as is, else the project
	// folder, else the exe folder
	std::filesystem::path resolveAvatarPath(const std::string& modelPath) const;

	AppConfig* getConfig() { return m_config.get(); }
	GlobalSettings* getGlobalSettings() { return m_globalSettings.get(); }
	LocalizationManager* getLocalization() { return m_localization.get(); }
	ProjectManager* getProjectManager() { return m_projectManager.get(); }
	VideoCaptureSystem* getVideoCapture() { return m_videoCapture.get(); }
	ImuService* getImuService() { return m_imuService.get(); }
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
	std::unique_ptr<LocalizationManager> m_localization;
	std::unique_ptr<ProjectManager> m_projectManager;
	std::unique_ptr<VideoCaptureSystem> m_videoCapture;
	// Started with a project and stopped with it; the vision thread is its
	// only caller while running, so it survives a vision thread restart
	std::unique_ptr<ImuService> m_imuService;
	std::unique_ptr<VisionThread> m_visionThread;
	std::unique_ptr<MainWindow> m_mainWindow;

	// Loads the configured avatar, if any, for the active project
	void loadConfiguredAvatar();
	std::shared_ptr<const AvatarModel> m_avatarModel;
	std::shared_ptr<const AvatarSkeleton> m_avatarSkeleton;
	uint32_t m_avatarGeneration= 0;
	std::string m_avatarLoadError;

	SDL_Window* m_sdlWindow= nullptr;
	SDL_GLContext m_glContext= nullptr;

	eAppState m_appState= eAppState::MainMenu;
	bool m_bStartSetupFlowOnEnter= false;

	enum class ePendingProjectAction
	{
		None,
		Load,
		Close,
		Discard,
	};
	// Services the pending project action at the top of tick()
	void applyPendingProjectAction();
	ePendingProjectAction m_pendingProjectAction= ePendingProjectAction::None;
	std::filesystem::path m_pendingLoadProjectFile;
	// The last-project pointer as it was before activateNewProject overwrote
	// it, so a discarded new project can hand Resume back to its predecessor
	std::filesystem::path m_lastProjectPathBeforeNew;
	bool m_bShutdownRequested= false;
};
