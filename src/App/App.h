#pragma once

#include <memory>

struct SDL_Window;
typedef void* SDL_GLContext;

class AppConfig;
class VideoCaptureSystem;
class VisionThread;
class MainWindow;

class App
{
public:
	App();
	~App();

	static App* getInstance() { return m_instance; }

	int exec(int argc, char** argv);
	void requestShutdown() { m_bShutdownRequested= true; }

	// Applies a config camera-count change: restarts the vision thread (its
	// context list is fixed while running) and resizes the capture slots
	void applyCameraCountChange();

	AppConfig* getConfig() { return m_config.get(); }
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
	std::unique_ptr<VideoCaptureSystem> m_videoCapture;
	std::unique_ptr<VisionThread> m_visionThread;
	std::unique_ptr<MainWindow> m_mainWindow;

	SDL_Window* m_sdlWindow= nullptr;
	SDL_GLContext m_glContext= nullptr;

	bool m_bShutdownRequested= false;
};
