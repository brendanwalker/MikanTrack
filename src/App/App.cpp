#include "App.h"

#include <chrono>

#include "GL/glew.h"
#include "SDL.h"

#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl2.h"

#include "AppConfig.h"
#include "FrameTimer.h"
#include "Logger.h"
#include "LogPanel.h"
#include "MainWindow.h"
#include "VideoCaptureSystem.h"
#include "VisionThread.h"

App* App::m_instance= nullptr;

App::App()
{
	m_instance= this;
}

App::~App()
{
	m_instance= nullptr;
}

int App::exec(int argc, char** argv)
{
	if (!startup())
	{
		shutdown();
		return 1;
	}

	auto lastFrameTime= std::chrono::steady_clock::now();
	while (!m_bShutdownRequested)
	{
		FrameTimer frameTimer(11); // ~90Hz UI cap

		const auto now= std::chrono::steady_clock::now();
		float deltaSeconds= std::chrono::duration<float>(now - lastFrameTime).count();
		lastFrameTime= now;
		deltaSeconds= std::min(deltaSeconds, 0.1f);

		tick(deltaSeconds);

		frameTimer.waitForNextFrame();
	}

	shutdown();
	return 0;
}

bool App::startup()
{
	// Logging first
	LoggerSettings loggerSettings= {};
	loggerSettings.min_log_level= LogSeverityLevel::debug;
	loggerSettings.log_filename= "MikanTrack.log";
	loggerSettings.enable_console= false;
	loggerSettings.log_callback= &LogPanel::logCallback;
	log_init(loggerSettings);

	MIKAN_LOG_INFO("App::startup") << "MikanTrack starting up";

	m_config= std::make_unique<AppConfig>();
	m_config->load();

	// SDL + GL context
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
	{
		MIKAN_LOG_ERROR("App::startup") << "SDL_Init failed: " << SDL_GetError();
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

	m_sdlWindow= SDL_CreateWindow(
		"MikanTrack",
		SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
		1600, 900,
		SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
	if (m_sdlWindow == nullptr)
	{
		MIKAN_LOG_ERROR("App::startup") << "SDL_CreateWindow failed: " << SDL_GetError();
		return false;
	}

	m_glContext= SDL_GL_CreateContext(m_sdlWindow);
	if (m_glContext == nullptr)
	{
		MIKAN_LOG_ERROR("App::startup") << "SDL_GL_CreateContext failed: " << SDL_GetError();
		return false;
	}
	SDL_GL_MakeCurrent(m_sdlWindow, m_glContext);
	SDL_GL_SetSwapInterval(1); // vsync

	glewExperimental= GL_TRUE;
	const GLenum glewResult= glewInit();
	if (glewResult != GLEW_OK)
	{
		MIKAN_LOG_ERROR("App::startup") << "glewInit failed: " << (const char*)glewGetErrorString(glewResult);
		return false;
	}
	MIKAN_LOG_INFO("App::startup") << "OpenGL " << (const char*)glGetString(GL_VERSION)
		<< " on " << (const char*)glGetString(GL_RENDERER);

	// ImGui (docking)
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io= ImGui::GetIO();
	io.ConfigFlags|= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags|= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui::StyleColorsDark();
	ImGui_ImplSDL2_InitForOpenGL(m_sdlWindow, m_glContext);
	ImGui_ImplOpenGL3_Init("#version 330 core");

	// Subsystems
	m_videoCapture= std::make_unique<VideoCaptureSystem>();
	if (!m_videoCapture->startup())
	{
		MIKAN_LOG_ERROR("App::startup") << "VideoCaptureSystem startup failed";
		return false;
	}
	m_videoCapture->setCameraSlotCount(m_config->cameraCount());

	m_visionThread= std::make_unique<VisionThread>(m_videoCapture.get(), m_config.get());
	m_visionThread->start();

	m_mainWindow= std::make_unique<MainWindow>(this);

	// Restore last-used device/mode
	m_mainWindow->tryRestoreVideoDeviceFromConfig();

	return true;
}

void App::applyCameraCountChange()
{
	MIKAN_LOG_INFO("App::applyCameraCountChange")
		<< "Applying camera count change: " << m_config->cameraCount() << " cameras";

	// Order matters: stop the consumer (vision thread) before touching the
	// capture slots, then restart with the new context count
	m_visionThread->stop();
	m_videoCapture->setCameraSlotCount(m_config->cameraCount());
	m_visionThread->start();
	m_config->markDirty();
}

void App::shutdown()
{
	MIKAN_LOG_INFO("App::shutdown") << "Shutting down";

	if (m_config != nullptr)
		m_config->save();

	if (m_visionThread != nullptr)
	{
		m_visionThread->stop();
		m_visionThread= nullptr;
	}

	if (m_videoCapture != nullptr)
	{
		m_videoCapture->shutdown();
		m_videoCapture= nullptr;
	}

	m_mainWindow= nullptr;

	if (ImGui::GetCurrentContext() != nullptr)
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplSDL2_Shutdown();
		ImGui::DestroyContext();
	}

	if (m_glContext != nullptr)
	{
		SDL_GL_DeleteContext(m_glContext);
		m_glContext= nullptr;
	}
	if (m_sdlWindow != nullptr)
	{
		SDL_DestroyWindow(m_sdlWindow);
		m_sdlWindow= nullptr;
	}
	SDL_Quit();

	log_dispose();
}

void App::tick(float deltaSeconds)
{
	// SDL events
	SDL_Event event;
	while (SDL_PollEvent(&event))
	{
		ImGui_ImplSDL2_ProcessEvent(&event);
		if (event.type == SDL_QUIT)
			m_bShutdownRequested= true;
		if (event.type == SDL_WINDOWEVENT &&
			event.window.event == SDL_WINDOWEVENT_CLOSE &&
			event.window.windowID == SDL_GetWindowID(m_sdlWindow))
		{
			m_bShutdownRequested= true;
		}
	}

	// Pump the device manager (hotplug HWND lives on this thread)
	m_videoCapture->update(deltaSeconds);

	// UI frame
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame();
	ImGui::NewFrame();

	m_mainWindow->update(deltaSeconds);

	ImGui::Render();

	int drawableWidth= 0, drawableHeight= 0;
	SDL_GL_GetDrawableSize(m_sdlWindow, &drawableWidth, &drawableHeight);
	glViewport(0, 0, drawableWidth, drawableHeight);
	glClearColor(0.08f, 0.08f, 0.09f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	SDL_GL_SwapWindow(m_sdlWindow);

	m_config->updateAutoSave(deltaSeconds);
}
