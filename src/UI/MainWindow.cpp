#include "MainWindow.h"

#include <filesystem>

#include "imgui.h"
#include "imgui_internal.h" // dock builder

#include "App.h"
#include "AppConfig.h"
#include "CalibrationPanel.h"
#include "DevicePanel.h"
#include "GlobalSettings.h"
#include "HandOverlay.h"
#include "LocText.h"
#include "LogPanel.h"
#include "Logger.h"
#include "MainMenuScreen.h"
#include "PathUtils.h"
#include "ProjectManager.h"
#include "Scene3dPanel.h"
#include "SettingsPanels.h"
#include "SetupFlow.h"
#include "TimelinePanel.h"
#include "VideoModeUtils.h"
#include "VideoCaptureSystem.h"
#include "VideoPreviewPanel.h"
#include "WizardHost.h"

#include "tinyfiledialogs.h"

MainWindow::MainWindow(App* app)
	: m_app(app)
	, m_mainMenuScreen(std::make_unique<MainMenuScreen>())
	, m_videoPreviewPanel(std::make_unique<VideoPreviewPanel>())
	, m_scene3dPanel(std::make_unique<Scene3dPanel>())
	, m_devicePanel(std::make_unique<DevicePanel>(app, app->getVideoCapture(), app->getConfig()))
	, m_calibrationPanel(std::make_unique<CalibrationPanel>(app->getConfig()))
	, m_wizardHost(std::make_unique<WizardHost>(app->getConfig(), app->getVisionThread()))
	, m_timelinePanel(std::make_unique<TimelinePanel>())
{
	SetupFlow::WizardSet wizards;
	wizards.intrinsics= m_wizardHost->getIntrinsicsWizard();
	wizards.extrinsics= m_wizardHost->getExtrinsicsWizard();
	wizards.hand= m_wizardHost->getHandCalibrationWizard();
	wizards.mounting= m_wizardHost->getMountingWizard();
	wizards.body= m_wizardHost->getBodyCalibrationWizard();
	m_setupFlow= std::make_unique<SetupFlow>(app, this, m_videoPreviewPanel.get(), wizards);
}

MainWindow::~MainWindow()= default;

bool MainWindow::restoreCameraDevice(int cameraIndex)
{
	AppConfig* config= m_app->getConfig();
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	const CameraProfile& profile= config->camera(cameraIndex);

	if (profile.video.devicePath.empty())
		return false;

	// Match by path first, then by friendly name
	std::string pathToOpen;
	const size_t deviceCount= videoCapture->getDeviceCount();
	for (size_t i= 0; i < deviceCount; ++i)
	{
		std::string path, name;
		if (!videoCapture->getDevicePath(i, path) || !videoCapture->getDeviceFriendlyName(i, name))
			continue;
		if (path == profile.video.devicePath)
		{
			pathToOpen= path;
			break;
		}
		if (pathToOpen.empty() && !profile.video.deviceName.empty() && name == profile.video.deviceName)
			pathToOpen= path;
	}

	if (pathToOpen.empty())
	{
		MIKAN_LOG_INFO("MainWindow") << "Camera " << cameraIndex
			<< ": persisted video device not found: " << profile.video.deviceName;
		return false;
	}

	if (!videoCapture->openDeviceByPath(cameraIndex, pathToOpen))
		return false;

	if (!profile.video.modeName.empty())
		videoCapture->setVideoModeByName(cameraIndex, profile.video.modeName);

	videoCapture->startStream(cameraIndex);

	MIKAN_LOG_INFO("MainWindow") << "Camera " << cameraIndex << " restored: "
		<< videoCapture->getCurrentDeviceFriendlyName(cameraIndex)
		<< " @ " << videoCapture->getCurrentVideoModeName(cameraIndex);
	return true;
}

void MainWindow::tryRestoreVideoDeviceFromConfig()
{
	AppConfig* config= m_app->getConfig();
	for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		restoreCameraDevice(cameraIndex);

	refreshDevicePanelState();
}

void MainWindow::refreshDevicePanelState()
{
	m_devicePanel->refreshDeviceList();
	for (int cameraIndex= 0; cameraIndex < (int)m_app->getConfig()->cameraCount(); ++cameraIndex)
		m_devicePanel->refreshModeOptions(cameraIndex);
}

void MainWindow::launchWizard(eWizardKind kind, int cameraIndex)
{
	// The host refuses while one of its wizards runs; the flow launches its
	// own wizards and must not be interrupted by a manual one
	if (!m_setupFlow->isActive())
		m_wizardHost->launch(kind, cameraIndex);
}

void MainWindow::drawMainMenu()
{
	GlobalSettings* globalSettings= m_app->getGlobalSettings();
	// Existence-checked so a project deleted on disk does not present a
	// Resume that can only fail
	const bool bHasLastProject= globalSettings->hasLastProjectPath() &&
								std::filesystem::is_regular_file(globalSettings->lastProjectPath);
	const std::string lastProjectName= bHasLastProject
		? globalSettings->lastProjectPath.parent_path().filename().string()
		: std::string();

	const MainMenuScreen::Action action= m_mainMenuScreen->draw(bHasLastProject, lastProjectName);
	if (action.type != MainMenuScreen::Action::Type::None)
		m_mainMenuScreen->setStatusMessage(std::string());

	switch (action.type)
	{
		case MainMenuScreen::Action::Type::Resume:
			if (!m_app->activateProject(globalSettings->lastProjectPath))
				m_mainMenuScreen->setStatusMessage(locText("errors.loadLastProjectFailed"));
			break;
		case MainMenuScreen::Action::Type::NewProject:
			if (!m_app->activateNewProject(action.projectName))
				m_mainMenuScreen->setStatusMessage(locText("errors.createProjectFailed"));
			break;
		case MainMenuScreen::Action::Type::LoadProject:
			if (!m_app->activateProject(action.projectFile))
				m_mainMenuScreen->setStatusMessage(locText("errors.loadProjectFailed"));
			break;
		case MainMenuScreen::Action::Type::Exit:
			m_app->requestShutdown();
			break;
		default:
			break;
	}
}

void MainWindow::drawDockspaceAndMenuBar()
{
	const ImGuiViewport* viewport= ImGui::GetMainViewport();
	ImGui::SetNextWindowPos(viewport->WorkPos);
	ImGui::SetNextWindowSize(viewport->WorkSize);
	ImGui::SetNextWindowViewport(viewport->ID);

	const ImGuiWindowFlags hostFlags=
		ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar |
		ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus | ImGuiWindowFlags_NoBackground;

	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	ImGui::Begin("MikanTrackDockHost", nullptr, hostFlags);
	ImGui::PopStyleVar();

	const ImGuiID dockspaceId= ImGui::GetID("MikanTrackDockspace");

	// Default layout on first run
	if (!m_bDockLayoutInitialized && ImGui::DockBuilderGetNode(dockspaceId) == nullptr)
	{
		ImGui::DockBuilderRemoveNode(dockspaceId);
		ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
		ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

		ImGuiID centerId= dockspaceId;
		const ImGuiID leftId= ImGui::DockBuilderSplitNode(centerId, ImGuiDir_Left, 0.20f, nullptr, &centerId);
		const ImGuiID rightId= ImGui::DockBuilderSplitNode(centerId, ImGuiDir_Right, 0.25f, nullptr, &centerId);
		const ImGuiID bottomId= ImGui::DockBuilderSplitNode(centerId, ImGuiDir_Down, 0.22f, nullptr, &centerId);

		ImGui::DockBuilderDockWindow(locWindowTitle("windows.videoPreview"), centerId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.scene3d"), centerId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.device"), leftId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.tracking"), leftId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.settings"), leftId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.calibration"), rightId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.oscOutput"), rightId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.log"), bottomId);
		ImGui::DockBuilderDockWindow(locWindowTitle("windows.timeline"), bottomId);
		ImGui::DockBuilderFinish(dockspaceId);
	}
	m_bDockLayoutInitialized= true;

	ImGui::DockSpace(dockspaceId, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu(locLabel("mainWindow.fileMenu")))
		{
			const bool bWizardActive= m_wizardHost->isAnyActive() || m_setupFlow->isActive();
			if (ImGui::MenuItem(locLabel("mainWindow.saveProject")))
				m_app->getConfig()->save();
			if (ImGui::MenuItem(locLabel("mainWindow.loadProject"), nullptr, false, !bWizardActive))
			{
				const std::string defaultDir=
					(ProjectManager::getProjectsRootDirectory() / "").string();
				const char* filterPatterns[]= {"project.json"};
				const char* selectedPath= tinyfd_openFileDialog(
					locText("mainWindow.loadProjectDialogTitle"), defaultDir.c_str(), 1, filterPatterns,
					locText("mainWindow.loadProjectDialogFilterDescription"), 0);
				if (selectedPath != nullptr)
					m_app->requestLoadProject(PathUtils::utf8ToPath(selectedPath));
			}
			if (ImGui::MenuItem(locLabel("mainWindow.closeProject"), nullptr, false, !bWizardActive))
				m_app->requestCloseProject();
			ImGui::Separator();
			if (ImGui::MenuItem(locLabel("mainWindow.quit"), "Alt+F4"))
				m_app->requestShutdown();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(locLabel("mainWindow.calibrationMenu")))
		{
			const bool bWizardActive= m_wizardHost->isAnyActive() || m_setupFlow->isActive();
			AppConfig* config= m_app->getConfig();

			for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
			{
				ImGui::PushID(cameraIndex);
				const std::string label= locFormat("mainWindow.cameraIntrinsicsFmt", cameraIndex + 1);
				if (ImGui::MenuItem(label.c_str(), nullptr, false, !bWizardActive))
					launchWizard(eWizardKind::Intrinsics, cameraIndex);
				ImGui::PopID();
			}

			// One shared session: every camera calibrates against the same
			// marker placement (separate sessions = disagreeing world frames)
			bool bAllIntrinsics= true;
			for (size_t i= 0; i < config->cameraCount(); ++i)
				bAllIntrinsics&= config->camera(i).intrinsics.present;
			if (ImGui::MenuItem(locLabel("mainWindow.extrinsicsAllCameras"), nullptr, false,
								!bWizardActive && bAllIntrinsics))
				launchWizard(eWizardKind::Extrinsics);
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(locLabel("mainWindow.viewMenu")))
		{
			ImGui::MenuItem(locLabel("mainWindow.logPanel"), nullptr, &m_bShowLogPanel);
			ImGui::MenuItem(locLabel("mainWindow.viewSettingsPanel"), nullptr, &m_bShowSettingsPanel);
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}

	ImGui::End();
}

// -- Global hotkeys ----------------------------------------------------------

void MainWindow::handleHotkeys()
{
	// Every key the tracking UI binds, in one place. Repeat is off: each
	// press fires once.
	struct HotkeyBinding
	{
		ImGuiKey key;
		void (MainWindow::*action)();
	};
	static const HotkeyBinding k_hotkeys[]= {
		{ImGuiKey_F9, &MainWindow::dumpDiagnostics},
		{ImGuiKey_F10, &MainWindow::toggleRecording},
	};

	for (const HotkeyBinding& binding : k_hotkeys)
	{
		if (ImGui::IsKeyPressed(binding.key, false))
			(this->*binding.action)();
	}
}

// Diagnostic dump: state history + camera frames + config
void MainWindow::dumpDiagnostics()
{
	m_app->getVisionThread()->requestDiagnosticDump(m_app->getConfig()->makeDumpDirectoryPath());
}

// Toggles the tracking recording (deterministic replay input capture;
// starting resets transient tracking state - brief blip)
void MainWindow::toggleRecording()
{
	VisionThread* visionThread= m_app->getVisionThread();
	if (visionThread->isRecording())
		visionThread->requestRecordingStop();
	else
		visionThread->requestRecordingStart(m_app->getConfig()->makeRecordingFilePath());
}

// -- Frame ------------------------------------------------------------------

void MainWindow::update(float deltaSeconds)
{
	if (m_app->getAppState() == App::eAppState::MainMenu)
	{
		drawMainMenu();
		return;
	}

	// A freshly created project starts the guided setup chain
	if (m_app->consumeStartSetupFlowFlag())
		m_setupFlow->begin();

	AppConfig* config= m_app->getConfig();
	VisionThread* visionThread= m_app->getVisionThread();
	const int cameraCount= (int)config->cameraCount();

	// Keep per-camera containers in sync with the config
	m_latestPreviews.resize(cameraCount);
	m_videoPreviewPanel->setCameraCount(cameraCount);

	// Pull the newest per-camera previews + the fused result
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		VisionPreviewFrame freshFrame;
		if (visionThread->fetchPreviewFrame(cameraIndex, freshFrame))
		{
			m_latestPreviews[cameraIndex]= std::move(freshFrame);
			m_videoPreviewPanel->setFrame(cameraIndex, m_latestPreviews[cameraIndex].bgr);
		}
	}
	visionThread->fetchFusedResult(m_latestFused);

	handleHotkeys();

	drawDockspaceAndMenuBar();

	const bool bWizardActive= m_wizardHost->isAnyActive() || m_setupFlow->isActive();

	// Panels
	m_devicePanel->draw();
	SettingsPanels::drawTrackingPanel(config, visionThread, m_app->getVideoCapture(), m_videoPreviewPanel.get(),
									  m_scene3dPanel.get(),
									  m_trackingPanelState, m_latestPreviews, m_latestFused);
	SettingsPanels::drawOscPanel(config, visionThread, m_latestFused);
	if (m_bShowSettingsPanel)
		SettingsPanels::drawAppSettingsPanel();
	m_timelinePanel->draw(config, visionThread);

	// Launch requests raised by the panels this frame
	if (m_trackingPanelState.bLaunchMountingWizard)
		launchWizard(eWizardKind::Mounting);
	if (m_trackingPanelState.bLaunchBodyCalibrationWizard)
		launchWizard(eWizardKind::Body);
	if (m_trackingPanelState.bLaunchHandCalibrationWizard)
		launchWizard(eWizardKind::Hand);
	m_trackingPanelState.bLaunchMountingWizard= false;
	m_trackingPanelState.bLaunchBodyCalibrationWizard= false;
	m_trackingPanelState.bLaunchHandCalibrationWizard= false;

	const CalibrationPanel::DrawResult calibrationAction= m_calibrationPanel->draw(bWizardActive);
	if (calibrationAction.bLaunchIntrinsicsWizard)
		launchWizard(eWizardKind::Intrinsics, calibrationAction.cameraIndex);
	if (calibrationAction.bLaunchExtrinsicsWizard)
		launchWizard(eWizardKind::Extrinsics);

	// Guided setup chain: draws its prompt modals on top of the panels, and
	// launches/watches the wizards updated below
	m_setupFlow->update();

	// Bring the Video Preview tab forward when a camera calibration wizard
	// starts: the pattern feed is the wizard's whole UI, and the 3D Scene tab
	// may be the selected one in the shared center dock
	if (m_wizardHost->consumeCameraWizardStarted())
		ImGui::SetWindowFocus(locWindowTitle("windows.videoPreview"));

	// Pin the preview highlight to the intrinsics wizard's camera while it is
	// active (the extrinsics wizard uses ALL cameras, so no pinning there)
	const int intrinsicsCamera= m_wizardHost->getIntrinsicsCameraIndex();
	if (intrinsicsCamera >= 0)
		m_videoPreviewPanel->setActiveCamera(intrinsicsCamera);

	// Newest per-camera results, for the preview overlays and the 3D scene
	std::vector<const TrackingFrameResult*> perCameraResults;
	for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
	{
		perCameraResults.push_back(
			m_latestPreviews[cameraIndex].valid ? &m_latestPreviews[cameraIndex].result : nullptr);
	}

	// Central: side-by-side previews with per-camera overlays
	{
		std::vector<const char*> executionProviders;
		std::vector<ForearmOverlay> forearmOverlays;
		for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
		{
			executionProviders.push_back(visionThread->getActiveExecutionProvider(cameraIndex));
			// The fused (world-space) forearm projected back into this camera:
			// the UI is what holds both the fused result and every camera's
			// calibration
			forearmOverlays.push_back(HandOverlay::makeForearmOverlay(
				config->camera(cameraIndex), m_latestFused, config->body.forearmLengthMeters));
		}
		m_videoPreviewPanel->draw(perCameraResults, executionProviders, &forearmOverlays);
	}

	// 3D scene: fused skeleton + all calibrated camera frustums. In replay
	// view the whole feed comes from the timeline's scrub position instead,
	// with the frustums built from the RECORDING's config snapshot.
	m_scene3dPanel->setForearmLength(config->body.forearmLengthMeters);
	if (m_timelinePanel->isReplayViewActive())
	{
		m_scene3dPanel->draw(m_timelinePanel->getDisplayFused(), m_timelinePanel->getSceneCameras(),
							 m_timelinePanel->getPerCameraResults());
	}
	else
	{
		m_scene3dPanel->draw(m_latestFused, makeSceneCameraViews(*config), perCameraResults);
	}

	// The active wizard, drawn last, on top
	m_wizardHost->update(deltaSeconds, m_latestPreviews, m_latestFused, m_videoPreviewPanel.get());

	if (m_bShowLogPanel)
		LogPanel::getInstance().draw(&m_bShowLogPanel);
}
