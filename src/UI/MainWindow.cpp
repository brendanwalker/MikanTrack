#include "MainWindow.h"

#include "imgui.h"
#include "imgui_internal.h" // dock builder

#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"

#include "App.h"
#include "AppConfig.h"
#include "CalibrationPanel.h"
#include "DevicePanel.h"
#include "ExtrinsicsWizard.h"
#include "IntrinsicsWizard.h"
#include "LogPanel.h"
#include "Logger.h"
#include "Scene3dPanel.h"
#include "SettingsPanels.h"
#include "VideoModeUtils.h"
#include "VideoCaptureSystem.h"
#include "VideoPreviewPanel.h"

MainWindow::MainWindow(App* app)
	: m_app(app)
	, m_videoPreviewPanel(std::make_unique<VideoPreviewPanel>())
	, m_scene3dPanel(std::make_unique<Scene3dPanel>())
	, m_devicePanel(std::make_unique<DevicePanel>(app->getVideoCapture(), app->getConfig()))
	, m_calibrationPanel(std::make_unique<CalibrationPanel>(app->getConfig()))
	, m_intrinsicsWizard(std::make_unique<IntrinsicsWizard>(app->getConfig(), app->getVisionThread()))
	, m_extrinsicsWizard(std::make_unique<ExtrinsicsWizard>(app->getConfig(), app->getVisionThread()))
{
	// Hotplug / disconnect notifications refresh the device panel
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	videoCapture->setDeviceListChangedCallback([this]() { m_devicePanel->refreshDeviceList(); });
	videoCapture->setVideoModeChangedCallback([this]() { m_devicePanel->refreshModeOptions(); });
	videoCapture->setDeviceDisconnectedCallback([this]() {
		MIKAN_LOG_WARNING("MainWindow") << "Video device disconnected";
		m_devicePanel->refreshDeviceList();
	});
}

MainWindow::~MainWindow()= default;

void MainWindow::tryRestoreVideoDeviceFromConfig()
{
	AppConfig* config= m_app->getConfig();
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();

	if (config->video.devicePath.empty())
		return;

	// Match by path first, then by friendly name
	std::string pathToOpen;
	const size_t deviceCount= videoCapture->getDeviceCount();
	for (size_t i= 0; i < deviceCount; ++i)
	{
		std::string path, name;
		if (!videoCapture->getDevicePath(i, path) || !videoCapture->getDeviceFriendlyName(i, name))
			continue;
		if (path == config->video.devicePath)
		{
			pathToOpen= path;
			break;
		}
		if (pathToOpen.empty() && !config->video.deviceName.empty() && name == config->video.deviceName)
			pathToOpen= path;
	}

	if (pathToOpen.empty())
	{
		MIKAN_LOG_INFO("MainWindow") << "Persisted video device not found: " << config->video.deviceName;
		return;
	}

	if (!videoCapture->openDeviceByPath(pathToOpen))
		return;

	if (!config->video.modeName.empty())
		videoCapture->setVideoModeByName(config->video.modeName);

	m_devicePanel->refreshDeviceList();
	m_devicePanel->refreshModeOptions();
	videoCapture->startStream();

	MIKAN_LOG_INFO("MainWindow") << "Restored video device: " << videoCapture->getCurrentDeviceFriendlyName()
		<< " @ " << videoCapture->getCurrentVideoModeName();
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
	ImGui::Begin("MikanMediaPipeDockHost", nullptr, hostFlags);
	ImGui::PopStyleVar();

	const ImGuiID dockspaceId= ImGui::GetID("MikanMediaPipeDockspace");

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

		ImGui::DockBuilderDockWindow("Video Preview", centerId);
		ImGui::DockBuilderDockWindow("3D Scene", centerId);
		ImGui::DockBuilderDockWindow("Device", leftId);
		ImGui::DockBuilderDockWindow("Tracking", leftId);
		ImGui::DockBuilderDockWindow("Calibration", rightId);
		ImGui::DockBuilderDockWindow("OSC Output", rightId);
		ImGui::DockBuilderDockWindow("Log", bottomId);
		ImGui::DockBuilderFinish(dockspaceId);
	}
	m_bDockLayoutInitialized= true;

	ImGui::DockSpace(dockspaceId, ImVec2(0, 0), ImGuiDockNodeFlags_PassthruCentralNode);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("File"))
		{
			if (ImGui::MenuItem("Save Config"))
				m_app->getConfig()->save();
			ImGui::Separator();
			if (ImGui::MenuItem("Quit", "Alt+F4"))
				m_app->requestShutdown();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("Calibration"))
		{
			const bool bWizardActive= m_intrinsicsWizard->isActive() || m_extrinsicsWizard->isActive();
			if (ImGui::MenuItem("Intrinsics Wizard...", nullptr, false, !bWizardActive))
				m_intrinsicsWizard->enter();
			if (ImGui::MenuItem("Extrinsics + Hand Scale Wizard...", nullptr, false,
								!bWizardActive && m_app->getConfig()->intrinsics.present))
				m_extrinsicsWizard->enter();
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu("View"))
		{
			ImGui::MenuItem("Log Panel", nullptr, &m_bShowLogPanel);
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}

	ImGui::End();
}

void MainWindow::update(float deltaSeconds)
{
	// Pull the newest processed frame from the vision thread
	VisionThread* visionThread= m_app->getVisionThread();
	VisionPreviewFrame freshFrame;
	if (visionThread->fetchPreviewFrame(freshFrame))
	{
		m_latestPreview= std::move(freshFrame);
		m_videoPreviewPanel->setFrame(m_latestPreview.bgr);
	}

	drawDockspaceAndMenuBar();

	const bool bWizardActive= m_intrinsicsWizard->isActive() || m_extrinsicsWizard->isActive();

	// Panels
	m_devicePanel->draw();
	SettingsPanels::drawTrackingPanel(m_app->getConfig(), visionThread, m_videoPreviewPanel.get());
	SettingsPanels::drawOscPanel(m_app->getConfig(), visionThread);

	const CalibrationPanel::DrawResult calibrationAction= m_calibrationPanel->draw(bWizardActive);
	if (calibrationAction.bLaunchIntrinsicsWizard)
		m_intrinsicsWizard->enter();
	if (calibrationAction.bLaunchExtrinsicsWizard)
		m_extrinsicsWizard->enter();

	// Central: video preview (with overlay) + 3D scene
	m_videoPreviewPanel->draw(
		m_latestPreview.result,
		visionThread->getActiveExecutionProvider(),
		m_latestPreview.valid);

	{
		AppConfig* config= m_app->getConfig();

		// markerFromCamera maps GL-convention camera space -> world space
		const glm::mat4 cameraToWorld= glm::mat4(config->extrinsics.markerFromCamera);
		m_scene3dPanel->draw(
			m_latestPreview.result,
			cameraToWorld,
			config->extrinsics.present,
			config->intrinsics.present ? &config->intrinsics.intrinsics : nullptr);
	}

	// Wizards (drawn last, on top)
	if (m_intrinsicsWizard->isActive())
	{
		if (!m_intrinsicsWizard->update(deltaSeconds, m_latestPreview.bgr,
										m_videoPreviewPanel->getLastDrawList(),
										m_videoPreviewPanel->getImageToScreenMapping()))
		{
			m_intrinsicsWizard->exit();
		}
	}
	else if (m_extrinsicsWizard->isActive())
	{
		if (!m_extrinsicsWizard->update(deltaSeconds, m_latestPreview.bgr, m_latestPreview.result,
										m_videoPreviewPanel->getLastDrawList(),
										m_videoPreviewPanel->getImageToScreenMapping()))
		{
			m_extrinsicsWizard->exit();
		}
	}

	if (m_bShowLogPanel)
		LogPanel::getInstance().draw(&m_bShowLogPanel);
}
