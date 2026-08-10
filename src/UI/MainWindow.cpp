#include "MainWindow.h"

#include "imgui.h"
#include "imgui_internal.h" // dock builder

#include "App.h"
#include "AppConfig.h"
#include "CalibrationPanel.h"
#include "DevicePanel.h"
#include "ExtrinsicsWizard.h"
#include "IntrinsicsWizard.h"
#include "MountingWizard.h"
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
	, m_devicePanel(std::make_unique<DevicePanel>(app, app->getVideoCapture(), app->getConfig()))
	, m_calibrationPanel(std::make_unique<CalibrationPanel>(app->getConfig()))
	, m_intrinsicsWizard(std::make_unique<IntrinsicsWizard>(app->getConfig(), app->getVisionThread()))
	, m_extrinsicsWizard(std::make_unique<ExtrinsicsWizard>(app->getConfig(), app->getVisionThread()))
	, m_mountingWizard(std::make_unique<MountingWizard>(app->getConfig(), app->getVisionThread()))
{
	// Hotplug / disconnect notifications refresh the device panel
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	videoCapture->setDeviceListChangedCallback([this]() { m_devicePanel->refreshDeviceList(); });
	videoCapture->setVideoModeChangedCallback([this](int cameraIndex) { m_devicePanel->refreshModeOptions(cameraIndex); });
	videoCapture->setDeviceDisconnectedCallback([this](int cameraIndex) {
		MIKAN_LOG_WARNING("MainWindow") << "Video device for camera " << cameraIndex << " disconnected";
		m_devicePanel->refreshDeviceList();
	});
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

	m_devicePanel->refreshDeviceList();
	for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
		m_devicePanel->refreshModeOptions(cameraIndex);
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
			const bool bWizardActive= m_intrinsicsWizard->isActive() || m_extrinsicsWizard->isActive() ||
							  m_mountingWizard->isActive();
			AppConfig* config= m_app->getConfig();

			for (int cameraIndex= 0; cameraIndex < (int)config->cameraCount(); ++cameraIndex)
			{
				ImGui::PushID(cameraIndex);
				char label[64];
				snprintf(label, sizeof(label), "Camera %d Intrinsics...", cameraIndex + 1);
				if (ImGui::MenuItem(label, nullptr, false, !bWizardActive))
					m_intrinsicsWizard->enter(cameraIndex);
				ImGui::PopID();
			}

			// One shared session: every camera calibrates against the same
			// marker placement (separate sessions = disagreeing world frames)
			bool bAllIntrinsics= true;
			for (size_t i= 0; i < config->cameraCount(); ++i)
				bAllIntrinsics&= config->camera(i).intrinsics.present;
			if (ImGui::MenuItem("Extrinsics (all cameras)...", nullptr, false,
								!bWizardActive && bAllIntrinsics))
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

	// F9 anywhere: diagnostic dump (state history + camera frames + config)
	if (ImGui::IsKeyPressed(ImGuiKey_F9, false))
		visionThread->requestDiagnosticDump(AppConfig::makeDumpDirectoryPath());

	drawDockspaceAndMenuBar();

	const bool bWizardActive= m_intrinsicsWizard->isActive() || m_extrinsicsWizard->isActive() ||
							  m_mountingWizard->isActive();

	// Panels
	m_devicePanel->draw();
	SettingsPanels::drawTrackingPanel(config, visionThread, m_videoPreviewPanel.get(), m_scene3dPanel.get(),
									  m_trackingPanelState, m_latestPreviews, m_latestFused);
	SettingsPanels::drawOscPanel(config, visionThread, m_latestFused);

	if (m_trackingPanelState.bLaunchMountingWizard)
	{
		m_trackingPanelState.bLaunchMountingWizard= false;
		if (!bWizardActive)
			m_mountingWizard->enter();
	}

	const CalibrationPanel::DrawResult calibrationAction= m_calibrationPanel->draw(bWizardActive);
	if (calibrationAction.bLaunchIntrinsicsWizard)
		m_intrinsicsWizard->enter(calibrationAction.cameraIndex);
	if (calibrationAction.bLaunchExtrinsicsWizard)
		m_extrinsicsWizard->enter();

	// Pin the preview highlight to the wizard's camera while one is active
	// (the extrinsics wizard uses ALL cameras, so no pinning there)
	if (m_intrinsicsWizard->isActive())
		m_videoPreviewPanel->setActiveCamera(m_intrinsicsWizard->getCameraIndex());

	// Central: side-by-side previews with per-camera overlays
	{
		std::vector<const TrackingFrameResult*> previewResults;
		std::vector<const char*> executionProviders;
		std::vector<ForearmOverlay> forearmOverlays;
		for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
		{
			previewResults.push_back(
				m_latestPreviews[cameraIndex].valid ? &m_latestPreviews[cameraIndex].result : nullptr);
			executionProviders.push_back(visionThread->getActiveExecutionProvider(cameraIndex));

			// Project the fused (world-space) forearm back into this camera.
			// Done here rather than on the vision thread because the UI is
			// what already holds both the fused result and every camera's
			// calibration.
			ForearmOverlay overlay;
			const CameraProfile& profile= config->camera(cameraIndex);
			if (profile.intrinsics.present && profile.extrinsics.present)
			{
				const glm::dmat4 cameraFromWorld= glm::inverse(profile.extrinsics.markerFromCamera);
				const MikanMatrix3d& cameraMatrix= profile.intrinsics.intrinsics.undistorted_camera_matrix;

				auto projectToImage= [&](const glm::vec3& worldPoint, ImVec2& outPixel) {
					const glm::dvec4 cameraPoint= cameraFromWorld * glm::dvec4(glm::dvec3(worldPoint), 1.0);
					if (cameraPoint.z < 1e-3)
						return false;
					outPixel= ImVec2((float)(cameraMatrix.x0 * cameraPoint.x / cameraPoint.z + cameraMatrix.z0),
									 (float)(cameraMatrix.y1 * cameraPoint.y / cameraPoint.z + cameraMatrix.z1));
					return true;
				};

				for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
				{
					const HandPose& pose= m_latestFused.poses[sideIndex];
					if (!pose.tracked || !pose.hasWorldPose || !pose.hasForearmPose)
						continue;

					const glm::vec3 wristWorld= pose.getWristPositionWorld();
					const glm::vec3 elbowWorld=
						pose.getElbowPositionWorld(config->imu.forearmLengthMeters);
					overlay.valid[sideIndex]= projectToImage(wristWorld, overlay.wristPx[sideIndex]) &&
						projectToImage(elbowWorld, overlay.elbowPx[sideIndex]);
				}
			}
			forearmOverlays.push_back(overlay);
		}
		m_videoPreviewPanel->draw(previewResults, executionProviders, &forearmOverlays);
	}

	// 3D scene: fused skeleton + all calibrated camera frustums
	{
		std::vector<SceneCameraView> sceneCameras;
		std::vector<const TrackingFrameResult*> perCameraResults;
		for (int cameraIndex= 0; cameraIndex < cameraCount; ++cameraIndex)
		{
			const CameraProfile& profile= config->camera(cameraIndex);
			SceneCameraView view;
			// markerFromCamera maps OpenCV-convention camera space -> world
			// (Scene3dPanel applies the GL flip for the frustum itself)
			view.cameraToWorld= glm::mat4(profile.extrinsics.markerFromCamera);
			view.bHasExtrinsics= profile.extrinsics.present;
			view.intrinsics= profile.intrinsics.present ? &profile.intrinsics.intrinsics : nullptr;
			sceneCameras.push_back(view);

			perCameraResults.push_back(
				m_latestPreviews[cameraIndex].valid ? &m_latestPreviews[cameraIndex].result : nullptr);
		}
		m_scene3dPanel->setForearmLength(config->imu.forearmLengthMeters);
		m_scene3dPanel->draw(m_latestFused, sceneCameras, perCameraResults);
	}

	// Wizards (drawn last, on top). They consume THEIR camera's per-camera
	// preview + result, not the fused output.
	static const VisionPreviewFrame s_emptyPreview{};
	if (m_intrinsicsWizard->isActive())
	{
		const int wizardCamera= m_intrinsicsWizard->getCameraIndex();
		const VisionPreviewFrame& preview=
			wizardCamera < cameraCount ? m_latestPreviews[wizardCamera] : s_emptyPreview;
		if (!m_intrinsicsWizard->update(deltaSeconds, preview.bgr,
										m_videoPreviewPanel->getLastDrawList(),
										m_videoPreviewPanel->getImageToScreenMapping(wizardCamera)))
		{
			m_intrinsicsWizard->exit();
		}
	}
	else if (m_extrinsicsWizard->isActive())
	{
		if (!m_extrinsicsWizard->update(deltaSeconds, m_latestPreviews, m_videoPreviewPanel.get()))
			m_extrinsicsWizard->exit();
	}
	else if (m_mountingWizard->isActive())
	{
		// Unlike the calibration wizards this one leaves tracking running - it
		// needs live tracked hands for the straight-wrist pose
		if (!m_mountingWizard->update(deltaSeconds, m_latestFused))
			m_mountingWizard->exit();
	}

	if (m_bShowLogPanel)
		LogPanel::getInstance().draw(&m_bShowLogPanel);
}
