#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "SettingsPanels.h" // TrackingPanelState
#include "VisionThread.h"    // VisionPreviewFrame

class App;
class CalibrationPanel;
class DevicePanel;
class ExtrinsicsWizard;
class IntrinsicsWizard;
class MainMenuScreen;
class MountingWizard;
class BodyCalibrationWizard;
class HandCalibrationWizard;
class Scene3dPanel;
class SetupFlow;
class TimelinePanel;
class VideoPreviewPanel;

// Owns the ImGui layout: dockspace, menu bar, all panels and the calibration
// wizards. Created after ImGui/GL are initialized.
class MainWindow
{
public:
	explicit MainWindow(App* app);
	~MainWindow();

	// Per-frame UI update (inside an active ImGui frame)
	void update(float deltaSeconds);

	// Reopens each configured camera's device/mode (called when a project is
	// activated)
	void tryRestoreVideoDeviceFromConfig();

	// Re-syncs the device panel's cached device list and per-camera state
	// with the config and the open devices
	void refreshDevicePanelState();

private:
	// The MainMenu app state: draws the startup menu and dispatches its
	// actions to App
	void drawMainMenu();
	void drawDockspaceAndMenuBar();
	bool isAnyWizardActive() const;
	// Restores one camera's persisted device by path, then by friendly name
	bool restoreCameraDevice(int cameraIndex);

	App* m_app;

	std::unique_ptr<MainMenuScreen> m_mainMenuScreen;
	std::unique_ptr<VideoPreviewPanel> m_videoPreviewPanel;
	std::unique_ptr<Scene3dPanel> m_scene3dPanel;
	std::unique_ptr<DevicePanel> m_devicePanel;
	std::unique_ptr<CalibrationPanel> m_calibrationPanel;
	std::unique_ptr<IntrinsicsWizard> m_intrinsicsWizard;
	std::unique_ptr<ExtrinsicsWizard> m_extrinsicsWizard;
	std::unique_ptr<MountingWizard> m_mountingWizard;
	std::unique_ptr<BodyCalibrationWizard> m_bodyCalibrationWizard;
	std::unique_ptr<HandCalibrationWizard> m_handCalibrationWizard;
	std::unique_ptr<TimelinePanel> m_timelinePanel;
	// Guided new-project setup chain; needs the wizard pointers above, so it
	// is constructed last (in the constructor body)
	std::unique_ptr<SetupFlow> m_setupFlow;

	// Latest per-camera previews + the fused result (kept between updates so
	// the UI still has data when no new frame arrived this tick)
	std::vector<VisionPreviewFrame> m_latestPreviews;
	TrackingFrameResult m_latestFused;

	TrackingPanelState m_trackingPanelState;

	// Project actions clicked in the menu bar (or requested by the setup
	// flow), applied at the top of the next update (never mid-frame)
	std::filesystem::path m_pendingLoadProjectFile;
	bool m_bPendingCloseProject= false;
	bool m_bPendingDiscardProject= false;

	// Rising-edge tracker for focusing the Video Preview tab when a camera
	// calibration wizard starts
	bool m_bCameraWizardWasActive= false;

	bool m_bShowLogPanel= true;
	bool m_bShowSettingsPanel= true;
	bool m_bDockLayoutInitialized= false;
};
