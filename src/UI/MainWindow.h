#pragma once

#include <memory>
#include <vector>

#include "AvatarRetarget.h"
#include "Scene3dPanel.h" // SceneGizmos
#include "SettingsPanels.h" // TrackingPanelState
#include "VisionThread.h"    // VisionPreviewFrame

class App;
class AvatarPanel;
class CalibrationPanel;
class DevicePanel;
class MainMenuScreen;
class Scene3dPanel;
class SetupFlow;
class TimelinePanel;
class VideoPreviewPanel;
class WizardHost;
enum class eWizardKind;

// Owns the ImGui layout: dockspace, menu bar, all panels and the wizard host.
// Created after ImGui/GL are initialized.
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
	// Manual wizard launches (menu, panels). Refused while a wizard or the
	// guided setup flow is running.
	void launchWizard(eWizardKind kind, int cameraIndex= 0);
	// Global hotkeys, active anywhere in the tracking UI: fires the action
	// bound to any key pressed this frame (the table is in the source file)
	void handleHotkeys();
	void dumpDiagnostics();
	void toggleRecording();
	// Restores one camera's persisted device by path, then by friendly name
	bool restoreCameraDevice(int cameraIndex);

	App* m_app;

	std::unique_ptr<MainMenuScreen> m_mainMenuScreen;
	std::unique_ptr<VideoPreviewPanel> m_videoPreviewPanel;
	std::unique_ptr<Scene3dPanel> m_scene3dPanel;
	std::unique_ptr<DevicePanel> m_devicePanel;
	std::unique_ptr<CalibrationPanel> m_calibrationPanel;
	std::unique_ptr<WizardHost> m_wizardHost;
	std::unique_ptr<TimelinePanel> m_timelinePanel;
	std::unique_ptr<AvatarPanel> m_avatarPanel;
	// Guided new-project setup chain; needs the wizards above, so it is
	// constructed last (in the constructor body)
	std::unique_ptr<SetupFlow> m_setupFlow;

	// Latest per-camera previews + the fused result (kept between updates so
	// the UI still has data when no new frame arrived this tick)
	std::vector<VisionPreviewFrame> m_latestPreviews;
	TrackingFrameResult m_latestFused;

	TrackingPanelState m_trackingPanelState;

	// The display-side avatar retarget, run on whatever fused result the 3D
	// scene shows (live, replay, or a preview pose); the OSC streamer runs its own on the
	// vision thread over the resolved poses
	AvatarRetarget m_avatarRetarget;
	AvatarPose m_avatarPose;
	// Which feed the display retarget last solved (live, replay, or a
	// preview pose), so a switch resets its root follow
	int m_avatarPoseFeed= -1;
	// Poses the scene's avatar and fills the elbow hint gizmos from it
	void updateDisplayAvatar(SceneGizmos& outGizmos);
	// Writes a dragged elbow hint back into the rig settings
	void applyGizmoDrags(const SceneGizmos& gizmos);

	bool m_bShowLogPanel= true;
	bool m_bShowSettingsPanel= true;
	bool m_bShowAvatarPanel= true;
	bool m_bDockLayoutInitialized= false;
};
