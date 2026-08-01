#pragma once

#include <memory>
#include <vector>

#include "SettingsPanels.h" // TrackingPanelState
#include "VisionThread.h"    // VisionPreviewFrame

class App;
class CalibrationPanel;
class DevicePanel;
class ExtrinsicsWizard;
class IntrinsicsWizard;
class Scene3dPanel;
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

	// Reopens each configured camera's device/mode (called once at startup)
	void tryRestoreVideoDeviceFromConfig();

private:
	void drawDockspaceAndMenuBar();
	// Restores one camera's persisted device by path, then by friendly name
	bool restoreCameraDevice(int cameraIndex);

	App* m_app;

	std::unique_ptr<VideoPreviewPanel> m_videoPreviewPanel;
	std::unique_ptr<Scene3dPanel> m_scene3dPanel;
	std::unique_ptr<DevicePanel> m_devicePanel;
	std::unique_ptr<CalibrationPanel> m_calibrationPanel;
	std::unique_ptr<IntrinsicsWizard> m_intrinsicsWizard;
	std::unique_ptr<ExtrinsicsWizard> m_extrinsicsWizard;

	// Latest per-camera previews + the fused result (kept between updates so
	// the UI still has data when no new frame arrived this tick)
	std::vector<VisionPreviewFrame> m_latestPreviews;
	TrackingFrameResult m_latestFused;

	TrackingPanelState m_trackingPanelState;

	bool m_bShowLogPanel= true;
	bool m_bDockLayoutInitialized= false;
};
