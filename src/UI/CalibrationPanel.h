#pragma once

class AppConfig;

// Calibration status panel: per-camera intrinsics/extrinsics state plus the
// global hand scale, and launches the wizards (launch requests are returned
// to MainWindow with the camera they apply to).
class CalibrationPanel
{
public:
	explicit CalibrationPanel(AppConfig* config);

	struct DrawResult
	{
		bool bLaunchIntrinsicsWizard= false;
		bool bLaunchExtrinsicsWizard= false;
		int cameraIndex= 0;
	};

	DrawResult draw(bool bWizardActive);

private:
	AppConfig* m_config;
};
