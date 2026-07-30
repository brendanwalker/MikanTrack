#pragma once

class AppConfig;

// Calibration status panel: shows intrinsics/extrinsics/hand-scale state and
// launches the wizards (launch requests are returned to MainWindow).
class CalibrationPanel
{
public:
	explicit CalibrationPanel(AppConfig* config);

	struct DrawResult
	{
		bool bLaunchIntrinsicsWizard= false;
		bool bLaunchExtrinsicsWizard= false;
	};

	DrawResult draw(bool bWizardActive);

private:
	AppConfig* m_config;
};
