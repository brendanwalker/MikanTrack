#pragma once

#include <string>
#include <vector>

#include "WizardResult.h"

class App;
class BodyCalibrationWizard;
class ExtrinsicsWizard;
class HandCalibrationWizard;
class IntrinsicsWizard;
class MainWindow;
class MountingWizard;
class VideoPreviewPanel;

// Guided new-project setup: a chain of modal prompts and the existing
// calibration wizards in dependency order (cameras, intrinsics per camera,
// extrinsics, hands, then IMU mounting or body measurement by variant, then
// output protocol). Runs inside the normal tracking UI so the wizards keep
// their preview-panel plumbing; the prompts are modals drawn on top, and
// manual wizard launches are suppressed while the flow is active.
//
// The flow only ever runs on a project created moments earlier, so
// cancelling anywhere (always through a confirmation, since it is
// destructive) deletes that project from disk and returns to the main menu,
// with Resume pointing back at whatever project preceded it.
class SetupFlow
{
public:
	enum class eTrackingSetup
	{
		DualOverhead,
		DualOverheadJoyCons,
		TriCameraFront,
	};

	enum class eStep
	{
		Inactive,
		TrackingSetup,
		CameraSelection,
		CharucoPrint,
		IntrinsicsRunning, // loops over every camera
		ArucoPrint,
		ExtrinsicsRunning,
		HandCalibRunning,
		MountingRunning,  // JoyCon variant only
		BodyCalibRunning, // tri-camera variant only
		OutputProtocol,
		ConfirmCancel,
	};

	struct WizardSet
	{
		IntrinsicsWizard* intrinsics= nullptr;
		ExtrinsicsWizard* extrinsics= nullptr;
		HandCalibrationWizard* hand= nullptr;
		MountingWizard* mounting= nullptr;
		BodyCalibrationWizard* body= nullptr;
	};

	SetupFlow(App* app, MainWindow* mainWindow, VideoPreviewPanel* previewPanel,
			  const WizardSet& wizards);

	void begin();
	bool isActive() const { return m_step != eStep::Inactive; }

	// Draws the current prompt, or watches the running wizard and advances on
	// its result. Called once per frame from MainWindow, after the panels and
	// before the wizard update dispatch.
	void update();

	// True once when the flow wants App::discardNewProjectAndReturnToMenu();
	// MainWindow consumes it and defers the switch to the top of the next
	// frame
	bool consumeDiscardProjectRequest();

private:
	void transitionTo(eStep step);
	void enterConfirmCancel();
	void requestDiscardProject();

	void updateTrackingSetupPrompt();
	void updateCameraSelectionPrompt();
	void updateCharucoPrintPrompt();
	void updateArucoPrintPrompt();
	void updateOutputProtocolPrompt();
	void updateConfirmCancelPrompt();
	// All five *Running steps: launch the step's wizard once, then wait for it
	// to close and branch on Completed vs Cancelled
	void updateRunningWizardStep();

	// Opens the picked device in a camera slot and starts its stream, so the
	// selection prompt can show a live preview of it
	void openSelectedDevice(int slotIndex);
	// Writes the open devices into the config as the camera selections
	void applyCameraSelection();
	int getRequiredCameraCount() const;

	App* m_app= nullptr;
	MainWindow* m_mainWindow= nullptr;
	VideoPreviewPanel* m_previewPanel= nullptr;
	WizardSet m_wizards;

	eStep m_step= eStep::Inactive;
	eStep m_stepBeforeConfirm= eStep::Inactive;
	eTrackingSetup m_trackingSetup= eTrackingSetup::DualOverhead;

	bool m_bWizardLaunched= false;
	int m_intrinsicsCameraIndex= 0;
	bool m_bDiscardProjectRequested= false;

	// CameraSelection state: chosen enumeration index per camera slot, -1 =
	// nothing picked yet
	std::vector<int> m_selectedDeviceIndices;

	// Print-prompt edit fields (seeded from the config on step entry)
	int m_boardCols= 11;
	int m_boardRows= 8;
	float m_boardSquareLengthMM= 16.f;
	float m_boardMarkerLengthMM= 12.f;
	float m_arucoMarkerLengthMM= 100.f;
};
