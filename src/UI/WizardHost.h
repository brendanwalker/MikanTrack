#pragma once

#include <memory>
#include <vector>

#include "TrackingTypes.h"

class AppConfig;
class BodyCalibrationWizard;
class ExtrinsicsWizard;
class HandCalibrationWizard;
class IntrinsicsWizard;
class MountingWizard;
class VideoPreviewPanel;
class VisionThread;
struct VisionPreviewFrame;

enum class eWizardKind
{
	None,
	Intrinsics,
	Extrinsics,
	Hand,
	Mounting,
	Body,
};

// Owns the five calibration wizards and the rule that at most one runs at a
// time. Launches go through launch(), which refuses while one is active;
// update() drives whichever is active and closes it when it asks to. The
// guided setup flow reaches the wizards directly through the accessors,
// since it watches each one's result rather than just its liveness.
class WizardHost
{
public:
	WizardHost(AppConfig* config, VisionThread* visionThread);
	~WizardHost();

	// Starts a wizard. False, and nothing happens, while any wizard is
	// active. cameraIndex is the intrinsics wizard's camera; the others
	// ignore it.
	bool launch(eWizardKind kind, int cameraIndex= 0);

	bool isAnyActive() const;
	eWizardKind getActiveKind() const;
	// The camera the intrinsics wizard is calibrating, -1 when it is not
	// active (the extrinsics wizard uses every camera at once)
	int getIntrinsicsCameraIndex() const;

	// True once on the frame a camera calibration wizard (intrinsics or
	// extrinsics) becomes active. The pattern feed is that wizard's whole
	// UI, so the caller fronts the video preview.
	bool consumeCameraWizardStarted();

	// Runs the active wizard on top of the panels and exits it when it wants
	// to close. The camera wizards consume THEIR camera's per-camera preview
	// and result; the others consume the fused output.
	void update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
				const TrackingFrameResult& fused, VideoPreviewPanel* previewPanel);

	IntrinsicsWizard* getIntrinsicsWizard() { return m_intrinsicsWizard.get(); }
	ExtrinsicsWizard* getExtrinsicsWizard() { return m_extrinsicsWizard.get(); }
	HandCalibrationWizard* getHandCalibrationWizard() { return m_handCalibrationWizard.get(); }
	MountingWizard* getMountingWizard() { return m_mountingWizard.get(); }
	BodyCalibrationWizard* getBodyCalibrationWizard() { return m_bodyCalibrationWizard.get(); }

private:
	std::unique_ptr<IntrinsicsWizard> m_intrinsicsWizard;
	std::unique_ptr<ExtrinsicsWizard> m_extrinsicsWizard;
	std::unique_ptr<HandCalibrationWizard> m_handCalibrationWizard;
	std::unique_ptr<MountingWizard> m_mountingWizard;
	std::unique_ptr<BodyCalibrationWizard> m_bodyCalibrationWizard;

	// Rising-edge tracker behind consumeCameraWizardStarted
	bool m_bCameraWizardWasActive= false;
};
