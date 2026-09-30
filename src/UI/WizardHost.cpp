#include "WizardHost.h"

#include "BodyCalibrationWizard.h"
#include "CameraContext.h" // VisionPreviewFrame
#include "ExtrinsicsWizard.h"
#include "HandCalibrationWizard.h"
#include "IntrinsicsWizard.h"
#include "MountingWizard.h"
#include "VideoPreviewPanel.h"

WizardHost::WizardHost(AppConfig* config, VisionThread* visionThread)
	: m_intrinsicsWizard(std::make_unique<IntrinsicsWizard>(config, visionThread))
	, m_extrinsicsWizard(std::make_unique<ExtrinsicsWizard>(config, visionThread))
	, m_handCalibrationWizard(std::make_unique<HandCalibrationWizard>(config, visionThread))
	, m_mountingWizard(std::make_unique<MountingWizard>(config, visionThread))
	, m_bodyCalibrationWizard(std::make_unique<BodyCalibrationWizard>(config, visionThread))
{
}

WizardHost::~WizardHost()= default;

bool WizardHost::launch(eWizardKind kind, int cameraIndex)
{
	if (isAnyActive())
		return false;

	switch (kind)
	{
		case eWizardKind::Intrinsics:
			m_intrinsicsWizard->enter(cameraIndex);
			return true;
		case eWizardKind::Extrinsics:
			m_extrinsicsWizard->enter();
			return true;
		case eWizardKind::Hand:
			m_handCalibrationWizard->enter();
			return true;
		case eWizardKind::Mounting:
			m_mountingWizard->enter();
			return true;
		case eWizardKind::Body:
			m_bodyCalibrationWizard->enter();
			return true;
		default:
			return false;
	}
}

bool WizardHost::isAnyActive() const
{
	return getActiveKind() != eWizardKind::None;
}

eWizardKind WizardHost::getActiveKind() const
{
	if (m_intrinsicsWizard->isActive())
		return eWizardKind::Intrinsics;
	if (m_extrinsicsWizard->isActive())
		return eWizardKind::Extrinsics;
	if (m_handCalibrationWizard->isActive())
		return eWizardKind::Hand;
	if (m_mountingWizard->isActive())
		return eWizardKind::Mounting;
	if (m_bodyCalibrationWizard->isActive())
		return eWizardKind::Body;
	return eWizardKind::None;
}

int WizardHost::getIntrinsicsCameraIndex() const
{
	return m_intrinsicsWizard->isActive() ? m_intrinsicsWizard->getCameraIndex() : -1;
}

bool WizardHost::consumeCameraWizardStarted()
{
	const eWizardKind active= getActiveKind();
	const bool bCameraWizardActive= active == eWizardKind::Intrinsics || active == eWizardKind::Extrinsics;
	const bool bStarted= bCameraWizardActive && !m_bCameraWizardWasActive;
	m_bCameraWizardWasActive= bCameraWizardActive;
	return bStarted;
}

void WizardHost::update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
						const TrackingFrameResult& fused, VideoPreviewPanel* previewPanel)
{
	static const VisionPreviewFrame s_emptyPreview{};

	switch (getActiveKind())
	{
		case eWizardKind::Intrinsics:
		{
			const int wizardCamera= m_intrinsicsWizard->getCameraIndex();
			const VisionPreviewFrame& preview=
				wizardCamera < (int)previews.size() ? previews[wizardCamera] : s_emptyPreview;
			if (!m_intrinsicsWizard->update(deltaSeconds, preview.bgr, previewPanel->getLastDrawList(),
											previewPanel->getImageToScreenMapping(wizardCamera)))
			{
				m_intrinsicsWizard->exit();
			}
			break;
		}
		case eWizardKind::Extrinsics:
			if (!m_extrinsicsWizard->update(deltaSeconds, previews, previewPanel))
				m_extrinsicsWizard->exit();
			break;
		case eWizardKind::Mounting:
			// Unlike the camera calibration wizards this one leaves tracking
			// running - it needs live tracked hands for the straight-wrist pose
			if (!m_mountingWizard->update(deltaSeconds, fused))
				m_mountingWizard->exit();
			break;
		case eWizardKind::Body:
			// Also leaves tracking running: the fused wrists ARE the
			// measurement's ruler, so they have to keep arriving
			if (!m_bodyCalibrationWizard->update(deltaSeconds, previews, fused))
				m_bodyCalibrationWizard->exit();
			break;
		case eWizardKind::Hand:
			// Also leaves tracking running: both stages measure the tracked hands
			if (!m_handCalibrationWizard->update(deltaSeconds, fused))
				m_handCalibrationWizard->exit();
			break;
		default:
			break;
	}
}
