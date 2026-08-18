#include "SetupFlow.h"

#include "imgui.h"

#include "App.h"
#include "AppConfig.h"
#include "BodyCalibrationWizard.h"
#include "ExtrinsicsWizard.h"
#include "HandCalibrationWizard.h"
#include "IntrinsicsWizard.h"
#include "MainWindow.h"
#include "MountingWizard.h"
#include "LocText.h"
#include "PatternExport.h"
#include "VideoCaptureSystem.h"
#include "VideoPreviewPanel.h"
#include "VisionThread.h"

static constexpr float k_promptWrapWidth= 440.f;
static const ImVec4 k_colorGood(0.4f, 1.f, 0.5f, 1.f);
static const ImVec4 k_colorWait(1.f, 0.85f, 0.3f, 1.f);

// Opens (if needed) and begins a viewport-centered modal for the current step
static bool beginCenteredModal(const char* title)
{
	const ImGuiViewport* viewport= ImGui::GetMainViewport();
	const ImVec2 center(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
						viewport->WorkPos.y + viewport->WorkSize.y * 0.5f);
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (!ImGui::IsPopupOpen(title))
		ImGui::OpenPopup(title);
	return ImGui::BeginPopupModal(title, nullptr, ImGuiWindowFlags_AlwaysAutoResize);
}

SetupFlow::SetupFlow(App* app, MainWindow* mainWindow, VideoPreviewPanel* previewPanel,
					 const WizardSet& wizards)
	: m_app(app)
	, m_mainWindow(mainWindow)
	, m_previewPanel(previewPanel)
	, m_wizards(wizards)
{
}

void SetupFlow::begin()
{
	m_bDiscardProjectRequested= false;
	m_trackingSetup= eTrackingSetup::DualOverhead;
	transitionTo(eStep::TrackingSetup);
}

void SetupFlow::update()
{
	switch (m_step)
	{
		case eStep::Inactive:
			break;
		case eStep::TrackingSetup:
			updateTrackingSetupPrompt();
			break;
		case eStep::CameraSelection:
			updateCameraSelectionPrompt();
			break;
		case eStep::CharucoPrint:
			updateCharucoPrintPrompt();
			break;
		case eStep::ArucoPrint:
			updateArucoPrintPrompt();
			break;
		case eStep::OutputProtocol:
			updateOutputProtocolPrompt();
			break;
		case eStep::ConfirmCancel:
			updateConfirmCancelPrompt();
			break;
		default:
			updateRunningWizardStep();
			break;
	}
}

bool SetupFlow::consumeDiscardProjectRequest()
{
	const bool bRequested= m_bDiscardProjectRequested;
	m_bDiscardProjectRequested= false;
	return bRequested;
}

int SetupFlow::getRequiredCameraCount() const
{
	return m_trackingSetup == eTrackingSetup::TriCameraFront ? 3 : 2;
}

void SetupFlow::transitionTo(eStep step)
{
	m_step= step;
	m_bWizardLaunched= false;

	AppConfig* config= m_app->getConfig();
	switch (step)
	{
		case eStep::CameraSelection:
		{
			// Size the config and capture slots up front, so each device can
			// stream a live preview the moment it is picked (a name alone
			// rarely says which physical camera is which)
			const int requiredCameras= getRequiredCameraCount();
			while ((int)config->cameras.size() < requiredCameras)
				config->cameras.emplace_back();
			for (int slotIndex= 0; slotIndex < requiredCameras; ++slotIndex)
			{
				// 720p default, matching DevicePanel's Add Camera: two
				// uncompressed 1080p streams can exceed one USB controller
				if (config->camera(slotIndex).video.modeName.empty())
					config->camera(slotIndex).video.modeName= "1280x720@30fps (NV12)";
			}
			m_app->applyCameraCountChange();
			m_mainWindow->refreshDevicePanelState();
			m_app->getVideoCapture()->refreshDeviceList();
			m_selectedDeviceIndices.assign(requiredCameras, -1);
			break;
		}
		case eStep::CharucoPrint:
			m_boardCols= config->charucoBoard.cols;
			m_boardRows= config->charucoBoard.rows;
			m_boardSquareLengthMM= (float)config->charucoBoard.squareLengthMM;
			m_boardMarkerLengthMM= (float)config->charucoBoard.markerLengthMM;
			break;
		case eStep::IntrinsicsRunning:
			m_intrinsicsCameraIndex= 0;
			break;
		case eStep::ArucoPrint:
			m_arucoMarkerLengthMM= (float)config->camera(0).extrinsics.markerLengthMM;
			break;
		default:
			break;
	}
}

void SetupFlow::enterConfirmCancel()
{
	m_stepBeforeConfirm= m_step;
	m_step= eStep::ConfirmCancel;
}

void SetupFlow::requestDiscardProject()
{
	m_step= eStep::Inactive;
	m_bDiscardProjectRequested= true;
}

void SetupFlow::updateTrackingSetupPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalTrackingSetup")))
		return;

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.trackingSetupPrompt"));
	ImGui::Spacing();

	int setup= (int)m_trackingSetup;
	ImGui::RadioButton(locLabel("setupFlow.dualOverheadRadio"), &setup, (int)eTrackingSetup::DualOverhead);
	ImGui::TextDisabled("%s", locText("setupFlow.dualOverheadHint"));
	ImGui::RadioButton(locLabel("setupFlow.dualOverheadJoyConsRadio"), &setup,
					   (int)eTrackingSetup::DualOverheadJoyCons);
	ImGui::TextDisabled("%s", locText("setupFlow.dualOverheadJoyConsHint"));
	ImGui::RadioButton(locLabel("setupFlow.triCameraFrontRadio"), &setup,
					   (int)eTrackingSetup::TriCameraFront);
	ImGui::TextDisabled("%s", locText("setupFlow.triCameraFrontHint"));
	m_trackingSetup= (eTrackingSetup)setup;
	ImGui::PopTextWrapPos();

	ImGui::Separator();
	if (ImGui::Button(locLabel("common.next"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		transitionTo(eStep::CameraSelection);
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		enterConfirmCancel();
	}
	ImGui::EndPopup();
}

void SetupFlow::updateCameraSelectionPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalSelectCameras")))
		return;

	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	const int requiredCameras= getRequiredCameraCount();

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.pickCameraPrompt"));
	ImGui::PopTextWrapPos();
	ImGui::Spacing();

	for (int slotIndex= 0; slotIndex < requiredCameras; ++slotIndex)
	{
		const bool bFrontCamera=
			m_trackingSetup == eTrackingSetup::TriCameraFront && slotIndex == 2;
		const std::string label= bFrontCamera
			? locFormat("setupFlow.frontCameraLabel")
			: locFormat("setupFlow.overheadCameraLabelFmt", slotIndex + 1);

		int& selected= m_selectedDeviceIndices[slotIndex];
		std::string previewName= locText("setupFlow.selectCameraPlaceholder");
		if (selected >= 0)
		{
			std::string name;
			if (videoCapture->getDeviceFriendlyName((size_t)selected, name))
				previewName= name;
			else
				selected= -1; // the device list changed underneath the pick
		}

		if (ImGui::BeginCombo(label.c_str(), previewName.c_str()))
		{
			for (int deviceIndex= 0; deviceIndex < (int)videoCapture->getDeviceCount(); ++deviceIndex)
			{
				std::string name;
				if (!videoCapture->getDeviceFriendlyName((size_t)deviceIndex, name))
					continue;

				bool bTakenElsewhere= false;
				for (int otherSlot= 0; otherSlot < requiredCameras; ++otherSlot)
					bTakenElsewhere|= otherSlot != slotIndex &&
									  m_selectedDeviceIndices[otherSlot] == deviceIndex;

				ImGui::PushID(deviceIndex);
				ImGui::BeginDisabled(bTakenElsewhere);
				if (ImGui::Selectable(name.c_str(), selected == deviceIndex) &&
					selected != deviceIndex)
				{
					selected= deviceIndex;
					openSelectedDevice(slotIndex);
				}
				ImGui::EndDisabled();
				ImGui::PopID();
			}
			ImGui::EndCombo();
		}

		// Live preview of the picked device, so the name can be matched to
		// the physical camera
		const uint32_t previewTextureId= m_previewPanel->getCameraTextureId(slotIndex);
		if (selected >= 0 && previewTextureId != 0)
		{
			const float previewWidth= 220.f;
			const float previewHeight= previewWidth / m_previewPanel->getCameraAspect(slotIndex);
			ImGui::Image((ImTextureID)(intptr_t)previewTextureId,
						 ImVec2(previewWidth, previewHeight));
		}
		ImGui::Spacing();
	}

	if (ImGui::Button(locLabel("setupFlow.refreshDevicesButton")))
		videoCapture->refreshDeviceList();

	bool bReady= true;
	for (int slotIndex= 0; slotIndex < requiredCameras; ++slotIndex)
	{
		bReady&= m_selectedDeviceIndices[slotIndex] >= 0 &&
				 videoCapture->getIsDeviceOpen(slotIndex);
	}

	if (m_trackingSetup == eTrackingSetup::DualOverheadJoyCons)
	{
		ImGui::SeparatorText(locText("setupFlow.wristJoyConsHeader"));
		VisionThread* visionThread= m_app->getVisionThread();
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const ImuSideStatus status= visionThread->getImuSideStatus((eHandSide)sideIndex);
			const char* connectedKey= sideIndex == 0 ? "setupFlow.leftConnectedFmt" : "setupFlow.rightConnectedFmt";
			const char* notConnectedKey= sideIndex == 0 ? "setupFlow.leftNotConnected" : "setupFlow.rightNotConnected";
			if (status.deviceConnected)
				ImGui::TextColored(k_colorGood, locText(connectedKey), status.deviceName.c_str());
			else
			{
				ImGui::TextColored(k_colorWait, "%s", locText(notConnectedKey));
				bReady= false;
			}
		}
		ImGui::PushTextWrapPos(k_promptWrapWidth);
		ImGui::TextDisabled("%s", locText("setupFlow.pairingHint"));
		ImGui::PopTextWrapPos();
	}

	ImGui::Separator();
	ImGui::BeginDisabled(!bReady);
	if (ImGui::Button(locLabel("common.ok"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		applyCameraSelection();
		transitionTo(eStep::CharucoPrint);
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		enterConfirmCancel();
	}
	ImGui::EndPopup();
}

void SetupFlow::openSelectedDevice(int slotIndex)
{
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	const int deviceIndex= m_selectedDeviceIndices[slotIndex];

	videoCapture->closeDevice(slotIndex);
	if (deviceIndex < 0)
		return;

	std::string devicePath;
	if (!videoCapture->getDevicePath((size_t)deviceIndex, devicePath))
		return;

	if (videoCapture->openDeviceByPath(slotIndex, devicePath))
	{
		videoCapture->setVideoModeByName(slotIndex,
										 m_app->getConfig()->camera(slotIndex).video.modeName);
		videoCapture->startStream(slotIndex);
	}
}

void SetupFlow::applyCameraSelection()
{
	AppConfig* config= m_app->getConfig();
	VideoCaptureSystem* videoCapture= m_app->getVideoCapture();
	const int requiredCameras= getRequiredCameraCount();

	// The devices were already opened and streamed as they were picked (for
	// the live previews); this persists what is open as the project's cameras
	for (int slotIndex= 0; slotIndex < requiredCameras; ++slotIndex)
	{
		CameraProfile& profile= config->camera(slotIndex);
		profile.video.devicePath= videoCapture->getCurrentDevicePath(slotIndex);
		profile.video.deviceName= videoCapture->getCurrentDeviceFriendlyName(slotIndex);
	}

	// The front camera is the one that sees the user upright, so it alone
	// runs the body-pose stage; dual variants leave body pose off everywhere
	if (m_trackingSetup == eTrackingSetup::TriCameraFront)
		config->camera(2).bodyPose.enabled= true;
	config->imu.enabled= m_trackingSetup == eTrackingSetup::DualOverheadJoyCons;

	config->markDirty();
	m_mainWindow->refreshDevicePanelState();
}

void SetupFlow::updateCharucoPrintPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalPrintIntrinsicsBoard")))
		return;

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.charucoPrintPrompt"));
	ImGui::PopTextWrapPos();
	ImGui::Spacing();

	ImGui::InputInt(locLabel("setupFlow.boardColumnsLabel"), &m_boardCols);
	ImGui::InputInt(locLabel("setupFlow.boardRowsLabel"), &m_boardRows);
	ImGui::InputFloat(locLabel("setupFlow.blackSquareSizeMmLabel"), &m_boardSquareLengthMM, 0.f, 0.f, "%.2f");
	ImGui::InputFloat(locLabel("setupFlow.markerSizeMmLabel"), &m_boardMarkerLengthMM, 0.f, 0.f, "%.2f");

	if (ImGui::Button(locLabel("setupFlow.printBoardButton"), ImVec2(-1, 0)))
	{
		exportAndOpenCharucoBoard(m_boardCols, m_boardRows, m_boardSquareLengthMM,
								  m_boardMarkerLengthMM, eCharucoDictionaryType::DICT_6X6);
	}

	const bool bParamsValid= m_boardCols >= 2 && m_boardRows >= 2 &&
							 m_boardMarkerLengthMM > 0.f &&
							 m_boardSquareLengthMM > m_boardMarkerLengthMM;
	if (!bParamsValid)
		ImGui::TextColored(k_colorWait, "%s", locText("setupFlow.charucoValidationWarning"));

	ImGui::Separator();
	ImGui::BeginDisabled(!bParamsValid);
	if (ImGui::Button(locLabel("common.continueLabel"), ImVec2(120, 0)))
	{
		AppConfig* config= m_app->getConfig();
		config->charucoBoard.cols= m_boardCols;
		config->charucoBoard.rows= m_boardRows;
		config->charucoBoard.squareLengthMM= m_boardSquareLengthMM;
		config->charucoBoard.markerLengthMM= m_boardMarkerLengthMM;
		config->markDirty();
		ImGui::CloseCurrentPopup();
		transitionTo(eStep::IntrinsicsRunning);
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		enterConfirmCancel();
	}
	ImGui::EndPopup();
}

void SetupFlow::updateArucoPrintPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalPrintExtrinsicsMarker")))
		return;

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.arucoPrintPrompt"));
	ImGui::PopTextWrapPos();
	ImGui::Spacing();

	ImGui::InputFloat(locLabel("setupFlow.markerSizeMmLabel"), &m_arucoMarkerLengthMM, 0.f, 0.f, "%.2f");

	AppConfig* config= m_app->getConfig();
	if (ImGui::Button(locLabel("setupFlow.printMarkerButton"), ImVec2(-1, 0)))
	{
		exportAndOpenArucoMarker(config->camera(0).extrinsics.markerId,
								 eCharucoDictionaryType::DICT_6X6, m_arucoMarkerLengthMM);
	}

	const bool bParamsValid= m_arucoMarkerLengthMM > 0.f;
	ImGui::Separator();
	ImGui::BeginDisabled(!bParamsValid);
	if (ImGui::Button(locLabel("common.continueLabel"), ImVec2(120, 0)))
	{
		// The extrinsics wizard seeds its marker params from camera 0, but the
		// length is written everywhere so a per-camera read stays consistent
		for (size_t cameraIndex= 0; cameraIndex < config->cameraCount(); ++cameraIndex)
			config->camera(cameraIndex).extrinsics.markerLengthMM= m_arucoMarkerLengthMM;
		config->markDirty();
		ImGui::CloseCurrentPopup();
		transitionTo(eStep::ExtrinsicsRunning);
	}
	ImGui::EndDisabled();
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		enterConfirmCancel();
	}
	ImGui::EndPopup();
}

void SetupFlow::updateOutputProtocolPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalOutputProtocol")))
		return;

	AppConfig* config= m_app->getConfig();

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.outputProtocolPrompt"));
	ImGui::PopTextWrapPos();
	ImGui::Spacing();

	static const char* k_outputModeKeys[]= {"setupFlow.outputFormatMikan", "setupFlow.outputFormatVmc"};

	int outputMode= (int)config->osc.outputMode;
	if (ImGui::BeginCombo(locLabel("setupFlow.outputFormatLabel"), locText(k_outputModeKeys[outputMode])))
	{
		for (int modeIndex= 0; modeIndex < 2; ++modeIndex)
		{
			if (ImGui::Selectable(locLabel(k_outputModeKeys[modeIndex]), outputMode == modeIndex))
			{
				outputMode= modeIndex;
				config->osc.outputMode= (eOscOutputMode)outputMode;
				config->markDirty();
				m_app->getVisionThread()->requestConfigRefresh();
			}
		}
		ImGui::EndCombo();
	}

	ImGui::Separator();
	if (ImGui::Button(locLabel("common.finish"), ImVec2(120, 0)))
	{
		config->save();
		ImGui::CloseCurrentPopup();
		m_step= eStep::Inactive;
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("common.cancel"), ImVec2(120, 0)))
	{
		ImGui::CloseCurrentPopup();
		enterConfirmCancel();
	}
	ImGui::EndPopup();
}

void SetupFlow::updateConfirmCancelPrompt()
{
	if (!beginCenteredModal(locWindowTitle("windows.modalCancelSetup")))
		return;

	ImGui::PushTextWrapPos(k_promptWrapWidth);
	ImGui::TextUnformatted(locText("setupFlow.confirmCancelPrompt"));
	ImGui::PopTextWrapPos();

	ImGui::Separator();
	if (ImGui::Button(locLabel("setupFlow.continueSetupButton"), ImVec2(160, 0)))
	{
		ImGui::CloseCurrentPopup();
		m_step= m_stepBeforeConfirm;
		m_bWizardLaunched= false; // re-enter the wizard the user backed out of
	}
	ImGui::SameLine();
	if (ImGui::Button(locLabel("setupFlow.deleteProjectButton"), ImVec2(140, 0)))
	{
		ImGui::CloseCurrentPopup();
		requestDiscardProject();
	}
	ImGui::EndPopup();
}

void SetupFlow::updateRunningWizardStep()
{
	if (!m_bWizardLaunched)
	{
		switch (m_step)
		{
			case eStep::IntrinsicsRunning:
				m_wizards.intrinsics->enter(m_intrinsicsCameraIndex);
				break;
			case eStep::ExtrinsicsRunning:
				m_wizards.extrinsics->enter();
				break;
			case eStep::HandCalibRunning:
				m_wizards.hand->enter();
				break;
			case eStep::MountingRunning:
				m_wizards.mounting->enter();
				break;
			case eStep::BodyCalibRunning:
				m_wizards.body->enter();
				break;
			default:
				break;
		}
		m_bWizardLaunched= true;
		return;
	}

	bool bWizardActive= false;
	eWizardResult result= eWizardResult::None;
	switch (m_step)
	{
		case eStep::IntrinsicsRunning:
			bWizardActive= m_wizards.intrinsics->isActive();
			result= m_wizards.intrinsics->getResult();
			break;
		case eStep::ExtrinsicsRunning:
			bWizardActive= m_wizards.extrinsics->isActive();
			result= m_wizards.extrinsics->getResult();
			break;
		case eStep::HandCalibRunning:
			bWizardActive= m_wizards.hand->isActive();
			result= m_wizards.hand->getResult();
			break;
		case eStep::MountingRunning:
			bWizardActive= m_wizards.mounting->isActive();
			result= m_wizards.mounting->getResult();
			break;
		case eStep::BodyCalibRunning:
			bWizardActive= m_wizards.body->isActive();
			result= m_wizards.body->getResult();
			break;
		default:
			break;
	}

	if (bWizardActive)
		return;

	if (result != eWizardResult::Completed)
	{
		enterConfirmCancel();
		return;
	}

	switch (m_step)
	{
		case eStep::IntrinsicsRunning:
			++m_intrinsicsCameraIndex;
			if (m_intrinsicsCameraIndex < (int)m_app->getConfig()->cameraCount())
				m_bWizardLaunched= false; // same step, next camera
			else
				transitionTo(eStep::ArucoPrint);
			break;
		case eStep::ExtrinsicsRunning:
			transitionTo(eStep::HandCalibRunning);
			break;
		case eStep::HandCalibRunning:
			// The JoyCon variant calibrates the IMU mounting next; the
			// tri-camera variant measures the body on its front camera; the
			// plain dual variant has neither (body pose stays off by design)
			if (m_trackingSetup == eTrackingSetup::DualOverheadJoyCons)
				transitionTo(eStep::MountingRunning);
			else if (m_trackingSetup == eTrackingSetup::TriCameraFront)
				transitionTo(eStep::BodyCalibRunning);
			else
				transitionTo(eStep::OutputProtocol);
			break;
		case eStep::MountingRunning:
		case eStep::BodyCalibRunning:
			transitionTo(eStep::OutputProtocol);
			break;
		default:
			break;
	}
}
