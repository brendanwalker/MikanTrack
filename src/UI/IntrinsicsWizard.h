#pragma once

#include <memory>
#include <string>

#include "opencv2/core/mat.hpp"

#include "HandOverlay.h"

class AppConfig;
class MonoLensDistortionCalibrator;
class VisionThread;

// Charuco camera-intrinsics calibration wizard.
// Runs pattern detection on the main thread from the preview frames
// (ML tracking and undistortion are paused while active).
class IntrinsicsWizard
{
public:
	enum class eState
	{
		SelectBoardParams,
		Capture,
		Solving,
		TestUndistort,
		Failed,
	};

	IntrinsicsWizard(AppConfig* config, VisionThread* visionThread);
	~IntrinsicsWizard();

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }

	// Called each frame while active. bgrPreview is the newest preview frame
	// (may be empty when no stream). Overlay drawing goes to the video panel's
	// draw list via the mapping. Returns false when the wizard wants to close.
	bool update(float deltaSeconds, const cv::Mat& bgrPreview, ImDrawList* overlayDrawList,
				const ImageToScreenMapping& mapping);

private:
	void drawWizardWindow(float deltaSeconds, const cv::Mat& bgrPreview);
	void drawPatternOverlay(ImDrawList* drawList, const ImageToScreenMapping& mapping);
	void beginCapture(int frameWidth, int frameHeight);
	bool areBoardParamsValid(std::string& outError) const;
	void applyResultToConfig();

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::SelectBoardParams;
	bool m_bActive= false;
	bool m_bWantsClose= false;

	std::unique_ptr<MonoLensDistortionCalibrator> m_calibrator;
	cv::Mat m_grayFrame;

	// Board params being edited (copied from config on enter)
	int m_boardCols= 11;
	int m_boardRows= 8;
	float m_squareLengthMM= 16.f;
	float m_markerLengthMM= 12.f;

	// Shown in the Failed state (e.g. cv::Exception message)
	std::string m_lastErrorMessage;
};
