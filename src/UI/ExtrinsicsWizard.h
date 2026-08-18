#pragma once

#include <memory>
#include <string>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"
#include "opencv2/core/mat.hpp"

#include "ExtrinsicsValidation.h"
#include "HandOverlay.h"
#include "TrackingTypes.h"
#include "WizardResult.h"

class AppConfig;
class PatternPoseSampler;
class VideoPreviewPanel;
class VisionThread;
struct VisionPreviewFrame;

// Aruco-marker camera-extrinsics calibration wizard.
//
// Calibrates ALL cameras in one session, against the same physical marker
// placement, and saves atomically. Per-camera sessions are deliberately not
// offered: every camera's extrinsics are relative to the marker AS PLACED
// when that camera was calibrated, so calibrating cameras separately (with
// the marker nudged in between) bakes a silent world-frame disagreement into
// fusion that then masquerades as tracking noise.
//
// Requires intrinsics on every camera (frames arrive undistorted).
class ExtrinsicsWizard
{
public:
	enum class eState
	{
		VerifySetup,
		CaptureCameraPose,
		Review,
	};

	ExtrinsicsWizard(AppConfig* config, VisionThread* visionThread);
	~ExtrinsicsWizard();

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }
	eWizardResult getResult() const { return m_result; }

	bool update(float deltaSeconds, const std::vector<VisionPreviewFrame>& previews,
				VideoPreviewPanel* previewPanel);

	// -- Pure math helpers (static for the --test-extrinsics self test) -----

	// GL-convention camera space -> Z-up world space (origin at the marker,
	// +Z = marker plane normal toward the camera), given the sampler's
	// camera-from-marker transform. The solvePnP object points lie in the
	// marker's local XZ plane, so the plane normal is the marker's +/-Y axis.
	static glm::dmat4 computeWorldFromCameraPose(const glm::dmat4& cameraFromMarker);

	// Intersects the camera ray through an (undistorted) pixel with the
	// marker/table plane. Returns the hit point in GL camera space, meters.
	static bool raycastPixelOntoPlane(const struct MikanMonoIntrinsics& intrinsics,
									  const glm::dmat4& cameraFromMarker,
									  const glm::vec2& pixel,
									  glm::dvec3& outPoint);

private:
	// One camera's calibration-in-progress
	struct CameraCapture
	{
		std::unique_ptr<PatternPoseSampler> sampler;
		cv::Mat grayFrame;
		glm::dmat4 cameraFromMarker{1.0};
		bool bHasPose= false;
	};

	void drawWizardWindow(const std::vector<VisionPreviewFrame>& previews);
	void drawMarkerOverlays(VideoPreviewPanel* previewPanel);
	bool beginPoseCapture(const std::vector<VisionPreviewFrame>& previews);
	bool areMarkerParamsValid(std::string& outError) const;
	// True when every configured camera has calibrated intrinsics
	bool allCamerasHaveIntrinsics() const;
	// Cross-camera validation for the Review stage: triangulates the marker
	// corners each pair of cameras both saw and checks the reconstruction
	// against the marker's known geometry (fills m_pairQuality)
	void evaluateCalibrationQuality();

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifySetup;
	bool m_bActive= false;
	bool m_bWantsClose= false;
	eWizardResult m_result= eWizardResult::None;

	std::vector<CameraCapture> m_captures; // indexed by camera

	// Marker params being edited
	int m_markerId= 0;
	float m_markerLengthMM= 100.f;

	// Review-stage validation results (one entry per camera pair)
	std::vector<ExtrinsicsPairQuality> m_pairQuality;
	int m_worstPairIndex= -1;
};
