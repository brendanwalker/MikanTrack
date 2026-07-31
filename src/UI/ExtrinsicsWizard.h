#pragma once

#include <memory>
#include <string>
#include <vector>

#include "glm/ext/matrix_double4x4.hpp"
#include "opencv2/core/mat.hpp"

#include "HandOverlay.h"
#include "TrackingTypes.h"

class AppConfig;
class ArucoMarkerPoseSampler;
class VisionThread;

// Aruco-marker camera-extrinsics + hand-scale calibration wizard.
// Requires intrinsics to be calibrated first (frames arrive undistorted).
//
// - CaptureCameraPose: averages N solvePnP samples of the printed origin
//   marker -> markerFromCamera (world anchor).
// - CaptureHandScale: with the hand flat on the table next to the marker,
//   ray-casts the wrist and middle-MCP landmarks onto the marker plane to
//   measure the user's real wrist->knuckle length (depth scale reference).
class ExtrinsicsWizard
{
public:
	enum class eState
	{
		VerifySetup,
		CaptureCameraPose,
		PoseTest,
		CaptureHandScale,
		Review,
	};

	ExtrinsicsWizard(AppConfig* config, VisionThread* visionThread);
	~ExtrinsicsWizard();

	void enter(int cameraIndex);
	void exit();
	bool isActive() const { return m_bActive; }
	int getCameraIndex() const { return m_cameraIndex; }

	bool update(float deltaSeconds, const cv::Mat& bgrPreview, const TrackingFrameResult& trackingResult,
				ImDrawList* overlayDrawList, const ImageToScreenMapping& mapping);

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
	void drawWizardWindow(const cv::Mat& bgrPreview, const TrackingFrameResult& trackingResult);
	void drawMarkerOverlay(ImDrawList* drawList, const ImageToScreenMapping& mapping);
	void beginPoseCapture(int frameWidth, int frameHeight);
	bool areMarkerParamsValid(std::string& outError) const;

	// Uses the wizard's captured pose; valid once m_bHasCameraPose
	glm::dmat4 computeWorldFromCamera() const;
	void updateHandScaleCapture(const TrackingFrameResult& trackingResult);

	// Member wrappers over the static helpers using the wizard's current state
	bool raycastPixelOntoMarkerPlane(const glm::vec2& pixel, glm::dvec3& outPoint) const;

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifySetup;
	int m_cameraIndex= 0;
	bool m_bActive= false;
	bool m_bWantsClose= false;

	std::unique_ptr<ArucoMarkerPoseSampler> m_poseSampler;
	cv::Mat m_grayFrame;

	// Marker params being edited
	int m_markerId= 0;
	float m_markerLengthMM= 100.f;

	// Captured camera pose (camera-from-marker, GL convention)
	glm::dmat4 m_cameraFromMarker{1.0};
	bool m_bHasCameraPose= false;

	// Hand-scale sampling
	std::vector<double> m_handScaleSamples;
	double m_measuredHandScale= 0.0;
	// Countdown before sampling starts, so the hand can settle into a flat
	// pose after clicking the button
	float m_handScaleCountdown= 0.f;
	static constexpr int k_handScaleSampleCount= 12;
	static constexpr float k_handScaleCountdownSeconds= 3.f;
};
