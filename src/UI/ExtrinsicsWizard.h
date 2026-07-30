#pragma once

#include <memory>
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

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }

	bool update(float deltaSeconds, const cv::Mat& bgrPreview, const TrackingFrameResult& trackingResult,
				ImDrawList* overlayDrawList, const ImageToScreenMapping& mapping);

private:
	void drawWizardWindow(const cv::Mat& bgrPreview, const TrackingFrameResult& trackingResult);
	void drawMarkerOverlay(ImDrawList* drawList, const ImageToScreenMapping& mapping);
	void beginPoseCapture(int frameWidth, int frameHeight);
	void updateHandScaleCapture(const TrackingFrameResult& trackingResult);

	// Intersects the camera ray through an (undistorted) pixel with the marker
	// plane; returns false if the ray is parallel. GL camera space, meters.
	bool raycastPixelOntoMarkerPlane(const glm::vec2& pixel, glm::dvec3& outPoint) const;

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifySetup;
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
	static constexpr int k_handScaleSampleCount= 12;
};
