#pragma once

#include <memory>
#include <vector>

#include "glm/ext/matrix_float4x4.hpp"

#include "TrackingTypes.h"

class GlFrameBuffer;
class GlLineRenderer;
class OrbitCamera;
struct MikanMonoIntrinsics;

// A camera's pose/intrinsics for frustum rendering.
// cameraToWorld maps OpenCV-convention camera space to world space (this is
// the markerFromCamera transform stored in ExtrinsicsConfig).
struct SceneCameraView
{
	glm::mat4 cameraToWorld{1.f};
	bool bHasExtrinsics= false;
	const MikanMonoIntrinsics* intrinsics= nullptr;
};

// Alternate 3D view: renders the marker-plane grid, marker axes, one frustum
// per calibrated camera, the FUSED hand/arm skeletons (full brightness) and
// optionally each camera's unfused skeleton (dimmed, in that camera's color)
// into an FBO shown as an ImGui image.
//
// World convention (from calibration): right-handed, meters, origin at the
// marker center, +Z out of the table. For display this is rotated into the
// renderer's Y-up space (world +Z becomes view up).
class Scene3dPanel
{
public:
	Scene3dPanel();
	~Scene3dPanel();

	void draw(const TrackingFrameResult& fusedResult,
			  const std::vector<SceneCameraView>& cameras,
			  const std::vector<const TrackingFrameResult*>& perCameraResults);

	bool getShowPerCameraSkeletons() const { return m_bShowPerCameraSkeletons; }
	void setShowPerCameraSkeletons(bool bShow) { m_bShowPerCameraSkeletons= bShow; }

private:
	void renderScene(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
					 const std::vector<const TrackingFrameResult*>& perCameraResults, float aspect);
	void drawSkeleton(const TrackingFrameResult& result, float brightness, const glm::vec3* colorOverride);

	std::unique_ptr<GlFrameBuffer> m_frameBuffer;
	std::unique_ptr<GlLineRenderer> m_lineRenderer;
	std::unique_ptr<OrbitCamera> m_camera;
	bool m_bRenderInitialized= false;
	bool m_bShowPerCameraSkeletons= false;
};
