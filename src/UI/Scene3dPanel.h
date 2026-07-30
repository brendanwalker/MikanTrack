#pragma once

#include <memory>

#include "glm/ext/matrix_float4x4.hpp"

#include "TrackingTypes.h"

class GlFrameBuffer;
class GlLineRenderer;
class OrbitCamera;
struct MikanMonoIntrinsics;

// Alternate 3D view: renders the marker-plane grid, marker axes, the camera
// frustum and the tracked hand/arm skeletons into an FBO shown as an ImGui image.
//
// World convention (from calibration): right-handed, meters, origin at the
// marker center, +Z out of the table. For display this is rotated into the
// renderer's Y-up space (world +Z becomes view up).
class Scene3dPanel
{
public:
	Scene3dPanel();
	~Scene3dPanel();

	// cameraToWorld: transform mapping GL-convention camera space (-Z forward)
	// into world space; only used when bHasExtrinsics is true.
	void draw(const TrackingFrameResult& result,
			  const glm::mat4& cameraToWorld,
			  bool bHasExtrinsics,
			  const MikanMonoIntrinsics* intrinsics);

private:
	void renderScene(const TrackingFrameResult& result, const glm::mat4& cameraToWorld, bool bHasExtrinsics,
					 const MikanMonoIntrinsics* intrinsics, float aspect);

	std::unique_ptr<GlFrameBuffer> m_frameBuffer;
	std::unique_ptr<GlLineRenderer> m_lineRenderer;
	std::unique_ptr<OrbitCamera> m_camera;
	bool m_bRenderInitialized= false;
};
