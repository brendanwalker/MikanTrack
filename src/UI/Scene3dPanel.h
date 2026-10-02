#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "glm/ext/matrix_float4x4.hpp"

#include "AvatarRetarget.h"
#include "TrackingTypes.h"

class AppConfig;
struct AvatarModel;
class AvatarSkeleton;
class GlFrameBuffer;
class GlSkinnedMeshRenderer;
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

// One view per configured camera, from the profiles' calibration.
// markerFromCamera maps OpenCV-convention camera space to world; the panel
// applies the GL flip for the frustum itself.
std::vector<SceneCameraView> makeSceneCameraViews(const AppConfig& config);

// World is Z-up (marker plane = XY); the renderer and orbit camera are Y-up.
// The one world-to-display conversion: world +Z becomes display +Y, so
// (x, y, z) -> (x, z, -y). Shared with the headless avatar render tool.
const glm::mat4& displayFromWorld();

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

	// Forearm length used to place the elbow estimate (meters); the
	// direction is measured by the wrist IMU, only the length is assumed
	void setForearmLength(float meters) { m_forearmLengthMeters= meters; }

	// The avatar to draw (null = none). The generation counter tells the panel
	// when the model changed and its GPU copy must be rebuilt.
	void setAvatar(std::shared_ptr<const AvatarModel> model, std::shared_ptr<const AvatarSkeleton> skeleton,
				   uint32_t generation);
	void setShowAvatar(bool bShow) { m_bShowAvatar= bShow; }
	// The retargeted pose to draw (copied); null draws the rest pose at the
	// world origin
	void setAvatarPose(const AvatarPose* pose);

	bool getShowPerCameraSkeletons() const { return m_bShowPerCameraSkeletons; }
	void setShowPerCameraSkeletons(bool bShow) { m_bShowPerCameraSkeletons= bShow; }

private:
	void renderScene(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
					 const std::vector<const TrackingFrameResult*>& perCameraResults, float aspect);
	void drawSkeleton(const TrackingFrameResult& result, float brightness, const glm::vec3* colorOverride);
	// Draws the avatar when one is shown; returns whether anything was drawn
	bool drawAvatar();

	std::unique_ptr<GlFrameBuffer> m_frameBuffer;
	std::unique_ptr<GlLineRenderer> m_lineRenderer;
	std::unique_ptr<OrbitCamera> m_camera;
	std::unique_ptr<GlSkinnedMeshRenderer> m_meshRenderer;
	bool m_bMeshRendererInitialized= false;

	std::shared_ptr<const AvatarModel> m_avatarModel;
	std::shared_ptr<const AvatarSkeleton> m_avatarSkeleton;
	uint32_t m_avatarGeneration= 0;
	uint32_t m_uploadedAvatarGeneration= 0;
	bool m_bShowAvatar= true;
	bool m_bHasAvatarPose= false;
	AvatarPose m_avatarPose;
	std::vector<glm::mat4> m_posedGlobals;
	bool m_bRenderInitialized= false;
	bool m_bShowPerCameraSkeletons= false;
	float m_forearmLengthMeters= 0.25f;
};
