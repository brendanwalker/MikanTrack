#pragma once

#include "MikanRendererFwd.h"
#include "IMkCamera.h"

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"

#include <string>

enum eCameraMovementMode : int
{
	orbit,
	fly,
	stationary,
	ortho,
};

enum class eCameraProjectionMode : int
{
	perspective,
	orthographic,
};

// Axis-aligned orthographic viewpoints (UE-style)
enum class eCameraViewpoint : int
{
	top,
	bottom,
	front,
	back,
	left,
	right,
};

class MikanCamera : public IMkCamera
{
public:
	MikanCamera();
	virtual ~MikanCamera()= default;

	virtual void setName(const std::string& name) override { m_cameraName= name; }
	virtual const std::string& getName() const override { return m_cameraName; }

	virtual const glm::mat4& getProjectionMatrix() const override { return m_projectionMatrix; }
	virtual const glm::mat4& getViewMatrix() const override { return m_viewMatrix; }
	virtual glm::mat4 getViewProjectionMatrix() const override { return m_projectionMatrix * m_viewMatrix; }

	virtual float getAspectRatio() const override { return m_aspectRatio; }
	virtual float getHorizontalFOVDegrees() const override { return m_hFOVDegrees; }
	virtual float getVerticalFOVDegrees() const override { return m_vFOVDegrees; }
	virtual float getZNear() const override { return m_zNear; }
	virtual float getZFar() const override { return m_zFar; }

	virtual glm::vec3 getCameraPositionFromViewMatrix() const override;
	virtual glm::vec3 getCameraForwardFromViewMatrix() const override;
	virtual glm::vec3 getCameraRightFromViewMatrix() const override;
	virtual glm::vec3 getCameraUpFromViewMatrix() const override;
	virtual glm::mat4 getCameraTransformFromViewMatrix() const override;

	eCameraMovementMode getCameraMovementMode() const { return m_movementMode; }
	void setCameraMovementMode(eCameraMovementMode mode);

	// Projection mode (perspective vs orthographic)
	eCameraProjectionMode getProjectionMode() const { return m_projectionMode; }
	bool isOrthographic() const { return m_projectionMode == eCameraProjectionMode::orthographic; }
	void setProjectionMode(eCameraProjectionMode mode);

	// Orthographic viewpoint controls
	eCameraViewpoint getOrthoViewpoint() const { return m_orthoViewpoint; }
	void setOrthographicViewpoint(eCameraViewpoint viewpoint);
	float getOrthoExtent() const { return m_orthoExtent; }
	void setOrthoExtent(float extent);
	const glm::vec3& getOrthoTargetPosition() const { return m_orthoTargetPosition; }
	void setOrthoTargetPosition(const glm::vec3& target);
	void adjustOrthoTargetPosition(const glm::vec3& deltaTarget);

	// Update the projection to match the actual viewport aspect ratio
	void setViewportAspect(float aspectRatio);

	void applyMonoCameraIntrinsics(struct MikanVideoSourceIntrinsics* cameraIntrinsics);

	void setCameraTransform(const glm::mat4& poseXform);
	void setPosition(const glm::vec3& location);
	void lookAt(const glm::vec3& target);

	void adjustFlyForward(float distance);
	void adjustFlyRight(float distance);
	void adjustFlyUp(float distance);
	void adjustFlyYaw(float deltaDegrees);
	void adjustFlyPitch(float deltaDegrees);

	void setOrbitLocation(float yawDegrees, float pitchDegrees, float radius);
	void setOrbitYaw(float yawDegrees);
	void setOrbitPitch(float pitchDegrees);
	void setOrbitRadius(float radius);
	void setOrbitTargetPosition(const glm::vec3& cameraTarget);
	void adjustOrbitAngles(float deltaYaw, float deltaPitch);
	void adjustOrbitRadius(float deltaRadius);
	void adjustOrbitTargetPosition(const glm::vec3& deltaTarget);

	void computeCameraRayThruPixel(MikanViewportConstPtr viewportPtr, const glm::vec2& pixelLocation,
								   glm::vec3& outRayOrigin, glm::vec3& outRayDirection) const;

protected:
	void applyStationaryParamsToViewMatrix();
	void applyFlyParamsToViewMatrix();
	void applyOrbitParamsToViewMatrix();
	void applyOrthoParamsToViewMatrix();
	void rebuildProjectionMatrix();

	const float k_default_aspect_ratio= 16.f / 9.f;
	const float k_default_camera_vfov= 35.f;
	const float k_default_camera_z_near= 0.1f;
	const float k_default_camera_z_far= 5000.f;
	const float k_camera_min_zoom= 0.01f;

	std::string m_cameraName;

	float m_aspectRatio;
	float m_hFOVDegrees;
	float m_vFOVDegrees;
	float m_zNear;
	float m_zFar;
	glm::mat4 m_projectionMatrix;
	glm::mat4 m_viewMatrix;

	// Projection parameters
	eCameraProjectionMode m_projectionMode= eCameraProjectionMode::perspective;
	float m_viewportAspectRatio= 16.f / 9.f;

	// Orthographic camera parameters
	eCameraViewpoint m_orthoViewpoint= eCameraViewpoint::top;
	glm::vec3 m_orthoTargetPosition= glm::vec3(0.f);
	float m_orthoExtent= 5.f; // half-height of the view in world meters

	// Stationary camera parameters
	glm::mat4 m_stationaryTransform;

	// Stationary camera parameters
	float m_flyYawDegrees;
	float m_flyPitchDegrees;
	glm::mat4 m_flyTransform;

	// Orbit camera parameters
	float m_orbitYawDegrees;
	float m_orbitPitchDegrees;
	float m_orbitRadius;
	glm::vec3 m_orbitTargetPosition;

	eCameraMovementMode m_movementMode= eCameraMovementMode::orbit;
};
