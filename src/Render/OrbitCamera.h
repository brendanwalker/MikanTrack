#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"

// Free orbit camera for the 3D scene viewport, ported from the orbit-mode math
// in MikanXR's MikanCamera. Right-handed, +Y up.
//
// Pitch convention: pitch is the camera's look-direction pitch, so a NEGATIVE
// pitch places the camera above the target looking down (default -45 degrees).
// (MikanXR's MikanCamera used the opposite sign: its pitch was the elevation of
// the camera position above the target, clamped to [0, 90].)
class OrbitCamera
{
public:
	OrbitCamera();

	// -- Orbit controls --
	void setOrbitLocation(float yawDegrees, float pitchDegrees, float radius);
	void setOrbitTargetPosition(const glm::vec3& targetPosition);
	void adjustOrbitAngles(float deltaYawDegrees, float deltaPitchDegrees);
	void adjustOrbitRadius(float deltaRadius);
	void adjustOrbitTargetPosition(const glm::vec3& deltaTarget);

	float getOrbitYawDegrees() const { return m_orbitYawDegrees; }
	float getOrbitPitchDegrees() const { return m_orbitPitchDegrees; }
	float getOrbitRadius() const { return m_orbitRadius; }
	const glm::vec3& getOrbitTargetPosition() const { return m_orbitTargetPosition; }

	// -- Projection --
	void setPerspectiveProjection(float fovYDegrees, float aspectRatio, float zNear, float zFar);

	// -- Matrices --
	const glm::mat4& getViewMatrix() const { return m_viewMatrix; }
	const glm::mat4& getProjectionMatrix() const { return m_projectionMatrix; }
	glm::mat4 getViewProjection() const { return m_projectionMatrix * m_viewMatrix; }

	// Camera world position derived from the view matrix (assumes no scaling)
	glm::vec3 getCameraPosition() const;

private:
	void applyOrbitParamsToViewMatrix();

	static constexpr float k_camera_min_zoom= 0.01f;
	static constexpr float k_camera_max_abs_pitch= 89.f;

	// Orbit camera parameters
	float m_orbitYawDegrees;
	float m_orbitPitchDegrees;
	float m_orbitRadius;
	glm::vec3 m_orbitTargetPosition;

	// Projection parameters
	float m_fovYDegrees;
	float m_aspectRatio;
	float m_zNear;
	float m_zFar;

	glm::mat4 m_viewMatrix;
	glm::mat4 m_projectionMatrix;
};
