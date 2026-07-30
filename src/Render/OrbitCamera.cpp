#include "OrbitCamera.h"
#include "MathUtility.h"

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/ext/matrix_transform.hpp"

#include <cmath>

OrbitCamera::OrbitCamera()
	: m_orbitYawDegrees(0.f)
	, m_orbitPitchDegrees(-45.f) // Looking down at the target from above
	, m_orbitRadius(1.5f)
	, m_orbitTargetPosition(0.f)
	, m_fovYDegrees(45.f)
	, m_aspectRatio(16.f / 9.f)
	, m_zNear(0.1f)
	, m_zFar(100.f)
	, m_viewMatrix(1.f)
	, m_projectionMatrix(1.f)
{
	setPerspectiveProjection(m_fovYDegrees, m_aspectRatio, m_zNear, m_zFar);
	applyOrbitParamsToViewMatrix();
}

void OrbitCamera::setOrbitLocation(float yawDegrees, float pitchDegrees, float radius)
{
	m_orbitYawDegrees= wrap_degrees(yawDegrees);
	m_orbitPitchDegrees= clampf(pitchDegrees, -k_camera_max_abs_pitch, k_camera_max_abs_pitch);
	m_orbitRadius= fmaxf(radius, k_camera_min_zoom);
	applyOrbitParamsToViewMatrix();
}

void OrbitCamera::setOrbitTargetPosition(const glm::vec3& targetPosition)
{
	m_orbitTargetPosition= targetPosition;
	applyOrbitParamsToViewMatrix();
}

void OrbitCamera::adjustOrbitAngles(float deltaYawDegrees, float deltaPitchDegrees)
{
	setOrbitLocation(m_orbitYawDegrees + deltaYawDegrees, m_orbitPitchDegrees + deltaPitchDegrees, m_orbitRadius);
}

void OrbitCamera::adjustOrbitRadius(float deltaRadius)
{
	setOrbitLocation(m_orbitYawDegrees, m_orbitPitchDegrees, m_orbitRadius + deltaRadius);
}

void OrbitCamera::adjustOrbitTargetPosition(const glm::vec3& deltaTarget)
{
	m_orbitTargetPosition+= deltaTarget;
	applyOrbitParamsToViewMatrix();
}

void OrbitCamera::setPerspectiveProjection(float fovYDegrees, float aspectRatio, float zNear, float zFar)
{
	m_fovYDegrees= fovYDegrees;
	m_aspectRatio= (aspectRatio > 0.f) ? aspectRatio : m_aspectRatio;
	m_zNear= zNear;
	m_zFar= zFar;

	m_projectionMatrix= glm::perspective(degrees_to_radians(m_fovYDegrees), m_aspectRatio, m_zNear, m_zFar);
}

void OrbitCamera::applyOrbitParamsToViewMatrix()
{
	// Same math as MikanXR MikanCamera::applyOrbitParamsToViewMatrix, except the
	// camera's elevation above the target is -pitch (see pitch convention in the header)
	const float yawRadians= degrees_to_radians(m_orbitYawDegrees);
	const float elevationRadians= degrees_to_radians(-m_orbitPitchDegrees);
	const float xzRadiusAtElevation= m_orbitRadius * cosf(elevationRadians);
	const glm::vec3 cameraPosition(m_orbitTargetPosition.x + xzRadiusAtElevation * sinf(yawRadians),
								   m_orbitTargetPosition.y + m_orbitRadius * sinf(elevationRadians),
								   m_orbitTargetPosition.z + xzRadiusAtElevation * cosf(yawRadians));

	if (fabsf(m_orbitPitchDegrees) < 85.0f)
	{
		m_viewMatrix= glm::lookAt(cameraPosition,
								  m_orbitTargetPosition, // Look at the orbit target
								  glm::vec3(0, 1, 0));   // +Y is up.
	}
	else
	{
		// Near the poles the world-up vector becomes degenerate for lookAt;
		// use a yaw-derived horizontal up vector instead
		m_viewMatrix= glm::lookAt(cameraPosition,
								  m_orbitTargetPosition, // Look at the orbit target
								  glm::vec3(sinf(yawRadians), 0.0f, -cosf(yawRadians)));
	}
}

glm::vec3 OrbitCamera::getCameraPosition() const
{
	// Assumes no scaling
	const glm::mat3 rotMat(m_viewMatrix);
	const glm::vec3 d(m_viewMatrix[3]);
	const glm::vec3 position= -d * rotMat;

	return position;
}
