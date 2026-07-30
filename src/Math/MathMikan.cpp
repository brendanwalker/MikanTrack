#include "MathMikan.h"
#include "MathUtility.h"
#include "MikanMathTypes.h"

#include <glm/gtx/euler_angles.hpp>

void MikanOrientationToEulerAngles(const MikanVector3f& x_axis, const MikanVector3f& y_axis,
								   const MikanVector3f& z_axis, float& out_x_angle, float& out_y_angle,
								   float& out_z_angle)
{
	const glm::mat3 R= glm::mat3(glm::vec3(x_axis.x, x_axis.y, x_axis.z), glm::vec3(y_axis.x, y_axis.y, y_axis.z),
								 glm::vec3(z_axis.x, z_axis.y, z_axis.z));

	float xRadians= 0, yRadians= 0, zRadians= 0;
	glm::extractEulerAngleXYZ(glm::mat4(R), xRadians, yRadians, zRadians);

	out_x_angle= xRadians * k_radians_to_degrees;
	out_y_angle= yRadians * k_radians_to_degrees;
	out_z_angle= zRadians * k_radians_to_degrees;
}

void EulerAnglesToMikanOrientation(const float x_angle, const float y_angle, const float z_angle,
								   MikanVector3f& out_x_axis, MikanVector3f& out_y_axis, MikanVector3f& out_z_axis)
{
	const float xRadians= x_angle * k_degrees_to_radians;
	const float yRadians= y_angle * k_degrees_to_radians;
	const float zRadians= z_angle * k_degrees_to_radians;
	const glm::mat3 R= glm::mat3(glm::eulerAngleXYZ(xRadians, yRadians, zRadians));

	out_x_axis= {R[0].x, R[0].y, R[0].z};
	out_y_axis= {R[1].x, R[1].y, R[1].z};
	out_z_axis= {R[2].x, R[2].y, R[2].z};
}
