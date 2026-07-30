#pragma once

#include "MikanMathExport.h"

MIKAN_MATH_FUNC(void) MikanOrientationToEulerAngles(const struct MikanVector3f& x_axis,
													const struct MikanVector3f& y_axis,
													const struct MikanVector3f& z_axis, float& out_x_angle,
													float& out_y_angle, float& out_z_angle);
MIKAN_MATH_FUNC(void) EulerAnglesToMikanOrientation(const float x_angle, const float y_angle, const float z_angle,
													struct MikanVector3f& out_x_axis, struct MikanVector3f& out_y_axis,
													struct MikanVector3f& out_z_axis);
