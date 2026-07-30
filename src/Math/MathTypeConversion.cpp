#include "MathTypeConversion.h"
#include "MathGLM.h"
#include "Transform.h"
#include <assert.h>

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif
#include "glm/gtx/euler_angles.hpp"

// GLM types to OpenCV types
cv::Matx33f glm_mat3_to_cv_mat33f(const glm::mat3& in)
{
	cv::Matx33f out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= in[col][row];
		}
	}

	return out;
}

cv::Matx33f glm_dmat3_to_cv_mat33f(const glm::dmat3& in)
{
	cv::Matx33f out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= (float)in[col][row];
		}
	}

	return out;
}

cv::Matx33d glm_dmat3_to_cv_mat33d(const glm::dmat3& in)
{
	cv::Matx33d out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= in[col][row];
		}
	}

	return out;
}

cv::Matx34d glm_dmat4x3_to_cv_mat34d(const glm::dmat4x3& in)
{
	cv::Matx34d out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= in[col][row];
		}
	}

	return out;
}

cv::Matx44d glm_dmat4x4_to_cv_mat44d(const glm::dmat4& in)
{
	cv::Matx44d out;
	for (int row= 0; row < 4; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= in[col][row];
		}
	}

	return out;
}

cv::Vec3d glm_dvec3_to_cv_vec3d(const glm::dvec3& in) { return cv::Vec3d(in.x, in.y, in.z); }

cv::Quatd glm_dquat_to_cv_quatd(const glm::dquat& in) { return cv::Quatd(in.w, in.x, in.y, in.z); }

// OpenCV types to Mikan types
glm::mat3 cv_mat33f_to_glm_mat3(const cv::Matx33f& in)
{
	glm::mat3 out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out[col][row]= in(row, col);
		}
	}

	return out;
}

glm::dmat3 cv_mat33d_to_glm_dmat3(const cv::Matx33d& in)
{
	glm::dmat3 out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out[col][row]= in(row, col);
		}
	}

	return out;
}

glm::dmat4 cv_mat44d_to_glm_dmat4(const cv::Matx44d& in)
{
	glm::dmat4 out;
	for (int row= 0; row < 4; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out[col][row]= in(row, col);
		}
	}

	return out;
}

glm::dmat4x3 cv_mat34d_to_glm_dmat4x3(const cv::Matx34d& in)
{
	glm::dmat4x3 out;
	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out[col][row]= in(row, col);
		}
	}

	return out;
}

glm::vec3 cv_vec3f_to_glm_vec3(const cv::Vec3f& in) { return glm::vec3(in(0), in(1), in(2)); }

glm::dvec3 cv_vec3d_to_glm_dvec3(const cv::Vec3d& in) { return glm::dvec3(in(0), in(1), in(2)); }

glm::dquat cv_quatd_to_glm_dquat(const cv::Quatd& in)
{
	glm::dquat result= glm::dquat(in.w, in.x, in.y, in.z);
	assert(is_double_nearly_equal(in.w, result.w, DBL_EPSILON));
	assert(is_double_nearly_equal(in.x, result.x, DBL_EPSILON));
	assert(is_double_nearly_equal(in.y, result.y, DBL_EPSILON));
	assert(is_double_nearly_equal(in.z, result.z, DBL_EPSILON));

	return result;
}

// OpenCV to Mikan types
MikanVector3d cv_vec3d_to_MikanVector3d(const cv::Vec3d& in) { return {in(0), in(1), in(2)}; }

MikanQuatd cv_quatd_to_MikanQuatd(const cv::Quatd& in)
{
	MikanQuatd result= {in.w, in.x, in.y, in.z};
	assert(is_double_nearly_equal(in.w, result.w, DBL_EPSILON));
	assert(is_double_nearly_equal(in.x, result.x, DBL_EPSILON));
	assert(is_double_nearly_equal(in.y, result.y, DBL_EPSILON));
	assert(is_double_nearly_equal(in.z, result.z, DBL_EPSILON));

	return result;
}

MikanDistortionCoefficients cv_vec8_to_Mikan_distortion(const cv::Matx81d& cv_distortion_coeffs)
{
	MikanDistortionCoefficients distortion_coeffs;

	distortion_coeffs.k1= cv_distortion_coeffs(0, 0);
	distortion_coeffs.k2= cv_distortion_coeffs(1, 0);
	distortion_coeffs.p1= cv_distortion_coeffs(2, 0);
	distortion_coeffs.p2= cv_distortion_coeffs(3, 0);
	distortion_coeffs.k3= cv_distortion_coeffs(4, 0);
	distortion_coeffs.k4= cv_distortion_coeffs(5, 0);
	distortion_coeffs.k5= cv_distortion_coeffs(6, 0);
	distortion_coeffs.k6= cv_distortion_coeffs(7, 0);

	return distortion_coeffs;
}

cv::Matx81d Mikan_distortion_to_cv_vec8(const MikanDistortionCoefficients& distortion_coeffs)
{
	cv::Matx81d cv_distortion_coeffs;
	cv_distortion_coeffs(0, 0)= distortion_coeffs.k1;
	cv_distortion_coeffs(1, 0)= distortion_coeffs.k2;
	cv_distortion_coeffs(2, 0)= distortion_coeffs.p1;
	cv_distortion_coeffs(3, 0)= distortion_coeffs.p2;
	cv_distortion_coeffs(4, 0)= distortion_coeffs.k3;
	cv_distortion_coeffs(5, 0)= distortion_coeffs.k4;
	cv_distortion_coeffs(6, 0)= distortion_coeffs.k5;
	cv_distortion_coeffs(7, 0)= distortion_coeffs.k6;

	return cv_distortion_coeffs;
}

cv::Matx33d MikanMatrix3d_to_cv_mat33d(const MikanMatrix3d& in)
{
	cv::Matx33d out;
	auto m= reinterpret_cast<const double(*)[3][3]>(&in);

	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// Mikan indexed by column first
			// OpenCV indexed by row first
			out(row, col)= (*m)[col][row];
		}
	}

	return out;
}

cv::Matx33f MikanMatrix3d_to_cv_mat33f(const MikanMatrix3d& in)
{
	cv::Matx33f out;
	auto m= reinterpret_cast<const double(*)[3][3]>(&in);

	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// Mikan indexed by column first
			// OpenCV indexed by row first
			out(row, col)= (float)(*m)[col][row];
		}
	}

	return out;
}

MikanMatrix3d cv_mat33d_to_MikanMatrix3d(const cv::Matx33d& in)
{
	MikanMatrix3d out;
	auto m= reinterpret_cast<double(*)[3][3]>(&out);

	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 3; ++col)
		{
			// Mikan indexed by column first
			// OpenCV indexed by row first
			(*m)[col][row]= in(row, col);
		}
	}

	return out;
}

cv::Matx34d MikanMatrix4x3d_to_cv_mat34d(const MikanMatrix4x3d& in)
{
	cv::Matx34d out;
	auto m= reinterpret_cast<const double(*)[4][3]>(&in);

	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			out(row, col)= (*m)[col][row];
		}
	}

	return out;
}

MikanMatrix4x3d MikanMatrix4x3d_to_cv_mat34d(const cv::Matx34d& in)
{
	MikanMatrix4x3d out;
	auto m= reinterpret_cast<double(*)[4][3]>(&out);

	for (int row= 0; row < 3; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			(*m)[col][row]= in(row, col);
		}
	}

	return out;
}

// Mikan types to GLM tyoes
glm::dvec3 MikanVector3d_to_glm_dvec3(const MikanVector3d& in) { return glm::dvec3(in.x, in.y, in.z); }

glm::vec3 MikanVector3f_to_glm_vec3(const MikanVector3f& in) { return glm::vec3(in.x, in.y, in.z); }

MikanVector3f glm_vec3_to_MikanVector3f(const glm::vec3& in) { return {in.x, in.y, in.z}; }

glm::dquat MikanQuatd_to_glm_dquat(const MikanQuatd& in) { return glm::dquat(in.w, in.x, in.y, in.z); }

GlmTransform MikanTransform_to_glm_transform(const MikanTransform& in)
{
	return GlmTransform(MikanVector3f_to_glm_vec3(in.position), MikanQuatf_to_glm_quat(in.rotation),
						MikanVector3f_to_glm_vec3(in.scale));
}

MikanMatrix4f glm_mat4_to_MikanMatrix4f(const glm::mat4& in)
{
	MikanMatrix4f out;
	auto m= reinterpret_cast<float(*)[4][4]>(&out);

	for (int row= 0; row < 4; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			(*m)[col][row]= in[col][row];
		}
	}

	return out;
}

glm::mat4 MikanMatrix4f_to_glm_mat4(const MikanMatrix4f& in)
{
	glm::mat4 out;
	auto m= reinterpret_cast<const float(*)[4][4]>(&in);

	for (int row= 0; row < 4; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			out[col][row]= (*m)[col][row];
		}
	}

	return out;
}

glm::quat MikanRotator3f_to_glm_quat(const MikanRotator3f& in)
{
	const float xRadians= in.x_angle * k_degrees_to_radians;
	const float yRadians= in.y_angle * k_degrees_to_radians;
	const float zRadians= in.z_angle * k_degrees_to_radians;

	return glm::quat(glm::mat3(glm::eulerAngleXYZ(xRadians, yRadians, zRadians)));
}

glm::quat MikanQuatf_to_glm_quat(const MikanQuatf& in) { return glm::quat(in.w, in.x, in.y, in.z); }

glm::quat MikanQuatd_to_glm_quat(const MikanQuatd& in) { return glm::quat(in.w, in.x, in.y, in.z); }

MikanTransform glm_transform_to_MikanTransform(const GlmTransform& in)
{
	MikanTransform xform;
	xform.rotation= glm_quat_to_MikanQuatf(in.getRotation());
	xform.scale= glm_vec3_to_MikanVector3f(in.getScale());
	xform.position= glm_vec3_to_MikanVector3f(in.getPosition());

	return xform;
}

MikanRotator3f glm_quat_to_MikanRotator3f(const glm::quat& in)
{
	float xRadians, yRadians, zRadians;
	glm_quat_to_euler_angles(in, xRadians, yRadians, zRadians);

	return {xRadians * k_radians_to_degrees, yRadians * k_radians_to_degrees, zRadians * k_radians_to_degrees};
}

MikanQuatf glm_quat_to_MikanQuatf(const glm::quat& in) { return {in.w, in.x, in.y, in.z}; }