#pragma once

//-- includes -----
#include "MikanMathTypes.h"
#include "MikanVideoSourceTypes.h"
#include "MikanCameraTypes.h"
#include "OpenCVFwd.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>

#define k_meters_to_millimeters 1000.0f
#define k_millimeters_to_meters (1.0f / k_meters_to_millimeters)

#define k_meters_to_centimeters 100.0f
#define k_centimeters_to_meters (1.0f / k_meters_to_centimeters)

// pre-declarations
class GlmTransform;

//-- methods -----
// GLM types to OpenCV types
cv::Matx33f glm_mat3_to_cv_mat33f(const glm::mat3& in);
cv::Matx33f glm_dmat3_to_cv_mat33f(const glm::dmat3& in);
cv::Matx33d glm_dmat3_to_cv_mat33d(const glm::dmat3& in);
cv::Matx34d glm_dmat4x3_to_cv_mat34d(const glm::dmat4x3& in);
cv::Matx44d glm_dmat4x4_to_cv_mat44d(const glm::dmat4& in);
cv::Vec3d glm_dvec3_to_cv_vec3d(const glm::dvec3& in);
cv::Quatd glm_dquat_to_cv_quatd(const glm::dquat& in);

// OpenCV types to GLM types
glm::mat3 cv_mat33f_to_glm_mat3(const cv::Matx33f& in);
glm::dmat3 cv_mat33d_to_glm_dmat3(const cv::Matx33d& in);
glm::dmat4 cv_mat44d_to_glm_dmat4(const cv::Matx44d& in);
glm::dmat4x3 cv_mat34d_to_glm_dmat4x3(const cv::Matx34d& in);
glm::vec3 cv_vec3f_to_glm_vec3(const cv::Vec3f& in);
glm::dvec3 cv_vec3d_to_glm_dvec3(const cv::Vec3d& in);
glm::dquat cv_quatd_to_glm_dquat(const cv::Quatd& in);

// OpenCV <-> Mikan types
MikanVector3d cv_vec3d_to_MikanVector3d(const cv::Vec3d& in);
MikanQuatd cv_quatd_to_MikanQuatd(const cv::Quatd& in);
MikanDistortionCoefficients cv_vec8_to_Mikan_distortion(const cv::Matx81d& cv_distortion_coeffs);
cv::Matx81d Mikan_distortion_to_cv_vec8(const MikanDistortionCoefficients& distortion_coeffs);
cv::Matx33d MikanMatrix3d_to_cv_mat33d(const MikanMatrix3d& in);
cv::Matx33f MikanMatrix3d_to_cv_mat33f(const MikanMatrix3d& in);
MikanMatrix3d cv_mat33d_to_MikanMatrix3d(const cv::Matx33d& in);
cv::Matx34d MikanMatrix4x3d_to_cv_mat34d(const MikanMatrix4x3d& in);
MikanMatrix4x3d MikanMatrix4x3d_to_cv_mat34d(const cv::Matx34d& in);

// Mikan types <-> GLM tyoes
glm::dvec3 MikanVector3d_to_glm_dvec3(const MikanVector3d& in);
glm::vec3 MikanVector3f_to_glm_vec3(const MikanVector3f& in);
MikanVector3f glm_vec3_to_MikanVector3f(const glm::vec3& in);
glm::dquat MikanQuatd_to_glm_dquat(const MikanQuatd& in);
MikanMatrix4f glm_mat4_to_MikanMatrix4f(const glm::mat4& in);
GlmTransform MikanTransform_to_glm_transform(const MikanTransform& in);
glm::mat4 MikanMatrix4f_to_glm_mat4(const MikanMatrix4f& in);
glm::quat MikanRotator3f_to_glm_quat(const MikanRotator3f& in);
glm::quat MikanQuatf_to_glm_quat(const MikanQuatf& in);
glm::quat MikanQuatd_to_glm_quat(const MikanQuatd& in);
MikanTransform glm_transform_to_MikanTransform(const GlmTransform& in);
MikanRotator3f glm_quat_to_MikanRotator3f(const glm::quat& in);
MikanQuatf glm_quat_to_MikanQuatf(const glm::quat& in);