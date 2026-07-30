#pragma once

#include "OpenCVFwd.h"
#include <opencv2/core/quaternion.hpp>
#include <vector>

// Compute the average of an array of 2d vectors
bool opencv_point2f_compute_average(const std::vector<cv::Point2f>& points, cv::Point2f& outAveragePoint);

// Compute the average of an array of 3d vectors
bool opencv_vec3d_compute_average(const std::vector<cv::Vec3d>& vectors, cv::Vec3d& outAverageVec);

// Compute the average of an array of quaternions
bool opencv_quaternion_compute_average(const std::vector<cv::Quatd>& quaternions, cv::Quatd& outAverageQuat);