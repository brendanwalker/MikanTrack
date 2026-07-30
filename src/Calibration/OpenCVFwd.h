#pragma once

#include <vector>
#include <memory>

// -- pre-declarations -----
namespace cv
{
class VideoWriter;

class Mat;
class UMat;

template <typename _Tp, int m, int n>
class Matx;

using Matx34f= Matx<float, 3, 4>;
using Matx33f= Matx<float, 3, 3>;
using Matx34d= Matx<double, 3, 4>;
using Matx33d= Matx<double, 3, 3>;
using Matx44d= Matx<double, 4, 4>;
using Matx81f= Matx<float, 8, 1>;
using Matx81d= Matx<double, 8, 1>;

template <typename _Tp, int cn>
class Vec;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;

template <typename _Tp>
class Quat;

using Quatd= Quat<double>;
using Quatf= Quat<float>;

// Rodrigues Vectors, used to represent rotation
// Rotation Angle(radians) = norm(R)
// Rotation Unit Vector = R / angle
using Rvecf= Matx<float, 3, 1>;
using Rvecd= Matx<float, 3, 1>;

template <typename _Tp>
class Rect_;

using Rect2i= Rect_<int>;
using Rect2f= Rect_<float>;
using Rect2d= Rect_<double>;
using Rect= Rect2i;

template <typename _Tp>
class Point_;

typedef Point_<int> Point2i;
typedef Point_<float> Point2f;
typedef Point_<double> Point2d;

template <typename _Tp>
class Point3_;

typedef Point3_<int> Point3i;
typedef Point3_<float> Point3f;
typedef Point3_<double> Point3d;

template <typename _Tp>
class Size_;
typedef Size_<int> Size;

namespace aruco
{
class Dictionary;
}
}; // namespace cv

typedef std::vector<int> t_opencv_pointID_list;
typedef std::vector<cv::Point2f> t_opencv_point2d_list;
typedef std::vector<cv::Point3f> t_opencv_point3d_list;

class DeepNeuralNetwork;
using DeepNeuralNetworkPtr= std::shared_ptr<DeepNeuralNetwork>;

using ArucoDictionaryPtr= std::shared_ptr<cv::aruco::Dictionary>;