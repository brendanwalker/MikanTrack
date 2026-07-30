#pragma once

// Plain-struct retype of MikanXR's camera intrinsics types
// (MikanXR: src/Libraries/MikanClientAPI/Public/MikanVideoSourceTypes.h).
// The reflection/serialization machinery is removed and the polymorphic
// intrinsics_ptr wrapper is replaced with by-value mono/stereo members, so
// code copied from MikanXR (CameraMath, calibration classes, MikanCamera)
// compiles unmodified without Refureku.

#include "MikanMathTypes.h"

// -- Constants -----
enum class MikanVideoSourceType
{
	MONO,
	STEREO
};

enum class MikanIntrinsicsType
{
	INVALID_CAMERA_INTRINSICS,
	MONO_CAMERA_INTRINSICS,
	STEREO_CAMERA_INTRINSICS,
};

// -- Structures -----

/// Radial and tangential lens distortion coefficients computed during lens calibration
/// See the OpenCV camera calibration docs for details
struct MikanDistortionCoefficients
{
	double k1= 0.0; ///< Radial Distortion Parameter 1 (r^2 numerator constant)
	double k2= 0.0; ///< Radial Distortion Parameter 2 (r^4 numerator constant)
	double k3= 0.0; ///< Radial Distortion Parameter 3 (r^6 numerator constant)
	double k4= 0.0; ///< Radial Distortion Parameter 4 (r^2 divisor constant)
	double k5= 0.0; ///< Radial Distortion Parameter 5 (r^4 divisor constant)
	double k6= 0.0; ///< Radial Distortion Parameter 6 (r^6 divisor constant)
	double p1= 0.0; ///< Tangential Distortion Parameter 1
	double p2= 0.0; ///< Tangential Distortion Parameter 2
};

/// Camera intrinsic common properties
struct MikanBaseIntrinsics
{
	double pixel_width= 0.0;  ///< Width of the camera buffer in pixels
	double pixel_height= 0.0; ///< Height of the camera buffer in pixels
	double aspect_ratio= 0.0; ///< The aspect ratio of each pixel (y focal length / x focal length)
	double hfov= 0.0;         ///< The horizontal field of view camera in degrees
	double vfov= 0.0;         ///< The vertical field of view camera in degrees
	double znear= 0.0;        ///< The distance of the near clipping plane in meters
	double zfar= 0.0;         ///< The distance of the far clipping plane in meters
};

/// Camera intrinsic properties for a monoscopic camera
struct MikanMonoIntrinsics : public MikanBaseIntrinsics
{
	// Distortion coefficients computed for the physical camera lens
	MikanDistortionCoefficients distortion_coefficients;
	// Intrinsic camera matrix containing focal lengths and principal point for the raw distorted image
	MikanMatrix3d distorted_camera_matrix;
	// Intrinsic camera matrix containing focal lengths and principal point for the undistorted image
	// NOTE: The hfov and vfov are computed from this matrix
	MikanMatrix3d undistorted_camera_matrix;
};

/// Camera intrinsic properties for a stereoscopic camera
struct MikanStereoIntrinsics : public MikanBaseIntrinsics
{
	MikanDistortionCoefficients left_distortion_coefficients;
	MikanMatrix3d left_camera_matrix;
	MikanDistortionCoefficients right_distortion_coefficients;
	MikanMatrix3d right_camera_matrix;
	MikanMatrix3d left_rectification_rotation;
	MikanMatrix3d right_rectification_rotation;
	MikanMatrix4x3d left_rectification_projection;
	MikanMatrix4x3d right_rectification_projection;
	MikanMatrix3d rotation_between_cameras;
	MikanVector3d translation_between_cameras;
	MikanMatrix3d essential_matrix;
	MikanMatrix3d fundamental_matrix;
	MikanMatrix4d reprojection_matrix;
};

/// Bundle containing all intrinsic video source properties.
/// Simplified from MikanXR's polymorphic-pointer version: mono and stereo
/// intrinsics are held by value and selected via intrinsics_type.
struct MikanVideoSourceIntrinsics
{
	MikanMonoIntrinsics mono;
	MikanStereoIntrinsics stereo;
	MikanIntrinsicsType intrinsics_type= MikanIntrinsicsType::INVALID_CAMERA_INTRINSICS;

	const MikanMonoIntrinsics& getMonoIntrinsics() const { return mono; }
	MikanMonoIntrinsics& getMonoIntrinsicsMutable() { return mono; }
	const MikanStereoIntrinsics& getStereoIntrinsics() const { return stereo; }
	MikanStereoIntrinsics& getStereoIntrinsicsMutable() { return stereo; }

	MikanMonoIntrinsics& makeMonoIntrinsics()
	{
		intrinsics_type= MikanIntrinsicsType::MONO_CAMERA_INTRINSICS;
		return mono;
	}

	MikanStereoIntrinsics& makeStereoIntrinsics()
	{
		intrinsics_type= MikanIntrinsicsType::STEREO_CAMERA_INTRINSICS;
		return stereo;
	}
};
