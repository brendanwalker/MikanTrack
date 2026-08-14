#pragma once

#include "glm/ext/matrix_double4x4.hpp"

#include "TrackingTypes.h"

// Camera-space -> world (marker-anchored) space for tracking results.
//
// COORDINATE CONVENTIONS
//   Camera space (input): OpenCV convention — +X right, +Y down, +Z forward
//     (out of the lens toward the scene), meters, origin at the camera's
//     optical center. This is what LandmarkTo3D produces by back-projecting
//     through the calibrated (undistorted) camera matrix.
//   World space (output): right-handed, meters, origin at the calibration
//     marker's center, +Z out of the marker plane (toward the camera when the
//     marker faces it).
//
//   markerFromCamera comes from the calibration module: it is the inverse of
//   the marker pose solved by cv::solvePnP (i.e. cameraFromMarker^-1), so it
//   is itself expressed in OpenCV axis conventions end-to-end. No OpenCV->GL
//   axis flip is applied here: a renderer or engine that wants a GL-style
//   camera (+Y up, -Z forward) must apply its own axis conversion AFTER this
//   transform (conventionally diag(1,-1,-1,1) on camera-space vectors, i.e.
//   negate Y and Z). Keeping this module purely in OpenCV conventions means
//   the world-space output only depends on how the marker frame is defined,
//   not on any rendering choice.
//
// Applies worldPoint= markerFromCamera * cameraPoint to every camera-space
// hand landmark and arm point that is present (hasCameraSpace), and sets the
// corresponding hasWorldSpace flags.
void applyWorldTransform(TrackingFrameResult& ioResult, const glm::dmat4& markerFromCamera);

// Camera's optical center in world space (the translation column)
glm::vec3 cameraPositionWorld(const glm::dmat4& markerFromCamera);

// World-space direction (normalized) of the ray from the camera's optical
// center through an undistorted pixel. fx/fy/cx/cy are the UNDISTORTED camera
// matrix the pixel lives in.
glm::vec3 pixelRayDirWorld(
	const glm::dmat4& markerFromCamera,
	float fx, float fy, float cx, float cy,
	const glm::vec2& px);
