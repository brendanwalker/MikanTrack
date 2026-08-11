#pragma once

#include <vector>

#include "glm/ext/matrix_double4x4.hpp"
#include "opencv2/core/types.hpp"

#include "MikanVideoSourceTypes.h"

// Cross-camera validation of calibrated extrinsics, measured against the
// calibration board itself.
//
// WHY NOT PER-CAMERA REPROJECTION: each camera's solvePnP already minimizes
// its own reprojection error, so a low value there says the pose fits that
// camera's own detections - it says nothing about whether two cameras AGREE.
// What the tracking pipeline actually depends on is the RELATIVE pose, so this
// measures the same operation the hand pipeline performs: triangulate corners
// both cameras saw, then check the reconstruction against the board's known
// geometry. Errors show up in millimeters, the units that matter downstream.

// One camera's averaged board observations plus its calibration
struct CameraBoardObservations
{
	int cameraIndex= -1;
	MikanMonoIntrinsics intrinsics;
	// CV-convention camera space -> world, exactly as ExtrinsicsConfig stores it
	glm::dmat4 markerFromCamera{1.0};
	// Parallel arrays: pattern point ID, its averaged pixel, and its known
	// object-space position in millimeters
	std::vector<int> pointIDs;
	std::vector<cv::Point2f> imagePoints;
	std::vector<cv::Point3f> objectPointsMM;
};

struct ExtrinsicsPairQuality
{
	bool valid= false;
	int cameraA= -1;
	int cameraB= -1;
	// Corners both cameras saw - the correspondences everything below uses
	int sharedCornerCount= 0;
	// Triangulated corners reprojected into BOTH views (px). This is the same
	// residual the hand pipeline reports, so the numbers are comparable.
	float reprojectionRmsPx= 0.f;
	float reprojectionMaxPx= 0.f;
	// Triangulated inter-corner distances vs the board's known distances (mm).
	// Catches a wrong baseline, which reprojection alone can hide.
	float spacingErrorMm= 0.f;
	// Scale of the reconstruction: triangulated distance / known distance.
	// 1.0 = correct; 1.02 = the rig thinks the board is 2% bigger than it is.
	float spacingScale= 1.f;
	// The board is flat, so this is pure measurement error (mm)
	float planarityRmsMm= 0.f;
};

// Triangulates every shared corner from the two cameras' calibrated poses and
// fills the quality metrics. Returns false when there are too few shared
// corners (fewer than 4) or the geometry is degenerate.
bool evaluateExtrinsicsPair(const CameraBoardObservations& observationsA,
							const CameraBoardObservations& observationsB,
							ExtrinsicsPairQuality& outQuality);

// -- Pieces exposed for the self test ------------------------------------

// Back-projects an undistorted pixel into a world-space ray
void buildWorldRay(const MikanMonoIntrinsics& intrinsics, const glm::dmat4& markerFromCamera,
				   const cv::Point2f& imagePoint, glm::dvec3& outOriginWorld, glm::dvec3& outDirectionWorld);

// Midpoint of the closest approach between two world rays. Returns false when
// the rays are parallel (no usable intersection).
bool triangulateRayMidpoint(const glm::dvec3& originA, const glm::dvec3& directionA,
							const glm::dvec3& originB, const glm::dvec3& directionB,
							glm::dvec3& outPointWorld);

// Projects a world point into a camera; false when it lands behind the camera
bool projectWorldPoint(const MikanMonoIntrinsics& intrinsics, const glm::dmat4& markerFromCamera,
					   const glm::dvec3& pointWorld, cv::Point2f& outImagePoint);
