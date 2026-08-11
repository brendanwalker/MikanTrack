#include "ExtrinsicsValidation.h"

#include <algorithm>
#include <cmath>
#include <map>

#include "glm/geometric.hpp"
#include "opencv2/core.hpp"

namespace
{
// Undistorted intrinsics: the wizard and the tracking pipeline both consume
// undistorted frames, so the undistorted camera matrix is the right one here
void getUndistortedIntrinsics(const MikanMonoIntrinsics& intrinsics, double& outFx, double& outFy,
							  double& outCx, double& outCy)
{
	const MikanMatrix3d& cameraMatrix= intrinsics.undistorted_camera_matrix;
	outFx= cameraMatrix.x0;
	outFy= cameraMatrix.y1;
	outCx= cameraMatrix.z0;
	outCy= cameraMatrix.z1;
}
} // namespace

void buildWorldRay(const MikanMonoIntrinsics& intrinsics, const glm::dmat4& markerFromCamera,
				   const cv::Point2f& imagePoint, glm::dvec3& outOriginWorld, glm::dvec3& outDirectionWorld)
{
	double fx, fy, cx, cy;
	getUndistortedIntrinsics(intrinsics, fx, fy, cx, cy);

	// OpenCV camera convention: +X right, +Y down, +Z forward
	const glm::dvec3 directionCamera(((double)imagePoint.x - cx) / fx, ((double)imagePoint.y - cy) / fy, 1.0);

	outOriginWorld= glm::dvec3(markerFromCamera[3]);
	outDirectionWorld= glm::normalize(glm::dvec3(markerFromCamera * glm::dvec4(directionCamera, 0.0)));
}

bool triangulateRayMidpoint(const glm::dvec3& originA, const glm::dvec3& directionA,
							const glm::dvec3& originB, const glm::dvec3& directionB,
							glm::dvec3& outPointWorld)
{
	const glm::dvec3 between= originB - originA;
	const double dirDot= glm::dot(directionA, directionB);
	const double denominator= 1.0 - dirDot * dirDot;
	if (std::abs(denominator) < 1e-9)
		return false; // parallel views carry no depth information

	const double alongA= (glm::dot(between, directionA) - dirDot * glm::dot(between, directionB)) / denominator;
	const double alongB= (dirDot * glm::dot(between, directionA) - glm::dot(between, directionB)) / denominator;

	const glm::dvec3 closestA= originA + directionA * alongA;
	const glm::dvec3 closestB= originB + directionB * alongB;
	outPointWorld= (closestA + closestB) * 0.5;
	return true;
}

bool projectWorldPoint(const MikanMonoIntrinsics& intrinsics, const glm::dmat4& markerFromCamera,
					   const glm::dvec3& pointWorld, cv::Point2f& outImagePoint)
{
	double fx, fy, cx, cy;
	getUndistortedIntrinsics(intrinsics, fx, fy, cx, cy);

	const glm::dvec4 pointCamera= glm::inverse(markerFromCamera) * glm::dvec4(pointWorld, 1.0);
	if (pointCamera.z < 1e-6)
		return false;

	outImagePoint= cv::Point2f((float)(fx * pointCamera.x / pointCamera.z + cx),
							   (float)(fy * pointCamera.y / pointCamera.z + cy));
	return true;
}

bool evaluateExtrinsicsPair(const CameraBoardObservations& observationsA,
							const CameraBoardObservations& observationsB,
							ExtrinsicsPairQuality& outQuality)
{
	outQuality= ExtrinsicsPairQuality();
	outQuality.cameraA= observationsA.cameraIndex;
	outQuality.cameraB= observationsB.cameraIndex;

	// Corner IDs are what make a correspondence: the same physical corner in
	// both images. A partially visible board still contributes its overlap.
	std::map<int, size_t> indexByIdB;
	for (size_t index= 0; index < observationsB.pointIDs.size(); ++index)
		indexByIdB[observationsB.pointIDs[index]]= index;

	std::vector<glm::dvec3> triangulatedPoints;
	std::vector<cv::Point3f> objectPointsMM;
	double squaredPixelErrorSum= 0.0;
	int pixelErrorCount= 0;
	double maxPixelError= 0.0;

	for (size_t indexA= 0; indexA < observationsA.pointIDs.size(); ++indexA)
	{
		const auto found= indexByIdB.find(observationsA.pointIDs[indexA]);
		if (found == indexByIdB.end())
			continue;
		const size_t indexB= found->second;

		glm::dvec3 originA, directionA, originB, directionB;
		buildWorldRay(observationsA.intrinsics, observationsA.markerFromCamera,
					  observationsA.imagePoints[indexA], originA, directionA);
		buildWorldRay(observationsB.intrinsics, observationsB.markerFromCamera,
					  observationsB.imagePoints[indexB], originB, directionB);

		glm::dvec3 pointWorld;
		if (!triangulateRayMidpoint(originA, directionA, originB, directionB, pointWorld))
			continue;

		// Reproject into both views: this is the residual the hand pipeline
		// reports, so it is directly comparable to triResidualRmsPx
		cv::Point2f reprojectedA, reprojectedB;
		if (!projectWorldPoint(observationsA.intrinsics, observationsA.markerFromCamera, pointWorld,
							   reprojectedA) ||
			!projectWorldPoint(observationsB.intrinsics, observationsB.markerFromCamera, pointWorld,
							   reprojectedB))
		{
			continue;
		}

		const double errorA= cv::norm(reprojectedA - observationsA.imagePoints[indexA]);
		const double errorB= cv::norm(reprojectedB - observationsB.imagePoints[indexB]);
		squaredPixelErrorSum+= errorA * errorA + errorB * errorB;
		pixelErrorCount+= 2;
		maxPixelError= std::max({maxPixelError, errorA, errorB});

		triangulatedPoints.push_back(pointWorld);
		objectPointsMM.push_back(indexA < observationsA.objectPointsMM.size()
									 ? observationsA.objectPointsMM[indexA]
									 : cv::Point3f(0.f, 0.f, 0.f));
	}

	outQuality.sharedCornerCount= (int)triangulatedPoints.size();
	if (outQuality.sharedCornerCount < 4 || pixelErrorCount == 0)
		return false;

	outQuality.reprojectionRmsPx= (float)std::sqrt(squaredPixelErrorSum / pixelErrorCount);
	outQuality.reprojectionMaxPx= (float)maxPixelError;

	// Metric check: every pair of corners has a known separation on the board,
	// so the reconstruction's scale and absolute error are both measurable. A
	// baseline error shows up here even when reprojection looks clean.
	double scaleSum= 0.0;
	double absoluteErrorSum= 0.0;
	int distanceCount= 0;
	for (size_t first= 0; first < triangulatedPoints.size(); ++first)
	{
		for (size_t second= first + 1; second < triangulatedPoints.size(); ++second)
		{
			const double knownMM= cv::norm(objectPointsMM[first] - objectPointsMM[second]);
			if (knownMM < 1e-3)
				continue;

			// Triangulated points are in world meters; the board geometry is mm
			const double measuredMM= glm::length(triangulatedPoints[first] - triangulatedPoints[second]) * 1000.0;
			scaleSum+= measuredMM / knownMM;
			absoluteErrorSum+= std::abs(measuredMM - knownMM);
			++distanceCount;
		}
	}
	if (distanceCount > 0)
	{
		outQuality.spacingScale= (float)(scaleSum / distanceCount);
		outQuality.spacingErrorMm= (float)(absoluteErrorSum / distanceCount);
	}

	// The board is flat, so any thickness in the reconstruction is measurement
	// error. Smallest-eigenvector plane fit about the centroid.
	{
		glm::dvec3 centroid(0.0);
		for (const glm::dvec3& point : triangulatedPoints)
			centroid+= point;
		centroid/= (double)triangulatedPoints.size();

		cv::Matx33d covariance= cv::Matx33d::zeros();
		for (const glm::dvec3& point : triangulatedPoints)
		{
			const glm::dvec3 offset= point - centroid;
			const double values[3]= {offset.x, offset.y, offset.z};
			for (int row= 0; row < 3; ++row)
				for (int col= 0; col < 3; ++col)
					covariance(row, col)+= values[row] * values[col];
		}

		cv::Matx31d eigenValues;
		cv::Matx33d eigenVectors;
		if (cv::eigen(covariance, eigenValues, eigenVectors))
		{
			// cv::eigen returns eigenvalues descending, so the last row is the
			// plane normal (least variance)
			const glm::dvec3 normal(eigenVectors(2, 0), eigenVectors(2, 1), eigenVectors(2, 2));
			double squaredDistanceSum= 0.0;
			for (const glm::dvec3& point : triangulatedPoints)
			{
				const double distance= glm::dot(point - centroid, normal);
				squaredDistanceSum+= distance * distance;
			}
			outQuality.planarityRmsMm=
				(float)(std::sqrt(squaredDistanceSum / triangulatedPoints.size()) * 1000.0);
		}
	}

	outQuality.valid= true;
	return true;
}
