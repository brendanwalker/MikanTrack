#include "PatternPoseSampler.h"

#include "CameraMath.h"
#include "Logger.h"

#include "opencv2/core/quaternion.hpp"

PatternPoseSampler::PatternPoseSampler(const MikanMonoIntrinsics& cameraIntrinsics,
									   CalibrationPatternFinderPtr patternFinder, int desiredSampleCount)
	: m_cameraIntrinsics(cameraIntrinsics)
	, m_patternFinder(patternFinder)
	, m_desiredSampleCount(desiredSampleCount > 0 ? desiredSampleCount : 1)
{
	if (m_patternFinder != nullptr)
		m_patternFinder->setCameraIntrinsics(cameraIntrinsics);
}

void PatternPoseSampler::setGrayscaleFrame(const cv::Mat* grayscaleFrame)
{
	if (m_patternFinder != nullptr)
		m_patternFinder->setGrayscaleFrame(grayscaleFrame);
}

bool PatternPoseSampler::hasFinishedSampling() const
{
	return m_capturedSampleCount >= m_desiredSampleCount;
}

float PatternPoseSampler::getCalibrationProgress() const
{
	return (float)m_capturedSampleCount / (float)m_desiredSampleCount;
}

void PatternPoseSampler::resetCalibrationState()
{
	m_capturedSampleCount= 0;
	m_bLastDetectionValid= false;
	m_observations.clear();
	m_meanReprojectionErrorPx= 0.f;
}

bool PatternPoseSampler::captureSample()
{
	m_bLastDetectionValid= false;
	if (m_patternFinder == nullptr)
		return false;

	// The board is held still, so no minimum separation is required between
	// captures - that gate exists for the intrinsics wizard, which wants
	// DIFFERENT views rather than repeated measurements of one view
	if (!m_patternFinder->findNewCalibrationPattern())
		return false;

	cv::Point2f boundingQuad[4];
	t_opencv_point2d_list imagePoints;
	t_opencv_pointID_list imagePointIDs;
	if (!m_patternFinder->fetchLastFoundCalibrationPattern(imagePoints, imagePointIDs, boundingQuad))
		return false;

	// A pattern that reports no IDs (a fixed full-pattern layout) is indexed
	// in order, matching how estimateNewCalibrationPatternPose reads geometry
	if (imagePointIDs.size() != imagePoints.size())
	{
		imagePointIDs.clear();
		for (int pointIndex= 0; pointIndex < (int)imagePoints.size(); ++pointIndex)
			imagePointIDs.push_back(pointIndex);
	}

	for (size_t pointIndex= 0; pointIndex < imagePoints.size(); ++pointIndex)
	{
		CornerObservation& observation= m_observations[imagePointIDs[pointIndex]];
		observation.pixelSum.x+= imagePoints[pointIndex].x;
		observation.pixelSum.y+= imagePoints[pointIndex].y;
		observation.sampleCount++;
	}

	m_bLastDetectionValid= true;
	m_capturedSampleCount++;
	return true;
}

void PatternPoseSampler::getAveragedObservations(std::vector<int>& outPointIDs,
												 std::vector<cv::Point2f>& outImagePoints) const
{
	outPointIDs.clear();
	outImagePoints.clear();
	for (const auto& entry : m_observations)
	{
		const CornerObservation& observation= entry.second;
		if (observation.sampleCount <= 0)
			continue;

		outPointIDs.push_back(entry.first);
		outImagePoints.push_back(
			cv::Point2f((float)(observation.pixelSum.x / observation.sampleCount),
						(float)(observation.pixelSum.y / observation.sampleCount)));
	}
}

bool PatternPoseSampler::getObjectPoint(int pointID, cv::Point3f& outObjectPointMM) const
{
	if (m_patternFinder == nullptr)
		return false;

	OpenCVCalibrationGeometry geometry;
	m_patternFinder->getOpenCVSolvePnPGeometry(&geometry);
	if (pointID < 0 || pointID >= (int)geometry.points.size())
		return false;

	outObjectPointMM= geometry.points[pointID];
	return true;
}

bool PatternPoseSampler::computeCalibratedPatternPose(glm::dmat4& outCameraFromPattern)
{
	if (!hasFinishedSampling())
		return false;

	std::vector<int> pointIDs;
	std::vector<cv::Point2f> imagePoints;
	getAveragedObservations(pointIDs, imagePoints);

	// solvePnP needs at least 4 correspondences; a planar target wants more to
	// condition the out-of-plane rotation at all well
	if (imagePoints.size() < 4)
	{
		MIKAN_LOG_ERROR("PatternPoseSampler")
			<< "Only " << imagePoints.size() << " pattern corners observed - too few to solve a pose";
		return false;
	}

	t_opencv_point3d_list objectPointsMM;
	objectPointsMM.reserve(pointIDs.size());
	for (const int pointID : pointIDs)
	{
		cv::Point3f objectPointMM;
		if (!getObjectPoint(pointID, objectPointMM))
			return false;

		objectPointsMM.push_back(objectPointMM);
	}

	cv::Quatd cameraToPatternRotation;
	cv::Vec3d cameraToPatternPositionMM;
	double meanReprojectionError= 0.0;
	if (!computeOpenCVCameraRelativePatternTransform(m_cameraIntrinsics, imagePoints, objectPointsMM,
													 cameraToPatternRotation, cameraToPatternPositionMM,
													 &meanReprojectionError))
	{
		return false;
	}

	m_meanReprojectionErrorPx= (float)meanReprojectionError;
	convertOpenCVCameraRelativePoseToGLMMat(cameraToPatternRotation, cameraToPatternPositionMM,
											outCameraFromPattern);
	return true;
}
