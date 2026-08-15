#include "BodyDimensionCalibrator.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

#include "SpaceTransforms.h"

// Same gate the solver applies, so calibration never measures a landmark the
// solve would refuse to use
static constexpr float kMinVisibility= 0.5f;

namespace
{
float medianOf(std::vector<float> values)
{
	if (values.empty())
		return 0.f;
	std::sort(values.begin(), values.end());
	return values[values.size() / 2];
}

// Spread as a fraction of the median: the capture's own quality readout
float relativeSpread(std::vector<float> values)
{
	if (values.size() < 3)
		return 0.f;
	std::sort(values.begin(), values.end());
	const float median= values[values.size() / 2];
	if (median <= 1e-6f)
		return 0.f;
	const float low= values[values.size() / 10];
	const float high= values[(values.size() * 9) / 10];
	return (high - low) / median;
}
} // namespace

void BodyDimensionCalibrator::reset()
{
	m_frontalSamples.clear();
	m_noseForwardSamples.clear();
	m_headRangeMeters= 0.f;
}

bool BodyDimensionCalibrator::addFrontalSample(
	const CameraFrameResult& camera,
	const TrackingFrameResult& fused,
	const BodyDimensions& currentDimensions,
	Sample& outSample)
{
	outSample= Sample();
	if (!camera.valid || !camera.hasExtrinsics || !camera.hasIntrinsics || !camera.result.body.valid)
		return false;

	const BodyPoseObservation& body= camera.result.body;
	const glm::vec3 cameraPos= cameraPositionWorld(camera.markerFromCamera);

	auto pixelRay= [&](int landmarkIndex) {
		return pixelRayDirWorld(camera.markerFromCamera, camera.fx, camera.fy, camera.cx, camera.cy,
								glm::vec2(body.imagePoints[landmarkIndex]));
	};
	auto isVisible= [&](int landmarkIndex) {
		return body.isProvided(landmarkIndex) && body.visibility[landmarkIndex] >= kMinVisibility;
	};
	// World point -> this camera's pixel, so a measured wrist can be compared
	// against the landmarks in the space they were detected in
	auto projectToPixel= [&](const glm::vec3& worldPoint) {
		const glm::dvec4 cameraPoint=
			glm::inverse(camera.markerFromCamera) * glm::dvec4(glm::dvec3(worldPoint), 1.0);
		if (cameraPoint.z < 1e-3)
			return glm::vec2(-1e6f);
		return glm::vec2((float)(camera.fx * cameraPoint.x / cameraPoint.z + camera.cx),
						 (float)(camera.fy * cameraPoint.y / cameraPoint.z + camera.cy));
	};

	// Only the landmarks actually measured. The elbows used to be required
	// too, back when the upper arm was measured here rather than derived.
	const int landmarks[]= {
		(int)ePoseLandmark::LEFT_SHOULDER, (int)ePoseLandmark::RIGHT_SHOULDER,
		(int)ePoseLandmark::LEFT_EAR,      (int)ePoseLandmark::RIGHT_EAR,
	};
	for (int landmark : landmarks)
	{
		if (!isVisible(landmark))
			return false;
	}

	// The scale reference: a fused, world-anchored wrist. Judged per arm,
	// because at a desk the resting hand is usually reaching toward the
	// camera and would otherwise veto the arm that IS held correctly.
	const glm::vec3 cameraForward=
		glm::normalize(glm::mat3(glm::mat4(camera.markerFromCamera)) * glm::vec3(0.f, 0.f, 1.f));
	auto depthOf= [&](const glm::vec3& point) { return glm::dot(point - cameraPos, cameraForward); };

	// DEPTH along the view axis, not distance from the camera: the pose puts
	// the body in a plane facing the camera, and a plane is constant depth. A
	// constant distance would be a sphere, which reads every landmark away
	// from the image center as further out than it is - with the arms spread
	// wide that inflated every measurement by about 15%.
	auto atPlaneDepth= [&](int landmarkIndex, float planeDepth) {
		const glm::vec3 rayDir= pixelRay(landmarkIndex);
		const float alongAxis= glm::dot(rayDir, cameraForward);
		if (alongAxis < 1e-3f)
			return cameraPos; // a ray parallel to the plane never reaches it
		return cameraPos + rayDir * (planeDepth / alongAxis);
	};

	const int shoulderIndices[2]= {
		(int)ePoseLandmark::LEFT_SHOULDER, (int)ePoseLandmark::RIGHT_SHOULDER};

	// One raised hand is the whole pose. It only has to sit near the plane of
	// the torso, which a hand held up beside the shoulder does naturally -
	// unlike the straight, square-to-camera arm the upper-arm measurement
	// needed, which proved impractical at a desk. The upper arm is derived
	// from the shoulder width instead.
	float shoulderWidthSum= 0.f;
	float headWidthSum= 0.f;
	int acceptedCount= 0;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= fused.poses[sideIndex];
		if (!pose.tracked || !pose.hasWorldPose)
			continue;

		const glm::vec3 wrist= pose.getWristPositionWorld();
		const float planeDepth= depthOf(wrist);
		if (planeDepth < 0.2f)
			continue;

		// The hand has to be RAISED beside its shoulder, not resting out in
		// front. Judged in the IMAGE, because that is where the statement
		// means anything: comparing 3D positions cannot work, since the
		// shoulder is placed at the WRIST's depth by construction and so can
		// never disagree with it about depth. A hand on the desk projects far
		// below and inboard of its shoulder; a raised one lands right beside
		// it. Measured against the shoulders' own separation so it holds at
		// any distance or resolution.
		const glm::vec2 wristPixel= projectToPixel(wrist);
		const glm::vec2 shoulderPixel(body.imagePoints[shoulderIndices[sideIndex]]);
		const glm::vec2 otherShoulderPixel(body.imagePoints[shoulderIndices[1 - sideIndex]]);
		const float shoulderSeparationPixels= glm::length(shoulderPixel - otherShoulderPixel);
		if (shoulderSeparationPixels < 1e-3f)
			continue;

		const float handOffset= glm::length(wristPixel - shoulderPixel) / shoulderSeparationPixels;
		if (sideIndex == 0)
			outSample.handOffsetLeft= handOffset;
		else
			outSample.handOffsetRight= handOffset;
		if (handOffset > k_maxRaisedHandOffset)
			continue;

		if (isVisible(shoulderIndices[0]) && isVisible(shoulderIndices[1]))
		{
			shoulderWidthSum+= glm::length(atPlaneDepth(shoulderIndices[0], planeDepth) -
										   atPlaneDepth(shoulderIndices[1], planeDepth));
		}
		if (isVisible((int)ePoseLandmark::LEFT_EAR) && isVisible((int)ePoseLandmark::RIGHT_EAR))
		{
			headWidthSum+= glm::length(atPlaneDepth((int)ePoseLandmark::LEFT_EAR, planeDepth) -
									   atPlaneDepth((int)ePoseLandmark::RIGHT_EAR, planeDepth));
		}

		(sideIndex == 0 ? outSample.bAcceptedLeft : outSample.bAcceptedRight)= true;
		acceptedCount++;
		// Remembered for the head-turn step, whose own depth cue (the ear
		// separation) disappears the moment the head turns and the ears overlap
		m_headRangeMeters= planeDepth;
	}

	if (acceptedCount == 0)
		return false;

	outSample.shoulderWidth= shoulderWidthSum / (float)acceptedCount;
	outSample.headWidth= headWidthSum / (float)acceptedCount;
	if (outSample.shoulderWidth <= 0.f || outSample.headWidth <= 0.f)
		return false;

	m_frontalSamples.push_back(outSample);
	return true;
}

bool BodyDimensionCalibrator::addHeadTurnSample(const CameraFrameResult& camera, float& outNoseForward)
{
	outNoseForward= 0.f;
	if (!camera.valid || !camera.hasExtrinsics || !camera.hasIntrinsics || !camera.result.body.valid)
		return false;
	if (m_headRangeMeters <= 0.f)
		return false; // the frontal step has to establish the head distance first

	const BodyPoseObservation& body= camera.result.body;
	const glm::vec3 cameraPos= cameraPositionWorld(camera.markerFromCamera);

	auto pixelRay= [&](int landmarkIndex) {
		return pixelRayDirWorld(camera.markerFromCamera, camera.fx, camera.fy, camera.cx, camera.cy,
								glm::vec2(body.imagePoints[landmarkIndex]));
	};
	auto isVisible= [&](int landmarkIndex) {
		return body.isProvided(landmarkIndex) && body.visibility[landmarkIndex] >= kMinVisibility;
	};

	const int noseIndex= (int)ePoseLandmark::NOSE;
	const int leftEarIndex= (int)ePoseLandmark::LEFT_EAR;
	const int rightEarIndex= (int)ePoseLandmark::RIGHT_EAR;
	if (!isVisible(noseIndex) || !isVisible(leftEarIndex) || !isVisible(rightEarIndex))
		return false;

	// At the head's distance, the nose's offset from the ear midpoint is
	// metric. Facing the camera that offset points along the view ray and
	// projects to nothing; turned, it lies across the image and measures.
	// Depth, matching the frontal step: the head has not moved toward or away
	// from the camera, it has only rotated
	const glm::vec3 cameraForward=
		glm::normalize(glm::mat3(glm::mat4(camera.markerFromCamera)) * glm::vec3(0.f, 0.f, 1.f));
	auto atHeadDepth= [&](int landmarkIndex) {
		const glm::vec3 rayDir= pixelRay(landmarkIndex);
		const float alongAxis= glm::dot(rayDir, cameraForward);
		if (alongAxis < 1e-3f)
			return cameraPos;
		return cameraPos + rayDir * (m_headRangeMeters / alongAxis);
	};

	const glm::vec3 leftEar= atHeadDepth(leftEarIndex);
	const glm::vec3 rightEar= atHeadDepth(rightEarIndex);
	const glm::vec3 nose= atHeadDepth(noseIndex);
	const glm::vec3 earMid= 0.5f * (leftEar + rightEar);

	const float earSeparation= glm::length(leftEar - rightEar);
	const float offset= glm::length(nose - earMid);
	// Only count a genuinely turned head: face-on, the ears are wide apart
	// and the nose sits between them, which measures nothing
	if (offset < earSeparation * 0.5f)
		return false;

	outNoseForward= offset;
	m_noseForwardSamples.push_back(offset);
	return true;
}

BodyDimensionCalibrator::Result BodyDimensionCalibrator::solve(
	const BodyDimensions& currentDimensions) const
{
	Result result;
	result.sampleCount= (int)m_frontalSamples.size();
	if (result.sampleCount < k_minSamples)
		return result;

	std::vector<float> shoulderWidths, headWidths;
	shoulderWidths.reserve(m_frontalSamples.size());
	headWidths.reserve(m_frontalSamples.size());
	for (const Sample& sample : m_frontalSamples)
	{
		shoulderWidths.push_back(sample.shoulderWidth);
		headWidths.push_back(sample.headWidth);
	}

	result.shoulderWidth= medianOf(shoulderWidths);
	result.headWidth= medianOf(headWidths);
	// The upper arm is no longer measured here: it is derived from the
	// shoulder width, because measuring it needed a pose that could not be
	// hit reliably at a desk and was 20% wrong when missed.
	result.upperArmLength= result.shoulderWidth * currentDimensions.upperArmPerShoulderWidth;
	result.shoulderWidthSpread= relativeSpread(shoulderWidths);
	result.headWidthSpread= relativeSpread(headWidths);

	if ((int)m_noseForwardSamples.size() >= k_minSamples)
	{
		result.noseForward= medianOf(m_noseForwardSamples);
		result.bHaveNoseForward= true;
	}

	result.bValid= result.shoulderWidth > 0.05f && result.headWidth > 0.02f && result.upperArmLength > 0.05f;
	return result;
}
