#include "SpaceTransforms.h"

#include "glm/ext/vector_double4.hpp"
#include "glm/gtc/quaternion.hpp"

static glm::vec3 transformPoint(const glm::dmat4& transform, const glm::vec3& point)
{
	const glm::dvec4 transformed= transform * glm::dvec4((double)point.x, (double)point.y, (double)point.z, 1.0);
	return glm::vec3((float)transformed.x, (float)transformed.y, (float)transformed.z);
}

void applyWorldTransform(TrackingFrameResult& ioResult, const glm::dmat4& markerFromCamera)
{
	const glm::mat3 rotation= glm::mat3(glm::dmat3(markerFromCamera));
	const glm::quat rotationQuat= glm::quat_cast(rotation);

	for (TrackedHand& hand : ioResult.hands)
	{
		if (!hand.tracked || !hand.hasCameraSpace)
			continue;

		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			hand.worldPoints[i]= transformPoint(markerFromCamera, hand.cameraPoints[i]);

		hand.hasWorldSpace= true;
	}

	for (HandPose& pose : ioResult.poses)
	{
		if (!pose.tracked || !pose.hasCameraPose)
			continue;

		// Palm transform: position through the full transform, orientation
		// through its rotation. Finger angles and skeleton are palm-local
		// and unaffected by the world transform.
		pose.palmPositionWorld= transformPoint(markerFromCamera, pose.palmPositionCamera);
		pose.palmOrientationWorld= glm::normalize(rotationQuat * pose.palmOrientationCamera);
		pose.hasWorldPose= true;
	}
}
