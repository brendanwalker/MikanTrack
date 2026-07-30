#include "SpaceTransforms.h"

#include "glm/ext/vector_double4.hpp"

static glm::vec3 transformPoint(const glm::dmat4& transform, const glm::vec3& point)
{
	const glm::dvec4 transformed= transform * glm::dvec4((double)point.x, (double)point.y, (double)point.z, 1.0);
	return glm::vec3((float)transformed.x, (float)transformed.y, (float)transformed.z);
}

void applyWorldTransform(TrackingFrameResult& ioResult, const glm::dmat4& markerFromCamera)
{
	for (TrackedHand& hand : ioResult.hands)
	{
		if (!hand.tracked || !hand.hasCameraSpace)
			continue;

		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			hand.worldPoints[i]= transformPoint(markerFromCamera, hand.cameraPoints[i]);

		hand.hasWorldSpace= true;
	}

	for (TrackedArm& arm : ioResult.arms)
	{
		if (!arm.valid || !arm.hasCameraSpace)
			continue;

		arm.elbowWorld= transformPoint(markerFromCamera, arm.elbowCamera);
		arm.wristWorld= transformPoint(markerFromCamera, arm.wristCamera);
		arm.hasWorldSpace= true;
	}
}
