#include "FaceHead.h"

#include "glm/ext/quaternion_trigonometric.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/trigonometric.hpp"

glm::quat faceHeadDeltaFromPhoneEuler(const glm::vec3& eulerDegrees)
{
	const glm::quat yaw= glm::angleAxis(glm::radians(eulerDegrees.y), glm::vec3(0.f, 0.f, 1.f));
	const glm::quat pitch= glm::angleAxis(glm::radians(eulerDegrees.x), glm::vec3(0.f, 1.f, 0.f));
	const glm::quat roll= glm::angleAxis(glm::radians(eulerDegrees.z), glm::vec3(1.f, 0.f, 0.f));
	return glm::normalize(yaw * pitch * roll);
}

glm::quat faceHeadWorldOrientation(const glm::quat& anchor, const glm::vec3& eulerDegrees)
{
	return glm::normalize(anchor * faceHeadDeltaFromPhoneEuler(eulerDegrees));
}

glm::quat faceHeadAnchorFromWorld(const glm::quat& headWorld, const glm::vec3& eulerDegrees)
{
	return glm::normalize(headWorld * glm::inverse(faceHeadDeltaFromPhoneEuler(eulerDegrees)));
}
