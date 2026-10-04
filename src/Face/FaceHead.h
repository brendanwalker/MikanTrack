#pragma once

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

// The phone's head rotation in the app's head frame (+X facing, +Y toward
// the person's left, +Z up). The single conversion point for the face
// stream's head: everything downstream works in the head frame.

// The phone's head Euler angles (x pitch, y yaw, z roll, degrees, measured in
// a frame facing the performer) as a rotation away from looking straight at
// the phone, expressed in the head frame. The phone applies yaw, then pitch,
// then roll about the head's own axes; facing the performer turns its pitch
// and roll axes half way round, so in the head frame the three become turns
// about +Z, +Y and +X with the phone's own signs.
glm::quat faceHeadDeltaFromPhoneEuler(const glm::vec3& eulerDegrees);

// The world head orientation the phone reports: the delta applied to the
// anchor (the world orientation of the head looking straight at the phone)
glm::quat faceHeadWorldOrientation(const glm::quat& anchor, const glm::vec3& eulerDegrees);

// The anchor that makes the phone agree with a world head orientation
// measured at the same instant (the camera head)
glm::quat faceHeadAnchorFromWorld(const glm::quat& headWorld, const glm::vec3& eulerDegrees);
