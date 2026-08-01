#include "HandPoseModel.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"
#include "glm/gtc/quaternion.hpp"

namespace
{
// Signed angle from vector a to vector b about the given axis (all normalized)
float signedAngle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& axis)
{
	const float cosAngle= std::clamp(glm::dot(a, b), -1.f, 1.f);
	const float sinAngle= glm::dot(glm::cross(a, b), axis);
	return atan2f(sinAngle, cosAngle);
}

glm::vec3 safeNormalize(const glm::vec3& v)
{
	const float length= glm::length(v);
	return length > 1e-6f ? v / length : glm::vec3(1.f, 0.f, 0.f);
}
} // namespace

glm::mat4 HandPoseModel::computePalmFrame(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side)
{
	const glm::vec3& wrist= points[(int)eHandLandmark::WRIST];
	const glm::vec3& indexMcp= points[(int)eHandLandmark::INDEX_MCP];
	const glm::vec3& middleMcp= points[(int)eHandLandmark::MIDDLE_MCP];
	const glm::vec3& pinkyMcp= points[(int)eHandLandmark::PINKY_MCP];

	// +X toward the fingers
	const glm::vec3 xAxis= safeNormalize(middleMcp - wrist);

	// Palm normal: cross of the two palm edges. For a RIGHT hand,
	// (index - wrist) x (pinky - wrist) points out of the BACK of the hand,
	// so negate; for a LEFT hand the same cross points out of the palmar
	// side already (mirrored chirality).
	glm::vec3 normal= glm::cross(indexMcp - wrist, pinkyMcp - wrist);
	if (side == eHandSide::Right)
		normal= -normal;

	// Orthonormalize: Z out of the palmar surface, Y completes right-handed
	glm::vec3 zAxis= safeNormalize(normal - xAxis * glm::dot(normal, xAxis));
	const glm::vec3 yAxis= glm::cross(zAxis, xAxis);

	const glm::vec3 palmCenter= (wrist + middleMcp) * 0.5f;

	glm::mat4 frame(1.f);
	frame[0]= glm::vec4(xAxis, 0.f);
	frame[1]= glm::vec4(yAxis, 0.f);
	frame[2]= glm::vec4(zAxis, 0.f);
	frame[3]= glm::vec4(palmCenter, 1.f);
	return frame;
}

void HandPoseModel::computeFingerAngles(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
										std::array<FingerAngles, FINGER_COUNT>& outAngles)
{
	const glm::mat4 palmFrame= computePalmFrame(points, side);
	const glm::vec3 palmX= glm::vec3(palmFrame[0]);
	const glm::vec3 palmZ= glm::vec3(palmFrame[2]);
	const glm::vec3& wrist= points[(int)eHandLandmark::WRIST];

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const int* joints= FINGER_JOINTS[finger];
		const glm::vec3& base= points[joints[0]];

		// Neutral direction: the metacarpal (wrist -> finger base), projected
		// into the palm plane so lateral splay is measured in-plane. Always
		// well-conditioned: metacarpals never point along the palm normal.
		glm::vec3 metacarpal= base - wrist;
		glm::vec3 neutralDir= safeNormalize(metacarpal - palmZ * glm::dot(metacarpal, palmZ));

		const glm::vec3 proximalBone= safeNormalize(points[joints[1]] - base);
		const glm::vec3 intermediateBone= safeNormalize(points[joints[2]] - points[joints[1]]);
		const glm::vec3 distalBone= safeNormalize(points[joints[3]] - points[joints[2]]);

		// Lateral: signed splay of the proximal bone's palm-plane projection
		// vs the neutral direction, about the palm normal. (The projection
		// degenerates only at exactly 90 degrees of proximal curl, where
		// lateral is visually meaningless anyway - safeNormalize guards it.)
		const glm::vec3 proximalInPlane=
			safeNormalize(proximalBone - palmZ * glm::dot(proximalBone, palmZ));
		const float lateralGeometric= signedAngle(neutralDir, proximalInPlane, palmZ);

		// ONE fixed hinge axis per finger, exactly as the FK side builds it:
		// from the post-lateral in-plane direction. All three bend angles are
		// measured as signed rotations about this axis - measuring each
		// joint's sign against cross(bone, palmZ) (the old approach) breaks
		// down when a curled bone points along the palm normal and that cross
		// degenerates, which flipped distal signs mid-curl (Z-shaped fingers).
		const glm::quat lateralRotation= glm::angleAxis(lateralGeometric, palmZ);
		const glm::vec3 directionLat= lateralRotation * neutralDir;
		const glm::vec3 hingeAxis= safeNormalize(glm::cross(directionLat, -palmZ));

		// FK rotates by angleAxis(-bend, hinge), so the extracted bend is the
		// NEGATED signed angle about the hinge (positive = toward palmar +Z)
		const float proximal= -signedAngle(directionLat, proximalBone, hingeAxis);
		const float intermediate= -signedAngle(proximalBone, intermediateBone, hingeAxis);
		const float distal= -signedAngle(intermediateBone, distalBone, hingeAxis);

		// "+ toward the thumb side" for both hands: the palm Y axis points
		// toward the thumb on a right hand and away on a left hand
		outAngles[finger].lateral= side == eHandSide::Left ? -lateralGeometric : lateralGeometric;
		outAngles[finger].proximal= proximal;
		outAngles[finger].intermediate= intermediate;
		outAngles[finger].distal= distal;
	}
}

void HandPoseModel::computeSkeleton(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points, eHandSide side,
									HandSkeleton& outSkeleton)
{
	const glm::mat4 palmFrame= computePalmFrame(points, side);
	const glm::mat4 palmInverse= glm::inverse(palmFrame);

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const int* joints= FINGER_JOINTS[finger];

		outSkeleton.baseInPalm[finger]= glm::vec3(palmInverse * glm::vec4(points[joints[0]], 1.f));
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			outSkeleton.phalanxLengths[finger][phalanx]=
				glm::length(points[joints[phalanx + 1]] - points[joints[phalanx]]);
		}
	}
}

void HandPoseModel::buildFingerJoints(const glm::mat4& palmTransform, const HandSkeleton& skeleton,
									  const std::array<FingerAngles, FINGER_COUNT>& angles,
									  std::array<std::array<glm::vec3, 4>, FINGER_COUNT>& outJoints)
{
	const glm::vec3 palmZLocal(0.f, 0.f, 1.f);

	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		const FingerAngles& fingerAngles= angles[finger];
		const glm::vec3& base= skeleton.baseInPalm[finger];

		// Neutral direction in the palm frame: the metacarpal projected into
		// the palm plane (palm origin is between wrist and middle MCP; the
		// wrist sits at -x from the origin along palm X by construction)
		// wristInPalm = -baseMiddleDistance... derive from geometry: the palm
		// frame origin is midway wrist<->middleMCP, so the wrist in palm
		// coordinates is at (-|middleMCP-wrist|/2, 0, 0). Recover that from
		// the middle finger's base:
		const glm::vec3& middleBase= skeleton.baseInPalm[(int)eFinger::Middle];
		const glm::vec3 wristInPalm(-middleBase.x, 0.f, 0.f);

		glm::vec3 metacarpal= base - wristInPalm;
		glm::vec3 neutralDir= metacarpal - palmZLocal * metacarpal.z;
		neutralDir= glm::length(neutralDir) > 1e-6f ? glm::normalize(neutralDir) : glm::vec3(1.f, 0.f, 0.f);

		// Apply lateral (about palm Z; sign flipped back for left hands to
		// mirror computeFingerAngles) then proximal bend (about the finger's
		// lateral axis, curling toward the palmar +Z side being positive)
		float lateral= fingerAngles.lateral;
		// note: computeFingerAngles negates for left hands; undo here
		// (the caller passes the side implicitly via the skeleton chirality,
		// which was captured in the palm frame - the baseInPalm y signs)
		// We can detect chirality from the index finger's y sign:
		if (skeleton.baseInPalm[(int)eFinger::Index].y < 0.f)
			lateral= -lateral;

		const glm::quat lateralRotation= glm::angleAxis(lateral, palmZLocal);
		glm::vec3 direction= lateralRotation * neutralDir;

		const glm::vec3 hingeAxis= glm::normalize(glm::cross(direction, -palmZLocal));
		const glm::quat proximalRotation= glm::angleAxis(-fingerAngles.proximal, hingeAxis);
		direction= proximalRotation * direction;

		std::array<glm::vec3, 4>& joints= outJoints[finger];
		joints[0]= glm::vec3(palmTransform * glm::vec4(base, 1.f));

		glm::vec3 position= base + direction * skeleton.phalanxLengths[finger][0];
		joints[1]= glm::vec3(palmTransform * glm::vec4(position, 1.f));

		const glm::quat intermediateRotation= glm::angleAxis(-fingerAngles.intermediate, hingeAxis);
		direction= intermediateRotation * direction;
		position+= direction * skeleton.phalanxLengths[finger][1];
		joints[2]= glm::vec3(palmTransform * glm::vec4(position, 1.f));

		const glm::quat distalRotation= glm::angleAxis(-fingerAngles.distal, hingeAxis);
		direction= distalRotation * direction;
		position+= direction * skeleton.phalanxLengths[finger][2];
		joints[3]= glm::vec3(palmTransform * glm::vec4(position, 1.f));
	}
}
