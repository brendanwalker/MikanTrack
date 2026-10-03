#include "AvatarPreviewPoses.h"

#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "AvatarSkeleton.h"
#include "MathGLM.h"

namespace
{
using B= eHumanoidBone;

void setAvatarLengths(const AvatarSkeleton& skeleton, BodyDimensions& outUser)
{
	outUser.upperArmLengthMeters= skeleton.getUpperArmLength(eHandSide::Left);
	outUser.forearmLengthMeters= skeleton.getForearmLength(eHandSide::Left);
	outUser.shoulderWidthMeters= skeleton.getShoulderWidth();
}

HandPose makeTrackedHand(const AvatarSkeleton& skeleton, eHandSide side)
{
	HandPose pose;
	pose.tracked= true;
	pose.side= side;
	pose.presence= 1.f;
	pose.confidence= 1.f;
	pose.skeleton= skeleton.getHand(side).skeleton;
	pose.hasWorldPose= true;
	return pose;
}
} // namespace

void AvatarPreviewPoses::makeRestFrame(const AvatarSkeleton& skeleton, TrackingFrameResult& outFrame,
									   BodyDimensions& outUser)
{
	setAvatarLengths(skeleton, outUser);
	outFrame= TrackingFrameResult();
	outFrame.timestampMs= 0.0;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const eHandSide side= (eHandSide)sideIndex;
		const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
		HandPose& pose= outFrame.poses[sideIndex];
		pose= makeTrackedHand(skeleton, side);
		pose.palmOrientationWorld= glm::quat_cast(glm::mat3(hand.palmFrameWorld));
		pose.palmPositionWorld= glm::vec3(hand.palmFrameWorld[3]);
		pose.hasShoulder= true;
		pose.shoulderPositionWorld=
			skeleton.getBone(sideIndex == 0 ? B::LeftUpperArm : B::RightUpperArm).restPositionWorld;
		pose.shoulderConfidence= 1.f;
	}
	outFrame.head.valid= true;
	outFrame.head.positionWorld= skeleton.getBone(B::Head).restPositionWorld;
	outFrame.head.orientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
	outFrame.head.confidence= 1.f;
}

void AvatarPreviewPoses::makeDemoFrame(const AvatarSkeleton& skeleton, TrackingFrameResult& outFrame,
									   BodyDimensions& outUser)
{
	setAvatarLengths(skeleton, outUser);
	outFrame= TrackingFrameResult();
	outFrame.timestampMs= 0.0;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const eHandSide side= (eHandSide)sideIndex;
		const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
		const float sign= sideIndex == 0 ? 1.f : -1.f;
		HandPose& pose= outFrame.poses[sideIndex];
		pose= makeTrackedHand(skeleton, side);

		// Palm facing inward toward the body's midline, fingers forward
		const glm::quat restPalm= glm::quat_cast(glm::mat3(hand.palmFrameWorld));
		const glm::quat fingersForward= glm_shortest_arc(glm::vec3(0.f, sign, 0.f), glm::vec3(1.f, 0.f, 0.f));
		const glm::quat palmInward=
			glm::angleAxis(sign * glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f)) * fingersForward;
		pose.palmOrientationWorld= palmInward * restPalm;

		const glm::vec3 shoulder= skeleton.getBone(sideIndex == 0 ? B::LeftUpperArm : B::RightUpperArm).restPositionWorld;
		const glm::vec3 wrist= shoulder + glm::vec3(0.28f, -sign * 0.08f, -0.12f);
		const float halfPalm= hand.skeleton.baseInPalm[(int)eFinger::Middle].x;
		pose.palmPositionWorld= wrist + pose.palmOrientationWorld * glm::vec3(halfPalm, 0.f, 0.f);

		if (sideIndex == 1)
		{
			for (int finger= 1; finger < FINGER_COUNT; ++finger)
				pose.fingers[finger]= {0.f, 1.3f, 1.2f, 0.7f};
			pose.fingers[(int)eFinger::Thumb]= {0.3f, 0.6f, 0.8f, 0.4f};
		}
		else
		{
			pose.fingers[(int)eFinger::Index].lateral= 0.25f;
			pose.fingers[(int)eFinger::Pinky].lateral= -0.3f;
			pose.fingers[(int)eFinger::Thumb].lateral= 0.6f;
		}
	}
	outFrame.head.valid= true;
	outFrame.head.orientationWorld= glm::angleAxis(glm::radians(25.f), glm::vec3(0.f, 0.f, 1.f)) *
		glm::angleAxis(glm::radians(10.f), glm::vec3(0.f, 1.f, 0.f));
	outFrame.head.confidence= 1.f;
}
