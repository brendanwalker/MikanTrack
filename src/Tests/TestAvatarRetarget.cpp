#include "TestCommon.h"

#include <cstring>

#include "glm/gtx/quaternion.hpp"

#include "AvatarRetarget.h"
#include "AvatarSkeleton.h"
#include "MathGLM.h"
#include "SyntheticAvatar.h"

// The avatar retarget and the avatar-driven VMC stream on the shared
// synthetic rig. Every case is exact: a user resting in the avatar's own
// T-pose must leave every bone at identity, a reach scales by the arm ratio,
// the two-bone solve keeps the avatar's lengths, and the fingers follow the
// forward kinematics on the avatar's hand.

namespace
{
using B= eHumanoidBone;
using namespace SyntheticAvatar;

constexpr float kTolerance= 1e-4f;

bool nearlyEqual(const glm::vec3& a, const glm::vec3& b, float tolerance)
{
	return glm::length(a - b) <= tolerance;
}

bool nearlyEqual(float a, float b, float tolerance)
{
	return std::fabs(a - b) <= tolerance;
}

// Two rotations within an angular tolerance of each other
bool nearlySameRotation(const glm::quat& a, const glm::quat& b, float toleranceRadians)
{
	const float dot= std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
	return 2.f * acosf(std::min(dot, 1.f)) <= toleranceRadians;
}

bool isIdentity(const glm::quat& q, float toleranceRadians)
{
	return nearlySameRotation(q, glm::quat(1.f, 0.f, 0.f, 0.f), toleranceRadians);
}

eHumanoidBone sideBone(eHandSide side, B left, B right)
{
	return side == eHandSide::Left ? left : right;
}

eHumanoidBone fingerBone(eHandSide side, int finger, int phalanx)
{
	return (eHumanoidBone)((int)firstHumanoidFingerBone((int)side) + finger * 3 + phalanx);
}

// The user's body lengths equal to the avatar's, so a rest pose maps 1:1
BodyDimensions makeMatchingUser()
{
	BodyDimensions user;
	user.upperArmLengthMeters= kUpperArmLength;
	user.forearmLengthMeters= kForearmLength;
	user.shoulderWidthMeters= kShoulderWidth;
	return user;
}

// A hand resting in the avatar's own T-pose: the palm at the avatar's rest
// palm frame, the wrist on the avatar's hand joint, zero finger angles,
// a neutral wrist, and the shoulder on the avatar's upper-arm joint
HandPose makeRestHand(const AvatarSkeleton& skeleton, eHandSide side)
{
	const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
	const glm::quat restPalm= glm::quat_cast(glm::mat3(hand.palmFrameWorld));

	HandPose pose;
	pose.tracked= true;
	pose.side= side;
	pose.presence= 1.f;
	pose.confidence= 1.f;
	pose.skeleton= hand.skeleton; // the user's hand IS the avatar's for these cases
	pose.hasWorldPose= true;
	pose.palmOrientationWorld= restPalm;
	pose.palmPositionWorld= glm::vec3(hand.palmFrameWorld[3]);
	pose.hasForearmPose= true;
	pose.forearmOrientationWorld= restPalm;
	pose.forearmConfidence= 1.f;
	pose.hasShoulder= true;
	pose.shoulderPositionWorld= skeleton.getBone(sideBone(side, B::LeftUpperArm, B::RightUpperArm)).restPositionWorld;
	pose.shoulderConfidence= 1.f;
	return pose;
}

// Moves a rest hand so its WRIST lands on `wrist` with the palm rotated by
// `rotation` about the wrist, forearm following (neutral wrist)
void placeHand(HandPose& pose, const glm::vec3& wrist, const glm::quat& rotation)
{
	const float halfPalm= pose.skeleton.baseInPalm[(int)eFinger::Middle].x;
	pose.palmOrientationWorld= rotation * pose.palmOrientationWorld;
	pose.forearmOrientationWorld= pose.palmOrientationWorld;
	pose.palmPositionWorld= wrist + pose.palmOrientationWorld * glm::vec3(halfPalm, 0.f, 0.f);
}

struct Scenario
{
	AvatarModel model;
	std::unique_ptr<AvatarSkeleton> skeleton;
	std::array<HandPose, 2> poses;
	bool bSideValid[2]= {true, true};
	TrackingFrameResult::HeadPose head;
	BodyDimensions user;
	AvatarRetargetConfig config;
};

Scenario makeRestScenario(eVrmVersion version, bool rotateLeftUpperArmNode)
{
	Scenario scenario;
	scenario.model= makeModel(version, rotateLeftUpperArmNode);
	scenario.skeleton= std::make_unique<AvatarSkeleton>(scenario.model);
	scenario.poses[0]= makeRestHand(*scenario.skeleton, eHandSide::Left);
	scenario.poses[1]= makeRestHand(*scenario.skeleton, eHandSide::Right);
	scenario.head.valid= true;
	scenario.head.positionWorld= scenario.skeleton->getBone(B::Head).restPositionWorld;
	scenario.user= makeMatchingUser();
	return scenario;
}

void solveScenario(const Scenario& scenario, AvatarRetarget& retarget, double timestampMs, AvatarPose& outPose)
{
	retarget.solve(scenario.poses, scenario.bSideValid, scenario.head, timestampMs, scenario.user,
				   *scenario.skeleton, scenario.config, outPose);
}

// Composes the VMC stream the way a receiver does (see TestVmc), starting
// from an absolute parent so composed positions can be compared with the
// posed world bones
struct BoneWorld
{
	glm::quat rotation{1.f, 0.f, 0.f, 0.f};
	glm::vec3 position{0.f};
};

glm::vec3 unityToWorldPosition(const glm::vec3& unity)
{
	return glm::vec3(unity.z, -unity.x, unity.y);
}

glm::quat unityToWorldRotation(const glm::quat& unity)
{
	const glm::mat3 toWorld(glm::vec3(0.f, -1.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 0.f, 0.f));
	return glm::quat_cast(toWorld * glm::mat3_cast(unity) * glm::transpose(toWorld));
}

BoneWorld composeBone(const VmcRetarget::VmcPose& pose, VmcRetarget::eVmcBone bone, const BoneWorld& parent)
{
	const VmcRetarget::VmcBone& streamed= pose.bones[(int)bone];
	if (!streamed.present)
		return parent;
	BoneWorld out;
	out.rotation= parent.rotation * unityToWorldRotation(streamed.localRotation);
	out.position= parent.position + parent.rotation * unityToWorldPosition(streamed.localPosition);
	return out;
}
} // namespace

static int runAvatarRetargetTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-avatar-retarget") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-avatar-retarget") << "FAIL " << name;
			failures++;
		}
	};

	// (a) Rest identity: a user resting in the avatar's own T-pose leaves
	// every placed bone at identity and every posed bone on its rest
	// position, with the root at the origin
	{
		Scenario rest= makeRestScenario(eVrmVersion::Vrm1, false);
		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(rest, retarget, 0.0, pose);

		bool bAllIdentity= pose.valid;
		int presentCount= 0;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		{
			if (!pose.bones[index].present)
				continue;
			++presentCount;
			bAllIdentity&= isIdentity(pose.bones[index].deltaWorld, 1e-3f);
		}
		check(bAllIdentity && presentCount == 1 + 1 + 2 * (1 + 3 + 15),
			  "(a) rest input places hips, head, clavicles, arms, hands and fingers at identity");
		check(nearlyEqual(pose.rootPositionWorld, glm::vec3(0.f), kTolerance) && isIdentity(pose.rootRotationWorld, 1e-4f),
			  "(a) rest shoulders put the root at the origin");

		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*rest.skeleton, pose, posed);
		bool bOnRest= true;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		{
			const AvatarSkeleton::Bone& bone= rest.skeleton->getBone((B)index);
			if (bone.present)
				bOnRest&= nearlyEqual(posed[index].positionWorld, bone.restPositionWorld, kTolerance);
		}
		check(bOnRest, "(a) every posed bone sits on its rest position");

		std::vector<glm::mat4> globals;
		computePosedGlobals(rest.model, *rest.skeleton, pose, globals);
		bool bGlobalsRest= globals.size() == rest.model.nodes.size();
		for (size_t node= 0; bGlobalsRest && node < globals.size(); ++node)
		{
			for (int column= 0; column < 4; ++column)
				bGlobalsRest&= nearlyEqual(glm::vec3(globals[node][column]),
										   glm::vec3(rest.skeleton->getRestGlobalsAvatar()[node][column]), kTolerance);
		}
		check(bGlobalsRest, "(a) posed node globals equal the rest globals");

		// The VMC stream of the rest pose: identity rotations, rest offsets
		VmcRetarget::VmcPose vmc;
		VmcRetarget::buildPoseFromAvatar(pose, *rest.skeleton, vmc);
		bool bVmcRest= true;
		int streamedCount= 0;
		for (int index= 0; index < VmcRetarget::VMC_BONE_COUNT; ++index)
		{
			const VmcRetarget::VmcBone& bone= vmc.bones[index];
			if (!bone.present)
				continue;
			++streamedCount;
			const eHumanoidBone humanoid= VmcRetarget::humanoidBoneForVmc((VmcRetarget::eVmcBone)index);
			bVmcRest&= isIdentity(bone.localRotation, 1e-3f);
			bVmcRest&= nearlyEqual(unityToWorldPosition(bone.localPosition),
								   rest.skeleton->getBone(humanoid).restOffsetFromParentWorld, kTolerance);
		}
		check(bVmcRest && streamedCount == VmcRetarget::VMC_BONE_COUNT,
			  "(a) the VMC stream of the rest pose is all 39 bones at identity with the avatar's own offsets");
		check(VmcRetarget::humanoidBoneForVmc(VmcRetarget::eVmcBone::LeftThumbProximal) == B::LeftThumbProximal &&
				  VmcRetarget::humanoidBoneForVmc(VmcRetarget::eVmcBone::RightLittleDistal) == B::RightLittleDistal &&
				  VmcRetarget::humanoidBoneForVmc(VmcRetarget::eVmcBone::Head) == B::Head,
			  "(a) VMC bones name the matching humanoid bones");
	}

	// (b) Reach scaling: a user with twice the avatar's arm reaching forward
	// moves the avatar's wrist half as far from its shoulder; beyond the
	// avatar's reach the wrist is clamped
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		scenario.user.upperArmLengthMeters= 2.f * kUpperArmLength;
		scenario.user.forearmLengthMeters= 2.f * kForearmLength;
		const glm::vec3 shoulder= scenario.poses[0].shoulderPositionWorld;
		placeHand(scenario.poses[0], shoulder + glm::vec3(0.6f, 0.f, 0.f), glm::quat(1.f, 0.f, 0.f, 0.f));

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);
		check(nearlyEqual(posed[(int)B::LeftHand].positionWorld, shoulder + glm::vec3(0.3f, 0.f, 0.f), kTolerance),
			  "(b) a reach scales by the avatar-to-user arm ratio");

		placeHand(scenario.poses[0], shoulder + glm::vec3(2.f, 0.f, 0.f), glm::quat(1.f, 0.f, 0.f, 0.f));
		solveScenario(scenario, retarget, 16.0, pose);
		computePosedBones(*scenario.skeleton, pose, posed);
		const float reach= glm::length(posed[(int)B::LeftHand].positionWorld - shoulder);
		check(nearlyEqual(reach, kUpperArmLength + kForearmLength, 1e-4f) &&
				  nearlyEqual(posed[(int)B::LeftLowerArm].positionWorld, shoulder + glm::vec3(kUpperArmLength, 0.f, 0.f), 1e-3f),
			  "(b) a wrist beyond reach is clamped and the arm straightens toward it");
	}

	// (c) Two-bone solve: the avatar's lengths hold, the elbow bends in the
	// plane of the measured elbow, and without one the default pole bends it
	// down with the forearm taking the hand's own frame
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		HandPose& left= scenario.poses[0];
		const glm::vec3 shoulder= left.shoulderPositionWorld;
		// Hand in front of the chest, elbow hinted out to the side
		const glm::vec3 wrist= shoulder + glm::vec3(0.30f, -0.05f, -0.05f);
		placeHand(left, wrist, glm::angleAxis(-glm::half_pi<float>(), glm::vec3(0.f, 0.f, 1.f)));
		const glm::vec3 hintedElbow= shoulder + glm::vec3(0.1f, 0.2f, -0.1f);
		// The forearm frame: +X from the elbow toward the wrist
		const glm::vec3 forearmX= glm::normalize(wrist - hintedElbow);
		const glm::quat forearm= glm_shortest_arc(glm::vec3(1.f, 0.f, 0.f), forearmX);
		left.forearmOrientationWorld= forearm;
		// getElbowPositionWorld walks back along the forearm by the USER length, so
		// put the hint exactly one user forearm behind the wrist
		scenario.user.forearmLengthMeters= glm::length(wrist - hintedElbow);
		scenario.user.upperArmLengthMeters= glm::length(hintedElbow - shoulder);

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);
		const glm::vec3 elbow= posed[(int)B::LeftLowerArm].positionWorld;
		const glm::vec3 hand= posed[(int)B::LeftHand].positionWorld;
		check(nearlyEqual(glm::length(elbow - shoulder), kUpperArmLength, 1e-5f) &&
				  nearlyEqual(glm::length(hand - elbow), kForearmLength, 1e-5f),
			  "(c) the posed arm keeps the avatar's upper arm and forearm lengths");
		const glm::vec3 planeNormal= glm::normalize(glm::cross(hand - shoulder, hintedElbow - shoulder));
		check(std::fabs(glm::dot(elbow - shoulder, planeNormal)) < 1e-4f &&
				  glm::dot(elbow - shoulder, hintedElbow - shoulder) > 0.f,
			  "(c) the elbow bends in the plane of the measured elbow, on its side");

		// No measured elbow: default pole, down and back
		left.hasForearmPose= false;
		AvatarRetarget retargetNoElbow;
		solveScenario(scenario, retargetNoElbow, 0.0, pose);
		computePosedBones(*scenario.skeleton, pose, posed);
		const glm::vec3 elbowDefault= posed[(int)B::LeftLowerArm].positionWorld;
		const glm::vec3 midpoint= (shoulder + hand) * 0.5f;
		check(elbowDefault.z < midpoint.z - 0.02f, "(c) without a measured elbow the arm bends down");
		// Neutral wrist: the forearm's palm normal matches the hand's
		const glm::vec3 forearmDirection= glm::normalize(hand - elbowDefault);
		const AvatarSkeleton::HandRest& handRest= scenario.skeleton->getHand(eHandSide::Left);
		const glm::vec3 restNormal(handRest.palmFrameWorld[2]);
		const glm::vec3 posedForearmNormal= pose.bones[(int)B::LeftLowerArm].deltaWorld * restNormal;
		const glm::vec3 palmNormal= left.palmOrientationWorld * glm::vec3(0.f, 0.f, 1.f);
		auto perpendicular= [&](const glm::vec3& v) { return glm::normalize(v - forearmDirection * glm::dot(v, forearmDirection)); };
		check(glm::dot(perpendicular(posedForearmNormal), perpendicular(palmNormal)) > 0.999f,
			  "(c) without a measured forearm the forearm roll follows the palm");
	}

	// (d) Hand and forearm frames: the posed hand reproduces the measured
	// palm orientation, and a measured pronation reappears as forearm roll
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		HandPose& right= scenario.poses[1];
		const glm::quat turn= glm::angleAxis(0.7f, glm::normalize(glm::vec3(0.3f, -1.f, 0.4f)));
		const glm::vec3 wrist= right.shoulderPositionWorld + glm::vec3(0.25f, 0.05f, -0.15f);
		placeHand(right, wrist, turn);

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		const AvatarSkeleton::HandRest& handRest= scenario.skeleton->getHand(eHandSide::Right);
		const glm::quat restPalm= glm::quat_cast(glm::mat3(handRest.palmFrameWorld));
		const glm::quat posedPalm= pose.bones[(int)B::RightHand].deltaWorld * restPalm;
		check(nearlySameRotation(posedPalm, right.palmOrientationWorld, 1e-4f),
			  "(d) the posed hand reproduces the measured palm orientation");

		// Pronate the forearm 60 degrees about its own axis relative to the palm
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);
		const glm::vec3 forearmAxis= glm::normalize(posed[(int)B::RightHand].positionWorld - posed[(int)B::RightLowerArm].positionWorld);
		right.hasForearmPose= true;
		right.forearmOrientationWorld= glm::angleAxis(1.0f, forearmAxis) * right.palmOrientationWorld;
		// Keep the elbow hint consistent with where the arm already is
		scenario.user.forearmLengthMeters= kForearmLength;
		solveScenario(scenario, retarget, 16.0, pose);
		const glm::vec3 posedForearmNormal= pose.bones[(int)B::RightLowerArm].deltaWorld * glm::vec3(handRest.palmFrameWorld[2]);
		const glm::vec3 measuredNormal= right.forearmOrientationWorld * glm::vec3(0.f, 0.f, 1.f);
		computePosedBones(*scenario.skeleton, pose, posed);
		const glm::vec3 axis= glm::normalize(posed[(int)B::RightHand].positionWorld - posed[(int)B::RightLowerArm].positionWorld);
		auto perpendicular= [&](const glm::vec3& v) { return glm::normalize(v - axis * glm::dot(v, axis)); };
		check(glm::dot(perpendicular(posedForearmNormal), perpendicular(measuredNormal)) > 0.999f,
			  "(d) a measured forearm pronation reappears as forearm roll");
	}

	// (e) Fingers: the posed phalanges point where the forward kinematics on
	// the avatar hand puts them, a curl touches only its finger, zero angles
	// stay identity, and the thumb's first bone answers to its proximal angle
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		HandPose& left= scenario.poses[0];
		left.fingers[(int)eFinger::Index]= {0.f, 1.0f, 0.8f, 0.5f};

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);

		const AvatarSkeleton::HandRest& handRest= scenario.skeleton->getHand(eHandSide::Left);
		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
		HandPoseModel::buildFingerJoints(handRest.palmFrameWorld, handRest.skeleton, left.fingers, joints);

		bool bDirectionsMatch= true;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const int finger= (int)eFinger::Index;
			const glm::vec3 expected= glm::normalize(joints[finger][phalanx + 1] - joints[finger][phalanx]);
			// Bone direction: the posed delta applied to the rest phalanx direction
			const glm::vec3 actual=
				pose.bones[(int)fingerBone(eHandSide::Left, finger, phalanx)].deltaWorld * handRest.restPhalanxDirWorld[finger][phalanx];
			bDirectionsMatch&= glm::dot(expected, actual) > 1.f - 1e-6f;
		}
		check(bDirectionsMatch, "(e) curled index phalanges point where the avatar-hand FK puts them");
		check(nearlyEqual(posed[(int)fingerBone(eHandSide::Left, (int)eFinger::Index, 1)].positionWorld, joints[(int)eFinger::Index][1], kTolerance) &&
				  nearlyEqual(posed[(int)fingerBone(eHandSide::Left, (int)eFinger::Index, 2)].positionWorld, joints[(int)eFinger::Index][2], kTolerance),
			  "(e) the posed index joints land on the FK joints");

		bool bOthersIdentity= true;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			if (finger == (int)eFinger::Index)
				continue;
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				bOthersIdentity&= isIdentity(pose.bones[(int)fingerBone(eHandSide::Left, finger, phalanx)].deltaWorld, 1e-3f);
		}
		check(bOthersIdentity, "(e) a curled index leaves the other fingers at identity");

		left.fingers[(int)eFinger::Index]= {};
		left.fingers[(int)eFinger::Thumb].proximal= 0.5f;
		solveScenario(scenario, retarget, 16.0, pose);
		check(!isIdentity(pose.bones[(int)B::LeftThumbProximal].deltaWorld, 0.1f) &&
				  isIdentity(pose.bones[(int)B::LeftIndexProximal].deltaWorld, 1e-3f),
			  "(e) the thumb's proximal angle moves its first bone (the metacarpal)");
	}

	// (f) Head: a measured yaw reappears as the head delta; an invalid head
	// leaves the bone unplaced
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		scenario.head.orientationWorld= glm::angleAxis(glm::radians(30.f), glm::vec3(0.f, 0.f, 1.f));
		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		check(pose.bones[(int)B::Head].present &&
				  nearlySameRotation(pose.bones[(int)B::Head].deltaWorld, scenario.head.orientationWorld, 1e-4f),
			  "(f) a measured head yaw is the head delta");
		scenario.head.valid= false;
		solveScenario(scenario, retarget, 16.0, pose);
		check(!pose.bones[(int)B::Head].present, "(f) an invalid head leaves the bone at rest");
	}

	// (g) Root: both shoulders place and yaw the avatar, one shoulder keeps
	// the fixed root, and the follow filters over time
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		const glm::quat yaw= glm::angleAxis(glm::radians(20.f), glm::vec3(0.f, 0.f, 1.f));
		const glm::vec3 shift(0.1f, 0.2f, 0.05f);
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			HandPose& hand= scenario.poses[sideIndex];
			hand.shoulderPositionWorld= yaw * hand.shoulderPositionWorld + shift;
			const float halfPalm= hand.skeleton.baseInPalm[(int)eFinger::Middle].x;
			const glm::vec3 wrist= yaw * (hand.palmPositionWorld - hand.palmOrientationWorld * glm::vec3(halfPalm, 0.f, 0.f)) + shift;
			hand.palmOrientationWorld= yaw * hand.palmOrientationWorld;
			hand.forearmOrientationWorld= hand.palmOrientationWorld;
			hand.palmPositionWorld= wrist + hand.palmOrientationWorld * glm::vec3(halfPalm, 0.f, 0.f);
		}

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);
		const glm::vec3 midpoint= (posed[(int)B::LeftUpperArm].positionWorld + posed[(int)B::RightUpperArm].positionWorld) * 0.5f;
		const glm::vec3 measuredMidpoint=
			(scenario.poses[0].shoulderPositionWorld + scenario.poses[1].shoulderPositionWorld) * 0.5f;
		check(nearlyEqual(midpoint, measuredMidpoint, kTolerance) && nearlySameRotation(pose.rootRotationWorld, yaw, 1e-4f),
			  "(g) both shoulders place the root on their midpoint with the shoulder-line yaw");
		bool bArmsIdentity= true;
		for (B bone : {B::LeftUpperArm, B::LeftLowerArm, B::LeftHand, B::RightUpperArm, B::RightLowerArm, B::RightHand})
			bArmsIdentity&= isIdentity(pose.bones[(int)bone].deltaWorld, 1e-3f);
		check(bArmsIdentity, "(g) a whole-body yaw lands in the root, not the arms");

		// The follow filters: a jump in the shoulders moves the root part way
		// after one 16 ms step
		const glm::vec3 jump(0.f, 0.f, 0.3f);
		scenario.poses[0].shoulderPositionWorld+= jump;
		scenario.poses[1].shoulderPositionWorld+= jump;
		solveScenario(scenario, retarget, 16.0, pose);
		const float moved= pose.rootPositionWorld.z - (shift.z);
		check(moved > 0.005f && moved < 0.29f, "(g) the root follows a shoulder jump through a filter");

		// A lost shoulder holds the last followed placement
		scenario.poses[1].hasShoulder= false;
		solveScenario(scenario, retarget, 32.0, pose);
		check(nearlyEqual(pose.rootPositionWorld.z, shift.z + moved, 1e-5f), "(g) a lost shoulder holds the followed root");

		// Never followed: the fixed root
		Scenario fixed= makeRestScenario(eVrmVersion::Vrm1, false);
		fixed.poses[0].hasShoulder= false;
		fixed.config.fixedRootPositionWorld= glm::vec3(1.f, 2.f, 3.f);
		fixed.config.fixedRootYawDegrees= 45.f;
		AvatarRetarget fixedRetarget;
		solveScenario(fixed, fixedRetarget, 0.0, pose);
		check(nearlyEqual(pose.rootPositionWorld, glm::vec3(1.f, 2.f, 3.f), kTolerance) &&
				  nearlySameRotation(pose.rootRotationWorld, glm::angleAxis(glm::radians(45.f), glm::vec3(0.f, 0.f, 1.f)), 1e-4f),
			  "(g) with no shoulder pair ever seen the fixed root applies");
		fixed.config.followShoulders= false;
		fixed.poses[0].hasShoulder= true;
		solveScenario(fixed, fixedRetarget, 16.0, pose);
		check(nearlyEqual(pose.rootPositionWorld, glm::vec3(1.f, 2.f, 3.f), kTolerance),
			  "(g) follow off keeps the fixed root even with both shoulders");
	}

	// (h) Degradation: no measured shoulder aims the arm from the avatar's
	// own shoulder, an invalid side rests, missing finger bones and clavicles
	// are simply absent
	{
		Scenario scenario= makeRestScenario(eVrmVersion::Vrm1, false);
		HandPose& left= scenario.poses[0];
		left.hasShoulder= false;
		left.hasForearmPose= false;
		const glm::vec3 avatarShoulder= scenario.skeleton->getBone(B::LeftUpperArm).restPositionWorld;
		const glm::vec3 wrist= avatarShoulder + glm::vec3(0.2f, 0.1f, 0.1f);
		placeHand(left, wrist, glm::quat(1.f, 0.f, 0.f, 0.f));
		scenario.bSideValid[1]= false;

		AvatarRetarget retarget;
		AvatarPose pose;
		solveScenario(scenario, retarget, 0.0, pose);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
		computePosedBones(*scenario.skeleton, pose, posed);
		check(nearlyEqual(posed[(int)B::LeftHand].positionWorld, wrist, kTolerance),
			  "(h) without a measured shoulder the hand lands where it was seen");
		check(!pose.bones[(int)B::RightUpperArm].present && !pose.bones[(int)B::RightHand].present &&
				  !pose.bones[(int)B::LeftShoulder].present,
			  "(h) an invalid side and an unmeasured clavicle stay absent");

		// An avatar without fingers or clavicles
		AvatarModel bare= makeModel(eVrmVersion::Vrm1, false);
		for (int index= (int)B::LeftThumbProximal; index < HUMANOID_BONE_COUNT; ++index)
			bare.humanoidNodes[index]= -1;
		bare.humanoidNodes[(int)B::LeftShoulder]= -1;
		bare.humanoidNodes[(int)B::RightShoulder]= -1;
		const AvatarSkeleton bareSkeleton(bare);
		Scenario full= makeRestScenario(eVrmVersion::Vrm1, false);
		AvatarRetarget bareRetarget;
		bareRetarget.solve(full.poses, full.bSideValid, full.head, 0.0, full.user, bareSkeleton, full.config, pose);
		bool bNoFingers= true;
		for (int index= (int)B::LeftThumbProximal; index < HUMANOID_BONE_COUNT; ++index)
			bNoFingers&= !pose.bones[index].present;
		check(bNoFingers && !pose.bones[(int)B::LeftShoulder].present && pose.bones[(int)B::LeftUpperArm].present &&
				  pose.bones[(int)B::LeftHand].present && !bareSkeleton.getHand(eHandSide::Left).hasAnyFinger(),
			  "(h) an avatar without fingers or clavicles poses its arms and hands and nothing else");
		{
			// The fingerless hand still takes the measured palm orientation
			// through the T-pose stand-in palm frame
			const AvatarSkeleton::HandRest& bareHand= bareSkeleton.getHand(eHandSide::Left);
			const glm::quat bareRestPalm= glm::quat_cast(glm::mat3(bareHand.palmFrameWorld));
			check(nearlySameRotation(bareRestPalm, glm::quat_cast(VmcRetarget::restPalmFrame(eHandSide::Left)), 1e-3f) &&
					  nearlySameRotation(pose.bones[(int)B::LeftHand].deltaWorld * bareRestPalm, full.poses[0].palmOrientationWorld, 1e-4f),
				  "(h) a fingerless hand builds the T-pose palm frame and follows the measured palm");
		}

		// A rig with two bones per finger mapped into the intermediate and
		// distal slots and no little finger (a Blender-authored VRM): the hand
		// poses, each finger poses its two bones, and the bends past the
		// second bone fold into it
		AvatarModel twoBone= makeModel(eVrmVersion::Vrm1, false);
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				twoBone.humanoidNodes[(int)fingerBone((eHandSide)sideIndex, finger, 0)]= -1;
				if (finger == (int)eFinger::Pinky)
				{
					twoBone.humanoidNodes[(int)fingerBone((eHandSide)sideIndex, finger, 1)]= -1;
					twoBone.humanoidNodes[(int)fingerBone((eHandSide)sideIndex, finger, 2)]= -1;
				}
			}
		}
		const AvatarSkeleton twoBoneSkeleton(twoBone);
		const AvatarSkeleton::HandRest& twoBoneHand= twoBoneSkeleton.getHand(eHandSide::Left);
		check(twoBoneHand.fingerBoneCount[(int)eFinger::Index] == 2 && twoBoneHand.fingerSlots[(int)eFinger::Index][0] == 1 &&
				  twoBoneHand.fingerSlots[(int)eFinger::Index][1] == 2 && twoBoneHand.fingerBoneCount[(int)eFinger::Pinky] == 0 &&
				  twoBoneHand.skeleton.baseInPalm[(int)eFinger::Index].y > 0.01f,
			  "(h) a two-bone finger rig reports its physical bones and slots with the tracked chirality");

		Scenario twoBoneScenario= makeRestScenario(eVrmVersion::Vrm1, false);
		HandPose& twoBoneLeft= twoBoneScenario.poses[0];
		twoBoneLeft.fingers[(int)eFinger::Index]= {0.f, 1.0f, 0.5f, 0.3f};
		const glm::quat palmTurn= glm::angleAxis(0.5f, glm::normalize(glm::vec3(0.2f, 1.f, 0.3f)));
		placeHand(twoBoneLeft, twoBoneLeft.shoulderPositionWorld + glm::vec3(0.2f, 0.15f, -0.1f), palmTurn);
		AvatarRetarget twoBoneRetarget;
		twoBoneRetarget.solve(twoBoneScenario.poses, twoBoneScenario.bSideValid, twoBoneScenario.head, 0.0,
							  twoBoneScenario.user, twoBoneSkeleton, twoBoneScenario.config, pose);
		const glm::quat twoBoneRestPalm= glm::quat_cast(glm::mat3(twoBoneHand.palmFrameWorld));
		check(pose.bones[(int)B::LeftHand].present &&
				  nearlySameRotation(pose.bones[(int)B::LeftHand].deltaWorld * twoBoneRestPalm, twoBoneLeft.palmOrientationWorld, 1e-4f),
			  "(h) a two-bone finger rig still takes the measured palm orientation");
		const int indexFinger= (int)eFinger::Index;
		const glm::vec3 firstBone=
			pose.bones[(int)B::LeftIndexIntermediate].deltaWorld * twoBoneHand.restPhalanxDirWorld[indexFinger][0];
		const glm::vec3 secondBone=
			pose.bones[(int)B::LeftIndexDistal].deltaWorld * twoBoneHand.restPhalanxDirWorld[indexFinger][1];
		const float bend= acosf(std::clamp(glm::dot(firstBone, secondBone), -1.f, 1.f));
		check(!pose.bones[(int)B::LeftIndexProximal].present && pose.bones[(int)B::LeftIndexIntermediate].present &&
				  pose.bones[(int)B::LeftIndexDistal].present && nearlyEqual(bend, 0.8f, 1e-3f) &&
				  !pose.bones[(int)B::LeftLittleIntermediate].present,
			  "(h) the two present finger bones pose, the second folding in the distal bend");
		VmcRetarget::VmcPose vmc;
		VmcRetarget::buildPoseFromAvatar(pose, bareSkeleton, vmc);
		check(vmc.bones[(int)VmcRetarget::eVmcBone::LeftUpperArm].present &&
				  !vmc.bones[(int)VmcRetarget::eVmcBone::LeftShoulder].present &&
				  !vmc.bones[(int)VmcRetarget::eVmcBone::LeftIndexProximal].present,
			  "(h) the VMC stream of a bare avatar carries only what it has");
	}

	// (i) Both VRM generations and a non-normalized rig give the same world
	// result for the same input, and the same input twice gives bit-identical
	// output
	{
		Scenario a= makeRestScenario(eVrmVersion::Vrm1, false);
		Scenario b= makeRestScenario(eVrmVersion::Vrm0, false);
		Scenario c= makeRestScenario(eVrmVersion::Vrm1, true);
		auto poseAll= [&](Scenario& scenario) {
			HandPose& right= scenario.poses[1];
			right.fingers[(int)eFinger::Middle]= {0.1f, 0.6f, 0.4f, 0.3f};
			placeHand(right, right.shoulderPositionWorld + glm::vec3(0.2f, 0.1f, -0.2f),
					  glm::angleAxis(0.4f, glm::normalize(glm::vec3(1.f, 1.f, 0.f))));
			scenario.head.orientationWorld= glm::angleAxis(0.3f, glm::vec3(0.f, 1.f, 0.f));
		};
		poseAll(a);
		poseAll(b);
		poseAll(c);
		AvatarRetarget ra, rb, rc;
		AvatarPose pa, pb, pc, pa2;
		solveScenario(a, ra, 0.0, pa);
		solveScenario(b, rb, 0.0, pb);
		solveScenario(c, rc, 0.0, pc);
		std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posedA, posedB, posedC;
		computePosedBones(*a.skeleton, pa, posedA);
		computePosedBones(*b.skeleton, pb, posedB);
		computePosedBones(*c.skeleton, pc, posedC);
		bool bSame= true;
		for (int index= 0; index < HUMANOID_BONE_COUNT; ++index)
		{
			if (!a.skeleton->getBone((B)index).present)
				continue;
			bSame&= nearlyEqual(posedA[index].positionWorld, posedB[index].positionWorld, kTolerance);
			bSame&= nearlyEqual(posedA[index].positionWorld, posedC[index].positionWorld, kTolerance);
		}
		check(bSame, "(i) VRM 0.x, VRM 1.0 and a rotated-rest rig pose to the same world positions");

		AvatarRetarget ra2;
		solveScenario(a, ra2, 0.0, pa2);
		check(std::memcmp(&pa.bones, &pa2.bones, sizeof(pa.bones)) == 0 &&
				  std::memcmp(&pa.rootPositionWorld, &pa2.rootPositionWorld, sizeof(pa.rootPositionWorld)) == 0,
			  "(i) the same input solves to bit-identical output");

		// The VMC stream composes back to the posed world bones
		VmcRetarget::VmcPose vmc;
		VmcRetarget::buildPoseFromAvatar(pa, *a.skeleton, vmc);
		BoneWorld chest;
		chest.position= a.skeleton->getBone(B::Chest).restPositionWorld;
		const BoneWorld clavicle= composeBone(vmc, VmcRetarget::eVmcBone::RightShoulder, chest);
		const BoneWorld upperArm= composeBone(vmc, VmcRetarget::eVmcBone::RightUpperArm, clavicle);
		const BoneWorld lowerArm= composeBone(vmc, VmcRetarget::eVmcBone::RightLowerArm, upperArm);
		const BoneWorld hand= composeBone(vmc, VmcRetarget::eVmcBone::RightHand, lowerArm);
		const BoneWorld middleProximal= composeBone(vmc, VmcRetarget::eVmcBone::RightMiddleProximal, hand);
		const BoneWorld middleIntermediate= composeBone(vmc, VmcRetarget::eVmcBone::RightMiddleIntermediate, middleProximal);
		check(nearlyEqual(upperArm.position, posedA[(int)B::RightUpperArm].positionWorld, kTolerance) &&
				  nearlyEqual(lowerArm.position, posedA[(int)B::RightLowerArm].positionWorld, kTolerance) &&
				  nearlyEqual(hand.position, posedA[(int)B::RightHand].positionWorld, kTolerance) &&
				  nearlyEqual(middleIntermediate.position, posedA[(int)B::RightMiddleIntermediate].positionWorld, kTolerance) &&
				  nearlySameRotation(hand.rotation, pa.bones[(int)B::RightHand].deltaWorld, 1e-4f),
			  "(i) composing the VMC stream rebuilds the posed arm and fingers");
	}

	MIKAN_LOG_INFO("test-avatar-retarget") << (failures == 0 ? "ALL PASSED" : "FAILURES") << " (" << failures
										   << " failed)";
	return failures == 0 ? 0 : 1;
}
MIKAN_REGISTER_TEST("--test-avatar-retarget", "Avatar retarget: rest identity, reach scaling, two-bone IK, hands, fingers, root, VMC",
					eTestCategory::SelfTest, runAvatarRetargetTest);
