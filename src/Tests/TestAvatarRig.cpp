#include "TestCommon.h"

#include "glm/gtx/quaternion.hpp"

#include "AvatarRetarget.h"
#include "AvatarRig.h"
#include "AvatarSkeleton.h"
#include "MathGLM.h"
#include "PathUtils.h"
#include "SyntheticAvatar.h"

// The avatar rig settings: the sidecar round trip, the mapping overrides
// applied by name, and what each retarget adjustment does to the solve on
// the shared synthetic rig. Every case compares two solves that differ only
// in the adjustment, so the expected change is exact.

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

bool nearlySameRotation(const glm::quat& a, const glm::quat& b, float toleranceRadians)
{
	const float dot= std::fabs(glm::dot(glm::normalize(a), glm::normalize(b)));
	return 2.f * acosf(std::min(dot, 1.f)) <= toleranceRadians;
}

float rotationAngle(const glm::quat& q)
{
	return 2.f * acosf(std::min(std::fabs(glm::normalize(q).w), 1.f));
}

eHumanoidBone fingerBone(eHandSide side, int finger, int phalanx)
{
	return (eHumanoidBone)((int)firstHumanoidFingerBone((int)side) + finger * 3 + phalanx);
}

// The 0.x lowerCamel spelling the synthetic rig names its nodes with
std::string rigNodeName(eHumanoidBone bone)
{
	std::string name= humanoidBoneName(bone);
	name[0]= (char)std::tolower((unsigned char)name[0]);
	return name;
}

// The left hand's index and middle swapped in the FILE's own map, the way the
// Jasper Mozu VRM ships
AvatarModel makeSwappedModel()
{
	AvatarModel model= makeModel(eVrmVersion::Vrm1, false);
	for (int phalanx= 0; phalanx < 3; ++phalanx)
	{
		std::swap(model.humanoidNodes[(int)fingerBone(eHandSide::Left, (int)eFinger::Index, phalanx)],
				  model.humanoidNodes[(int)fingerBone(eHandSide::Left, (int)eFinger::Middle, phalanx)]);
	}
	model.fileHumanoidNodes= model.humanoidNodes;
	return model;
}

// A hand resting in the avatar's own T-pose with a measured forearm and
// shoulder (see the retarget test)
HandPose makeRestHand(const AvatarSkeleton& skeleton, eHandSide side)
{
	const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
	const glm::quat restPalm= glm::quat_cast(glm::mat3(hand.palmFrameWorld));

	HandPose pose;
	pose.tracked= true;
	pose.side= side;
	pose.presence= 1.f;
	pose.confidence= 1.f;
	pose.skeleton= hand.skeleton;
	pose.hasWorldPose= true;
	pose.palmOrientationWorld= restPalm;
	pose.palmPositionWorld= glm::vec3(hand.palmFrameWorld[3]);
	pose.hasForearmPose= true;
	pose.forearmOrientationWorld= restPalm;
	pose.forearmConfidence= 1.f;
	pose.hasShoulder= true;
	pose.shoulderPositionWorld=
		skeleton.getBone(side == eHandSide::Left ? B::LeftUpperArm : B::RightUpperArm).restPositionWorld;
	pose.shoulderConfidence= 1.f;
	return pose;
}

// Moves a rest hand so its wrist lands on `wrist` with the palm rotated by
// `rotation`, forearm following
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
	AvatarRigSettings rig;
	AvatarConfig placement;
};

Scenario makeRestScenario(AvatarModel model)
{
	Scenario scenario;
	scenario.model= std::move(model);
	scenario.skeleton= std::make_unique<AvatarSkeleton>(scenario.model);
	scenario.poses[0]= makeRestHand(*scenario.skeleton, eHandSide::Left);
	scenario.poses[1]= makeRestHand(*scenario.skeleton, eHandSide::Right);
	scenario.head.valid= true;
	scenario.head.positionWorld= scenario.skeleton->getBone(B::Head).restPositionWorld;
	scenario.user.upperArmLengthMeters= kUpperArmLength;
	scenario.user.forearmLengthMeters= kForearmLength;
	scenario.user.shoulderWidthMeters= kShoulderWidth;
	return scenario;
}

AvatarPose solveScenario(const Scenario& scenario)
{
	AppConfig config;
	config.avatar= scenario.placement;
	AvatarRetarget retarget;
	AvatarPose pose;
	retarget.solve(scenario.poses, scenario.bSideValid, scenario.head, 0.0, scenario.user, *scenario.skeleton,
				   makeAvatarRetargetConfig(config, scenario.rig), pose);
	return pose;
}

std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posedBones(const Scenario& scenario, const AvatarPose& pose)
{
	std::array<AvatarPosedBone, HUMANOID_BONE_COUNT> posed;
	computePosedBones(*scenario.skeleton, pose, posed);
	return posed;
}
} // namespace

static int runAvatarRigTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-avatar-rig") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-avatar-rig") << "FAIL " << name;
			failures++;
		}
	};

	// (a) The sidecar round trip, through a non-ASCII path: overrides (one
	// of them "unmapped"), trims and every per-side field survive, and a
	// missing sidecar yields the defaults
	{
		AvatarRigSettings saved;
		saved.boneNodeOverrides[(int)B::LeftIndexProximal]= "J_Bip_L_Middle1";
		saved.boneNodeOverrides[(int)B::LeftEye]= "";
		saved.rotationTrimDegrees[(int)B::Head]= glm::vec3(1.f, -2.f, 3.5f);
		saved.rotationTrimDegrees[(int)B::RightLowerArm]= glm::vec3(12.f, 0.f, 0.f);
		saved.sides[1].elbowHintOffset= glm::vec3(-0.1f, -0.05f, -0.2f);
		saved.sides[1].elbowHintConfidence= 0.7f;
		saved.sides[1].bThumbPronationAuto= false;
		saved.sides[1].thumbPronationDegrees= -40.f;
		saved.sides[1].curlGain= 0.8f;
		saved.sides[1].splayGain= 1.3f;
		saved.sides[1].fingerEnabled[(int)eFinger::Ring]= false;

		const std::filesystem::path folder=
			std::filesystem::temp_directory_path() / PathUtils::utf8ToPath("mikan-test-avatar-rig-\xE3\x82\xA2\xE3\x83\x90\xE3\x82\xBF\xE3\x83\xBC");
		std::filesystem::create_directories(folder);
		const std::filesystem::path vrmPath= folder / PathUtils::utf8ToPath("\xE3\x83\x86\xE3\x82\xB9\xE3\x83\x88.vrm");
		check(getAvatarRigPath(vrmPath).filename() == PathUtils::utf8ToPath("\xE3\x83\x86\xE3\x82\xB9\xE3\x83\x88.mikanrig.json"),
			  "(a) the sidecar sits beside the model as <stem>.mikanrig.json");

		const bool bSaved= saveAvatarRig(vrmPath, saved);
		const AvatarRigSettings loaded= loadAvatarRig(vrmPath);
		bool bSame= bSaved && loaded.boneNodeOverrides == saved.boneNodeOverrides &&
			loaded.rotationTrimDegrees == saved.rotationTrimDegrees;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const AvatarRigSettings::Side& a= saved.sides[sideIndex];
			const AvatarRigSettings::Side& b= loaded.sides[sideIndex];
			bSame&= a.elbowHintOffset == b.elbowHintOffset && a.elbowHintConfidence == b.elbowHintConfidence &&
				a.bThumbPronationAuto == b.bThumbPronationAuto && a.thumbPronationDegrees == b.thumbPronationDegrees &&
				a.curlGain == b.curlGain && a.splayGain == b.splayGain && a.fingerEnabled == b.fingerEnabled;
		}
		check(bSame, "(a) the sidecar round-trips through a non-ASCII path");
		check(loaded.boneNodeOverrides[(int)B::LeftEye].has_value() && loaded.boneNodeOverrides[(int)B::LeftEye]->empty() &&
				  !loaded.boneNodeOverrides[(int)B::RightEye].has_value(),
			  "(a) an unmapped override stays distinct from no override");

		std::error_code error;
		std::filesystem::remove_all(folder, error);
		const AvatarRigSettings missing= loadAvatarRig(vrmPath);
		check(!missing.boneNodeOverrides[(int)B::LeftIndexProximal].has_value() && missing.sides[1].curlGain == 1.f,
			  "(a) a missing sidecar yields the defaults");

		AvatarRigSettings corrupt;
		std::string parseError;
		check(!avatarRigFromJson("{ not json", corrupt, parseError) && !parseError.empty(),
			  "(a) a corrupt sidecar is rejected with an error");
	}

	// (b) Overrides resolve by node name; an unknown name or unmapping a
	// required bone warns and keeps the file's mapping
	{
		AvatarModel model= makeModel(eVrmVersion::Vrm1, false);
		AvatarRigSettings rig;
		rig.boneNodeOverrides[(int)B::LeftEye]= rigNodeName(B::Head);
		rig.boneNodeOverrides[(int)B::LeftShoulder]= "";
		rig.boneNodeOverrides[(int)B::RightShoulder]= "noSuchNode";
		rig.boneNodeOverrides[(int)B::Hips]= "";
		std::vector<std::string> warnings;
		applyRigToModel(rig, model, warnings);
		check(model.humanoidNodes[(int)B::LeftEye] == model.fileHumanoidNodes[(int)B::Head] &&
				  model.humanoidNodes[(int)B::LeftShoulder] < 0,
			  "(b) overrides map by name and unmap");
		check(model.humanoidNodes[(int)B::RightShoulder] == model.fileHumanoidNodes[(int)B::RightShoulder] &&
				  model.humanoidNodes[(int)B::Hips] == model.fileHumanoidNodes[(int)B::Hips],
			  "(b) an unknown name and an unmapped required bone keep the file's mapping");
		// Unknown name, required unmap, and LeftEye sharing the head's node
		check(warnings.size() == 3, "(b) each rejected override and the shared node warns");

		applyRigToModel(AvatarRigSettings(), model, warnings);
		check(model.humanoidNodes == model.fileHumanoidNodes, "(b) no overrides restores the file's map");
	}

	// (c) A rig whose file swaps index and middle: the skeleton warns and its
	// index base sits on the pinky side; overriding the six slots back by
	// name clears the warning and restores the sign
	{
		AvatarModel model= makeSwappedModel();
		const AvatarSkeleton swapped(model);
		const float swappedIndexY= swapped.getHand(eHandSide::Left).skeleton.baseInPalm[(int)eFinger::Index].y;
		check(!swapped.getWarnings().empty() && swappedIndexY < 0.f,
			  "(c) the swapped file map warns and puts the left index base on palm -Y");

		AvatarRigSettings rig;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const B index= fingerBone(eHandSide::Left, (int)eFinger::Index, phalanx);
			const B middle= fingerBone(eHandSide::Left, (int)eFinger::Middle, phalanx);
			rig.boneNodeOverrides[(int)index]= rigNodeName(index);
			rig.boneNodeOverrides[(int)middle]= rigNodeName(middle);
		}
		std::vector<std::string> warnings;
		applyRigToModel(rig, model, warnings);
		const AvatarSkeleton fixed(model);
		check(warnings.empty() && fixed.getWarnings().empty() &&
				  fixed.getHand(eHandSide::Left).skeleton.baseInPalm[(int)eFinger::Index].y > 0.f,
			  "(c) swapping index and middle back clears the warning and restores the index sign");
	}

	// (d) Trims. A 30 degree hand trim about palm X rotates the rest-input
	// hand delta by exactly that; a forearm roll trim turns the forearm about
	// its own axis by its angle; a head trim leaves the inverse of itself
	{
		Scenario base= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		const AvatarPose basePose= solveScenario(base);

		Scenario trimmed= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		trimmed.rig.rotationTrimDegrees[(int)B::LeftHand]= glm::vec3(30.f, 0.f, 0.f);
		trimmed.rig.rotationTrimDegrees[(int)B::LeftLowerArm]= glm::vec3(20.f, 0.f, 0.f);
		trimmed.rig.rotationTrimDegrees[(int)B::Head]= glm::vec3(0.f, 0.f, 15.f);
		const AvatarPose trimmedPose= solveScenario(trimmed);

		const glm::quat handDelta= trimmedPose.bones[(int)B::LeftHand].deltaWorld;
		const glm::vec3 palmX(trimmed.skeleton->getHand(eHandSide::Left).palmFrameWorld[0]);
		check(nearlySameRotation(handDelta, glm::angleAxis(glm::radians(-30.f), palmX), 1e-4f),
			  "(d) a 30 degree hand trim turns the rest hand by 30 degrees about its palm axis");

		const glm::quat forearmChange= trimmedPose.bones[(int)B::LeftLowerArm].deltaWorld *
			glm::inverse(basePose.bones[(int)B::LeftLowerArm].deltaWorld);
		const glm::vec3 forearmAxis= glm::normalize(trimmed.skeleton->getBone(B::LeftHand).restPositionWorld -
													trimmed.skeleton->getBone(B::LeftLowerArm).restPositionWorld);
		check(nearlySameRotation(forearmChange, glm::angleAxis(glm::radians(20.f), forearmAxis), 1e-4f),
			  "(d) a forearm roll trim rolls the forearm about its axis by the trim");

		check(nearlySameRotation(trimmedPose.bones[(int)B::Head].deltaWorld,
								 glm::angleAxis(glm::radians(-15.f), glm::vec3(0.f, 0.f, 1.f)), 1e-4f),
			  "(d) a head trim offsets the head delta by its inverse");
		check(nearlySameRotation(trimmedPose.bones[(int)B::RightHand].deltaWorld,
								 basePose.bones[(int)B::RightHand].deltaWorld, 1e-6f),
			  "(d) a trim on one hand leaves the other alone");
	}

	// (e) Finger tweaks: curl gain 0.5 poses the index exactly as half the
	// bend at gain 1; a disabled finger is absent
	{
		Scenario gained= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		gained.poses[0].fingers[(int)eFinger::Index]= {0.2f, 0.8f, 0.6f, 0.4f};
		gained.rig.sides[0].curlGain= 0.5f;
		gained.rig.sides[0].splayGain= 0.5f;
		gained.rig.sides[0].fingerEnabled[(int)eFinger::Ring]= false;
		const AvatarPose gainedPose= solveScenario(gained);

		Scenario halved= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		halved.poses[0].fingers[(int)eFinger::Index]= {0.1f, 0.4f, 0.3f, 0.2f};
		const AvatarPose halvedPose= solveScenario(halved);

		bool bSame= true;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
		{
			const int bone= (int)fingerBone(eHandSide::Left, (int)eFinger::Index, phalanx);
			bSame&= nearlySameRotation(gainedPose.bones[bone].deltaWorld, halvedPose.bones[bone].deltaWorld, 1e-5f);
		}
		check(bSame && rotationAngle(halvedPose.bones[(int)B::LeftIndexDistal].deltaWorld) > 0.1f,
			  "(e) curl and splay gain 0.5 pose the index as the halved angles do");

		bool bRingAbsent= true;
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			bRingAbsent&= !gainedPose.bones[(int)fingerBone(eHandSide::Left, (int)eFinger::Ring, phalanx)].present;
		check(bRingAbsent && gainedPose.bones[(int)B::LeftLittleProximal].present &&
				  halvedPose.bones[(int)B::LeftRingProximal].present,
			  "(e) a disabled finger is absent and its neighbors are not");
	}

	// (f) Thumb pronation: on the swapped rig the automatic sign is the
	// wrong one; the explicit opposite angle moves the curled thumb to the
	// other side of its uncurled direction, the side the correct rig uses
	{
		auto thumbSide= [](Scenario& scenario) {
			scenario.poses[0].fingers[(int)eFinger::Thumb]= {0.f, 0.f, 0.9f, 0.f};
			const auto curled= posedBones(scenario, solveScenario(scenario));
			scenario.poses[0].fingers[(int)eFinger::Thumb]= {0.f, 0.f, 0.f, 0.f};
			const auto straight= posedBones(scenario, solveScenario(scenario));
			const glm::vec3 palmY(scenario.skeleton->getHand(eHandSide::Left).palmFrameWorld[1]);
			const glm::vec3 offset=
				curled[(int)B::LeftThumbDistal].positionWorld - straight[(int)B::LeftThumbDistal].positionWorld;
			return glm::dot(offset, palmY);
		};

		Scenario correct= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		const float correctSide= thumbSide(correct);

		Scenario swappedAuto= makeRestScenario(makeSwappedModel());
		const float autoSide= thumbSide(swappedAuto);

		Scenario swappedExplicit= makeRestScenario(makeSwappedModel());
		swappedExplicit.rig.sides[0].bThumbPronationAuto= false;
		swappedExplicit.rig.sides[0].thumbPronationDegrees= -glm::degrees(
			HandPoseModel::getThumbPronationRad(swappedExplicit.skeleton->getHand(eHandSide::Left).skeleton));
		const float explicitSide= thumbSide(swappedExplicit);

		check(std::fabs(correctSide) > 1e-3f && autoSide * explicitSide < 0.f && explicitSide * correctSide > 0.f,
			  "(f) an explicit thumb pronation flips the swapped rig's thumb curl back to the correct side");
	}

	// (g) The elbow hint lives in the rooted torso frame: the same input
	// yawed 90 degrees about the origin, with the fixed root yawed with it,
	// bends the elbow to the same place in the root frame
	{
		auto elbowInRoot= [](float yawDegrees) {
			Scenario scenario= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
			scenario.placement.followShoulders= false;
			scenario.placement.rootYawDegrees= yawDegrees;
			scenario.bSideValid[1]= false;
			HandPose& left= scenario.poses[0];
			left.hasForearmPose= false;
			left.forearmConfidence= 0.f;
			const glm::vec3 shoulder= left.shoulderPositionWorld;
			placeHand(left, shoulder + glm::vec3(0.25f, -0.05f, -0.15f), glm::quat(1.f, 0.f, 0.f, 0.f));

			const glm::quat yaw= glm::angleAxis(glm::radians(yawDegrees), glm::vec3(0.f, 0.f, 1.f));
			left.palmOrientationWorld= yaw * left.palmOrientationWorld;
			left.palmPositionWorld= yaw * left.palmPositionWorld;
			left.shoulderPositionWorld= yaw * left.shoulderPositionWorld;
			scenario.head.positionWorld= yaw * scenario.head.positionWorld;
			scenario.head.orientationWorld= yaw;

			const AvatarPose pose= solveScenario(scenario);
			const auto posed= posedBones(scenario, pose);
			return glm::vec3(glm::inverse(pose.rootRotationWorld) *
							 (posed[(int)B::LeftLowerArm].positionWorld - pose.rootPositionWorld));
		};
		const glm::vec3 unyawed= elbowInRoot(0.f);
		const glm::vec3 yawed= elbowInRoot(90.f);
		Scenario reference= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		const glm::vec3 straightElbow= reference.skeleton->getBone(B::LeftLowerArm).restPositionWorld;
		check(nearlyEqual(unyawed, yawed, kTolerance) && !nearlyEqual(unyawed, straightElbow, 1e-2f),
			  "(g) the elbow hint turns with the root yaw");
	}

	// (h) The confidence blend: at zero forearm confidence the hint alone
	// decides, at or above elbowHintConfidence the measured elbow alone, and
	// halfway the pole is the midpoint of the two
	{
		auto solveElbow= [](float forearmConfidence, bool bForearm, const glm::vec3& hintOffset) {
			Scenario scenario= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
			scenario.bSideValid[1]= false;
			scenario.poses[1].hasShoulder= false;
			HandPose& left= scenario.poses[0];
			const glm::vec3 shoulder= left.shoulderPositionWorld;
			placeHand(left, shoulder + glm::vec3(0.3f, 0.12f, -0.05f),
					  glm::angleAxis(glm::radians(-50.f), glm::vec3(0.f, 0.f, 1.f)));
			left.hasForearmPose= bForearm;
			left.forearmConfidence= forearmConfidence;
			scenario.rig.sides[0].elbowHintOffset= hintOffset;
			scenario.rig.sides[0].elbowHintConfidence= 0.5f;
			const AvatarPose pose= solveScenario(scenario);
			return posedBones(scenario, pose)[(int)B::LeftLowerArm].positionWorld;
		};

		const glm::vec3 hint(-0.15f, 0.f, -0.30f);
		Scenario measure= makeRestScenario(makeModel(eVrmVersion::Vrm1, false));
		HandPose& left= measure.poses[0];
		const glm::vec3 shoulder= left.shoulderPositionWorld;
		placeHand(left, shoulder + glm::vec3(0.3f, 0.12f, -0.05f),
				  glm::angleAxis(glm::radians(-50.f), glm::vec3(0.f, 0.f, 1.f)));
		const glm::vec3 measuredOffset= left.getElbowPositionWorld(kForearmLength) - shoulder;

		const glm::vec3 hintOnly= solveElbow(0.f, false, hint);
		const glm::vec3 measuredOnly= solveElbow(0.f, false, measuredOffset);
		check(nearlyEqual(solveElbow(0.f, true, hint), hintOnly, kTolerance),
			  "(h) zero forearm confidence bends toward the hint alone");
		check(nearlyEqual(solveElbow(0.5f, true, hint), measuredOnly, kTolerance) &&
				  nearlyEqual(solveElbow(1.f, true, hint), measuredOnly, kTolerance),
			  "(h) confidence at the blend threshold bends toward the measured elbow alone");
		check(nearlyEqual(solveElbow(0.25f, true, hint), solveElbow(0.f, false, (hint + measuredOffset) * 0.5f), kTolerance) &&
				  !nearlyEqual(hintOnly, measuredOnly, 1e-2f),
			  "(h) half the threshold bends toward the midpoint of hint and measurement");
	}

	MIKAN_LOG_INFO("test-avatar-rig") << (failures == 0 ? "ALL PASSED" : "FAILURES") << " (" << failures << " failed)";
	return failures == 0 ? 0 : 1;
}
MIKAN_REGISTER_TEST("--test-avatar-rig", "Avatar rig: sidecar round trip, mapping overrides, trims, finger tweaks, elbow hint",
					eTestCategory::SelfTest, runAvatarRigTest);
