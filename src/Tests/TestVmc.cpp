#include "TestCommon.h"

#include "glm/matrix.hpp"

#include "OscStreamer.h"

// Synthetic tests for the VMC retarget. The whole conversion is pure, so every
// case here is exact: a hand at the rest pose must emit identity rotations, and
// composing the emitted bones the way a VMC receiver does must rebuild the
// measured skeleton.

namespace
{
using namespace VmcRetarget;

constexpr float kShoulderWidth= 0.30f;
constexpr float kUpperArm= 0.32f;
constexpr float kForearm= 0.26f;
constexpr float kHalfPalm= 0.045f;

VmcBodyLengths makeLengths()
{
	VmcBodyLengths lengths;
	lengths.shoulderWidthMeters= kShoulderWidth;
	lengths.upperArmLengthMeters= kUpperArm;
	lengths.forearmLengthMeters= kForearm;
	lengths.headOffsetMeters= 0.08f;
	return lengths;
}

// A plausible metric hand: the four fingers along palm +X, the thumb off to
// the side, with the flat-hand neutral directions the app defaults to
HandSkeleton makeSkeleton(eHandSide side)
{
	HandSkeleton skeleton;
	const float thumbSign= side == eHandSide::Right ? 1.f : -1.f;

	skeleton.baseInPalm[(int)eFinger::Thumb]= glm::vec3(-0.01f, thumbSign * 0.025f, 0.005f);
	skeleton.baseInPalm[(int)eFinger::Index]= glm::vec3(kHalfPalm, thumbSign * 0.020f, 0.f);
	skeleton.baseInPalm[(int)eFinger::Middle]= glm::vec3(kHalfPalm, 0.f, 0.f);
	skeleton.baseInPalm[(int)eFinger::Ring]= glm::vec3(kHalfPalm, thumbSign * -0.019f, 0.f);
	skeleton.baseInPalm[(int)eFinger::Pinky]= glm::vec3(kHalfPalm - 0.005f, thumbSign * -0.038f, 0.f);

	const float lengths[FINGER_COUNT][3]= {
		{0.035f, 0.032f, 0.026f}, // thumb
		{0.040f, 0.024f, 0.018f}, // index
		{0.044f, 0.027f, 0.020f}, // middle
		{0.041f, 0.025f, 0.019f}, // ring
		{0.032f, 0.019f, 0.017f}, // pinky
	};
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int phalanx= 0; phalanx < 3; ++phalanx)
			skeleton.phalanxLengths[finger][phalanx]= lengths[finger][phalanx];

	skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
	return skeleton;
}

// A hand whose palm sits at `palmRotation` about `palmPosition`, with the
// forearm and shoulder filled in so the whole arm chain is exercised
HandPose makePose(eHandSide side, const glm::quat& palmRotation, const glm::vec3& palmPosition,
				  const glm::quat& forearmRotation, const glm::vec3& shoulderPosition)
{
	HandPose pose;
	pose.tracked= true;
	pose.side= side;
	pose.presence= 1.f;
	pose.confidence= 1.f;
	pose.skeleton= makeSkeleton(side);

	pose.hasWorldPose= true;
	pose.palmPositionWorld= palmPosition;
	pose.palmOrientationWorld= palmRotation;

	pose.hasForearmPose= true;
	pose.forearmOrientationWorld= forearmRotation;
	pose.forearmConfidence= 1.f;

	pose.hasShoulder= true;
	pose.shoulderPositionWorld= shoulderPosition;
	pose.shoulderConfidence= 1.f;

	return pose;
}

// The pose a side rests in: arm straight out to the side, palm down, fingers
// flat. This is the reference the whole retarget is measured against, so it
// must emit nothing but identity rotations.
HandPose makeRestPose(eHandSide side, const glm::vec3& chestOrigin, const VmcBodyLengths& lengths)
{
	const glm::vec3 armAxis= restArmDirection(side);
	const glm::quat restPalm= glm::quat_cast(restPalmFrame(side));

	const glm::vec3 shoulder= chestOrigin + armAxis * (lengths.shoulderWidthMeters * 0.5f);
	const glm::vec3 elbow= shoulder + armAxis * lengths.upperArmLengthMeters;
	const glm::vec3 wrist= elbow + armAxis * lengths.forearmLengthMeters;
	// The palm origin is half a palm PAST the wrist along the palm's +X, which
	// at rest points down the arm
	const glm::vec3 palm= wrist + armAxis * kHalfPalm;

	return makePose(side, restPalm, palm, restPalm, shoulder);
}

// Composes the emitted bones the way a VMC receiver does: each bone's local
// transform applied against its parent's, with an absent bone left at the
// avatar's rest pose (identity rotation, and a position this test does not
// need since it only walks streamed chains).
struct BoneWorld
{
	glm::quat rotation{1.f, 0.f, 0.f, 0.f};
	glm::vec3 position{0.f};
};

// Unity -> avatar frame, so the composed chain can be compared against the
// measured world geometry it came from. Inverse of the emitted conversion.
glm::vec3 unityToWorldPosition(const glm::vec3& unity)
{
	return glm::vec3(unity.z, -unity.x, unity.y);
}
glm::quat unityToWorldRotation(const glm::quat& unity)
{
	const glm::mat3 toWorld(
		glm::vec3(0.f, -1.f, 0.f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(1.f, 0.f, 0.f));
	return glm::quat_cast(toWorld * glm::mat3_cast(unity) * glm::transpose(toWorld));
}

BoneWorld composeBone(const VmcPose& pose, eVmcBone bone, const BoneWorld& parent)
{
	const VmcBone& streamed= pose.bones[(int)bone];
	BoneWorld out;
	if (!streamed.present)
	{
		// Unstreamed bones sit at the avatar's rest pose, whose local rotation
		// is the identity - which is exactly why identity is the reference
		out= parent;
		return out;
	}

	out.rotation= parent.rotation * unityToWorldRotation(streamed.localRotation);
	out.position= parent.position + parent.rotation * unityToWorldPosition(streamed.localPosition);
	return out;
}

bool nearlyEqual(const glm::vec3& a, const glm::vec3& b, float tolerance)
{
	return glm::length(a - b) <= tolerance;
}

bool isIdentity(const glm::quat& q, float tolerance)
{
	return glm::angle(glm::normalize(q.w < 0.f ? -q : q)) <= tolerance;
}

// Angle between two directions, radians
float angleBetween(const glm::vec3& a, const glm::vec3& b)
{
	const float cosAngle= std::clamp(glm::dot(glm::normalize(a), glm::normalize(b)), -1.f, 1.f);
	return std::acos(cosAngle);
}

eVmcBone armBone(eHandSide side, int index) // 0 clavicle, 1 upper, 2 lower, 3 hand
{
	const int base= side == eHandSide::Left ? (int)eVmcBone::LeftShoulder : (int)eVmcBone::RightShoulder;
	return (eVmcBone)(base + index);
}

// -- Minimal OSC 1.0 bundle reader ------------------------------------------
// Deliberately written against the spec rather than against OscWriter, so it
// reads the bundle the way a receiver does: a decoder built from the encoder
// would agree with it however wrong both were.
struct DecodedMessage
{
	std::string address;
	std::string tags; // without the leading comma
	std::vector<float> floats;
	std::vector<int32_t> ints;
	std::vector<std::string> strings;
};

uint32_t readBigEndian32(const uint8_t* data)
{
	return ((uint32_t)data[0] << 24) | ((uint32_t)data[1] << 16) | ((uint32_t)data[2] << 8) |
		   (uint32_t)data[3];
}

// Reads a null-terminated string padded to a 4-byte boundary
bool readPaddedString(const std::vector<uint8_t>& buffer, size_t& offset, std::string& outText)
{
	const size_t start= offset;
	while (offset < buffer.size() && buffer[offset] != 0)
		offset++;
	if (offset >= buffer.size())
		return false;

	outText.assign((const char*)buffer.data() + start, offset - start);
	offset= start + ((offset - start) / 4 + 1) * 4;
	return offset <= buffer.size();
}

bool decodeMessage(const std::vector<uint8_t>& buffer, size_t offset, size_t size, DecodedMessage& out)
{
	const size_t end= offset + size;
	if (end > buffer.size() || !readPaddedString(buffer, offset, out.address))
		return false;

	std::string tagText;
	if (!readPaddedString(buffer, offset, tagText) || tagText.empty() || tagText[0] != ',')
		return false;
	out.tags= tagText.substr(1);

	for (char tag : out.tags)
	{
		if (tag == 'f' || tag == 'i')
		{
			if (offset + 4 > end)
				return false;
			const uint32_t raw= readBigEndian32(buffer.data() + offset);
			if (tag == 'f')
			{
				float value= 0.f;
				std::memcpy(&value, &raw, sizeof(value));
				out.floats.push_back(value);
			}
			else
			{
				out.ints.push_back((int32_t)raw);
			}
			offset+= 4;
		}
		else if (tag == 's')
		{
			std::string text;
			if (!readPaddedString(buffer, offset, text))
				return false;
			out.strings.push_back(text);
		}
		else
		{
			return false;
		}
	}
	return offset == end;
}

bool decodeBundle(const std::vector<uint8_t>& buffer, std::vector<DecodedMessage>& outMessages)
{
	if (buffer.size() < 16 || std::memcmp(buffer.data(), "#bundle\0", 8) != 0)
		return false;

	size_t offset= 16; // "#bundle\0" + the 8-byte time tag
	while (offset + 4 <= buffer.size())
	{
		const size_t size= readBigEndian32(buffer.data() + offset);
		offset+= 4;

		DecodedMessage message;
		if (!decodeMessage(buffer, offset, size, message))
			return false;
		outMessages.push_back(message);
		offset+= size;
	}
	return offset == buffer.size();
}

const DecodedMessage* findMessage(const std::vector<DecodedMessage>& messages, const char* address)
{
	for (const DecodedMessage& message : messages)
	{
		if (message.address == address)
			return &message;
	}
	return nullptr;
}
} // namespace

static int runVmcTest(const TestArgs&)
{
	int failures= 0;
	auto check= [&](bool bCondition, const char* name) {
		if (bCondition)
		{
			MIKAN_LOG_INFO("test-vmc") << "PASS " << name;
		}
		else
		{
			MIKAN_LOG_ERROR("test-vmc") << "FAIL " << name;
			failures++;
		}
	};

	const VmcBodyLengths lengths= makeLengths();
	const bool bBothValid[2]= {true, true};
	TrackingFrameResult::HeadPose noHead;

	// (a) The basis change. World is right-handed (+X forward, +Y the person's
	// left, +Z up); Unity is left-handed (+X right, +Y up, +Z forward). This is
	// the one place a handedness error can hide, so pin both halves down.
	{
		check(nearlyEqual(worldToUnityPosition(glm::vec3(1.f, 2.f, 3.f)), glm::vec3(-2.f, 3.f, 1.f), 1e-6f),
			  "world position maps onto the Unity axes");

		// Yaw to the person's left is a POSITIVE turn about world +Z and a
		// NEGATIVE one about Unity +Y: the flip reverses the rotation sense
		const glm::quat worldYaw= glm::angleAxis(glm::half_pi<float>(), glm::vec3(0.f, 0.f, 1.f));
		const glm::quat unityYaw= worldToUnityRotation(worldYaw);
		const glm::vec3 turnedForward= unityYaw * glm::vec3(0.f, 0.f, 1.f);
		check(nearlyEqual(turnedForward, glm::vec3(-1.f, 0.f, 0.f), 1e-5f),
			  "a left turn in world reads as a left turn in Unity");

		// A rotation must survive as a rotation: conjugating by a determinant
		// -1 basis change cancels, so no mirrored quaternion comes out
		const glm::mat3 rebuilt= glm::mat3_cast(unityYaw);
		check(fabsf(glm::determinant(rebuilt) - 1.f) < 1e-5f, "the converted rotation stays proper");

		// Round trip through the test's own inverse, which every later case
		// leans on
		const glm::quat tilted= glm::angleAxis(0.7f, glm::normalize(glm::vec3(0.3f, -0.5f, 0.8f)));
		check(isIdentity(glm::inverse(tilted) * unityToWorldRotation(worldToUnityRotation(tilted)), 1e-4f),
			  "the Unity conversion round trips");
	}

	// (b) The rest palm frame, stated rather than derived. Everything else here
	// uses restPalmFrame on both the producing and the checking side, so a
	// mirrored or swapped pair would cancel out and go unnoticed - this is the
	// only place the avatar's rest hand is pinned to an anatomical fact.
	{
		bool bFramesValid= true;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const eHandSide side= (eHandSide)sideIndex;
			const glm::mat3 frame= restPalmFrame(side);
			const glm::vec3 fingers= frame[0]; // palm +X
			const glm::vec3 thumbSide= frame[1]; // palm +Y
			const glm::vec3 palmar= frame[2]; // palm +Z, out of the palmar surface

			// Arms out to the sides, so the fingers run down the arm
			bFramesValid&= nearlyEqual(fingers, restArmDirection(side), 1e-6f);
			// Palms DOWN, which is the VRM rest pose
			bFramesValid&= nearlyEqual(palmar, glm::vec3(0.f, 0.f, -1.f), 1e-6f);
			// A left hand palm-down has its thumb forward, a right hand back.
			// This is the chirality, and it is the whole difference between the
			// two matrices.
			const float forward= side == eHandSide::Left ? 1.f : -1.f;
			bFramesValid&= nearlyEqual(thumbSide, glm::vec3(forward, 0.f, 0.f), 1e-6f);
			// Right-handed, not mirrored: a determinant of -1 here would flip
			// every finger silently
			bFramesValid&= fabsf(glm::determinant(frame) - 1.f) < 1e-6f;
		}
		check(bFramesValid, "the rest palm frames are palms-down T-pose hands of the right chirality");
	}

	// (c) Rest identity. A body in the avatar's rest pose must emit nothing but
	// identity rotations - that is what makes the streamed values deltas from
	// the avatar's own T-pose rather than from an assumed anatomy.
	{
		const glm::vec3 chest(0.f, 0.f, 1.35f);
		std::array<HandPose, 2> poses= {makeRestPose(eHandSide::Left, chest, lengths),
										makeRestPose(eHandSide::Right, chest, lengths)};

		VmcPose vmc;
		buildPose(poses, bBothValid, noHead, lengths, vmc);

		int emitted= 0;
		int nonIdentity= 0;
		for (int boneIndex= 0; boneIndex < VMC_BONE_COUNT; ++boneIndex)
		{
			const VmcBone& bone= vmc.bones[boneIndex];
			if (!bone.present)
				continue;
			emitted++;
			if (!isIdentity(bone.localRotation, 1e-3f))
				nonIdentity++;
		}
		// 8 arm bones + 30 finger bones; the head is absent (no head estimate)
		check(emitted == 38, "the rest pose emits every arm, hand and finger bone");
		check(nonIdentity == 0, "the rest pose emits only identity rotations");

		// Negative control: the finger rest directions come from the measured
		// skeleton, so bending the hand must break the identity
		poses[0].fingers[(int)eFinger::Index].proximal= 0.6f;
		buildPose(poses, bBothValid, noHead, lengths, vmc);
		check(!isIdentity(vmc.bones[(int)eVmcBone::LeftIndexProximal].localRotation, 1e-3f),
			  "a bent finger stops emitting identity");
		check(isIdentity(vmc.bones[(int)eVmcBone::LeftMiddleProximal].localRotation, 1e-3f),
			  "bending one finger leaves the others alone");
	}

	// (d) Chain round trip: compose the emitted bones the way the receiver does
	// and check the rebuilt skeleton against the geometry it was measured from.
	// This is what proves the retarget rather than just the encoder.
	{
		// Shoulders raised and rolled forward, mirrored about the chest: the
		// clavicle carries the DIRECTION and takes its length from the measured
		// shoulder width, so the composed joint lands on the measured one
		// exactly when the pair is symmetric at that separation - which is the
		// definition the width was measured under.
		const glm::vec3 chest(0.f, 0.f, 1.35f);
		const glm::vec3 clavicleDir= glm::normalize(glm::vec3(0.10f, 1.f, 0.20f));
		const glm::vec3 shoulderL= chest + clavicleDir * (kShoulderWidth * 0.5f);
		const glm::vec3 shoulderR= chest - clavicleDir * (kShoulderWidth * 0.5f);

		// A left arm reaching forward and down with a rolled forearm, i.e. a
		// pose whose every joint differs from the rest reference
		const glm::quat forearmRotation=
			glm::angleAxis(0.9f, glm::vec3(0.f, 0.f, 1.f)) *
			glm::angleAxis(-0.4f, glm::vec3(0.f, 1.f, 0.f)) *
			glm::angleAxis(0.8f, glm::vec3(1.f, 0.f, 0.f));
		const glm::quat palmRotation= forearmRotation * glm::angleAxis(0.35f, glm::vec3(0.f, 1.f, 0.f));

		// Place the wrist so the forearm bone is exactly kForearm long, which
		// is what the streamed offsets assert
		const glm::vec3 elbow= shoulderL + glm::normalize(glm::vec3(0.35f, 0.5f, -0.6f)) * kUpperArm;
		const glm::vec3 wrist= elbow + forearmRotation * glm::vec3(1.f, 0.f, 0.f) * kForearm;
		const glm::vec3 palm= wrist + palmRotation * glm::vec3(kHalfPalm, 0.f, 0.f);

		HandPose left= makePose(eHandSide::Left, palmRotation, palm, forearmRotation, shoulderL);
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			left.fingers[finger].lateral= 0.05f * (finger - 2);
			left.fingers[finger].proximal= 0.30f + 0.05f * finger;
			left.fingers[finger].intermediate= 0.45f;
			left.fingers[finger].distal= 0.25f;
		}

		// The right side is here only to fix the shoulder midpoint the left
		// clavicle is measured against; nothing below asserts on it
		HandPose right= makeRestPose(eHandSide::Right, chest, lengths);
		right.shoulderPositionWorld= shoulderR;

		std::array<HandPose, 2> poses= {left, right};
		VmcPose vmc;
		buildPose(poses, bBothValid, noHead, lengths, vmc);

		// Walk the chain from the chest, the bone this deliberately does not
		// stream (so the receiver holds it at rest, i.e. the identity)
		BoneWorld chestBone;
		chestBone.position= chest;

		const BoneWorld clavicle= composeBone(vmc, armBone(eHandSide::Left, 0), chestBone);
		const BoneWorld upperArm= composeBone(vmc, armBone(eHandSide::Left, 1), clavicle);
		const BoneWorld lowerArm= composeBone(vmc, armBone(eHandSide::Left, 2), upperArm);
		const BoneWorld hand= composeBone(vmc, armBone(eHandSide::Left, 3), lowerArm);

		// The clavicle points at the measured shoulder joint, so the upper arm
		// bone lands on it
		check(nearlyEqual(upperArm.position, shoulderL, 1e-4f),
			  "the composed chain reaches the measured shoulder");
		check(nearlyEqual(lowerArm.position, elbow, 1e-4f),
			  "the composed chain reaches the measured elbow");
		check(nearlyEqual(hand.position, wrist, 1e-4f),
			  "the composed chain reaches the measured wrist");

		// Directions, independently of the lengths: this is what a receiver
		// with its own proportions actually consumes
		const glm::vec3 restLeft= restArmDirection(eHandSide::Left);
		check(angleBetween(upperArm.rotation * restLeft, elbow - shoulderL) < 1e-4f,
			  "the upper arm points down the measured upper arm");
		check(angleBetween(lowerArm.rotation * restLeft, wrist - elbow) < 1e-4f,
			  "the forearm points down the measured forearm");

		// The hand carries the full palm frame, roll included, not just its
		// direction
		const glm::mat3 restPalm= restPalmFrame(eHandSide::Left);
		const glm::quat rebuiltPalm= hand.rotation * glm::quat_cast(restPalm);
		check(isIdentity(glm::inverse(palmRotation) * rebuiltPalm, 1e-4f),
			  "the hand rebuilds the measured palm orientation");

		// Fingers: rebuild every joint through the composed chain and compare
		// against the same forward kinematics the app draws and streams
		glm::mat4 palmTransform= glm::mat4_cast(palmRotation);
		palmTransform[3]= glm::vec4(palm, 1.f);
		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> truthJoints;
		HandPoseModel::buildFingerJoints(palmTransform, left.skeleton, left.fingers, truthJoints);

		float worstJointError= 0.f;
		const int firstBone= (int)firstFingerBone(eHandSide::Left);
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			BoneWorld parent= hand;
			for (int phalanx= 0; phalanx < 3; ++phalanx)
			{
				const BoneWorld joint=
					composeBone(vmc, (eVmcBone)(firstBone + finger * 3 + phalanx), parent);
				worstJointError=
					std::max(worstJointError, glm::length(joint.position - truthJoints[finger][phalanx]));
				parent= joint;
			}
			// The tip is one more offset along the distal bone
			const glm::vec3 distalDir= parent.rotation * (restPalm * left.skeleton.neutralDirInPalm[finger]);
			const glm::vec3 tip= parent.position + distalDir * left.skeleton.phalanxLengths[finger][2];
			worstJointError= std::max(worstJointError, glm::length(tip - truthJoints[finger][3]));
		}
		MIKAN_LOG_INFO("test-vmc") << "worst rebuilt finger joint error " << worstJointError * 1000.f << " mm";
		check(worstJointError < 0.001f, "the composed fingers rebuild the forward-kinematics joints");
	}

	// (e) Degradation. A bone the receiver never hears about stays at the
	// avatar's REST pose - which for an arm is the T-pose, not a neutral. The
	// hand's world orientation survives either way (rest locals are identity),
	// but skipping the arm leaves the T-posed forearm under a correctly
	// oriented hand, so the arm's whole rotation surfaces as WRIST BEND. That
	// is what the elbow dropouts looked like in VSeeFace, so the arm chain is
	// always streamed and the wrist has to stay near neutral without an elbow.
	{
		const glm::quat palmRotation= glm::angleAxis(0.8f, glm::normalize(glm::vec3(0.2f, 0.4f, 0.9f)));
		HandPose left= makePose(eHandSide::Left, palmRotation, glm::vec3(0.3f, 0.2f, 0.9f),
								glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(0.f));
		left.hasShoulder= false;
		left.hasForearmPose= false;

		std::array<HandPose, 2> poses= {left, HandPose()};
		const bool bLeftOnly[2]= {true, false};
		VmcPose vmc;
		buildPose(poses, bLeftOnly, noHead, lengths, vmc);

		check(vmc.bones[(int)eVmcBone::LeftUpperArm].present &&
				  vmc.bones[(int)eVmcBone::LeftLowerArm].present,
			  "an arm with no elbow still streams its chain");
		check(!vmc.bones[(int)eVmcBone::RightHand].present, "an invalid side streams nothing");
		check(vmc.bones[(int)eVmcBone::LeftHand].present, "the hand still streams without an arm");

		BoneWorld chestBone;
		const BoneWorld forearm= composeBone(vmc, eVmcBone::LeftLowerArm,
											 composeBone(vmc, eVmcBone::LeftUpperArm,
														 composeBone(vmc, eVmcBone::LeftShoulder,
																	 chestBone)));
		const BoneWorld hand= composeBone(vmc, eVmcBone::LeftHand, forearm);

		const glm::quat rebuiltPalm= hand.rotation * glm::quat_cast(restPalmFrame(eHandSide::Left));
		check(isIdentity(glm::inverse(palmRotation) * rebuiltPalm, 1e-4f),
			  "the hand orientation survives the missing arm");

		// The point of the change: with no elbow the forearm takes the hand's
		// own frame, so the wrist reads neutral instead of carrying the arm
		const glm::quat wristBend= glm::inverse(forearm.rotation) * hand.rotation;
		check(isIdentity(wristBend, 1e-4f),
			  "a missing elbow leaves the wrist neutral, not carrying the arm's rotation");
	}

	// (f) Head. Its measured frame is already the avatar frame, so the streamed
	// rotation must be the measured one with nothing but the basis change.
	{
		TrackingFrameResult::HeadPose head;
		head.valid= true;
		head.positionWorld= glm::vec3(0.1f, 0.f, 1.5f);
		head.orientationWorld= glm::angleAxis(0.5f, glm::vec3(0.f, 0.f, 1.f));
		head.confidence= 0.8f;

		std::array<HandPose, 2> poses;
		const bool bNone[2]= {false, false};
		VmcPose vmc;
		buildPose(poses, bNone, head, lengths, vmc);

		const VmcBone& headBone= vmc.bones[(int)eVmcBone::Head];
		check(headBone.present, "a valid head estimate streams the head bone");
		check(isIdentity(glm::inverse(head.orientationWorld) * unityToWorldRotation(headBone.localRotation),
						 1e-4f),
			  "the head rotation is the measured one");
		// The offset is the setting, carried up the avatar's own up axis
		check(nearlyEqual(headBone.localPosition, glm::vec3(0.f, lengths.headOffsetMeters, 0.f), 1e-6f),
			  "the head offset rides the avatar's up axis");

		head.valid= false;
		buildPose(poses, bNone, head, lengths, vmc);
		check(!vmc.bones[(int)eVmcBone::Head].present, "an invalid head streams nothing");
	}

	// (g) Freeze on loss. VMC has no confidence, so a hand that stops being
	// measured can only be expressed as an arm that stops moving.
	{
		HandPose live= makePose(eHandSide::Left, glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(0.2f, 0.1f, 0.9f),
								glm::quat(1.f, 0.f, 0.f, 0.f), glm::vec3(0.f, 0.15f, 1.3f));
		HandPose lost;

		OscStreamer::HeldPoseState held;
		HandPose out;

		check(OscStreamer::resolveVmcOutputPose(live, true, true, held, out) &&
				  nearlyEqual(out.palmPositionWorld, live.palmPositionWorld, 1e-6f),
			  "a live pose streams unchanged");

		check(OscStreamer::resolveVmcOutputPose(lost, false, true, held, out) &&
				  nearlyEqual(out.palmPositionWorld, live.palmPositionWorld, 1e-6f),
			  "a lost hand freezes at its last streamed pose");

		OscStreamer::HeldPoseState heldOff;
		check(OscStreamer::resolveVmcOutputPose(live, true, false, heldOff, out),
			  "freeze off still streams a live pose");
		check(!OscStreamer::resolveVmcOutputPose(lost, false, false, heldOff, out),
			  "freeze off releases the arm on loss");

		// A camera-space pose has no world anchor to hang a skeleton off
		HandPose cameraOnly= live;
		cameraOnly.hasWorldPose= false;
		OscStreamer::HeldPoseState heldCamera;
		check(!OscStreamer::resolveVmcOutputPose(cameraOnly, true, true, heldCamera, out),
			  "a camera-space pose is not streamable as bones");
	}

	// (h) The wire itself. Everything above checks the retarget; this checks
	// what a receiver actually parses, decoded straight from the encoded bundle
	// against the OSC 1.0 spec. A wrong address or type tag is invisible from
	// inside - a VMC receiver is required to ignore both silently.
	{
		const glm::vec3 chest(0.f, 0.f, 1.35f);
		TrackingFrameResult frame;
		frame.frameIndex= 7;
		frame.timestampMs= 1000.0;
		frame.poses[0]= makeRestPose(eHandSide::Left, chest, lengths);
		frame.poses[1]= makeRestPose(eHandSide::Right, chest, lengths);
		frame.head.valid= true;
		frame.head.positionWorld= glm::vec3(0.f, 0.f, 1.6f);
		frame.head.orientationWorld= glm::angleAxis(0.3f, glm::vec3(0.f, 0.f, 1.f));
		frame.head.confidence= 0.9f;

		OscStreamerConfig config;
		config.outputMode= eOscOutputMode::Vmc;
		config.maxRateHz= 0.f; // no decimation, so one call encodes one bundle
		config.shoulderWidthMeters= kShoulderWidth;
		config.upperArmLengthMeters= kUpperArm;
		config.forearmLengthMeters= kForearm;
		config.vmcHeadOffsetMeters= lengths.headOffsetMeters;

		OscStreamer streamer;
		streamer.setConfig(config);

		std::vector<std::vector<uint8_t>> packets;
		streamer.encodeFrame(frame, packets);

		// Every datagram must be a COMPLETE bundle in its own right - UDP does
		// not reassemble at the OSC layer - and must fit the size a receiver
		// will actually read
		size_t totalBytes= 0;
		size_t largestPacket= 0;
		bool bPacketsWellFormed= !packets.empty();
		std::vector<DecodedMessage> messages;
		for (const std::vector<uint8_t>& packet : packets)
		{
			bPacketsWellFormed&= packet.size() <= OscStreamer::k_maxDatagramBytes;
			bPacketsWellFormed&= decodeBundle(packet, messages);
			totalBytes+= packet.size();
			largestPacket= std::max(largestPacket, packet.size());
		}
		MIKAN_LOG_INFO("test-vmc") << "fully tracked VMC frame is " << packets.size() << " datagrams, "
								   << totalBytes << " bytes, largest " << largestPacket;
		check(bPacketsWellFormed, "every VMC datagram is a complete bundle within the size limit");

		const DecodedMessage* ok= findMessage(messages, "/VMC/Ext/OK");
		check(ok != nullptr && ok->tags == "iiii" && ok->ints.size() == 4 && ok->ints[0] == 1 &&
				  ok->ints[1] == 3 && ok->ints[3] == 1,
			  "/VMC/Ext/OK reports loaded, calibrated and tracking");

		const DecodedMessage* time= findMessage(messages, "/VMC/Ext/T");
		check(time != nullptr && time->tags == "f", "/VMC/Ext/T carries one float");

		const DecodedMessage* root= findMessage(messages, "/VMC/Ext/Root/Pos");
		check(root != nullptr && root->tags == "sfffffff" && root->strings.size() == 1 &&
				  root->strings[0] == "root" && root->floats.size() == 7 &&
				  root->floats[6] == 1.f,
			  "/VMC/Ext/Root/Pos is an identity root named \"root\"");

		// Every bone, exactly once, with the name and argument layout the
		// receiver matches on
		std::vector<std::string> boneNames;
		bool bBonesWellFormed= true;
		for (const DecodedMessage& message : messages)
		{
			if (message.address != "/VMC/Ext/Bone/Pos")
				continue;
			bBonesWellFormed&= message.tags == "sfffffff" && message.strings.size() == 1 &&
							   message.floats.size() == 7;
			if (!message.strings.empty())
				boneNames.push_back(message.strings[0]);
		}
		check(bBonesWellFormed, "every bone message is ,sfffffff");
		check(boneNames.size() == 39, "a fully tracked frame streams the head and all 38 body bones");

		std::sort(boneNames.begin(), boneNames.end());
		check(std::adjacent_find(boneNames.begin(), boneNames.end()) == boneNames.end(),
			  "no bone is streamed twice");

		bool bNamesKnown= true;
		for (const std::string& name : boneNames)
		{
			bool bFound= false;
			for (int boneIndex= 0; boneIndex < VMC_BONE_COUNT; ++boneIndex)
				bFound|= name == boneName((eVmcBone)boneIndex);
			bNamesKnown&= bFound;
		}
		check(bNamesKnown, "every streamed name is one this retarget knows");

		// The arguments carry the retarget's own values, in order
		VmcPose expected;
		buildPose(frame.poses, bBothValid, frame.head, lengths, expected);
		const VmcBone& expectedHand= expected.bones[(int)eVmcBone::LeftHand];

		bool bHandMatches= false;
		for (const DecodedMessage& message : messages)
		{
			if (message.address != "/VMC/Ext/Bone/Pos" || message.strings.empty() ||
				message.strings[0] != "LeftHand")
				continue;
			bHandMatches=
				nearlyEqual(glm::vec3(message.floats[0], message.floats[1], message.floats[2]),
							expectedHand.localPosition, 1e-6f) &&
				fabsf(message.floats[3] - expectedHand.localRotation.x) < 1e-6f &&
				fabsf(message.floats[4] - expectedHand.localRotation.y) < 1e-6f &&
				fabsf(message.floats[5] - expectedHand.localRotation.z) < 1e-6f &&
				fabsf(message.floats[6] - expectedHand.localRotation.w) < 1e-6f;
		}
		check(bHandMatches, "bone arguments are position xyz then rotation xyzw");

		// The modes are mutually exclusive, which is the point of the toggle
		config.outputMode= eOscOutputMode::Mikan;
		streamer.setConfig(config);
		std::vector<std::vector<uint8_t>> mikanPackets;
		streamer.encodeFrame(frame, mikanPackets);

		std::vector<DecodedMessage> mikanMessages;
		bool bMikanWellFormed= !mikanPackets.empty();
		size_t mikanBytes= 0;
		for (const std::vector<uint8_t>& packet : mikanPackets)
		{
			bMikanWellFormed&= decodeBundle(packet, mikanMessages);
			mikanBytes+= packet.size();
		}
		// The Mikan format is deliberately NOT chunked: its receiver treats one
		// bundle as one frame (it publishes a frame event per bundle and counts
		// loss off the sequence in /mikan/frame), so splitting a frame there is
		// a two-sided protocol change rather than a sender-side fix.
		check(mikanPackets.size() == 1, "Mikan mode is still one bundle per frame");

		// The frame just measured carried the 1 Hz skeleton and info messages.
		// The next one inside the same second does not, and THAT is the size
		// that has to stay inside a datagram - the 1 Hz frames are the only
		// ones exposed to IP fragmentation.
		std::vector<std::vector<uint8_t>> steadyPackets;
		streamer.encodeFrame(frame, steadyPackets);
		const size_t steadyBytes= steadyPackets.empty() ? 0 : steadyPackets[0].size();
		MIKAN_LOG_INFO("test-vmc") << "Mikan mode: " << mikanBytes << " bytes with the 1 Hz skeleton, "
								   << steadyBytes << " bytes steady state";
		check(steadyBytes > 0 && steadyBytes <= 1472,
			  "a steady-state Mikan frame fits one unfragmented datagram");
		check(bMikanWellFormed, "the Mikan stream still decodes");

		bool bAnyVmc= false;
		bool bAnyMikan= false;
		for (const DecodedMessage& message : mikanMessages)
		{
			bAnyVmc|= message.address.rfind("/VMC/", 0) == 0;
			bAnyMikan|= message.address.rfind("/mikan/", 0) == 0;
		}
		check(!bAnyVmc && bAnyMikan, "Mikan mode streams no VMC addresses");

		// The forearm message is the one address whose consumer rebuilds an arm
		// from it, so its exact layout is the contract: a receiver reads the
		// arguments positionally and a silently reordered or resized message
		// produces a plausible, wrong arm rather than a parse failure.
		{
			const HandPose& leftPose= frame.poses[0];
			bool bForearmOnWire= false;
			for (const DecodedMessage& message : mikanMessages)
			{
				if (message.address != "/mikan/hand/left/forearm")
					continue;

				bForearmOnWire=
					message.tags == "ifffffff" && message.ints.size() == 1 && message.ints[0] == 1 &&
					message.floats.size() == 7 &&
					// The WRIST JOINT, not the palm center - anchoring the frame
					// half a palm forward shifts the whole arm and still looks
					// like tracking
					nearlyEqual(glm::vec3(message.floats[0], message.floats[1], message.floats[2]),
								leftPose.getWristPositionWorld(), 1e-6f) &&
					fabsf(message.floats[3] - leftPose.forearmOrientationWorld.x) < 1e-6f &&
					fabsf(message.floats[4] - leftPose.forearmOrientationWorld.y) < 1e-6f &&
					fabsf(message.floats[5] - leftPose.forearmOrientationWorld.z) < 1e-6f &&
					fabsf(message.floats[6] - leftPose.forearmOrientationWorld.w) < 1e-6f;
			}
			check(bForearmOnWire, "/forearm is ,ifffffff: valid, wrist-joint xyz, forearm quat xyzw");
		}
	}

	// (i) Bone names, which are the whole contract with the receiver: a typo
	// here is silently ignored on the far end
	{
		bool bNamesValid= true;
		for (int boneIndex= 0; boneIndex < VMC_BONE_COUNT; ++boneIndex)
		{
			const char* name= boneName((eVmcBone)boneIndex);
			bNamesValid&= name != nullptr && name[0] != '\0';
			for (int other= 0; other < boneIndex; ++other)
				bNamesValid&= strcmp(name, boneName((eVmcBone)other)) != 0;
		}
		check(bNamesValid, "every bone has a unique non-empty name");
		check(strcmp(boneName(eVmcBone::LeftLittleDistal), "LeftLittleDistal") == 0 &&
				  strcmp(boneName(eVmcBone::RightThumbProximal), "RightThumbProximal") == 0,
			  "finger bones use Unity's HumanBodyBones spelling");
	}

	if (failures == 0)
		MIKAN_LOG_INFO("test-vmc") << "All VMC retarget tests passed";
	return failures == 0 ? 0 : 1;
}

MIKAN_REGISTER_TEST("--test-vmc", "VMC retarget: axis conversion, rest identity, chain round trip",
					eTestCategory::SelfTest, runVmcTest);
