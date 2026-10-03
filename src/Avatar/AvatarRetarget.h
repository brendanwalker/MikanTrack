#pragma once

#include <array>
#include <vector>

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

#include "AvatarTypes.h"
#include "BodyPoseSolver.h" // BodyDimensions
#include "TrackingTypes.h"

class AvatarSkeleton;

// Retargets the measured world-space pose onto a loaded avatar's humanoid
// skeleton. The avatar keeps its own proportions: every bone keeps its rest
// offset from its parent and only rotates, so the measured wrist is first
// scaled from the user's arm onto the avatar's, the elbow comes from a
// two-bone solve over the avatar's arm lengths hinted by the measured elbow,
// and the fingers are rebuilt through the shipping forward kinematics on the
// avatar's own hand skeleton. Pure math over the frame it is given plus a
// small filter state for the root, advanced on frame timestamps, so the same
// frames replay to the same poses.
struct AvatarRetargetConfig
{
	// Slide and yaw the avatar so its shoulders sit on the measured ones
	// whenever the body-pose stage tracks both. Held at the last followed
	// placement across a shoulder dropout.
	bool followShoulders= true;
	// Placement while the shoulders are not followed: where the avatar's
	// origin sits in the world frame, and its yaw about world +Z (0 = facing
	// world +X, the rest-pose facing)
	glm::vec3 fixedRootPositionWorld{0.f};
	float fixedRootYawDegrees= 0.f;
	// Time constant of the root follow filter, seconds. The measured
	// shoulders carry the body solver's jitter, which would otherwise shake
	// the whole character.
	float rootFollowTimeConstantS= 0.3f;

	// Per arm and hand, left then right (copied from the avatar's rig
	// settings, see AvatarRig.h)
	struct Side
	{
		// The point the elbow bends toward when nothing measured it, meters,
		// in the rooted torso frame relative to the avatar's upper-arm joint,
		// so it turns with the root yaw. Down and back, the way a resting arm
		// hangs.
		glm::vec3 elbowHintOffset{-0.15f, 0.f, -0.30f};
		// Forearm confidence at and above which the measured elbow alone
		// decides the bend; below it the pole blends toward the hint
		float elbowHintConfidence= 0.5f;
		// Trims on the rest frames the retarget assumes, radians: the palm
		// frame in its own axes (XYZ Euler), and a roll about the forearm
		glm::vec3 handTrimRadians{0.f};
		float forearmRollTrimRadians= 0.f;
		// Finger angle scales: curl on the three bends, splay on the lateral
		float curlGain= 1.f;
		float splayGain= 1.f;
		std::array<bool, FINGER_COUNT> fingerEnabled{true, true, true, true, true};
		// Explicit signed thumb hinge pronation; otherwise the hand
		// skeleton's own (HandPoseModel::getThumbPronationRad)
		bool bThumbPronationOverride= false;
		float thumbPronationRad= 0.f;
	};
	std::array<Side, 2> sides{};
	// Trim on the head's assumed rest frame (+X facing, +Y left, +Z up), in
	// its own axes, XYZ Euler radians
	glm::vec3 headTrimRadians{0.f};
};

// The posed avatar: a root placement plus one world rotation DELTA per
// humanoid bone, measured against the bone's rooted rest rotation. A bone's
// posed world rotation is `delta * rootRotation * restRotationWorld`, its
// posed position hangs off its humanoid parent's posed position by the
// parent's delta applied to the rest offset, and an absent bone (present ==
// false) sits at identity delta, following its parent. Defined this way so
// the renderer, the VMC stream, and a test can each compose it the same way.
struct AvatarPose
{
	bool valid= false;
	double timestampMs= 0.0;
	glm::vec3 rootPositionWorld{0.f};
	glm::quat rootRotationWorld{1.f, 0.f, 0.f, 0.f};

	struct Bone
	{
		bool present= false;
		glm::quat deltaWorld{1.f, 0.f, 0.f, 0.f};
	};
	std::array<Bone, HUMANOID_BONE_COUNT> bones{};

	void clear();
};

class AvatarRetarget
{
public:
	// Forgets the root follow state, so the next solve snaps to its
	// measurement instead of filtering from a stale placement
	void reset();

	// bSideValid selects which hands to pose at all (the caller has applied
	// its confidence gate and dropout hold); a side not valid leaves its arm
	// at rest. Always produces a valid pose with a root, tracked or not.
	void solve(const std::array<HandPose, 2>& poses, const bool bSideValid[2],
			   const TrackingFrameResult::HeadPose& head, double timestampMs, const BodyDimensions& user,
			   const AvatarSkeleton& skeleton, const AvatarRetargetConfig& config, AvatarPose& outPose);

	// Convenience over a fused frame: a side is valid when it is tracked
	// with a world pose
	void solve(const TrackingFrameResult& frame, const BodyDimensions& user, const AvatarSkeleton& skeleton,
			   const AvatarRetargetConfig& config, AvatarPose& outPose);

private:
	void resolveRoot(const std::array<HandPose, 2>& poses, const bool bSideValid[2], double timestampMs,
					 const AvatarSkeleton& skeleton, const AvatarRetargetConfig& config, AvatarPose& outPose);

	struct RootState
	{
		bool valid= false;
		glm::vec3 position{0.f};
		float yawRadians= 0.f;
		double timestampMs= 0.0;
	};
	RootState m_root;
};

// One humanoid bone's posed world transform
struct AvatarPosedBone
{
	glm::vec3 positionWorld{0.f};
	glm::quat rotationWorld{1.f, 0.f, 0.f, 0.f};
};

// Composes the pose over the rest skeleton: every bone's posed world position
// and rotation (absent bones follow their parents at rest)
void computePosedBones(const AvatarSkeleton& skeleton, const AvatarPose& pose,
					   std::array<AvatarPosedBone, HUMANOID_BONE_COUNT>& outBones);

// Node globals (avatar space) for rendering: present humanoid bones take
// their posed transform, every other node composes from its parent as
// authored, so helper nodes, hair and clothing bones ride along
void computePosedGlobals(const AvatarModel& model, const AvatarSkeleton& skeleton, const AvatarPose& pose,
						 std::vector<glm::mat4>& outGlobals);
