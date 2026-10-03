#pragma once

#include <array>
#include <string>
#include <vector>

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float3.hpp"

#include "AvatarTypes.h"
#include "TrackingTypes.h"

// Composes every node's global transform (avatar space) from the given
// parent-relative locals, walking the hierarchy from the roots so a child is
// always composed after its parent whatever the file's node order
void computeNodeGlobals(const AvatarModel& model, const std::vector<glm::mat4>& locals,
						std::vector<glm::mat4>& outGlobals);

// The avatar-to-world conversion for one VRM generation. glTF is right-handed
// Y-up; a VRM 1.0 model faces +Z and a 0.x model faces -Z. Both land facing
// world +X with arms along +/-Y and +Z up, a pure rotation (determinant +1).
glm::mat4 makeWorldFromAvatar(eVrmVersion version);

// Immutable rest-pose data derived from a loaded model, in the app's world
// frame: what the retarget measures against and the renderer poses from.
// Everything here comes from composed rest WORLD transforms, never from
// node-local rotations: a VRM 1.0 rig is not required to be normalized, and
// VRoid rigs put helper nodes between humanoid bones.
class AvatarSkeleton
{
public:
	explicit AvatarSkeleton(const AvatarModel& model);

	struct Bone
	{
		bool present= false;
		int node= -1;
		// Nearest PRESENT ancestor in the humanoid hierarchy (the spec parent
		// may be an optional bone the avatar lacks); HUMANOID_BONE_NONE at the hips
		eHumanoidBone parent= HUMANOID_BONE_NONE;
		glm::vec3 restPositionWorld{0.f};
		glm::quat restRotationWorld{1.f, 0.f, 0.f, 0.f};
		// restPositionWorld - parent's, zero at the hips. The offset a bone
		// keeps whatever rotation it is given.
		glm::vec3 restOffsetFromParentWorld{0.f};
	};

	// Rest hand geometry in the app's palm convention, built from the finger
	// rest joints through the shipping palm-frame code so the chirality the
	// forward kinematics reads matches a tracked hand exactly. Unlike a
	// tracked skeleton, neutralDirInPalm here carries the AVATAR's rest
	// finger directions (zero angles reproduce the avatar's own rest hand);
	// this skeleton is never put on the wire.
	//
	// Fingers are whatever the rig has. A finger's PHYSICAL bones are its
	// present humanoid slots in order (a rig that maps two bones into the
	// intermediate and distal slots has physical bones 0 and 1 there), and
	// the hand skeleton's three phalanges start at physical bone 0, with the
	// missing joints extrapolated. A finger with no bones gets a stand-in so
	// the palm frame still builds; a rig with no fingers at all gets the
	// T-pose palm frame off the hand joint.
	struct HandRest
	{
		bool valid= false;             // the hand bone exists (always, for a loaded avatar)
		glm::mat4 palmFrameWorld{1.f}; // columns: palm X, Y, Z axes, palm center
		HandSkeleton skeleton;
		// Physical bones per finger (0..3) and the humanoid phalanx slot
		// (0 proximal, 1 intermediate, 2 distal) each physical bone sits in
		std::array<int, FINGER_COUNT> fingerBoneCount{};
		std::array<std::array<int, 3>, FINGER_COUNT> fingerSlots{};
		// Rest direction of each PHYSICAL bone in world space, base to tip,
		// the direction a retarget swings onto a posed one. Entries past the
		// bone count run toward the extrapolated joints.
		std::array<std::array<glm::vec3, 3>, FINGER_COUNT> restPhalanxDirWorld{};

		bool hasFinger(int finger) const { return fingerBoneCount[finger] > 0; }
		bool hasAnyFinger() const
		{
			for (int count : fingerBoneCount)
				if (count > 0)
					return true;
			return false;
		}
	};

	eVrmVersion getVersion() const { return m_version; }
	const glm::mat4& getWorldFromAvatar() const { return m_worldFromAvatar; }

	// Rest globals per node, avatar space (the pose to render before any
	// retarget, and the reference a posed node is measured against)
	const std::vector<glm::mat4>& getRestGlobalsAvatar() const { return m_restGlobalsAvatar; }
	const std::vector<glm::mat4>& getRestLocalsAvatar() const { return m_restLocalsAvatar; }

	const Bone& getBone(eHumanoidBone bone) const { return m_bones[(int)bone]; }
	const HandRest& getHand(eHandSide side) const { return m_hands[(int)side]; }

	// Derived rest lengths, meters
	float getUpperArmLength(eHandSide side) const { return m_upperArmLength[(int)side]; }
	float getForearmLength(eHandSide side) const { return m_forearmLength[(int)side]; }
	float getShoulderWidth() const { return m_shoulderWidth; }
	// Midpoint between the upper-arm joints at rest, the point the measured
	// shoulder midpoint is matched to when the root follows the shoulders
	const glm::vec3& getRestShoulderMidpointWorld() const { return m_restShoulderMidpointWorld; }
	// Hand joint to middle finger base, 0 when the avatar has no middle finger
	float getHandLength(eHandSide side) const { return m_handLength[(int)side]; }
	// Highest rest joint (the head or an eye) above the hips, for framing
	float getHeightAboveHips() const { return m_heightAboveHips; }

	// Rig oddities found while building (a hand whose index sits on the
	// pinky side of its thumb, say). The avatar still loads and poses; these
	// tell the user what the mapping will do wrong.
	const std::vector<std::string>& getWarnings() const { return m_warnings; }

private:
	void buildHand(const AvatarModel& model, eHandSide side);

	eVrmVersion m_version= eVrmVersion::Vrm0;
	glm::mat4 m_worldFromAvatar{1.f};
	std::vector<glm::mat4> m_restLocalsAvatar;
	std::vector<glm::mat4> m_restGlobalsAvatar;
	std::array<Bone, HUMANOID_BONE_COUNT> m_bones{};
	std::array<HandRest, 2> m_hands{};
	float m_upperArmLength[2]= {0.f, 0.f};
	float m_forearmLength[2]= {0.f, 0.f};
	float m_handLength[2]= {0.f, 0.f};
	float m_shoulderWidth= 0.f;
	glm::vec3 m_restShoulderMidpointWorld{0.f};
	std::vector<std::string> m_warnings;
	float m_heightAboveHips= 0.f;
};
