#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "glm/ext/vector_float3.hpp"

#include "AvatarTypes.h"
#include "TrackingTypes.h"

// Per-avatar rig adjustments: the corrections a real rig needs on top of the
// file's humanoid map and the retarget's rest-frame assumptions (a T-pose,
// palms down, the head facing forward, a human thumb). Persisted in a sidecar
// JSON beside the model, `<stem>.mikanrig.json`, so they travel with the file
// and every project that loads it shares them.
struct AvatarRigSettings
{
	static constexpr int kVersion= 1;

	// Per humanoid bone: a node NAME replacing the file's mapping (empty =
	// unmapped), or nothing to keep the file's own. Names rather than indices
	// so the sidecar survives a re-export that reorders nodes.
	std::array<std::optional<std::string>, HUMANOID_BONE_COUNT> boneNodeOverrides{};

	// Per humanoid bone: Euler trims (XYZ, degrees) rotating the rest frame
	// the retarget assumes for that bone, in that frame's own axes. Only the
	// bones whose full frame is measured against an assumed rest use them:
	// the head and the hands (all three axes) and the forearms (x only, a
	// roll about the forearm).
	std::array<glm::vec3, HUMANOID_BONE_COUNT> rotationTrimDegrees{};

	struct Side
	{
		// Where the elbow bends toward when nothing measured it, meters, in
		// the rooted torso frame (+X facing, +Y the avatar's left, +Z up)
		// relative to the avatar's upper-arm joint
		glm::vec3 elbowHintOffset{-0.15f, 0.f, -0.30f};
		// The forearm confidence at and above which the measured elbow fully
		// decides the bend; below it the bend blends toward the hint
		float elbowHintConfidence= 0.5f;

		// The thumb's flexion hinge pronation. Auto reads its sign from which
		// side of the palm the index sits on, which a rig with swapped
		// fingers gets wrong; the explicit value is signed.
		bool bThumbPronationAuto= true;
		float thumbPronationDegrees= 0.f;

		// Scales on the measured finger angles: curl on the three bends,
		// splay on the lateral
		float curlGain= 1.f;
		float splayGain= 1.f;
		// A disabled finger is left at rest
		std::array<bool, FINGER_COUNT> fingerEnabled{true, true, true, true, true};
	};
	// Left, right. The default elbow hint lies in the sagittal plane, so it
	// is its own mirror image.
	std::array<Side, 2> sides{};

	// Whether the mapping (the only part that changes the skeleton) differs
	bool sameMapping(const AvatarRigSettings& other) const { return boneNodeOverrides == other.boneNodeOverrides; }
};

// The sidecar for a model file: `<stem>.mikanrig.json` beside it
std::filesystem::path getAvatarRigPath(const std::filesystem::path& vrmPath);

// A missing sidecar yields the defaults; an unreadable one warns (logged and
// appended) and yields the defaults
AvatarRigSettings loadAvatarRig(const std::filesystem::path& vrmPath, std::vector<std::string>* outWarnings= nullptr);
bool saveAvatarRig(const std::filesystem::path& vrmPath, const AvatarRigSettings& settings);

// JSON round trip, separate from the file so it can be tested in memory
std::string avatarRigToJson(const AvatarRigSettings& settings);
bool avatarRigFromJson(const std::string& text, AvatarRigSettings& outSettings, std::string& outError);

// Rebuilds model.humanoidNodes from the file's map plus the overrides,
// resolved by node name. An unknown name, or unmapping a bone every VRM must
// have, warns and keeps the file's mapping; two bones on one node warns.
void applyRigToModel(const AvatarRigSettings& settings, AvatarModel& model, std::vector<std::string>& outWarnings);
