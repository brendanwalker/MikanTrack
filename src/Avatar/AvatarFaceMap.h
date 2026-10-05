#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "FaceTypes.h"

struct AvatarModel;

// The phone face stream's ARKit blendshapes as the blendshape names a VMC
// receiver matches on (/VMC/Ext/Blend/Val), built once per avatar. Each
// output is a weighted sum of ARKit columns, clamped to [0, 1].
//
// Which names an avatar gets:
// - Expressions named after ARKit blendshapes (perfect sync): those
//   expressions alone, one column each, in the avatar's own spelling.
// - Otherwise the VRM presets the avatar carries, each derived from the ARKit
//   columns that shape it (blinks from the blinks, the vowels from the jaw and
//   mouth, the gaze directions from the eye looks). The emotion presets have
//   no ARKit counterpart and are never driven.
// - Plus, in that second case, the ARKit names the avatar carries as morph
//   targets. A standard receiver ignores them (it matches expressions only),
//   and a receiver that also matches morph targets gets the full face.
// - No avatar: the ARKit names verbatim.
class AvatarFaceMap
{
public:
	struct Term
	{
		int arkitIndex= -1;
		float weight= 1.f;
	};
	struct Output
	{
		std::string name;
		std::vector<Term> terms;
	};

	static std::shared_ptr<const AvatarFaceMap> buildRaw();
	static std::shared_ptr<const AvatarFaceMap> build(const AvatarModel& model);

	const std::vector<Output>& getOutputs() const { return m_outputs; }
	bool isPerfectSync() const { return m_bPerfectSync; }

	// One value per output, in getOutputs() order
	void evaluate(const std::array<float, ARKIT_BLENDSHAPE_COUNT>& arkit, std::vector<float>& outValues) const;

private:
	std::vector<Output> m_outputs;
	bool m_bPerfectSync= false;
};

// The face map's outputs as morph target weights on one avatar, the way a
// receiver holding the same file would apply them: an output naming an
// expression drives that expression's morph binds, an output naming a morph
// target drives it directly, and a morph driven by several outputs takes the
// largest weight rather than their sum, so a preset and the ARKit shape it
// came from never double up.
class AvatarFaceMorphs
{
public:
	AvatarFaceMorphs(const AvatarModel& model, std::shared_ptr<const AvatarFaceMap> faceMap);

	// Per mesh, one weight per morph target name. All zero without a face.
	void evaluate(const std::array<float, ARKIT_BLENDSHAPE_COUNT>* arkit,
				  std::vector<std::vector<float>>& outMeshWeights) const;

private:
	struct Bind
	{
		int output= -1;
		int mesh= -1;
		int morph= -1;
		float weight= 1.f;
	};

	std::shared_ptr<const AvatarFaceMap> m_faceMap;
	std::vector<Bind> m_binds;
	std::vector<size_t> m_meshMorphCounts;
	mutable std::vector<float> m_outputValues;
};
