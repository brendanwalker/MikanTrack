#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"

#include <cstdint>
#include <vector>

struct AvatarModel;

// Draws a loaded VRM avatar (GL 3.3 core) into whatever framebuffer is bound.
// upload() turns the CPU model into per-primitive VAOs, sRGB textures and a
// sorted draw list. draw() packs every skin's joint matrices into one texture
// buffer, skins on the GPU (4-weight linear blend) and shades with the MToon
// ramp. All GL state draw() touches is saved and restored around it.
class GlSkinnedMeshRenderer
{
public:
	GlSkinnedMeshRenderer();
	~GlSkinnedMeshRenderer();

	GlSkinnedMeshRenderer(const GlSkinnedMeshRenderer&)= delete;
	GlSkinnedMeshRenderer& operator=(const GlSkinnedMeshRenderer&)= delete;

	// Compiles the program and creates the joint texture buffer. Requires a current GL context.
	bool startup();
	// Releases the model resources and the program
	void shutdown();

	// Builds GPU resources for this model, replacing any previous. False on GL failure.
	bool upload(const AvatarModel& model);
	// Releases the model's GPU resources, keeps the program
	void clear();
	bool hasModel() const;

	// Draws every mesh node. modelMatrix maps avatar (glTF) space to the space viewProj expects.
	// nodeGlobalsAvatar is one global transform per model node in avatar space (size == node count).
	// lightDirection is the direction TOWARD the light in the same space as modelMatrix's target, normalized.
	void draw(const glm::mat4& viewProj, const glm::mat4& modelMatrix, const std::vector<glm::mat4>& nodeGlobalsAvatar,
			  const glm::vec3& lightDirection);

private:
	struct GpuPrimitive
	{
		uint32_t vao= 0;
		uint32_t vbo= 0;
		uint32_t ebo= 0;
		int32_t indexCount= 0;
		// Highest JOINTS_0 index with nonzero weight, -1 when the primitive carries no skinning data
		int32_t maxJointIndex= -1;
		int material= -1;
	};

	struct GpuMaterial
	{
		glm::vec4 baseColorFactor{1.f};
		glm::vec3 shadeColorFactor{0.f};
		float shadingShift= 0.f;
		float shadingToony= 0.9f;
		float alphaCutoff= 0.5f;
		int alphaMode= 0; // eAlphaMode as int
		bool doubleSided= false;
		int renderQueueOffset= 0;
		bool transparentWithZWrite= false;
		uint32_t baseTexture= 0; // GL texture, 0 = the white fallback
		uint32_t baseSampler= 0;
		uint32_t shadeTexture= 0;
		uint32_t shadeSampler= 0;
	};

	// One primitive drawn by one mesh node
	struct DrawItem
	{
		int node= -1;
		int primitive= -1; // index into m_primitives
		int skin= -1; // -1 draws with the node global and identity skinning
		int material= -1; // index into m_materials
	};

	// Where each draw-list phase begins and ends in m_drawList
	struct DrawPhase
	{
		size_t begin= 0;
		size_t end= 0;
		bool bBlend= false;
		bool bDepthWrite= true;
	};

	bool compileProgram();
	void releaseModelResources();

	uint32_t m_programId= 0;
	int32_t m_viewProjLocation= -1;
	int32_t m_modelLocation= -1;
	int32_t m_nodeLocation= -1;
	int32_t m_skinnedLocation= -1;
	int32_t m_jointTexelBaseLocation= -1;
	int32_t m_jointsSamplerLocation= -1;
	int32_t m_lightDirLocation= -1;
	int32_t m_baseColorFactorLocation= -1;
	int32_t m_shadeColorFactorLocation= -1;
	int32_t m_shadingShiftLocation= -1;
	int32_t m_shadingToonyLocation= -1;
	int32_t m_alphaModeLocation= -1;
	int32_t m_alphaCutoffLocation= -1;
	int32_t m_doubleSidedLocation= -1;
	int32_t m_baseTexLocation= -1;
	int32_t m_shadeTexLocation= -1;

	// Joint matrix texture buffer, kept across uploads
	uint32_t m_jointBuffer= 0;
	uint32_t m_jointTexture= 0;
	// 1x1 white texture standing in for an absent texture, kept across uploads
	uint32_t m_whiteTexture= 0;
	uint32_t m_defaultSampler= 0;

	// -- Per model --
	bool m_bHasModel= false;
	size_t m_nodeCount= 0;
	std::vector<GpuPrimitive> m_primitives;
	// m_primitives index of the first primitive of each mesh
	std::vector<int> m_meshPrimitiveBase;
	std::vector<GpuMaterial> m_materials;
	std::vector<uint32_t> m_imageTextures; // GL texture per image, 0 when the image failed
	std::vector<uint32_t> m_textureSamplers; // GL sampler per glTF texture
	std::vector<DrawItem> m_drawList;
	DrawPhase m_phases[3];
	// Per skin: its joint node indices and inverse bind matrices, and its first joint in the joint buffer
	std::vector<std::vector<int>> m_skinJoints;
	std::vector<std::vector<glm::mat4>> m_skinInverseBinds;
	std::vector<int> m_skinJointBase;
	size_t m_totalJointCount= 0;
	bool m_bLoggedNodeCountMismatch= false;

	// Per draw scratch, reused
	std::vector<glm::mat4> m_jointMatrices;
};
