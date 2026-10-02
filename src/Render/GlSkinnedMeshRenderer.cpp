#include "GlSkinnedMeshRenderer.h"
#include "AvatarTypes.h"
#include "GlUtils.h"
#include "Logger.h"

#include "GL/glew.h"

#include "glm/ext/matrix_float3x3.hpp"
#include "glm/geometric.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/matrix.hpp"

#include <algorithm>
#include <cstddef>

// Texture units the program samples from
static const int k_baseTextureUnit= 0;
static const int k_shadeTextureUnit= 1;
static const int k_jointTextureUnit= 2;

static const char* k_skinnedVertexShaderCode= R""""(
#version 330 core
uniform mat4 u_viewProj;
uniform mat4 u_model;
uniform mat4 u_node;
uniform int u_skinned;
uniform int u_jointTexelBase;
uniform samplerBuffer u_joints;
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in uvec4 in_joints;
layout(location = 4) in vec4 in_weights;
out vec3 v_normal;
out vec2 v_uv;

// A joint matrix is 4 RGBA32F texels, one per column
mat4 fetchJoint(uint joint)
{
	int texel = u_jointTexelBase + int(joint) * 4;
	return mat4(
		texelFetch(u_joints, texel),
		texelFetch(u_joints, texel + 1),
		texelFetch(u_joints, texel + 2),
		texelFetch(u_joints, texel + 3));
}

void main()
{
	mat4 skin = mat4(1.0);
	if (u_skinned != 0)
	{
		skin =
			in_weights.x * fetchJoint(in_joints.x) +
			in_weights.y * fetchJoint(in_joints.y) +
			in_weights.z * fetchJoint(in_joints.z) +
			in_weights.w * fetchJoint(in_joints.w);
	}

	mat4 world = u_model * u_node * skin;
	gl_Position = u_viewProj * world * vec4(in_position, 1.0);
	v_normal = mat3(world) * in_normal;
	v_uv = in_uv;
}
)"""";

static const char* k_skinnedFragmentShaderCode= R""""(
#version 330 core
uniform vec3 u_lightDir;
uniform vec4 u_baseColorFactor;
uniform vec3 u_shadeColorFactor;
uniform float u_shadingShift;
uniform float u_shadingToony;
uniform int u_alphaMode; // 0 opaque, 1 mask, 2 blend
uniform float u_alphaCutoff;
uniform int u_doubleSided;
uniform sampler2D u_baseTex;
uniform sampler2D u_shadeTex;
in vec3 v_normal;
in vec2 v_uv;
out vec4 out_FragColor;

void main()
{
	// sRGB textures sample as linear, so all of this runs in linear space
	vec4 base = u_baseColorFactor * texture(u_baseTex, v_uv);
	if (u_alphaMode == 1 && base.a < u_alphaCutoff)
	{
		discard;
	}

	vec3 shade = u_shadeColorFactor * texture(u_shadeTex, v_uv).rgb;

	vec3 n = normalize(v_normal);
	if (u_doubleSided != 0 && !gl_FrontFacing)
	{
		n = -n;
	}

	// MToon ramp: linearstep(-1 + toony, 1 - toony, NdotL + shift).
	// The max() keeps toony == 1 a hard step instead of a divide by zero.
	float NdotL = dot(n, u_lightDir);
	float rampLo = -1.0 + u_shadingToony;
	float rampHi = 1.0 - u_shadingToony;
	float shading = clamp((NdotL + u_shadingShift - rampLo) / max(rampHi - rampLo, 1e-5), 0.0, 1.0);

	// A small constant ambient so the shaded side is not flat shade color
	vec3 rgb = mix(shade, base.rgb, shading);
	rgb += base.rgb * 0.05;

	// The target is a plain RGBA8 buffer without GL_FRAMEBUFFER_SRGB, so encode here
	float alpha = (u_alphaMode == 2) ? base.a : 1.0;
	out_FragColor = vec4(pow(max(rgb, vec3(0.0)), vec3(1.0 / 2.2)), alpha);
}
)"""";

namespace
{
	// Interleaved vertex, 64 bytes
	struct SkinnedVertex
	{
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec2 uv;
		glm::uvec4 joints;
		glm::vec4 weights;
	};
	static_assert(sizeof(SkinnedVertex) == 64, "SkinnedVertex must stay tightly packed");

	// Area-weighted smooth normals: the unnormalized face cross product is twice
	// the triangle area, so summing it weights each face by its area for free
	void computeSmoothNormals(const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices,
							  std::vector<glm::vec3>& outNormals)
	{
		outNormals.assign(positions.size(), glm::vec3(0.f));

		const size_t triangleIndexCount= indices.size() - (indices.size() % 3);
		for (size_t i= 0; i < triangleIndexCount; i+= 3)
		{
			const uint32_t i0= indices[i];
			const uint32_t i1= indices[i + 1];
			const uint32_t i2= indices[i + 2];
			if (i0 >= positions.size() || i1 >= positions.size() || i2 >= positions.size())
			{
				continue;
			}

			const glm::vec3 faceNormal= glm::cross(positions[i1] - positions[i0], positions[i2] - positions[i0]);
			outNormals[i0]+= faceNormal;
			outNormals[i1]+= faceNormal;
			outNormals[i2]+= faceNormal;
		}

		for (glm::vec3& normal : outNormals)
		{
			const float length= glm::length(normal);
			// Unreferenced or fully degenerate vertices still need a unit normal
			normal= (length > 1e-12f) ? normal / length : glm::vec3(0.f, 1.f, 0.f);
		}
	}

	void setSamplerParameters(GLuint sampler, bool bRepeatS, bool bRepeatT, bool bNearestMag)
	{
		glSamplerParameteri(sampler, GL_TEXTURE_WRAP_S, bRepeatS ? GL_REPEAT : GL_CLAMP_TO_EDGE);
		glSamplerParameteri(sampler, GL_TEXTURE_WRAP_T, bRepeatT ? GL_REPEAT : GL_CLAMP_TO_EDGE);
		glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, bNearestMag ? GL_NEAREST : GL_LINEAR);
	}

	void setCapability(GLenum capability, GLboolean bEnabled)
	{
		if (bEnabled)
			glEnable(capability);
		else
			glDisable(capability);
	}
} // namespace

GlSkinnedMeshRenderer::GlSkinnedMeshRenderer() {}

GlSkinnedMeshRenderer::~GlSkinnedMeshRenderer() { shutdown(); }

bool GlSkinnedMeshRenderer::compileProgram()
{
	auto compileShader= [](GLenum shaderType, const char* source, const char* label) -> GLuint {
		GLuint shaderId= glCreateShader(shaderType);
		glShaderSource(shaderId, 1, &source, nullptr);
		glCompileShader(shaderId);

		GLint bSuccess= GL_FALSE;
		glGetShaderiv(shaderId, GL_COMPILE_STATUS, &bSuccess);
		if (bSuccess != GL_TRUE)
		{
			char infoLog[1024];
			glGetShaderInfoLog(shaderId, sizeof(infoLog), nullptr, infoLog);
			MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::compileProgram") << label << " compile failed: " << infoLog;
			glDeleteShader(shaderId);
			return 0;
		}

		return shaderId;
	};

	const GLuint vertexShaderId= compileShader(GL_VERTEX_SHADER, k_skinnedVertexShaderCode, "vertex shader");
	if (vertexShaderId == 0)
	{
		return false;
	}

	const GLuint fragmentShaderId= compileShader(GL_FRAGMENT_SHADER, k_skinnedFragmentShaderCode, "fragment shader");
	if (fragmentShaderId == 0)
	{
		glDeleteShader(vertexShaderId);
		return false;
	}

	m_programId= glCreateProgram();
	glAttachShader(m_programId, vertexShaderId);
	glAttachShader(m_programId, fragmentShaderId);
	glLinkProgram(m_programId);

	// The program keeps the shaders alive; flag them for deletion now
	glDeleteShader(vertexShaderId);
	glDeleteShader(fragmentShaderId);

	GLint bLinked= GL_FALSE;
	glGetProgramiv(m_programId, GL_LINK_STATUS, &bLinked);
	if (bLinked != GL_TRUE)
	{
		char infoLog[1024];
		glGetProgramInfoLog(m_programId, sizeof(infoLog), nullptr, infoLog);
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::compileProgram") << "program link failed: " << infoLog;
		glDeleteProgram(m_programId);
		m_programId= 0;
		return false;
	}

	m_viewProjLocation= glGetUniformLocation(m_programId, "u_viewProj");
	m_modelLocation= glGetUniformLocation(m_programId, "u_model");
	m_nodeLocation= glGetUniformLocation(m_programId, "u_node");
	m_skinnedLocation= glGetUniformLocation(m_programId, "u_skinned");
	m_jointTexelBaseLocation= glGetUniformLocation(m_programId, "u_jointTexelBase");
	m_jointsSamplerLocation= glGetUniformLocation(m_programId, "u_joints");
	m_lightDirLocation= glGetUniformLocation(m_programId, "u_lightDir");
	m_baseColorFactorLocation= glGetUniformLocation(m_programId, "u_baseColorFactor");
	m_shadeColorFactorLocation= glGetUniformLocation(m_programId, "u_shadeColorFactor");
	m_shadingShiftLocation= glGetUniformLocation(m_programId, "u_shadingShift");
	m_shadingToonyLocation= glGetUniformLocation(m_programId, "u_shadingToony");
	m_alphaModeLocation= glGetUniformLocation(m_programId, "u_alphaMode");
	m_alphaCutoffLocation= glGetUniformLocation(m_programId, "u_alphaCutoff");
	m_doubleSidedLocation= glGetUniformLocation(m_programId, "u_doubleSided");
	m_baseTexLocation= glGetUniformLocation(m_programId, "u_baseTex");
	m_shadeTexLocation= glGetUniformLocation(m_programId, "u_shadeTex");

	if (m_viewProjLocation == -1 || m_modelLocation == -1 || m_jointsSamplerLocation == -1)
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::compileProgram") << "Failed to find the transform uniforms";
		glDeleteProgram(m_programId);
		m_programId= 0;
		return false;
	}

	return true;
}

bool GlSkinnedMeshRenderer::startup()
{
	if (!compileProgram())
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::startup") << "Failed to build shader program";
		return false;
	}

	// Creating the textures binds them on the active unit; put the caller's bindings back after
	GLint prevTexture2d= 0;
	GLint prevTextureBuffer= 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture2d);
	glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prevTextureBuffer);

	// Joint matrix buffer. upload() sizes it to the model's total joint count;
	// one identity matrix keeps the data store non-empty until then.
	const glm::mat4 identity(1.f);
	glGenBuffers(1, &m_jointBuffer);
	glBindBuffer(GL_TEXTURE_BUFFER, m_jointBuffer);
	glBufferData(GL_TEXTURE_BUFFER, sizeof(glm::mat4), glm::value_ptr(identity), GL_DYNAMIC_DRAW);
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	// The texture views the buffer object, so later glBufferData resizes need no re-attach
	glGenTextures(1, &m_jointTexture);
	glBindTexture(GL_TEXTURE_BUFFER, m_jointTexture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, m_jointBuffer);
	glBindTexture(GL_TEXTURE_BUFFER, (GLuint)prevTextureBuffer);

	// 1x1 white stands in for an absent texture, so the shader never branches on presence.
	// MAX_LEVEL 0 keeps it complete under the samplers' mipmapped min filter.
	const uint8_t whitePixel[4]= {255, 255, 255, 255};
	glGenTextures(1, &m_whiteTexture);
	glBindTexture(GL_TEXTURE_2D, m_whiteTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, whitePixel);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture2d);

	glGenSamplers(1, &m_defaultSampler);
	setSamplerParameters(m_defaultSampler, true, true, false);

	if (checkGlError("GlSkinnedMeshRenderer::startup"))
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::startup") << "Failed to create the joint buffer";
		shutdown();
		return false;
	}

	return true;
}

void GlSkinnedMeshRenderer::shutdown()
{
	releaseModelResources();

	if (m_jointTexture != 0)
	{
		glDeleteTextures(1, &m_jointTexture);
		m_jointTexture= 0;
	}

	if (m_jointBuffer != 0)
	{
		glDeleteBuffers(1, &m_jointBuffer);
		m_jointBuffer= 0;
	}

	if (m_whiteTexture != 0)
	{
		glDeleteTextures(1, &m_whiteTexture);
		m_whiteTexture= 0;
	}

	if (m_defaultSampler != 0)
	{
		glDeleteSamplers(1, &m_defaultSampler);
		m_defaultSampler= 0;
	}

	if (m_programId != 0)
	{
		glDeleteProgram(m_programId);
		m_programId= 0;
	}
}

void GlSkinnedMeshRenderer::clear() { releaseModelResources(); }

bool GlSkinnedMeshRenderer::hasModel() const { return m_bHasModel; }

void GlSkinnedMeshRenderer::releaseModelResources()
{
	for (GpuPrimitive& primitive : m_primitives)
	{
		if (primitive.vao != 0)
			glDeleteVertexArrays(1, &primitive.vao);
		if (primitive.vbo != 0)
			glDeleteBuffers(1, &primitive.vbo);
		if (primitive.ebo != 0)
			glDeleteBuffers(1, &primitive.ebo);
	}
	m_primitives.clear();
	m_meshPrimitiveBase.clear();

	for (uint32_t texture : m_imageTextures)
	{
		if (texture != 0)
			glDeleteTextures(1, &texture);
	}
	m_imageTextures.clear();

	for (uint32_t sampler : m_textureSamplers)
	{
		if (sampler != 0)
			glDeleteSamplers(1, &sampler);
	}
	m_textureSamplers.clear();

	m_materials.clear();
	m_drawList.clear();
	for (DrawPhase& phase : m_phases)
	{
		phase= DrawPhase();
	}

	m_skinJoints.clear();
	m_skinInverseBinds.clear();
	m_skinJointBase.clear();
	m_totalJointCount= 0;
	m_jointMatrices.clear();

	m_nodeCount= 0;
	m_bLoggedNodeCountMismatch= false;
	m_bHasModel= false;
}

bool GlSkinnedMeshRenderer::upload(const AvatarModel& model)
{
	releaseModelResources();

	if (m_programId == 0)
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::upload") << "startup() has not succeeded";
		return false;
	}

	// Upload binds VAOs, buffers and textures; put the caller's bindings back after
	GLint prevVertexArray= 0;
	GLint prevArrayBuffer= 0;
	GLint prevTexture2d= 0;
	GLint prevUnpackAlignment= 4;
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVertexArray);
	glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &prevArrayBuffer);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture2d);
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpackAlignment);

	// RGBA8 rows are always a multiple of 4 bytes
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	// -- Images: one sRGB texture each, rows uploaded top-down as stored --
	m_imageTextures.assign(model.images.size(), 0);
	for (size_t imageIndex= 0; imageIndex < model.images.size(); ++imageIndex)
	{
		const AvatarImage& image= model.images[imageIndex];
		const size_t expectedBytes= (size_t)image.width * (size_t)image.height * 4;
		if (image.width <= 0 || image.height <= 0 || image.rgba.size() != expectedBytes)
		{
			MIKAN_LOG_WARNING("GlSkinnedMeshRenderer::upload")
				<< "Skipping image " << imageIndex << " (" << image.name << "): " << image.width << "x" << image.height
				<< " with " << image.rgba.size() << " bytes";
			continue;
		}

		GLuint texture= 0;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE,
					 image.rgba.data());
		glGenerateMipmap(GL_TEXTURE_2D);
		// The samplers carry the real filtering; these keep the texture sane if sampled without one
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_imageTextures[imageIndex]= texture;
	}

	// -- Textures: a sampler each, since two glTF textures may share an image with different wrap modes --
	m_textureSamplers.assign(model.textures.size(), 0);
	for (size_t textureIndex= 0; textureIndex < model.textures.size(); ++textureIndex)
	{
		const AvatarTexture& texture= model.textures[textureIndex];
		GLuint sampler= 0;
		glGenSamplers(1, &sampler);
		setSamplerParameters(sampler, texture.repeatS, texture.repeatT, texture.nearestFilter);
		m_textureSamplers[textureIndex]= sampler;
	}

	auto resolveTexture= [&](int textureIndex, uint32_t& outTexture, uint32_t& outSampler) {
		outTexture= m_whiteTexture;
		outSampler= m_defaultSampler;
		if (textureIndex < 0 || textureIndex >= (int)model.textures.size())
		{
			return false;
		}

		const int imageIndex= model.textures[textureIndex].image;
		if (imageIndex < 0 || imageIndex >= (int)m_imageTextures.size() || m_imageTextures[imageIndex] == 0)
		{
			return false;
		}

		outTexture= m_imageTextures[imageIndex];
		outSampler= m_textureSamplers[textureIndex];
		return true;
	};

	// -- Materials, plus one trailing default for primitives without a valid material --
	m_materials.reserve(model.materials.size() + 1);
	for (const AvatarMaterial& source : model.materials)
	{
		GpuMaterial material;
		material.baseColorFactor= source.baseColorFactor;
		material.alphaMode= (int)source.alphaMode;
		material.alphaCutoff= source.alphaCutoff;
		material.doubleSided= source.doubleSided;
		material.renderQueueOffset= source.renderQueueOffset;
		material.transparentWithZWrite= source.transparentWithZWrite;
		resolveTexture(source.baseColorTexture, material.baseTexture, material.baseSampler);

		if (source.isMToon)
		{
			material.shadeColorFactor= source.shadeColorFactor;
			material.shadingShift= source.shadingShiftFactor;
			material.shadingToony= source.shadingToonyFactor;
			resolveTexture(source.shadeMultiplyTexture, material.shadeTexture, material.shadeSampler);
		}
		else
		{
			// A plain glTF material gets a half-brightness shade of its own base color.
			// The shade also samples the base texture, otherwise the shaded side of a
			// textured material would lose its texture and read as a flat color.
			material.shadeColorFactor= glm::vec3(source.baseColorFactor) * 0.5f;
			material.shadingShift= 0.f;
			material.shadingToony= 0.9f;
			material.shadeTexture= material.baseTexture;
			material.shadeSampler= material.baseSampler;
		}

		m_materials.push_back(material);
	}
	const int defaultMaterialIndex= (int)m_materials.size();
	{
		GpuMaterial material;
		material.shadeColorFactor= glm::vec3(0.5f);
		material.baseTexture= m_whiteTexture;
		material.baseSampler= m_defaultSampler;
		material.shadeTexture= m_whiteTexture;
		material.shadeSampler= m_defaultSampler;
		m_materials.push_back(material);
	}

	// -- Skins: joint lists and inverse binds, laid end to end in the joint buffer --
	m_skinJoints.resize(model.skins.size());
	m_skinInverseBinds.resize(model.skins.size());
	m_skinJointBase.resize(model.skins.size());
	m_totalJointCount= 0;
	for (size_t skinIndex= 0; skinIndex < model.skins.size(); ++skinIndex)
	{
		const AvatarSkin& skin= model.skins[skinIndex];
		const size_t jointCount= skin.joints.size();

		// A joint node out of range is kept as -1 and contributes identity, so a
		// broken joint list degrades one joint instead of the whole skin
		m_skinJoints[skinIndex].resize(jointCount);
		for (size_t j= 0; j < jointCount; ++j)
		{
			const int jointNode= skin.joints[j];
			m_skinJoints[skinIndex][j]= (jointNode >= 0 && jointNode < (int)model.nodes.size()) ? jointNode : -1;
		}

		m_skinInverseBinds[skinIndex]= skin.inverseBindMatrices;
		m_skinInverseBinds[skinIndex].resize(jointCount, glm::mat4(1.f));

		m_skinJointBase[skinIndex]= (int)m_totalJointCount;
		m_totalJointCount+= jointCount;
	}

	GLint maxTextureBufferTexels= 0;
	glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE, &maxTextureBufferTexels);
	if (m_totalJointCount * 4 > (size_t)maxTextureBufferTexels)
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::upload")
			<< "Model needs " << m_totalJointCount * 4 << " joint texels, GL_MAX_TEXTURE_BUFFER_SIZE is "
			<< maxTextureBufferTexels;
		glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpackAlignment);
		glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture2d);
		releaseModelResources();
		return false;
	}

	m_jointMatrices.assign(m_totalJointCount, glm::mat4(1.f));
	if (m_totalJointCount > 0)
	{
		glBindBuffer(GL_TEXTURE_BUFFER, m_jointBuffer);
		glBufferData(GL_TEXTURE_BUFFER, m_totalJointCount * sizeof(glm::mat4), m_jointMatrices.data(),
					 GL_DYNAMIC_DRAW);
		glBindBuffer(GL_TEXTURE_BUFFER, 0);
	}

	// -- Meshes: one VAO + interleaved VBO + EBO per primitive --
	std::vector<SkinnedVertex> vertices;
	std::vector<glm::vec3> generatedNormals;
	m_meshPrimitiveBase.resize(model.meshes.size());
	for (size_t meshIndex= 0; meshIndex < model.meshes.size(); ++meshIndex)
	{
		const AvatarMesh& mesh= model.meshes[meshIndex];
		m_meshPrimitiveBase[meshIndex]= (int)m_primitives.size();

		for (size_t primitiveIndex= 0; primitiveIndex < mesh.primitives.size(); ++primitiveIndex)
		{
			const AvatarPrimitive& source= mesh.primitives[primitiveIndex];
			GpuPrimitive primitive;
			primitive.material= (source.material >= 0 && source.material < (int)model.materials.size())
									? source.material
									: defaultMaterialIndex;

			const size_t vertexCount= source.positions.size();
			const size_t indexCount= source.indices.size() - (source.indices.size() % 3);
			bool bIndicesValid= vertexCount > 0 && indexCount > 0;
			for (size_t i= 0; bIndicesValid && i < indexCount; ++i)
			{
				bIndicesValid= source.indices[i] < vertexCount;
			}
			if (!bIndicesValid)
			{
				// Left in the list with zero indices so mesh-relative primitive indexing still holds
				MIKAN_LOG_WARNING("GlSkinnedMeshRenderer::upload")
					<< "Skipping primitive " << primitiveIndex << " of mesh " << meshIndex << " (" << mesh.name
					<< "): empty or out-of-range indices";
				m_primitives.push_back(primitive);
				continue;
			}

			const bool bHasNormals= source.normals.size() == vertexCount;
			const bool bHasUvs= source.uvs.size() == vertexCount;
			const bool bHasSkinning= source.joints.size() == vertexCount && source.weights.size() == vertexCount;
			if (!bHasNormals)
			{
				computeSmoothNormals(source.positions, source.indices, generatedNormals);
			}
			const std::vector<glm::vec3>& normals= bHasNormals ? source.normals : generatedNormals;

			vertices.resize(vertexCount);
			for (size_t v= 0; v < vertexCount; ++v)
			{
				SkinnedVertex& vertex= vertices[v];
				vertex.position= source.positions[v];
				vertex.normal= normals[v];
				vertex.uv= bHasUvs ? source.uvs[v] : glm::vec2(0.f);
				vertex.joints= glm::uvec4(0);
				vertex.weights= glm::vec4(0.f);

				if (bHasSkinning)
				{
					// An unweighted influence may carry any joint index, and the shader
					// fetches all four, so point it at joint 0 to keep every fetch in range
					for (int k= 0; k < 4; ++k)
					{
						if (source.weights[v][k] > 0.f)
						{
							vertex.joints[k]= source.joints[v][k];
							vertex.weights[k]= source.weights[v][k];
							primitive.maxJointIndex= std::max(primitive.maxJointIndex, (int32_t)vertex.joints[k]);
						}
					}
				}
			}

			glGenVertexArrays(1, &primitive.vao);
			glGenBuffers(1, &primitive.vbo);
			glGenBuffers(1, &primitive.ebo);

			glBindVertexArray(primitive.vao);
			glBindBuffer(GL_ARRAY_BUFFER, primitive.vbo);
			glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(SkinnedVertex), vertices.data(), GL_STATIC_DRAW);
			// The element binding is VAO state, so it is bound while the VAO is
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, primitive.ebo);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(uint32_t), source.indices.data(),
						 GL_STATIC_DRAW);

			// layout(location = 0) vec3 in_position
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SkinnedVertex),
								  (const void*)offsetof(SkinnedVertex, position));

			// layout(location = 1) vec3 in_normal
			glEnableVertexAttribArray(1);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SkinnedVertex),
								  (const void*)offsetof(SkinnedVertex, normal));

			// layout(location = 2) vec2 in_uv
			glEnableVertexAttribArray(2);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(SkinnedVertex), (const void*)offsetof(SkinnedVertex, uv));

			// layout(location = 3) uvec4 in_joints: the I variant keeps them integers
			glEnableVertexAttribArray(3);
			glVertexAttribIPointer(3, 4, GL_UNSIGNED_INT, sizeof(SkinnedVertex),
								   (const void*)offsetof(SkinnedVertex, joints));

			// layout(location = 4) vec4 in_weights
			glEnableVertexAttribArray(4);
			glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(SkinnedVertex),
								  (const void*)offsetof(SkinnedVertex, weights));

			glBindVertexArray(0);

			primitive.indexCount= (int32_t)indexCount;
			m_primitives.push_back(primitive);
		}
	}

	glBindBuffer(GL_ARRAY_BUFFER, (GLuint)prevArrayBuffer);
	glBindVertexArray((GLuint)prevVertexArray);
	glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture2d);
	glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpackAlignment);

	// -- Draw list: one item per (mesh node, primitive), bucketed by alpha phase --
	std::vector<DrawItem> opaqueItems;
	std::vector<DrawItem> blendZWriteItems;
	std::vector<DrawItem> blendItems;
	for (size_t nodeIndex= 0; nodeIndex < model.nodes.size(); ++nodeIndex)
	{
		const AvatarNode& node= model.nodes[nodeIndex];
		if (node.mesh < 0 || node.mesh >= (int)model.meshes.size())
		{
			continue;
		}

		const bool bNodeHasSkin= node.skin >= 0 && node.skin < (int)model.skins.size();
		const int primitiveBase= m_meshPrimitiveBase[node.mesh];
		const int primitiveCount= (int)model.meshes[node.mesh].primitives.size();
		for (int p= 0; p < primitiveCount; ++p)
		{
			const GpuPrimitive& primitive= m_primitives[primitiveBase + p];
			if (primitive.indexCount == 0)
			{
				continue;
			}

			DrawItem item;
			item.node= (int)nodeIndex;
			item.primitive= primitiveBase + p;
			item.material= primitive.material;

			// Skinned only when the node has a skin and the primitive's joints fit it;
			// otherwise it falls back to the node global with identity skinning
			if (bNodeHasSkin && primitive.maxJointIndex >= 0)
			{
				if (primitive.maxJointIndex < (int)m_skinJoints[node.skin].size())
				{
					item.skin= node.skin;
				}
				else
				{
					MIKAN_LOG_WARNING("GlSkinnedMeshRenderer::upload")
						<< "Node " << nodeIndex << " (" << node.name << ") primitive " << p << " references joint "
						<< primitive.maxJointIndex << " but skin " << node.skin << " has "
						<< m_skinJoints[node.skin].size() << " joints; drawing it unskinned";
				}
			}

			const GpuMaterial& material= m_materials[item.material];
			if (material.alphaMode != (int)eAlphaMode::Blend)
				opaqueItems.push_back(item);
			else if (material.transparentWithZWrite)
				blendZWriteItems.push_back(item);
			else
				blendItems.push_back(item);
		}
	}

	// Stable, so equal offsets keep node order
	auto byRenderQueue= [this](const DrawItem& a, const DrawItem& b) {
		return m_materials[a.material].renderQueueOffset < m_materials[b.material].renderQueueOffset;
	};
	std::stable_sort(blendZWriteItems.begin(), blendZWriteItems.end(), byRenderQueue);
	std::stable_sort(blendItems.begin(), blendItems.end(), byRenderQueue);

	m_drawList.reserve(opaqueItems.size() + blendZWriteItems.size() + blendItems.size());
	auto appendPhase= [this](DrawPhase& phase, const std::vector<DrawItem>& items, bool bBlend, bool bDepthWrite) {
		phase.begin= m_drawList.size();
		m_drawList.insert(m_drawList.end(), items.begin(), items.end());
		phase.end= m_drawList.size();
		phase.bBlend= bBlend;
		phase.bDepthWrite= bDepthWrite;
	};
	appendPhase(m_phases[0], opaqueItems, false, true);
	appendPhase(m_phases[1], blendZWriteItems, true, true);
	appendPhase(m_phases[2], blendItems, true, false);

	if (checkGlError("GlSkinnedMeshRenderer::upload"))
	{
		MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::upload") << "GL error while uploading the model";
		releaseModelResources();
		return false;
	}

	m_nodeCount= model.nodes.size();
	m_bHasModel= true;
	return true;
}

void GlSkinnedMeshRenderer::draw(const glm::mat4& viewProj, const glm::mat4& modelMatrix,
								 const std::vector<glm::mat4>& nodeGlobalsAvatar, const glm::vec3& lightDirection)
{
	if (m_programId == 0 || !m_bHasModel || m_drawList.empty())
	{
		return;
	}

	if (nodeGlobalsAvatar.size() != m_nodeCount)
	{
		// Logged once per model: the caller would otherwise spam it every frame
		if (!m_bLoggedNodeCountMismatch)
		{
			MIKAN_LOG_ERROR("GlSkinnedMeshRenderer::draw")
				<< "Got " << nodeGlobalsAvatar.size() << " node globals for a model with " << m_nodeCount << " nodes";
			m_bLoggedNodeCountMismatch= true;
		}
		return;
	}

	// Joint matrices per the glTF skinning rule: joint global times inverse bind.
	// The mesh node's own transform is deliberately left out of skinned draws.
	for (size_t skinIndex= 0; skinIndex < m_skinJoints.size(); ++skinIndex)
	{
		const std::vector<int>& joints= m_skinJoints[skinIndex];
		const std::vector<glm::mat4>& inverseBinds= m_skinInverseBinds[skinIndex];
		glm::mat4* out= m_jointMatrices.data() + m_skinJointBase[skinIndex];
		for (size_t j= 0; j < joints.size(); ++j)
		{
			out[j]= (joints[j] >= 0) ? nodeGlobalsAvatar[joints[j]] * inverseBinds[j] : glm::mat4(1.f);
		}
	}

	// -- Save the GL state we modify --
	GLint prevProgram= 0;
	GLint prevVertexArray= 0;
	GLint prevActiveTexture= GL_TEXTURE0;
	GLint prevTexture2d[2]= {0, 0};
	GLint prevSampler[2]= {0, 0};
	GLint prevTextureBuffer= 0;
	GLint prevCullFaceMode= GL_BACK;
	GLint prevFrontFace= GL_CCW;
	GLint prevBlendSrcRgb= GL_ONE;
	GLint prevBlendDstRgb= GL_ZERO;
	GLint prevBlendSrcAlpha= GL_ONE;
	GLint prevBlendDstAlpha= GL_ZERO;
	GLboolean prevDepthMask= GL_TRUE;
	glGetIntegerv(GL_CURRENT_PROGRAM, &prevProgram);
	glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prevVertexArray);
	glGetIntegerv(GL_ACTIVE_TEXTURE, &prevActiveTexture);
	glActiveTexture(GL_TEXTURE0 + k_baseTextureUnit);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture2d[0]);
	glGetIntegerv(GL_SAMPLER_BINDING, &prevSampler[0]);
	glActiveTexture(GL_TEXTURE0 + k_shadeTextureUnit);
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTexture2d[1]);
	glGetIntegerv(GL_SAMPLER_BINDING, &prevSampler[1]);
	glActiveTexture(GL_TEXTURE0 + k_jointTextureUnit);
	glGetIntegerv(GL_TEXTURE_BINDING_BUFFER, &prevTextureBuffer);
	const GLboolean bWasCullFaceEnabled= glIsEnabled(GL_CULL_FACE);
	glGetIntegerv(GL_CULL_FACE_MODE, &prevCullFaceMode);
	glGetIntegerv(GL_FRONT_FACE, &prevFrontFace);
	const GLboolean bWasBlendEnabled= glIsEnabled(GL_BLEND);
	glGetIntegerv(GL_BLEND_SRC_RGB, &prevBlendSrcRgb);
	glGetIntegerv(GL_BLEND_DST_RGB, &prevBlendDstRgb);
	glGetIntegerv(GL_BLEND_SRC_ALPHA, &prevBlendSrcAlpha);
	glGetIntegerv(GL_BLEND_DST_ALPHA, &prevBlendDstAlpha);
	glGetBooleanv(GL_DEPTH_WRITEMASK, &prevDepthMask);
	const GLboolean bWasDepthTestEnabled= glIsEnabled(GL_DEPTH_TEST);

	// -- Per-draw constants --
	if (m_totalJointCount > 0)
	{
		glBindBuffer(GL_TEXTURE_BUFFER, m_jointBuffer);
		glBufferSubData(GL_TEXTURE_BUFFER, 0, m_totalJointCount * sizeof(glm::mat4), m_jointMatrices.data());
		glBindBuffer(GL_TEXTURE_BUFFER, 0);
	}

	// The joint unit is still active from the state save
	glBindTexture(GL_TEXTURE_BUFFER, m_jointTexture);

	glUseProgram(m_programId);
	glUniformMatrix4fv(m_viewProjLocation, 1, GL_FALSE, glm::value_ptr(viewProj));
	glUniformMatrix4fv(m_modelLocation, 1, GL_FALSE, glm::value_ptr(modelMatrix));
	glUniform3fv(m_lightDirLocation, 1, glm::value_ptr(lightDirection));
	glUniform1i(m_baseTexLocation, k_baseTextureUnit);
	glUniform1i(m_shadeTexLocation, k_shadeTextureUnit);
	glUniform1i(m_jointsSamplerLocation, k_jointTextureUnit);

	glEnable(GL_DEPTH_TEST);
	glCullFace(GL_BACK);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// The skinned draws share one node matrix; the mirror test for them only sees modelMatrix
	const glm::mat4 identity(1.f);
	const bool bModelMirrors= glm::determinant(glm::mat3(modelMatrix)) < 0.f;

	int boundMaterial= -1;
	for (const DrawPhase& phase : m_phases)
	{
		if (phase.begin == phase.end)
		{
			continue;
		}

		glDepthMask(phase.bDepthWrite ? GL_TRUE : GL_FALSE);
		setCapability(GL_BLEND, phase.bBlend ? GL_TRUE : GL_FALSE);

		for (size_t itemIndex= phase.begin; itemIndex < phase.end; ++itemIndex)
		{
			const DrawItem& item= m_drawList[itemIndex];
			const GpuPrimitive& primitive= m_primitives[item.primitive];

			if (item.material != boundMaterial)
			{
				const GpuMaterial& material= m_materials[item.material];
				glUniform4fv(m_baseColorFactorLocation, 1, glm::value_ptr(material.baseColorFactor));
				glUniform3fv(m_shadeColorFactorLocation, 1, glm::value_ptr(material.shadeColorFactor));
				glUniform1f(m_shadingShiftLocation, material.shadingShift);
				glUniform1f(m_shadingToonyLocation, material.shadingToony);
				glUniform1i(m_alphaModeLocation, material.alphaMode);
				glUniform1f(m_alphaCutoffLocation, material.alphaCutoff);
				glUniform1i(m_doubleSidedLocation, material.doubleSided ? 1 : 0);

				glActiveTexture(GL_TEXTURE0 + k_baseTextureUnit);
				glBindTexture(GL_TEXTURE_2D, material.baseTexture);
				glBindSampler(k_baseTextureUnit, material.baseSampler);
				glActiveTexture(GL_TEXTURE0 + k_shadeTextureUnit);
				glBindTexture(GL_TEXTURE_2D, material.shadeTexture);
				glBindSampler(k_shadeTextureUnit, material.shadeSampler);

				setCapability(GL_CULL_FACE, material.doubleSided ? GL_FALSE : GL_TRUE);
				boundMaterial= item.material;
			}

			// A mirroring transform reverses winding, so CCW front faces become CW on screen
			bool bMirrored= bModelMirrors;
			if (item.skin >= 0)
			{
				glUniformMatrix4fv(m_nodeLocation, 1, GL_FALSE, glm::value_ptr(identity));
				glUniform1i(m_skinnedLocation, 1);
				glUniform1i(m_jointTexelBaseLocation, m_skinJointBase[item.skin] * 4);
			}
			else
			{
				const glm::mat4& nodeGlobal= nodeGlobalsAvatar[item.node];
				glUniformMatrix4fv(m_nodeLocation, 1, GL_FALSE, glm::value_ptr(nodeGlobal));
				glUniform1i(m_skinnedLocation, 0);
				glUniform1i(m_jointTexelBaseLocation, 0);
				bMirrored= bMirrored != (glm::determinant(glm::mat3(nodeGlobal)) < 0.f);
			}
			glFrontFace(bMirrored ? GL_CW : GL_CCW);

			glBindVertexArray(primitive.vao);
			glDrawElements(GL_TRIANGLES, primitive.indexCount, GL_UNSIGNED_INT, nullptr);
		}
	}

	// -- Restore GL state --
	glBindVertexArray((GLuint)prevVertexArray);
	glUseProgram((GLuint)prevProgram);

	glActiveTexture(GL_TEXTURE0 + k_jointTextureUnit);
	glBindTexture(GL_TEXTURE_BUFFER, (GLuint)prevTextureBuffer);
	glActiveTexture(GL_TEXTURE0 + k_shadeTextureUnit);
	glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture2d[1]);
	glBindSampler(k_shadeTextureUnit, (GLuint)prevSampler[1]);
	glActiveTexture(GL_TEXTURE0 + k_baseTextureUnit);
	glBindTexture(GL_TEXTURE_2D, (GLuint)prevTexture2d[0]);
	glBindSampler(k_baseTextureUnit, (GLuint)prevSampler[0]);
	glActiveTexture((GLenum)prevActiveTexture);

	setCapability(GL_CULL_FACE, bWasCullFaceEnabled);
	glCullFace((GLenum)prevCullFaceMode);
	glFrontFace((GLenum)prevFrontFace);
	setCapability(GL_BLEND, bWasBlendEnabled);
	glBlendFuncSeparate((GLenum)prevBlendSrcRgb, (GLenum)prevBlendDstRgb, (GLenum)prevBlendSrcAlpha,
						(GLenum)prevBlendDstAlpha);
	glDepthMask(prevDepthMask);
	setCapability(GL_DEPTH_TEST, bWasDepthTestEnabled);

	checkGlError("GlSkinnedMeshRenderer::draw");
}
