#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"

#include <cstdint>

// Batched debug line/point renderer (GL 3.3 core), self-contained port of the
// concept behind MikanXR's GlLineRenderer. Queue up 3d (world space) and 2d
// (pixel space) segments/points each frame, then flush with render3d/render2d.
// Draw order per flush: points then lines. Vertices are {vec3 pos, vec4 color+size}
// with point size driven by gl_PointSize (GL_PROGRAM_POINT_SIZE).
class GlLineRenderer
{
public:
	GlLineRenderer();
	~GlLineRenderer();

	GlLineRenderer(const GlLineRenderer&)= delete;
	GlLineRenderer& operator=(const GlLineRenderer&)= delete;

	// Compiles the shader and creates the vertex buffers. Requires a current GL context.
	bool startup();
	void shutdown();

	// -- 3d (world space) queueing --
	void addSegment3d(const glm::mat4& xform, const glm::vec3& pos0, const glm::vec3& pos1, const glm::vec3& color);
	void addSegment3d(const glm::mat4& xform, const glm::vec3& pos0, const glm::vec3& color0, const glm::vec3& pos1,
					  const glm::vec3& color1);
	void addPoint3d(const glm::mat4& xform, const glm::vec3& pos, const glm::vec3& color, float size= 1.f);

	// -- 2d (pixel space, origin top-left) queueing --
	void addSegment2d(const glm::vec2& pos0, const glm::vec2& pos1, const glm::vec3& color);
	void addSegment2d(const glm::vec2& pos0, const glm::vec3& color0, const glm::vec2& pos1, const glm::vec3& color1);
	void addPoint2d(const glm::vec2& pos, const glm::vec3& color, float size= 1.f);

	// Draws and clears the queued 3d points/lines using the given view-projection matrix.
	// Pass bDisableDepth=true to draw over everything (depth test off).
	void render3d(const glm::mat4& viewProj, bool bDisableDepth= false);

	// Draws and clears the queued 2d points/lines with a pixel-space ortho projection
	// (0,0 at the top-left, screenW/screenH at the bottom-right). Depth test is disabled.
	void render2d(float screenW, float screenH);

private:
	struct Point
	{
		glm::vec3 position;
		glm::vec4 colorAndSize;
	};

	class PointBufferState
	{
	public:
		explicit PointBufferState(int maxPoints);
		~PointBufferState();

		PointBufferState(const PointBufferState&)= delete;
		PointBufferState& operator=(const PointBufferState&)= delete;

		bool createGlBufferState();
		void destroyGlBufferState();
		void drawGlBufferState(uint32_t glEnumMode);

		inline bool hasPoints() const { return m_pointCount > 0; }

		void addPoint3d(const glm::mat4& xform, const glm::vec3& pos, const glm::vec3& color, float size);
		void addPoint2d(const glm::vec2& pos, const glm::vec3& color, float size);

	private:
		Point* m_points;
		int m_maxPoints;
		int m_pointCount;
		uint32_t m_pointVAO;
		uint32_t m_pointVBO;
	};

	bool compileProgram();
	void setMvpUniform(const glm::mat4& mvp);

	uint32_t m_programId= 0;
	int32_t m_mvpUniformLocation= -1;

	PointBufferState m_points3d;
	PointBufferState m_lines3d;
	PointBufferState m_points2d;
	PointBufferState m_lines2d;
};
