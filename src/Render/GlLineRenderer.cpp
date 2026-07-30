#include "GlLineRenderer.h"
#include "GlUtils.h"
#include "Logger.h"

#include "GL/glew.h"

#include "glm/ext/matrix_clip_space.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <cassert>

static const int k_max_segments= 0x8000;
static const int k_max_points= 0x8000;

static const char* k_lineVertexShaderCode= R""""(
#version 330 core
uniform mat4 mvpMatrix;
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec4 in_colorPointSize;
out vec4 v_Color;
void main()
{
	gl_Position = mvpMatrix * vec4(in_position.xyz, 1);
	gl_PointSize = in_colorPointSize.w;
	v_Color = vec4(in_colorPointSize.xyz, 1.0);
}
)"""";

static const char* k_lineFragmentShaderCode= R""""(
#version 330 core
in vec4 v_Color;
out vec4 out_FragColor;
void main()
{
	out_FragColor = v_Color;
}
)"""";

// -- PointBufferState --

GlLineRenderer::PointBufferState::PointBufferState(int maxPoints)
	: m_points(new Point[maxPoints])
	, m_maxPoints(maxPoints)
	, m_pointCount(0)
	, m_pointVAO(0)
	, m_pointVBO(0)
{
}

GlLineRenderer::PointBufferState::~PointBufferState() { delete[] m_points; }

bool GlLineRenderer::PointBufferState::createGlBufferState()
{
	glGenVertexArrays(1, &m_pointVAO);
	glGenBuffers(1, &m_pointVBO);

	glBindVertexArray(m_pointVAO);
	glBindBuffer(GL_ARRAY_BUFFER, m_pointVBO);
	glBufferData(GL_ARRAY_BUFFER, m_maxPoints * sizeof(Point), nullptr, GL_DYNAMIC_DRAW);

	// layout(location = 0) vec3 in_position
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Point), (const void*)offsetof(Point, position));

	// layout(location = 1) vec4 in_colorPointSize
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (const void*)offsetof(Point, colorAndSize));

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return !checkGlError("GlLineRenderer::PointBufferState::createGlBufferState");
}

void GlLineRenderer::PointBufferState::destroyGlBufferState()
{
	if (m_pointVAO != 0)
	{
		glDeleteVertexArrays(1, &m_pointVAO);
		m_pointVAO= 0;
	}

	if (m_pointVBO != 0)
	{
		glDeleteBuffers(1, &m_pointVBO);
		m_pointVBO= 0;
	}

	m_pointCount= 0;
}

void GlLineRenderer::PointBufferState::drawGlBufferState(uint32_t glEnumMode)
{
	assert(m_points != nullptr);
	assert(m_pointCount <= m_maxPoints);
	if (m_pointCount > 0)
	{
		glBindVertexArray(m_pointVAO);

		glBindBuffer(GL_ARRAY_BUFFER, m_pointVBO);
		glBufferSubData(GL_ARRAY_BUFFER, 0, m_pointCount * sizeof(Point), m_points);

		glDrawArrays(glEnumMode, 0, m_pointCount);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	m_pointCount= 0;
}

void GlLineRenderer::PointBufferState::addPoint3d(const glm::mat4& xform, const glm::vec3& pos, const glm::vec3& color,
												  float size)
{
	if (m_pointCount < m_maxPoints)
	{
		const glm::vec3 xformedPos= glm::vec3(xform * glm::vec4(pos, 1.0f));

		m_points[m_pointCount]= {xformedPos, glm::vec4(color.r, color.g, color.b, size)};
		++m_pointCount;
	}
}

void GlLineRenderer::PointBufferState::addPoint2d(const glm::vec2& pos, const glm::vec3& color, float size)
{
	if (m_pointCount < m_maxPoints)
	{
		const glm::vec3 pos3d= glm::vec3(pos.x, pos.y, 0.0f);

		m_points[m_pointCount]= {pos3d, glm::vec4(color.r, color.g, color.b, size)};
		++m_pointCount;
	}
}

// -- GlLineRenderer --

GlLineRenderer::GlLineRenderer()
	: m_points3d(k_max_points)
	, m_lines3d(k_max_segments * 2)
	, m_points2d(k_max_points)
	, m_lines2d(k_max_segments * 2)
{
}

GlLineRenderer::~GlLineRenderer() { shutdown(); }

bool GlLineRenderer::compileProgram()
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
			MIKAN_LOG_ERROR("GlLineRenderer::compileProgram") << label << " compile failed: " << infoLog;
			glDeleteShader(shaderId);
			return 0;
		}

		return shaderId;
	};

	const GLuint vertexShaderId= compileShader(GL_VERTEX_SHADER, k_lineVertexShaderCode, "vertex shader");
	if (vertexShaderId == 0)
	{
		return false;
	}

	const GLuint fragmentShaderId= compileShader(GL_FRAGMENT_SHADER, k_lineFragmentShaderCode, "fragment shader");
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
		MIKAN_LOG_ERROR("GlLineRenderer::compileProgram") << "program link failed: " << infoLog;
		glDeleteProgram(m_programId);
		m_programId= 0;
		return false;
	}

	m_mvpUniformLocation= glGetUniformLocation(m_programId, "mvpMatrix");
	if (m_mvpUniformLocation == -1)
	{
		MIKAN_LOG_ERROR("GlLineRenderer::compileProgram") << "Failed to find mvpMatrix uniform";
		glDeleteProgram(m_programId);
		m_programId= 0;
		return false;
	}

	return true;
}

bool GlLineRenderer::startup()
{
	if (!compileProgram())
	{
		MIKAN_LOG_ERROR("GlLineRenderer::startup") << "Failed to build shader program";
		return false;
	}

	bool bSuccess= true;
	bSuccess&= m_points2d.createGlBufferState();
	bSuccess&= m_lines2d.createGlBufferState();
	bSuccess&= m_points3d.createGlBufferState();
	bSuccess&= m_lines3d.createGlBufferState();

	if (!bSuccess)
	{
		MIKAN_LOG_ERROR("GlLineRenderer::startup") << "Failed to create vertex buffers";
		shutdown();
		return false;
	}

	return true;
}

void GlLineRenderer::shutdown()
{
	m_points2d.destroyGlBufferState();
	m_lines2d.destroyGlBufferState();
	m_points3d.destroyGlBufferState();
	m_lines3d.destroyGlBufferState();

	if (m_programId != 0)
	{
		glDeleteProgram(m_programId);
		m_programId= 0;
	}
	m_mvpUniformLocation= -1;
}

void GlLineRenderer::setMvpUniform(const glm::mat4& mvp)
{
	glUniformMatrix4fv(m_mvpUniformLocation, 1, GL_FALSE, glm::value_ptr(mvp));
}

void GlLineRenderer::render3d(const glm::mat4& viewProj, bool bDisableDepth)
{
	if (m_programId == 0 || (!m_points3d.hasPoints() && !m_lines3d.hasPoints()))
	{
		return;
	}

	// Save the GL state we modify
	const GLboolean bWasDepthTestEnabled= glIsEnabled(GL_DEPTH_TEST);
	const GLboolean bWasProgramPointSizeEnabled= glIsEnabled(GL_PROGRAM_POINT_SIZE);

	// The point drawing shader uses gl_PointSize
	glEnable(GL_PROGRAM_POINT_SIZE);

	if (bDisableDepth)
	{
		glDisable(GL_DEPTH_TEST);
	}
	else
	{
		glEnable(GL_DEPTH_TEST);
	}

	glUseProgram(m_programId);
	setMvpUniform(viewProj);

	m_points3d.drawGlBufferState(GL_POINTS);
	m_lines3d.drawGlBufferState(GL_LINES);

	glUseProgram(0);

	// Restore GL state
	if (bWasDepthTestEnabled)
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);
	if (!bWasProgramPointSizeEnabled)
		glDisable(GL_PROGRAM_POINT_SIZE);
}

void GlLineRenderer::render2d(float screenW, float screenH)
{
	if (m_programId == 0 || (!m_points2d.hasPoints() && !m_lines2d.hasPoints()))
	{
		return;
	}

	// Save the GL state we modify
	const GLboolean bWasDepthTestEnabled= glIsEnabled(GL_DEPTH_TEST);
	const GLboolean bWasProgramPointSizeEnabled= glIsEnabled(GL_PROGRAM_POINT_SIZE);

	glEnable(GL_PROGRAM_POINT_SIZE);

	// Disable the depth buffer to allow overdraw
	glDisable(GL_DEPTH_TEST);

	// Pixel space ortho projection: origin top-left, +y down
	const glm::mat4 orthoMat= glm::ortho(0.f, screenW, screenH, 0.f, 1.0f, -1.0f);

	glUseProgram(m_programId);
	setMvpUniform(orthoMat);

	m_points2d.drawGlBufferState(GL_POINTS);
	m_lines2d.drawGlBufferState(GL_LINES);

	glUseProgram(0);

	// Restore GL state
	if (bWasDepthTestEnabled)
		glEnable(GL_DEPTH_TEST);
	if (!bWasProgramPointSizeEnabled)
		glDisable(GL_PROGRAM_POINT_SIZE);
}

void GlLineRenderer::addSegment3d(const glm::mat4& xform, const glm::vec3& pos0, const glm::vec3& pos1,
								  const glm::vec3& color)
{
	addSegment3d(xform, pos0, color, pos1, color);
}

void GlLineRenderer::addSegment3d(const glm::mat4& xform, const glm::vec3& pos0, const glm::vec3& color0,
								  const glm::vec3& pos1, const glm::vec3& color1)
{
	m_lines3d.addPoint3d(xform, pos0, color0, 1.f);
	m_lines3d.addPoint3d(xform, pos1, color1, 1.f);
}

void GlLineRenderer::addPoint3d(const glm::mat4& xform, const glm::vec3& pos, const glm::vec3& color, float size)
{
	m_points3d.addPoint3d(xform, pos, color, size);
}

void GlLineRenderer::addSegment2d(const glm::vec2& pos0, const glm::vec2& pos1, const glm::vec3& color)
{
	addSegment2d(pos0, color, pos1, color);
}

void GlLineRenderer::addSegment2d(const glm::vec2& pos0, const glm::vec3& color0, const glm::vec2& pos1,
								  const glm::vec3& color1)
{
	m_lines2d.addPoint2d(pos0, color0, 1.f);
	m_lines2d.addPoint2d(pos1, color1, 1.f);
}

void GlLineRenderer::addPoint2d(const glm::vec2& pos, const glm::vec3& color, float size)
{
	m_points2d.addPoint2d(pos, color, size);
}
