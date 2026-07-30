#include "GlFrameBuffer.h"
#include "GlUtils.h"
#include "Logger.h"

#include "GL/glew.h"

GlFrameBuffer::~GlFrameBuffer() { dispose(); }

bool GlFrameBuffer::init(uint16_t width, uint16_t height)
{
	dispose();

	if (width == 0 || height == 0)
	{
		MIKAN_LOG_ERROR("GlFrameBuffer::init") << "Invalid framebuffer size: " << width << "x" << height;
		return false;
	}

	m_width= width;
	m_height= height;

	// Save the currently bound framebuffer so we can restore it after setup
	GLint prevFrameBufferId= 0;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prevFrameBufferId);

	// Color attachment: RGBA8 texture
	glGenTextures(1, &m_glColorTextureId);
	glBindTexture(GL_TEXTURE_2D, m_glColorTextureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Depth attachment: 24-bit depth renderbuffer
	glGenRenderbuffers(1, &m_glDepthRenderBufferId);
	glBindRenderbuffer(GL_RENDERBUFFER, m_glDepthRenderBufferId);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, m_width, m_height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	glGenFramebuffers(1, &m_glFrameBufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, m_glFrameBufferId);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_glColorTextureId, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, m_glDepthRenderBufferId);

	const GLenum status= glCheckFramebufferStatus(GL_FRAMEBUFFER);
	glBindFramebuffer(GL_FRAMEBUFFER, prevFrameBufferId);

	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		MIKAN_LOG_ERROR("GlFrameBuffer::init") << "Framebuffer incomplete, status: 0x" << std::hex << status << std::dec;
		dispose();
		return false;
	}

	if (checkGlError("GlFrameBuffer::init"))
	{
		dispose();
		return false;
	}

	return true;
}

bool GlFrameBuffer::resize(uint16_t width, uint16_t height)
{
	if (width == m_width && height == m_height && m_glFrameBufferId != 0)
	{
		return true;
	}

	return init(width, height);
}

void GlFrameBuffer::dispose()
{
	if (m_glFrameBufferId != 0)
	{
		glDeleteFramebuffers(1, &m_glFrameBufferId);
		m_glFrameBufferId= 0;
	}

	if (m_glDepthRenderBufferId != 0)
	{
		glDeleteRenderbuffers(1, &m_glDepthRenderBufferId);
		m_glDepthRenderBufferId= 0;
	}

	if (m_glColorTextureId != 0)
	{
		glDeleteTextures(1, &m_glColorTextureId);
		m_glColorTextureId= 0;
	}

	m_width= 0;
	m_height= 0;
	m_bIsBound= false;
}

void GlFrameBuffer::bindFrameBuffer()
{
	if (m_glFrameBufferId == 0 || m_bIsBound)
	{
		return;
	}

	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &m_prevFrameBufferId);
	glGetIntegerv(GL_VIEWPORT, m_prevViewport);

	glBindFramebuffer(GL_FRAMEBUFFER, m_glFrameBufferId);
	glViewport(0, 0, m_width, m_height);

	m_bIsBound= true;
}

void GlFrameBuffer::unbindFrameBuffer()
{
	if (!m_bIsBound)
	{
		return;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, m_prevFrameBufferId);
	glViewport(m_prevViewport[0], m_prevViewport[1], m_prevViewport[2], m_prevViewport[3]);

	m_bIsBound= false;
}
