#include "GlTexture.h"
#include "GlUtils.h"
#include "Logger.h"

#include "GL/glew.h"

#include "opencv2/core.hpp"

#include <cstring>

GlTexture::~GlTexture() { dispose(); }

bool GlTexture::init(uint16_t width, uint16_t height)
{
	dispose();

	if (width == 0 || height == 0)
	{
		MIKAN_LOG_ERROR("GlTexture::init") << "Invalid texture size: " << width << "x" << height;
		return false;
	}

	m_width= width;
	m_height= height;
	m_pboByteSize= (size_t)m_width * (size_t)m_height * 3; // tightly packed GL_BGR/GL_UNSIGNED_BYTE

	glGenTextures(1, &m_glTextureId);
	glBindTexture(GL_TEXTURE_2D, m_glTextureId);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, m_width, m_height, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Allocate the two streaming pixel-unpack buffers
	glGenBuffers(2, m_glPixelBufferObjectIDs);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPixelBufferObjectIDs[0]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_pboByteSize, nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPixelBufferObjectIDs[1]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_pboByteSize, nullptr, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	if (checkGlError("GlTexture::init"))
	{
		dispose();
		return false;
	}

	return true;
}

bool GlTexture::resize(uint16_t width, uint16_t height)
{
	if (width == m_width && height == m_height && m_glTextureId != 0)
	{
		return true;
	}

	return init(width, height);
}

void GlTexture::dispose()
{
	if (m_glTextureId != 0)
	{
		glDeleteTextures(1, &m_glTextureId);
		m_glTextureId= 0;
	}

	if (m_glPixelBufferObjectIDs[0] != 0 || m_glPixelBufferObjectIDs[1] != 0)
	{
		glDeleteBuffers(2, m_glPixelBufferObjectIDs);
		m_glPixelBufferObjectIDs[0]= 0;
		m_glPixelBufferObjectIDs[1]= 0;
	}

	m_pboWriteIndex= 0;
	m_pboByteSize= 0;
	m_width= 0;
	m_height= 0;
}

void GlTexture::uploadBGR(const cv::Mat& bgr)
{
	if (bgr.empty() || bgr.type() != CV_8UC3)
	{
		MIKAN_LOG_ERROR("GlTexture::uploadBGR") << "Expected a non-empty CV_8UC3 (BGR) mat";
		return;
	}

	// Lazily (re)create the texture to match the incoming frame size
	if (bgr.cols != (int)m_width || bgr.rows != (int)m_height || m_glTextureId == 0)
	{
		if (!resize((uint16_t)bgr.cols, (uint16_t)bgr.rows))
		{
			return;
		}
	}

	// Rows are tightly packed in the PBO (width*3 bytes), so tell GL not to assume 4-byte alignment
	GLint prevUnpackAlignment= 4;
	glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevUnpackAlignment);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glBindTexture(GL_TEXTURE_2D, m_glTextureId);

	// Double PBO streaming, see http://www.songho.ca/opengl/gl_pbo.html#create
	// (ported from MikanXR GlTexture::copyBufferIntoTexture, DoublePBOWrite mode):
	// upload the texture from the PBO filled last frame while the CPU fills the other one.
	m_pboWriteIndex= (m_pboWriteIndex + 1) % 2;
	const int nextPBOIndex= (m_pboWriteIndex + 1) % 2;

	// Bind the current PBO and copy its pixels into the texture object
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPixelBufferObjectIDs[m_pboWriteIndex]);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, GL_BGR, GL_UNSIGNED_BYTE,
					0); // Treated as byte offset into the bound PBO

	// Bind the other PBO and fill it from main memory.
	// Orphan the buffer first (glBufferData with nullptr) so glMapBuffer doesn't
	// stall if the GPU is still reading the previous contents.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_glPixelBufferObjectIDs[nextPBOIndex]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, m_pboByteSize, nullptr, GL_STREAM_DRAW);
	GLubyte* writePointer= (GLubyte*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
	if (writePointer != nullptr)
	{
		const size_t packedRowBytes= (size_t)m_width * 3;

		if (bgr.isContinuous() && bgr.step[0] == packedRowBytes)
		{
			std::memcpy(writePointer, bgr.data, m_pboByteSize);
		}
		else
		{
			// cv::Mat rows can be padded (step > width*3) - copy row by row into the packed PBO
			for (int row= 0; row < (int)m_height; ++row)
			{
				std::memcpy(writePointer + (size_t)row * packedRowBytes, bgr.ptr(row), packedRowBytes);
			}
		}

		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	}

	// Release the PBO binding so later pixel operations behave normally
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	glPixelStorei(GL_UNPACK_ALIGNMENT, prevUnpackAlignment);
}
