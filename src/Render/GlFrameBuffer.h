#pragma once

#include <cstdint>

// Offscreen render target for the 3D scene viewport:
// RGBA8 color texture + 24-bit depth renderbuffer.
// The color texture id is handed to ImGui::Image for display.
class GlFrameBuffer
{
public:
	GlFrameBuffer()= default;
	~GlFrameBuffer();

	GlFrameBuffer(const GlFrameBuffer&)= delete;
	GlFrameBuffer& operator=(const GlFrameBuffer&)= delete;

	// Creates the FBO + attachments. Requires a current GL context.
	bool init(uint16_t width, uint16_t height);

	// Recreates the FBO at a new resolution (no-op if the size is unchanged)
	bool resize(uint16_t width, uint16_t height);

	// Destroys the FBO and its attachments
	void dispose();

	// Binds the FBO for rendering, saving the previous framebuffer binding and
	// viewport, then sets the viewport to cover the full FBO
	void bindFrameBuffer();

	// Restores the previously bound framebuffer and viewport
	void unbindFrameBuffer();

	uint32_t getColorTextureId() const { return m_glColorTextureId; }
	uint16_t getWidth() const { return m_width; }
	uint16_t getHeight() const { return m_height; }
	bool getIsValid() const { return m_glFrameBufferId != 0; }

private:
	uint32_t m_glFrameBufferId= 0;
	uint32_t m_glColorTextureId= 0;
	uint32_t m_glDepthRenderBufferId= 0;
	uint16_t m_width= 0;
	uint16_t m_height= 0;

	// Saved state for bind/unbind
	int32_t m_prevFrameBufferId= 0;
	int32_t m_prevViewport[4]= {0, 0, 0, 0};
	bool m_bIsBound= false;
};
