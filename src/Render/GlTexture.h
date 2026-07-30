#pragma once

#include <cstdint>

namespace cv
{
class Mat;
}

// Streaming 2D texture used for the webcam video preview.
// Internally a GL_RGB8 texture updated from BGR cv::Mat frames
// via double pixel-unpack-buffer (PBO) streaming, ported from the
// PixelBufferObjectMode::DoublePBOWrite path in MikanXR's GlTexture.
class GlTexture
{
public:
	GlTexture()= default;
	~GlTexture();

	GlTexture(const GlTexture&)= delete;
	GlTexture& operator=(const GlTexture&)= delete;

	// Creates the texture and the two streaming PBOs. Requires a current GL context.
	bool init(uint16_t width, uint16_t height);

	// Recreates the texture/PBOs at a new resolution (no-op if the size is unchanged)
	bool resize(uint16_t width, uint16_t height);

	// Destroys the texture and PBOs
	void dispose();

	// Uploads a CV_8UC3 BGR frame (e.g. straight from cv::VideoCapture).
	// Auto-resizes if the frame dimensions differ from the current texture size.
	// Handles padded cv::Mat row strides (mat.step) via a per-row copy into the PBO.
	void uploadBGR(const cv::Mat& bgr);

	// Texture id for rendering (e.g. ImGui::Image with (ImTextureID)(intptr_t)getGlTextureId())
	uint32_t getGlTextureId() const { return m_glTextureId; }
	uint16_t getWidth() const { return m_width; }
	uint16_t getHeight() const { return m_height; }
	bool getIsValid() const { return m_glTextureId != 0; }

private:
	uint32_t m_glTextureId= 0;
	uint32_t m_glPixelBufferObjectIDs[2]= {0, 0};
	int m_pboWriteIndex= 0;
	size_t m_pboByteSize= 0;
	uint16_t m_width= 0;
	uint16_t m_height= 0;
};
