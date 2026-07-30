#pragma once

#include "OpenCVFwd.h"

class CVVideoFrameProcessor
{
public:
	CVVideoFrameProcessor();
	~CVVideoFrameProcessor();

	void ensureBufferSize(int width, int height);
	void computeUndistortion(const cv::Mat& srcBuffer, const cv::Mat& distortionMapX, const cv::Mat& distortionMapY);

	inline cv::Mat* getGrayscaleSourceBuffer() const { return m_gsSourceBuffer; }
	inline cv::Mat* getGrayscaleUndistortBuffer() const { return m_gsUndistortBuffer; }
	inline cv::Mat* getBGRUndistortBuffer() const { return m_bgrUndistortBuffer; }
	inline cv::Mat* getBGRGsDisplayBuffer() const { return m_bgrGsDisplayBuffer; }

	inline bool isColorUndistortDisabled() const { return m_bColorUndistortDisabled; }
	inline void setColorUndistortDisabled(bool bDisabled) { m_bColorUndistortDisabled= bDisabled; }

	inline bool isGrayscaleUndistortDisabled() const { return m_bGrayscaleUndistortDisabled; }
	inline void setGrayscaleUndistortDisabled(bool bDisabled) { m_bGrayscaleUndistortDisabled= bDisabled; }

private:
	// Video frame buffers (24-BPP, BGR color format)
	cv::Mat* m_bgrUndistortBuffer= nullptr;

	// Grayscale video frame buffers
	cv::Mat* m_gsSourceBuffer= nullptr;     // 8-BPP source buffer
	cv::Mat* m_gsUndistortBuffer= nullptr;  // 8-BPP undistorted buffer
	cv::Mat* m_bgrGsDisplayBuffer= nullptr; // 24-BPP(BGR color format) debug display buffer

	// Runtime flags
	bool m_bColorUndistortDisabled= false;
	bool m_bGrayscaleUndistortDisabled= false;
};
