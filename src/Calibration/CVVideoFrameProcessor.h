#pragma once

#include "OpenCVFwd.h"
#include "MikanVideoSourceTypes.h"

// CPU undistortion helper: builds cv::initUndistortRectifyMap X/Y maps from mono
// camera intrinsics and applies them with cv::remap to color/grayscale frames.
class CVVideoFrameProcessor
{
public:
	CVVideoFrameProcessor(const MikanMonoIntrinsics& intrinsics, int width, int height);
	~CVVideoFrameProcessor();

	// Rebuild the undistortion maps from a new set of mono camera intrinsics
	void applyMonoCameraIntrinsics(const MikanMonoIntrinsics& intrinsics);

	// Undistorts a 24-BPP BGR source frame into the given output buffer.
	// When color undistortion is disabled the source frame is copied through unmodified.
	void processColorFrame(const cv::Mat& bgrSourceBuffer, cv::Mat& bgrUndistortedOut);

	// Converts a 24-BPP BGR source frame to grayscale (and undistorts it,
	// unless grayscale undistortion is disabled)
	void processGrayscale(const cv::Mat& bgrSourceBuffer);

	inline int getFrameWidth() const { return m_frameWidth; }
	inline int getFrameHeight() const { return m_frameHeight; }

	inline cv::Mat* getGrayscaleSourceBuffer() const { return m_gsSourceBuffer; }
	inline cv::Mat* getGrayscaleUndistortBuffer() const { return m_gsUndistortBuffer; }
	inline cv::Mat* getBGRGsDisplayBuffer() const { return m_bgrGsDisplayBuffer; }

	// The grayscale frame pattern finders should search: the undistorted grayscale
	// buffer unless grayscale undistortion is explicitly disabled
	// (which should only be the case during distortion calibration)
	inline cv::Mat* getGrayscaleFrameOutput() const
	{
		return m_bGrayscaleUndistortDisabled ? m_gsSourceBuffer : m_gsUndistortBuffer;
	}

	inline bool isColorUndistortDisabled() const { return m_bColorUndistortDisabled; }
	inline void setColorUndistortDisabled(bool bDisabled) { m_bColorUndistortDisabled= bDisabled; }

	inline bool isGrayscaleUndistortDisabled() const { return m_bGrayscaleUndistortDisabled; }
	inline void setGrayscaleUndistortDisabled(bool bDisabled) { m_bGrayscaleUndistortDisabled= bDisabled; }

private:
	int m_frameWidth= 0;
	int m_frameHeight= 0;

	// Undistortion maps used by cv::remap (CV_32FC1)
	cv::Mat* m_distortionMapX= nullptr;
	cv::Mat* m_distortionMapY= nullptr;

	// Grayscale video frame buffers
	cv::Mat* m_gsSourceBuffer= nullptr;     // 8-BPP source buffer
	cv::Mat* m_gsUndistortBuffer= nullptr;  // 8-BPP undistorted buffer
	cv::Mat* m_bgrGsDisplayBuffer= nullptr; // 24-BPP(BGR color format) debug display buffer

	// Runtime flags
	bool m_bColorUndistortDisabled= false;
	bool m_bGrayscaleUndistortDisabled= false;
};
