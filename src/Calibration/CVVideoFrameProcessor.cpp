#include "CVVideoFrameProcessor.h"

#include "opencv2/opencv.hpp"

#include <easy/profiler.h>

CVVideoFrameProcessor::CVVideoFrameProcessor() {}

CVVideoFrameProcessor::~CVVideoFrameProcessor()
{
	if (m_bgrUndistortBuffer != nullptr)
		delete m_bgrUndistortBuffer;

	if (m_gsSourceBuffer != nullptr)
		delete m_gsSourceBuffer;

	if (m_gsUndistortBuffer != nullptr)
		delete m_gsUndistortBuffer;

	if (m_bgrGsDisplayBuffer != nullptr)
		delete m_bgrGsDisplayBuffer;
}

void CVVideoFrameProcessor::ensureBufferSize(int width, int height)
{
	if (m_bgrUndistortBuffer != nullptr)
		delete m_bgrUndistortBuffer;
	m_bgrUndistortBuffer= new cv::Mat(cv::Size(width, height), CV_8UC3);

	if (m_gsSourceBuffer != nullptr)
		delete m_gsSourceBuffer;
	m_gsSourceBuffer= new cv::Mat(height, width, CV_8UC1);

	if (m_gsUndistortBuffer != nullptr)
		delete m_gsUndistortBuffer;
	m_gsUndistortBuffer= new cv::Mat(height, width, CV_8UC1);

	if (m_bgrGsDisplayBuffer != nullptr)
		delete m_bgrGsDisplayBuffer;
	m_bgrGsDisplayBuffer= new cv::Mat(height, width, CV_8UC3);
}

void CVVideoFrameProcessor::computeUndistortion(const cv::Mat& srcBuffer, const cv::Mat& distortionMapX,
												const cv::Mat& distortionMapY)
{
	if (m_bgrUndistortBuffer == nullptr)
		return;

	EASY_BLOCK("OpenCV Undistort");

	// Apply the X and Y undistortion maps to create an undistorted 24-BPP image (for display)
	if (!m_bColorUndistortDisabled)
	{
		EASY_BLOCK("Color Undistort");

		cv::remap(srcBuffer, *m_bgrUndistortBuffer, distortionMapX, distortionMapY, cv::INTER_LINEAR,
				  cv::BORDER_CONSTANT);
	}

	// Also, optionally do grayscale conversion (and maybe undistortion)
	if (m_gsSourceBuffer != nullptr && m_bgrGsDisplayBuffer != nullptr)
	{
		if (!m_bGrayscaleUndistortDisabled && m_gsUndistortBuffer != nullptr)
		{
			{
				EASY_BLOCK("BGR -> Grayscale");
				cv::cvtColor(srcBuffer, *m_gsSourceBuffer, cv::COLOR_BGR2GRAY);
			}

			{
				EASY_BLOCK("Grayscale Undistort");
				cv::remap(*m_gsSourceBuffer, *m_gsUndistortBuffer, distortionMapX, distortionMapY, cv::INTER_LINEAR,
						  cv::BORDER_CONSTANT);
			}

			{
				EASY_BLOCK("Grayscale -> BGR");
				cv::cvtColor(*m_gsUndistortBuffer, *m_bgrGsDisplayBuffer, cv::COLOR_GRAY2BGR);
			}
		}
		else if (m_bGrayscaleUndistortDisabled)
		{
			{
				EASY_BLOCK("BGR -> Grayscale");
				cv::cvtColor(srcBuffer, *m_gsSourceBuffer, cv::COLOR_BGR2GRAY);
			}

			{
				EASY_BLOCK("Grayscale -> BGR");
				cv::cvtColor(*m_gsSourceBuffer, *m_bgrGsDisplayBuffer, cv::COLOR_GRAY2BGR);
			}
		}
	}
}
