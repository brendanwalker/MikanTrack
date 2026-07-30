#include "CVVideoFrameProcessor.h"
#include "MathTypeConversion.h"

#include "opencv2/opencv.hpp"

#include <easy/profiler.h>

CVVideoFrameProcessor::CVVideoFrameProcessor(const MikanMonoIntrinsics& intrinsics, int width, int height)
	: m_frameWidth(width)
	, m_frameHeight(height)
{
	m_distortionMapX= new cv::Mat(height, width, CV_32FC1);
	m_distortionMapY= new cv::Mat(height, width, CV_32FC1);

	m_gsSourceBuffer= new cv::Mat(height, width, CV_8UC1);
	m_gsUndistortBuffer= new cv::Mat(height, width, CV_8UC1);
	m_bgrGsDisplayBuffer= new cv::Mat(height, width, CV_8UC3);

	applyMonoCameraIntrinsics(intrinsics);
}

CVVideoFrameProcessor::~CVVideoFrameProcessor()
{
	delete m_distortionMapX;
	delete m_distortionMapY;

	delete m_gsSourceBuffer;
	delete m_gsUndistortBuffer;
	delete m_bgrGsDisplayBuffer;
}

void CVVideoFrameProcessor::applyMonoCameraIntrinsics(const MikanMonoIntrinsics& intrinsics)
{
	const cv::Matx33d distortedIntrinsicMatrix= MikanMatrix3d_to_cv_mat33d(intrinsics.distorted_camera_matrix);
	const cv::Matx81d distortionCoeffs= Mikan_distortion_to_cv_vec8(intrinsics.distortion_coefficients);
	const cv::Matx33d undistortedIntrinsicMatrix= MikanMatrix3d_to_cv_mat33d(intrinsics.undistorted_camera_matrix);

	// (Re)create the X and Y undistortion maps used by cv::remap
	cv::initUndistortRectifyMap(distortedIntrinsicMatrix, distortionCoeffs,
								cv::noArray(), // unneeded rectification transformation computed by stereoRectify()
								undistortedIntrinsicMatrix, cv::Size(m_frameWidth, m_frameHeight),
								CV_32FC1, // Distortion map type
								*m_distortionMapX, *m_distortionMapY);
}

void CVVideoFrameProcessor::processColorFrame(const cv::Mat& bgrSourceBuffer, cv::Mat& bgrUndistortedOut)
{
	EASY_BLOCK("OpenCV Undistort");

	// Apply the X and Y undistortion maps to create an undistorted 24-BPP image
	if (!m_bColorUndistortDisabled)
	{
		EASY_BLOCK("Color Undistort");

		cv::remap(bgrSourceBuffer, bgrUndistortedOut, *m_distortionMapX, *m_distortionMapY, cv::INTER_LINEAR,
				  cv::BORDER_CONSTANT);
	}
	else
	{
		bgrSourceBuffer.copyTo(bgrUndistortedOut);
	}
}

void CVVideoFrameProcessor::processGrayscale(const cv::Mat& bgrSourceBuffer)
{
	if (!m_bGrayscaleUndistortDisabled)
	{
		{
			EASY_BLOCK("BGR -> Grayscale");
			cv::cvtColor(bgrSourceBuffer, *m_gsSourceBuffer, cv::COLOR_BGR2GRAY);
		}

		{
			EASY_BLOCK("Grayscale Undistort");
			cv::remap(*m_gsSourceBuffer, *m_gsUndistortBuffer, *m_distortionMapX, *m_distortionMapY, cv::INTER_LINEAR,
					  cv::BORDER_CONSTANT);
		}

		{
			EASY_BLOCK("Grayscale -> BGR");
			cv::cvtColor(*m_gsUndistortBuffer, *m_bgrGsDisplayBuffer, cv::COLOR_GRAY2BGR);
		}
	}
	else
	{
		{
			EASY_BLOCK("BGR -> Grayscale");
			cv::cvtColor(bgrSourceBuffer, *m_gsSourceBuffer, cv::COLOR_BGR2GRAY);
		}

		{
			EASY_BLOCK("Grayscale -> BGR");
			cv::cvtColor(*m_gsSourceBuffer, *m_bgrGsDisplayBuffer, cv::COLOR_GRAY2BGR);
		}
	}
}
