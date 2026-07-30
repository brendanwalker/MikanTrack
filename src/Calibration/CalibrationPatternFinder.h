#pragma once

#include "ObjectSystemConfigFwd.h"
#include "ComponentFwd.h"
#include "OpenCVFwd.h"
#include "CameraMath.h"
#include "ProjectConfig.h"

#include <memory>
#include "opencv2/core/types.hpp"

#include "glm/ext/vector_float3.hpp"

class CalibrationPatternFinder;
typedef std::shared_ptr<CalibrationPatternFinder> CalibrationPatternFinderPtr;

// Helper use to implement OpenCV camera lens intrinsic/distortion calibration method.
// See https://docs.opencv.org/3.3.0/dc/dbb/tutorial_py_calibration.html for details.
class CalibrationPatternFinder
{
public:
	CalibrationPatternFinder(class VideoFrameDistortionView* distortionView);
	virtual ~CalibrationPatternFinder();

	cv::Mat* getGrayscaleVideoFrameInput() const;

	virtual eCalibrationPatternType getCalibrationPatternType() const= 0;
	virtual bool findNewCalibrationPattern(const float minSeperationDist= 0.f)= 0;
	virtual bool estimateNewCalibrationPatternPose(glm::dmat4& outCameraToPatternXform);
	virtual bool fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
												  t_opencv_pointID_list& outImagePointIDs,
												  cv::Point2f outBoundingQuad[4])= 0;

	bool areCurrentImagePointsValid() const;
	inline float getFrameWidth() const { return m_frameWidth; }
	inline float getFrameHeight() const { return m_frameHeight; }
	inline VideoFrameDistortionView* getDistortionView() const { return m_distortionView; }
	inline void getOpenCVLensCalibrationGeometry(OpenCVCalibrationGeometry* outGeometry) const
	{
		*outGeometry= m_opencvLensCalibrationGeometry;
	};
	inline void getOpenCVSolvePnPGeometry(OpenCVCalibrationGeometry* outGeometry) const
	{
		*outGeometry= m_opencvSolvePnPGeometry;
	};
	inline void getOpenGLSolvePnPGeometry(OpenGLCalibrationGeometry* outGeometry) const
	{
		*outGeometry= m_openglSolvePnPGeometry;
	};
	virtual void renderCalibrationPattern2D() const;
	virtual void renderSolvePnPPattern3D(const glm::mat4& xform) const;

	static ArucoDictionaryPtr getArucoDictionary(eCharucoDictionaryType dictionaryType);

protected:
	// Constructor for test subclasses that provide fixed poses and don't need a video frame
	CalibrationPatternFinder(int frameWidth, int frameHeight);

	// Video buffer state
	class VideoFrameDistortionView* m_distortionView;

	float m_frameWidth;
	float m_frameHeight;

	// Internal Calibration State
	OpenCVCalibrationGeometry m_opencvLensCalibrationGeometry;
	OpenCVCalibrationGeometry m_opencvSolvePnPGeometry;
	OpenGLCalibrationGeometry m_openglSolvePnPGeometry;
	t_opencv_point2d_list m_lastValidQuad;
	t_opencv_point2d_list m_lastValidImagePoints;
	t_opencv_point2d_list m_currentImagePoints;
};