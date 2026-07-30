#pragma once

#include "CalibrationPatternFinder.h"

class CalibrationPatternFinder_Aruco : public CalibrationPatternFinder
{
public:
	// Detects the stage's origin marker (default behavior)
	CalibrationPatternFinder_Aruco(CameraComponentConstPtr cameraComponent, VideoFrameDistortionView* distortionView);

	// Detects an explicit marker definition (e.g., a utility marker)
	CalibrationPatternFinder_Aruco(CameraComponentConstPtr cameraComponent, VideoFrameDistortionView* distortionView,
								   MarkerDefinitionConstPtr markerDefinition);
	virtual ~CalibrationPatternFinder_Aruco();

	virtual eCalibrationPatternType getCalibrationPatternType() const override
	{
		return eCalibrationPatternType::mode_aruco;
	}
	virtual bool findNewCalibrationPattern(const float minSeperationDist= 0.f) override;
	virtual bool fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
												  t_opencv_pointID_list& outImagePointIDs,
												  cv::Point2f outBoundingQuad[4]) override;
	virtual void renderCalibrationPattern2D() const override;

protected:
	class ArucoBoardData* m_markerData;
};
