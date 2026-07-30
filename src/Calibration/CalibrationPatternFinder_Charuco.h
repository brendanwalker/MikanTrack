#pragma once

#include "CalibrationPatternFinder.h"

class CalibrationPatternFinder_Charuco : public CalibrationPatternFinder
{
public:
	CalibrationPatternFinder_Charuco(VideoFrameDistortionView* distortionView, int charucoRows, int charucoCols,
									 float charucoSquareLengthMM, float charucoMarkerLengthMM,
									 eCharucoDictionaryType charucoDictionaryType);
	virtual ~CalibrationPatternFinder_Charuco();

	virtual eCalibrationPatternType getCalibrationPatternType() const override
	{
		return eCalibrationPatternType::mode_charuco;
	}
	virtual bool findNewCalibrationPattern(const float minSeperationDist= 0.f) override;
	virtual bool fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
												  t_opencv_pointID_list& outImagePointIDs,
												  cv::Point2f outBoundingQuad[4]) override;
	virtual void renderCalibrationPattern2D() const override;
	virtual void renderSolvePnPPattern3D(const glm::mat4& xform) const override;

protected:
	class CharucoBoardData* m_markerData;
};
