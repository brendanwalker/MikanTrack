#pragma once

#include "CalibrationPatternFinder.h"

class CalibrationPatternFinder_Aruco : public CalibrationPatternFinder
{
public:
	CalibrationPatternFinder_Aruco(int frameWidth, int frameHeight, float markerLengthMM, int desiredArucoId,
								   eCharucoDictionaryType arucoDictionaryType);
	virtual ~CalibrationPatternFinder_Aruco();

	virtual eCalibrationPatternType getCalibrationPatternType() const override
	{
		return eCalibrationPatternType::mode_aruco;
	}
	virtual bool findNewCalibrationPattern(const float minSeperationDist= 0.f) override;
	virtual bool fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
												  t_opencv_pointID_list& outImagePointIDs,
												  cv::Point2f outBoundingQuad[4]) override;

protected:
	class ArucoBoardData* m_markerData;
};
