#pragma once

#include "CalibrationPatternFinder.h"

class CalibrationPatternFinder_Charuco : public CalibrationPatternFinder
{
public:
	CalibrationPatternFinder_Charuco(int frameWidth, int frameHeight, int charucoCols, int charucoRows,
									 float charucoSquareLengthMM, float charucoMarkerLengthMM,
									 eCharucoDictionaryType charucoDictionaryType);
	virtual ~CalibrationPatternFinder_Charuco();

	virtual eCalibrationPatternType getCalibrationPatternType() const override
	{
		return eCalibrationPatternType::mode_charuco;
	}
	virtual bool findNewCalibrationPattern(const float minSeperationDist= 0.f) override;
	virtual bool getCurrentCalibrationPattern(t_opencv_point2d_list& outImagePoints,
											  cv::Point2f outBoundingQuad[4]) const override;
	virtual bool fetchLastFoundCalibrationPattern(t_opencv_point2d_list& outImagePoints,
												  t_opencv_pointID_list& outImagePointIDs,
												  cv::Point2f outBoundingQuad[4]) override;

protected:
	class CharucoBoardData* m_markerData;
};
