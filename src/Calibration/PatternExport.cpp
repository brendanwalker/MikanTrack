#include "PatternExport.h"
#include "Logger.h"

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

bool generateCharucoBoardPng(const std::filesystem::path& pngPath, int charucoCols, int charucoRows,
							 float charucoSquareLengthMM, float charucoMarkerLengthMM,
							 eCharucoDictionaryType charucoDictionaryType, float pixelsPerMM)
{
	if (charucoCols < 2 || charucoRows < 2 || charucoSquareLengthMM <= 0.f || charucoMarkerLengthMM <= 0.f
		|| pixelsPerMM <= 0.f)
	{
		MIKAN_LOG_ERROR("generateCharucoBoardPng") << "Invalid board parameters";
		return false;
	}

	ArucoDictionaryPtr dictionary= CalibrationPatternFinder::getArucoDictionary(charucoDictionaryType);

	// Create ChArUco board (dimensions in mm, matching MarkerObjectSystem::printMarker)
	cv::aruco::CharucoBoard charucoBoard(cv::Size(charucoCols, charucoRows), charucoSquareLengthMM,
										 charucoMarkerLengthMM, *dictionary.get());

	// Generate ChArUco board image at the requested print resolution
	const int imageWidthPx= (int)(float(charucoCols) * charucoSquareLengthMM * pixelsPerMM);
	const int imageHeightPx= (int)(float(charucoRows) * charucoSquareLengthMM * pixelsPerMM);
	cv::Mat boardImage;
	charucoBoard.generateImage(cv::Size(imageWidthPx, imageHeightPx), boardImage, 10, 1);

	if (!cv::imwrite(pngPath.string(), boardImage))
	{
		MIKAN_LOG_ERROR("generateCharucoBoardPng") << "Failed to write PNG to: " << pngPath.string();
		return false;
	}

	MIKAN_LOG_INFO("generateCharucoBoardPng") << "ChArUco board PNG saved to: " << pngPath.string();

	return true;
}

bool generateArucoMarkerPng(const std::filesystem::path& pngPath, int arucoId,
							eCharucoDictionaryType arucoDictionaryType, int markerSizePx)
{
	if (arucoId < 0 || markerSizePx <= 0)
	{
		MIKAN_LOG_ERROR("generateArucoMarkerPng") << "Invalid marker parameters";
		return false;
	}

	ArucoDictionaryPtr dictionary= CalibrationPatternFinder::getArucoDictionary(arucoDictionaryType);

	// Generate ArUco marker image using OpenCV
	cv::Mat markerImage;
	cv::aruco::generateImageMarker(*dictionary.get(), arucoId, markerSizePx, markerImage, 1);

	if (!cv::imwrite(pngPath.string(), markerImage))
	{
		MIKAN_LOG_ERROR("generateArucoMarkerPng") << "Failed to write PNG to: " << pngPath.string();
		return false;
	}

	MIKAN_LOG_INFO("generateArucoMarkerPng") << "ArUco marker PNG saved to: " << pngPath.string();

	return true;
}
