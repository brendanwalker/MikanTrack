#include "PatternExport.h"
#include "Logger.h"
#include "PathUtils.h"
#include <functional>
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

namespace
{
// Writes the pattern into resources/calibration and hands it to the OS viewer
std::filesystem::path openExportedPattern(const std::string& fileName,
										  const std::function<bool(const std::filesystem::path&)>& generate)
{
	const std::filesystem::path exportDir= PathUtils::getResourceDirectory() / "calibration";
	const std::filesystem::path exportPath= exportDir / fileName;
	// cv::imwrite does not create directories, and a missing one fails as a
	// bare false - create it up front so the only failures left are real
	std::error_code ec;
	std::filesystem::create_directories(exportDir, ec);
	try
	{
		if (!generate(exportPath))
			return std::filesystem::path();
	}
	catch (const cv::Exception& e)
	{
		MIKAN_LOG_ERROR("openExportedPattern") << "OpenCV error exporting pattern: " << e.what();
		return std::filesystem::path();
	}
	// Failing to open is not fatal (the file is written either way), but it
	// must not be silent - that reads as "the button did nothing"
	if (!PathUtils::openFileWithDefaultApplication(exportPath))
	{
		MIKAN_LOG_WARNING("openExportedPattern")
			<< "Pattern written but the system viewer would not open it - print it from "
			<< exportPath.string();
	}
	return exportPath;
}
} // namespace
std::filesystem::path exportAndOpenCharucoBoard(int charucoCols, int charucoRows,
											   float charucoSquareLengthMM, float charucoMarkerLengthMM,
											   eCharucoDictionaryType charucoDictionaryType)
{
	return openExportedPattern("charuco_board.png", [&](const std::filesystem::path& path) {
		// 10 px/mm: enough for the cells to survive printing, small enough
		// that the sheet stays a manageable PNG
		return generateCharucoBoardPng(path, charucoCols, charucoRows, charucoSquareLengthMM,
									   charucoMarkerLengthMM, charucoDictionaryType, 10.f);
	});
}
std::filesystem::path exportAndOpenArucoMarker(int arucoId, eCharucoDictionaryType arucoDictionaryType,
											  float markerLengthMM)
{
	return openExportedPattern("aruco_marker.png", [&](const std::filesystem::path& path) {
		return generateArucoMarkerPng(path, arucoId, arucoDictionaryType, 1000, markerLengthMM);
	});
}

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
							eCharucoDictionaryType arucoDictionaryType, int markerSizePx,
							float markerLengthMM)
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

	// Turn the pattern a quarter turn counter-clockwise on the page. The
	// detector reads the marker's axes off its own bit pattern, so rotating it
	// here rotates the tracking world with it: the pattern's own +X (world
	// FORWARD) ends up pointing at the top of the sheet. The payoff is that
	// the sheet is then placed the way anyone would place a page - top edge
	// away from you - instead of a quarter turn that is invisible once wrong.
	cv::Mat rotatedMarker;
	cv::rotate(markerImage, rotatedMarker, cv::ROTATE_90_COUNTERCLOCKWISE);

	// Room around the marker for the labels. The QUIET ZONE matters
	// independently: aruco detection needs white around the pattern, and a
	// marker cut out flush to its border detects poorly.
	const int margin= std::max(140, (int)(markerSizePx * 0.30f));
	const int canvasSize= markerSizePx + 2 * margin;

	cv::Mat canvas(canvasSize, canvasSize, CV_8UC1, cv::Scalar(255));
	rotatedMarker.copyTo(canvas(cv::Rect(margin, margin, markerSizePx, markerSizePx)));

	const double labelScale= (double)margin / 95.0;
	const double captionScale= labelScale * 0.42;
	const int labelThickness= std::max(2, margin / 38);
	const int captionThickness= std::max(1, margin / 90);
	const cv::Scalar ink(0);

	auto drawCentered= [&canvas, &ink](const std::string& text, cv::Point center, double scale, int thickness) {
		int baseline= 0;
		const cv::Size textSize= cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, &baseline);
		cv::putText(canvas, text, cv::Point(center.x - textSize.width / 2, center.y + textSize.height / 2),
					cv::FONT_HERSHEY_SIMPLEX, scale, ink, thickness, cv::LINE_AA);
	};

	const int centerX= canvasSize / 2;
	const int centerY= canvasSize / 2;

	// FORWARD gets the arrow: it is the only direction that has to be right,
	// since left and right follow from it. The axis letters are spelled out
	// because this marker defines the tracking world's axes and fusion now
	// treats "right hand toward -Y" as a fixed convention rather than a
	// setting - so the sheet is the documentation for that convention.
	drawCentered("FORWARD  +X", cv::Point(centerX, (int)(margin * 0.66f)), labelScale, labelThickness);
	cv::arrowedLine(canvas, cv::Point(centerX, (int)(margin * 0.40f)), cv::Point(centerX, (int)(margin * 0.10f)),
					ink, labelThickness, cv::LINE_AA, 0, 0.4);

	// Side labels are stacked (direction over axis) rather than run on one
	// line: the margin is only as wide as the quiet zone needs, and a wider
	// label would print over the marker, which breaks detection
	const double sideLabelScale= labelScale * 0.8;
	const int sideLineStep= (int)(margin * 0.26f);
	drawCentered("LEFT", cv::Point((int)(margin * 0.5f), centerY - sideLineStep / 2), sideLabelScale,
				 labelThickness);
	drawCentered("+Y", cv::Point((int)(margin * 0.5f), centerY + sideLineStep / 2), sideLabelScale,
				 labelThickness);
	drawCentered("RIGHT", cv::Point(canvasSize - (int)(margin * 0.5f), centerY - sideLineStep / 2),
				 sideLabelScale, labelThickness);
	drawCentered("-Y", cv::Point(canvasSize - (int)(margin * 0.5f), centerY + sideLineStep / 2),
				 sideLabelScale, labelThickness);

	// Bottom margin carries the two things that silently ruin a calibration:
	// laying the sheet down turned, and letting the print dialog rescale it
	const int captionY= canvasSize - (int)(margin * 0.58f);
	const int captionLineStep= (int)(margin * 0.30f);
	drawCentered("Lay flat with FORWARD pointing away from you", cv::Point(centerX, captionY), captionScale,
				 captionThickness);
	if (markerLengthMM > 0.f)
	{
		char sizeText[128];
		snprintf(sizeText, sizeof(sizeText),
				 "Print at 100%% scale - the black square must measure %.0f mm", markerLengthMM);
		drawCentered(sizeText, cv::Point(centerX, captionY + captionLineStep), captionScale, captionThickness);
	}

	if (!cv::imwrite(pngPath.string(), canvas))
	{
		MIKAN_LOG_ERROR("generateArucoMarkerPng") << "Failed to write PNG to: " << pngPath.string();
		return false;
	}

	MIKAN_LOG_INFO("generateArucoMarkerPng") << "ArUco marker PNG saved to: " << pngPath.string();

	return true;
}
