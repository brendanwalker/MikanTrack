#include "HandRoiQuality.h"

#include <algorithm>
#include <cmath>

#include "opencv2/imgproc.hpp"

namespace
{
// Clamps a rect to the frame and returns false if nothing usable remains
bool clampRectToFrame(cv::Rect& rect, const cv::Size& frameSize, int minDim)
{
	rect&= cv::Rect(0, 0, frameSize.width, frameSize.height);
	return rect.width >= minDim && rect.height >= minDim;
}

cv::Rect expandAboutCenter(const cv::Rect2f& box, float factor)
{
	const float centerX= box.x + box.width * 0.5f;
	const float centerY= box.y + box.height * 0.5f;
	const float halfWidth= box.width * 0.5f * factor;
	const float halfHeight= box.height * 0.5f * factor;
	return cv::Rect((int)std::floor(centerX - halfWidth), (int)std::floor(centerY - halfHeight),
					(int)std::ceil(halfWidth * 2.f), (int)std::ceil(halfHeight * 2.f));
}
} // namespace

void HandRoiQuality::analyzeHand(const cv::Mat& bgrFrame, TrackedHand& hand)
{
	hand.imageQuality= HandImageQuality();
	if (!hand.tracked || bgrFrame.empty())
		return;

	// Landmark bounding box in frame pixels
	glm::vec2 boxMin(bgrFrame.cols, bgrFrame.rows);
	glm::vec2 boxMax(0.f, 0.f);
	for (const glm::vec3& point : hand.imagePoints)
	{
		boxMin= glm::min(boxMin, glm::vec2(point.x, point.y));
		boxMax= glm::max(boxMax, glm::vec2(point.x, point.y));
	}
	const cv::Rect2f landmarkBox(boxMin.x, boxMin.y, boxMax.x - boxMin.x, boxMax.y - boxMin.y);
	if (landmarkBox.width <= 0.f || landmarkBox.height <= 0.f)
		return;

	cv::Rect innerRect= expandAboutCenter(landmarkBox, kRoiExpand);
	cv::Rect outerRect= expandAboutCenter(landmarkBox, kRoiExpand * kRingExpand);
	if (!clampRectToFrame(innerRect, bgrFrame.size(), 8) ||
		!clampRectToFrame(outerRect, bgrFrame.size(), 8))
		return;

	// Grayscale copy of the outer crop (its own buffer, so the filtered ops
	// below never reach outside it)
	cv::Mat grayOuter;
	cv::cvtColor(bgrFrame(outerRect), grayOuter, cv::COLOR_BGR2GRAY);

	// Decimate so the INNER ROI lands at kAnalysisMaxDim. Nearest-neighbor
	// SAMPLES pixels rather than averaging them, which keeps the clipping and
	// noise statistics intact.
	const int maxInnerDim= std::max(innerRect.width, innerRect.height);
	if (maxInnerDim > kAnalysisMaxDim)
	{
		const double scale= (double)kAnalysisMaxDim / (double)maxInnerDim;
		cv::resize(grayOuter, grayOuter, cv::Size(), scale, scale, cv::INTER_NEAREST);
		innerRect= cv::Rect((int)((innerRect.x - outerRect.x) * scale), (int)((innerRect.y - outerRect.y) * scale),
							(int)(innerRect.width * scale), (int)(innerRect.height * scale));
	}
	else
	{
		innerRect-= outerRect.tl();
	}
	if (!clampRectToFrame(innerRect, grayOuter.size(), 4))
		return;
	const cv::Mat inner= grayOuter(innerRect);

	// One histogram pass covers luminance, clipping and contrast
	int histogram[256]= {};
	for (int row= 0; row < inner.rows; ++row)
	{
		const uint8_t* pixel= inner.ptr<uint8_t>(row);
		for (int col= 0; col < inner.cols; ++col)
			++histogram[pixel[col]];
	}
	const double innerArea= (double)inner.rows * inner.cols;
	double innerSum= 0.0;
	double innerSumSq= 0.0;
	int shadowCount= 0;
	int highlightCount= 0;
	for (int value= 0; value < 256; ++value)
	{
		innerSum+= (double)value * histogram[value];
		innerSumSq+= (double)value * value * histogram[value];
		if (value <= 2)
			shadowCount+= histogram[value];
		if (value >= 253)
			highlightCount+= histogram[value];
	}
	const double innerMean= innerSum / innerArea;
	const double innerVariance= std::max(0.0, innerSumSq / innerArea - innerMean * innerMean);

	HandImageQuality& quality= hand.imageQuality;
	quality.meanLuma= (float)innerMean;
	quality.shadowClipRatio= (float)(shadowCount / innerArea);
	quality.highlightClipRatio= (float)(highlightCount / innerArea);
	quality.contrast= (float)std::sqrt(innerVariance);

	// Ring mean from the outer/inner sums (the band between the two boxes).
	// Internal contrast measures what the landmark regressor sees; separation
	// measures what the palm DETECTOR needs - a hand blending into the
	// background fails detection with perfectly good internal texture.
	const double outerArea= (double)grayOuter.rows * grayOuter.cols;
	const double ringArea= outerArea - innerArea;
	if (ringArea > 1.0)
	{
		const double ringMean= (cv::sum(grayOuter)[0] - innerSum) / ringArea;
		quality.backgroundSeparation= (float)std::abs(innerMean - ringMean);
	}

	// Noise = residual against a 3x3 median; sharpness = Laplacian variance of
	// the MEDIAN-FILTERED image. Splitting them matters: raw Laplacian
	// variance rewards sensor noise, scoring a noisy high-gain frame as
	// "sharp" in exactly the low-light case being diagnosed.
	cv::Mat median;
	cv::medianBlur(inner.clone(), median, 3);
	quality.noise= (float)cv::mean(cv::abs(inner - median))[0];

	cv::Mat laplacian;
	cv::Laplacian(median, laplacian, CV_16S, 3);
	cv::Scalar lapMean, lapStddev;
	cv::meanStdDev(laplacian, lapMean, lapStddev);
	quality.sharpness= (float)(lapStddev[0] * lapStddev[0]);

	quality.valid= true;
}
