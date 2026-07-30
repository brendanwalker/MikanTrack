#include "SsdAnchors.h"

#include <algorithm>
#include <cmath>

SsdAnchorsConfig makePalmDetectorAnchorConfig()
{
	SsdAnchorsConfig config;
	config.inputWidth= 192;
	config.inputHeight= 192;
	config.strides= {8, 16, 16, 16};
	config.minScale= 0.1484375f;
	config.maxScale= 0.75f;
	return config;
}

SsdAnchorsConfig makePersonDetectorAnchorConfig()
{
	SsdAnchorsConfig config;
	config.inputWidth= 224;
	config.inputHeight= 224;
	config.strides= {8, 16, 32, 32, 32};
	config.minScale= 0.1484375f;
	config.maxScale= 0.75f;
	return config;
}

static float calculateScale(float minScale, float maxScale, int strideIndex, int numStrides)
{
	if (numStrides == 1)
		return (minScale + maxScale) * 0.5f;

	return minScale + (maxScale - minScale) * (float)strideIndex / (float)(numStrides - 1);
}

std::vector<glm::vec2> generateSsdAnchors(const SsdAnchorsConfig& config)
{
	std::vector<glm::vec2> anchors;

	const int numStrides= (int)config.strides.size();
	int layerId= 0;
	while (layerId < numStrides)
	{
		// Merge all consecutive layers with the same stride; each contributes
		// aspectRatios.size() (+1 interpolated) anchors per grid cell
		int anchorsPerCell= 0;
		int lastSameStrideLayer= layerId;
		while (lastSameStrideLayer < numStrides &&
			   config.strides[lastSameStrideLayer] == config.strides[layerId])
		{
			anchorsPerCell+= (int)config.aspectRatios.size();
			if (config.interpolatedScaleAspectRatio > 0.f)
				anchorsPerCell+= 1;
			lastSameStrideLayer++;
		}

		const int stride= config.strides[layerId];
		const int featureMapWidth= (int)std::ceil((float)config.inputWidth / (float)stride);
		const int featureMapHeight= (int)std::ceil((float)config.inputHeight / (float)stride);

		for (int y= 0; y < featureMapHeight; ++y)
		{
			for (int x= 0; x < featureMapWidth; ++x)
			{
				for (int anchorId= 0; anchorId < anchorsPerCell; ++anchorId)
				{
					const float xCenter= ((float)x + config.anchorOffsetX) / (float)featureMapWidth;
					const float yCenter= ((float)y + config.anchorOffsetY) / (float)featureMapHeight;
					anchors.push_back(glm::vec2(xCenter, yCenter));
				}
			}
		}

		layerId= lastSameStrideLayer;
	}

	return anchors;
}

static float boxIntersectionOverUnion(const SsdDetection& a, const SsdDetection& b)
{
	const float interMinX= std::max(a.boxMin.x, b.boxMin.x);
	const float interMinY= std::max(a.boxMin.y, b.boxMin.y);
	const float interMaxX= std::min(a.boxMax.x, b.boxMax.x);
	const float interMaxY= std::min(a.boxMax.y, b.boxMax.y);

	const float interW= std::max(0.f, interMaxX - interMinX);
	const float interH= std::max(0.f, interMaxY - interMinY);
	const float interArea= interW * interH;

	const float areaA= std::max(0.f, a.boxMax.x - a.boxMin.x) * std::max(0.f, a.boxMax.y - a.boxMin.y);
	const float areaB= std::max(0.f, b.boxMax.x - b.boxMin.x) * std::max(0.f, b.boxMax.y - b.boxMin.y);
	const float unionArea= areaA + areaB - interArea;

	return unionArea > 0.f ? interArea / unionArea : 0.f;
}

void weightedNonMaxSuppression(std::vector<SsdDetection>& ioDetections, float iouThreshold)
{
	if (ioDetections.empty())
		return;

	std::vector<SsdDetection> sorted= ioDetections;
	std::sort(
		sorted.begin(), sorted.end(),
		[](const SsdDetection& a, const SsdDetection& b) { return a.score > b.score; });

	std::vector<SsdDetection> kept;
	std::vector<bool> consumed(sorted.size(), false);

	for (size_t i= 0; i < sorted.size(); ++i)
	{
		if (consumed[i])
			continue;

		// Blend every remaining candidate that overlaps the current top
		// candidate, weighted by score (MediaPipe weighted NMS)
		SsdDetection blended= sorted[i];
		glm::vec2 boxMinAccum(0.f);
		glm::vec2 boxMaxAccum(0.f);
		std::vector<glm::vec2> keypointAccum(sorted[i].keypoints.size(), glm::vec2(0.f));
		float totalWeight= 0.f;

		for (size_t j= i; j < sorted.size(); ++j)
		{
			if (consumed[j])
				continue;
			if (j != i && boxIntersectionOverUnion(sorted[i], sorted[j]) <= iouThreshold)
				continue;

			consumed[j]= true;
			const float weight= sorted[j].score;
			totalWeight+= weight;
			boxMinAccum+= sorted[j].boxMin * weight;
			boxMaxAccum+= sorted[j].boxMax * weight;
			for (size_t k= 0; k < keypointAccum.size() && k < sorted[j].keypoints.size(); ++k)
				keypointAccum[k]+= sorted[j].keypoints[k] * weight;
		}

		if (totalWeight > 0.f)
		{
			blended.boxMin= boxMinAccum / totalWeight;
			blended.boxMax= boxMaxAccum / totalWeight;
			for (size_t k= 0; k < keypointAccum.size(); ++k)
				blended.keypoints[k]= keypointAccum[k] / totalWeight;
		}

		kept.push_back(std::move(blended));
	}

	ioDetections= std::move(kept);
}
