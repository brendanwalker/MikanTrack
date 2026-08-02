#include "RealSenseDepthView.h"

#include <algorithm>
#include <cmath>

// The projection math mirrors librealsense's rsutil.h
// (rs2_project_color_pixel_to_depth_pixel and friends), reimplemented on the
// neutral structs so the tracking layer stays free of librealsense headers.
// Distortion model indices match rs2_distortion:
//   0 = none, 2 = inverse Brown-Conrady (color streams),
//   4 = Brown-Conrady (D400 depth streams; coefficients are zero in practice)

namespace
{
// project a 3D point in a stream's own frame to its pixel coordinates
void projectPoint(const RealSenseDepthView::Intrinsics& intrin, const glm::vec3& point, float& outU, float& outV)
{
	float x= point.x / point.z;
	float y= point.y / point.z;

	if (intrin.model == 2 /* inverse Brown-Conrady, forward-applied on projection */)
	{
		const float r2= x * x + y * y;
		const float f= 1.f + intrin.coeffs[0] * r2 + intrin.coeffs[1] * r2 * r2 + intrin.coeffs[4] * r2 * r2 * r2;
		const float xd= x * f + 2.f * intrin.coeffs[2] * x * y + intrin.coeffs[3] * (r2 + 2.f * x * x);
		const float yd= y * f + 2.f * intrin.coeffs[3] * x * y + intrin.coeffs[2] * (r2 + 2.f * y * y);
		x= xd;
		y= yd;
	}

	outU= x * intrin.fx + intrin.ppx;
	outV= y * intrin.fy + intrin.ppy;
}

// deproject a pixel + metric depth to a 3D point in the stream's own frame
glm::vec3 deprojectPixel(const RealSenseDepthView::Intrinsics& intrin, float u, float v, float depthM)
{
	// Depth streams use Brown-Conrady with zero coefficients on D400s;
	// treating it as an ideal pinhole matches librealsense's own fast path
	const float x= (u - intrin.ppx) / intrin.fx;
	const float y= (v - intrin.ppy) / intrin.fy;
	return glm::vec3(x * depthM, y * depthM, depthM);
}

glm::vec3 transformDepthToColor(const RealSenseDepthView& view, const glm::vec3& p)
{
	const float* r= view.depthToColorRotation;
	const float* t= view.depthToColorTranslation;
	// rs2_extrinsics rotation is column-major: out = R * p with columns r[0..2], r[3..5], r[6..8]
	return glm::vec3(
		r[0] * p.x + r[3] * p.y + r[6] * p.z + t[0],
		r[1] * p.x + r[4] * p.y + r[7] * p.z + t[1],
		r[2] * p.x + r[5] * p.y + r[8] * p.z + t[2]);
}
} // namespace

bool RealSenseDepthView::sampleCameraPointAtColorPixel(float colorU, float colorV, float minDepthM,
													   float maxDepthM, glm::vec3& outCameraPoint) const
{
	if (!valid || depthData == nullptr || depthWidth <= 0 || depthHeight <= 0)
		return false;

	// Find the depth pixel corresponding to this color pixel. The exact
	// correspondence depends on the unknown depth, so search along the
	// depth-pixel locus for the candidate depth range (librealsense's
	// rs2_project_color_pixel_to_depth_pixel approach) and keep the point
	// whose round-trip projection lands closest to the query color pixel.
	float bestDistSq= 1e12f;
	glm::vec3 bestDepthSpacePoint(0.f);
	bool bFound= false;

	constexpr int kSearchSteps= 24;
	for (int step= 0; step <= kSearchSteps; ++step)
	{
		const float candidateDepth= minDepthM + (maxDepthM - minDepthM) * (float)step / (float)kSearchSteps;

		// color pixel + assumed depth -> point in color frame -> depth frame -> depth pixel
		// (color->depth transform is the inverse of depthToColor; rotation transpose)
		glm::vec3 colorSpace= deprojectPixel(colorIntrinsics, colorU, colorV, candidateDepth);
		const float* r= depthToColorRotation;
		const float* t= depthToColorTranslation;
		const glm::vec3 d(colorSpace.x - t[0], colorSpace.y - t[1], colorSpace.z - t[2]);
		const glm::vec3 depthSpace( // R^T * d (column-major R -> transpose = row access)
			r[0] * d.x + r[1] * d.y + r[2] * d.z,
			r[3] * d.x + r[4] * d.y + r[5] * d.z,
			r[6] * d.x + r[7] * d.y + r[8] * d.z);
		if (depthSpace.z < 1e-4f)
			continue;

		float depthU= 0.f, depthV= 0.f;
		projectPoint(depthIntrinsics, depthSpace, depthU, depthV);
		const int px= (int)std::lround(depthU);
		const int py= (int)std::lround(depthV);
		if (px < 1 || px >= depthWidth - 1 || py < 1 || py >= depthHeight - 1)
			continue;

		// 3x3 median of valid, in-range samples (rejects speckle + holes)
		float samples[9];
		int sampleCount= 0;
		for (int dy= -1; dy <= 1; ++dy)
		{
			for (int dx= -1; dx <= 1; ++dx)
			{
				const uint16_t raw= depthData[(py + dy) * depthWidth + (px + dx)];
				if (raw == 0)
					continue;
				const float meters= (float)raw * depthUnitsMeters;
				if (meters < minDepthM || meters > maxDepthM)
					continue;
				samples[sampleCount++]= meters;
			}
		}
		if (sampleCount < 3)
			continue;
		std::nth_element(samples, samples + sampleCount / 2, samples + sampleCount);
		const float measuredDepth= samples[sampleCount / 2];

		// Reject if the measured depth wildly disagrees with the candidate
		// (we'd be reading a different surface along the epipolar locus)
		if (fabsf(measuredDepth - candidateDepth) > (maxDepthM - minDepthM) / (float)kSearchSteps + 0.03f)
			continue;

		// Score by round-trip: measured depth pixel -> 3D -> color pixel
		const glm::vec3 measuredDepthSpace= deprojectPixel(depthIntrinsics, (float)px, (float)py, measuredDepth);
		const glm::vec3 measuredColorSpace= transformDepthToColor(*this, measuredDepthSpace);
		if (measuredColorSpace.z < 1e-4f)
			continue;
		float roundU= 0.f, roundV= 0.f;
		projectPoint(colorIntrinsics, measuredColorSpace, roundU, roundV);
		const float distSq= (roundU - colorU) * (roundU - colorU) + (roundV - colorV) * (roundV - colorV);
		if (distSq < bestDistSq)
		{
			bestDistSq= distSq;
			bestDepthSpacePoint= measuredDepthSpace;
			bFound= true;
		}
	}

	// Accept only if the round-trip lands near the queried pixel (a few px:
	// beyond that we sampled a different surface, e.g. desk past a fingertip)
	if (!bFound || bestDistSq > 6.f * 6.f)
		return false;

	outCameraPoint= transformDepthToColor(*this, bestDepthSpacePoint);
	return outCameraPoint.z > 1e-4f;
}
