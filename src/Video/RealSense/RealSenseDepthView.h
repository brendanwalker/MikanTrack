#pragma once

#include <cstdint>

#include "glm/ext/vector_float3.hpp"

// Neutral (no librealsense types) description of one depth frame plus the
// calibration needed to look up metric depth at a COLOR-image pixel. Filled
// by the RealSense capture backend; consumed by the tracking layer.
//
// The depth image is NOT pre-aligned to color (full-frame alignment costs a
// per-pixel pass we don't need): sampleCameraPointAtColorPixel() projects a
// color pixel onto the depth image through the factory depth<->color
// calibration, which is exact and only pays for the pixels we actually
// query (21 landmarks).
struct RealSenseDepthView
{
	bool valid= false;

	const uint16_t* depthData= nullptr; // Z16 row-major
	int depthWidth= 0;
	int depthHeight= 0;
	float depthUnitsMeters= 0.001f; // meters per Z16 unit

	// Factory intrinsics (Brown-Conrady coefficient layout mirrors
	// rs2_intrinsics; model index: 0=none,2=inverse-BC,4=BC as in rs2_distortion)
	struct Intrinsics
	{
		int width= 0, height= 0;
		float fx= 0.f, fy= 0.f, ppx= 0.f, ppy= 0.f;
		int model= 0;
		float coeffs[5]= {};
	};
	Intrinsics colorIntrinsics;
	Intrinsics depthIntrinsics;

	// depth->color rigid transform (row-major 3x3 rotation + translation,
	// meters; matches rs2_extrinsics layout)
	float depthToColorRotation[9]= {1, 0, 0, 0, 1, 0, 0, 0, 1};
	float depthToColorTranslation[3]= {0, 0, 0};

	// Looks up metric depth for a color pixel and returns the 3D point in
	// the COLOR camera's frame (OpenCV convention, meters) - i.e. exactly
	// the cameraPoints space the tracking layer uses. Median-samples a small
	// window around the projected depth pixel and rejects holes.
	// minDepthM/maxDepthM bound plausible hand distances (rejects the desk
	// bleeding through around finger silhouettes and near-field garbage).
	// Returns false when no valid depth is found (caller falls back).
	bool sampleCameraPointAtColorPixel(float colorU, float colorV, float minDepthM, float maxDepthM,
									   glm::vec3& outCameraPoint) const;
};
