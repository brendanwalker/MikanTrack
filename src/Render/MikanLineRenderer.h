#pragma once

#include "IMkGraphicsContext.h"

#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "glm/ext/vector_float4.hpp"
#include "glm/ext/matrix_float4x4.hpp"

void drawPoint(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& point,
			   const glm::vec3& color, const float size);
void drawSegment(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
				 const glm::vec3& end, const glm::vec3& color);
void drawSegment(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
				 const glm::vec3& end, const glm::vec3& colorStart, const glm::vec3& colorEnd);
void drawArrow(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
			   const glm::vec3& end, const float headFraction, const glm::vec3& color);
void drawGrid(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xSize, float ySize, int xSubDiv,
			  int ySubDiv, const glm::vec3& color);
void drawTransformedQuad(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xSize, float ySize,
						 const glm::vec3& color);
void drawTransformedCircle(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float radius,
						   const glm::vec3& color, int segmentCount= 0);
void drawTransformedSpiralArc(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float radius,
							  float radiusFractionPerCircle, float totalAngle, const glm::vec3& color,
							  int segmentCount= 0);
void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float scale,
						 bool drawLabels= false);
void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, bool drawLabels= false);
void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, const glm::vec3& xColor, const glm::vec3& yColor, const glm::vec3& zColor,
						 bool drawLabels= false);
void drawTransformedTriangle(IMkGraphicsContext* graphicsContext, const glm::mat4& transform,
							 const struct GlmTriangle& tri, const glm::vec3& color);
void drawTransformedBox(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& half_extents,
						const glm::vec3& color);
void drawTransformedBox(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& box_min,
						const glm::vec3& box_max, const glm::vec3& color);
void drawTransformedFrustum(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const float hfov_radians,
							const float vfov_radians, const float zNear, const float zFar, const glm::vec3& color);