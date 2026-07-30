#pragma once

#include "glm/ext/matrix_float4x4.hpp"
#include "glm/ext/vector_float3.hpp"

class GlLineRenderer;

// Free-function debug draw helpers, ported from MikanXR's MikanLineRenderer draw
// functions but re-targeted to queue into a GlLineRenderer instead of an
// IMkLineRenderer. Text-label variants are omitted (no text renderer).

void drawPoint(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& point, const glm::vec3& color,
			   const float size);
void drawSegment(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
				 const glm::vec3& color);
void drawSegment(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
				 const glm::vec3& colorStart, const glm::vec3& colorEnd);
void drawArrow(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
			   const float headFraction, const glm::vec3& color);

// Draws an XZ-plane grid centered on the transform origin
void drawGrid(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xSize, float zSize, int xSubDiv,
			  int zSubDiv, const glm::vec3& color);

void drawTransformedQuad(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xSize, float ySize,
						 const glm::vec3& color);
void drawTransformedCircle(GlLineRenderer& lineRenderer, const glm::mat4& transform, float radius,
						   const glm::vec3& color, int segmentCount= 0);

// RGB = XYZ axes
void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float scale);
void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xScale, float yScale,
						 float zScale);
void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, const glm::vec3& xColor, const glm::vec3& yColor, const glm::vec3& zColor);

void drawTransformedBox(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& halfExtents,
						const glm::vec3& color);
void drawTransformedBox(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& boxMin,
						const glm::vec3& boxMax, const glm::vec3& color);

// Camera frustum wireframe looking down -Z (FOV angles in degrees)
void drawTransformedFrustum(GlLineRenderer& lineRenderer, const glm::mat4& transform, const float hfovDegrees,
							const float vfovDegrees, const float zNear, const float zFar, const glm::vec3& color);
