#include "DebugDraw.h"
#include "Colors.h"
#include "GlLineRenderer.h"
#include "MathUtility.h"

#include "glm/geometric.hpp"

#include <cmath>

void drawPoint(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& point, const glm::vec3& color,
			   const float size)
{
	lineRenderer.addPoint3d(transform, point, color, size);
}

void drawSegment(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
				 const glm::vec3& color)
{
	lineRenderer.addSegment3d(transform, start, color, end, color);
}

void drawSegment(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
				 const glm::vec3& colorStart, const glm::vec3& colorEnd)
{
	lineRenderer.addSegment3d(transform, start, colorStart, end, colorEnd);
}

void drawArrow(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& start, const glm::vec3& end,
			   const float headFraction, const glm::vec3& color)
{
	const glm::vec3 headAxis= end - start;
	const float headSize= glm::length(headAxis) * headFraction;
	const glm::vec3 headOrigin= glm::mix(end, start, headFraction);

	const glm::vec3 worldUp= glm::vec3(0, 1, 0);
	const glm::vec3 headForward= glm::normalize(headAxis);
	const glm::vec3 headLeft= glm::normalize(glm::cross(worldUp, headForward));
	const glm::vec3 headUp= glm::normalize(glm::cross(headForward, headLeft));

	const glm::vec3 headXPos= headOrigin - headLeft * headSize;
	const glm::vec3 headXNeg= headOrigin + headLeft * headSize;
	const glm::vec3 headYPos= headOrigin + headUp * headSize;
	const glm::vec3 headYNeg= headOrigin - headUp * headSize;

	lineRenderer.addSegment3d(transform, start, color, end, color);

	lineRenderer.addSegment3d(transform, headXPos, color, headYPos, color);
	lineRenderer.addSegment3d(transform, headYPos, color, headXNeg, color);
	lineRenderer.addSegment3d(transform, headXNeg, color, headYNeg, color);
	lineRenderer.addSegment3d(transform, headYNeg, color, headXPos, color);

	lineRenderer.addSegment3d(transform, headXPos, color, end, color);
	lineRenderer.addSegment3d(transform, headYPos, color, end, color);
	lineRenderer.addSegment3d(transform, headXNeg, color, end, color);
	lineRenderer.addSegment3d(transform, headYNeg, color, end, color);

	lineRenderer.addSegment3d(transform, headXPos, color, headXNeg, color);
	lineRenderer.addSegment3d(transform, headYPos, color, headYNeg, color);
}

void drawGrid(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xSize, float zSize, int xSubDiv,
			  int zSubDiv, const glm::vec3& color)
{
	const float x0= -xSize / 2.f;
	const float x1= xSize / 2.f;
	for (float z= -zSize / 2.f; z <= zSize / 2.f; z+= (zSize / (float)zSubDiv))
	{
		lineRenderer.addSegment3d(transform, glm::vec3(x0, 0.f, z), color, glm::vec3(x1, 0.f, z), color);
	}

	const float z0= -zSize / 2.f;
	const float z1= zSize / 2.f;
	for (float x= -xSize / 2.f; x <= xSize / 2.f; x+= (xSize / (float)xSubDiv))
	{
		lineRenderer.addSegment3d(transform, glm::vec3(x, 0.f, z0), color, glm::vec3(x, 0.f, z1), color);
	}
}

void drawTransformedQuad(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xSize, float ySize,
						 const glm::vec3& color)
{
	const glm::vec3 p0(xSize / 2.f, ySize / 2.f, 0.f);
	const glm::vec3 p1(xSize / 2.f, -ySize / 2.f, 0.f);
	const glm::vec3 p2(-xSize / 2.f, -ySize / 2.f, 0.f);
	const glm::vec3 p3(-xSize / 2.f, ySize / 2.f, 0.f);

	lineRenderer.addSegment3d(transform, p0, color, p1, color);
	lineRenderer.addSegment3d(transform, p1, color, p2, color);
	lineRenderer.addSegment3d(transform, p2, color, p3, color);
	lineRenderer.addSegment3d(transform, p3, color, p0, color);
}

void drawTransformedCircle(GlLineRenderer& lineRenderer, const glm::mat4& transform, float radius,
						   const glm::vec3& color, int segmentCount)
{
	float angleStep;
	if (segmentCount > 0)
	{
		angleStep= k_real_two_pi / (float)segmentCount;
	}
	else
	{
		static const float k_segmentMaxLength= 0.01f;
		static const float k_maxAngleStep= k_real_quarter_pi;
		angleStep= fminf(k_segmentMaxLength / radius, k_maxAngleStep);
	}

	glm::vec3 prevPoint= glm::vec3(radius, 0.f, 0.f);
	for (float angle= angleStep; angle < k_real_two_pi; angle+= angleStep)
	{
		const glm::vec3 nextPoint= glm::vec3(cosf(angle), 0.f, sinf(angle)) * radius;

		lineRenderer.addSegment3d(transform, prevPoint, color, nextPoint, color);
		prevPoint= nextPoint;
	}
}

void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float scale)
{
	drawTransformedAxes(lineRenderer, transform, scale, scale, scale);
}

void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xScale, float yScale,
						 float zScale)
{
	drawTransformedAxes(lineRenderer, transform, xScale, yScale, zScale, Colors::Red, Colors::Green, Colors::Blue);
}

void drawTransformedAxes(GlLineRenderer& lineRenderer, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, const glm::vec3& xColor, const glm::vec3& yColor, const glm::vec3& zColor)
{
	glm::vec3 origin(0.f, 0.f, 0.f);
	glm::vec3 xAxis(xScale, 0.f, 0.f);
	glm::vec3 yAxis(0.f, yScale, 0.f);
	glm::vec3 zAxis(0.f, 0.f, zScale);

	lineRenderer.addSegment3d(transform, origin, Colors::Red, xAxis, xColor);
	lineRenderer.addSegment3d(transform, origin, Colors::Green, yAxis, yColor);
	lineRenderer.addSegment3d(transform, origin, Colors::Blue, zAxis, zColor);
}

void drawTransformedBox(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& halfExtents,
						const glm::vec3& color)
{
	drawTransformedBox(lineRenderer, transform, -halfExtents, halfExtents, color);
}

void drawTransformedBox(GlLineRenderer& lineRenderer, const glm::mat4& transform, const glm::vec3& boxMin,
						const glm::vec3& boxMax, const glm::vec3& color)
{
	const glm::vec3 v0(boxMax.x, boxMax.y, boxMax.z);
	const glm::vec3 v1(boxMin.x, boxMax.y, boxMax.z);
	const glm::vec3 v2(boxMin.x, boxMax.y, boxMin.z);
	const glm::vec3 v3(boxMax.x, boxMax.y, boxMin.z);
	const glm::vec3 v4(boxMax.x, boxMin.y, boxMax.z);
	const glm::vec3 v5(boxMin.x, boxMin.y, boxMax.z);
	const glm::vec3 v6(boxMin.x, boxMin.y, boxMin.z);
	const glm::vec3 v7(boxMax.x, boxMin.y, boxMin.z);

	lineRenderer.addSegment3d(transform, v0, color, v1, color);
	lineRenderer.addSegment3d(transform, v1, color, v2, color);
	lineRenderer.addSegment3d(transform, v2, color, v3, color);
	lineRenderer.addSegment3d(transform, v3, color, v0, color);

	lineRenderer.addSegment3d(transform, v4, color, v5, color);
	lineRenderer.addSegment3d(transform, v5, color, v6, color);
	lineRenderer.addSegment3d(transform, v6, color, v7, color);
	lineRenderer.addSegment3d(transform, v7, color, v4, color);

	lineRenderer.addSegment3d(transform, v0, color, v4, color);
	lineRenderer.addSegment3d(transform, v1, color, v5, color);
	lineRenderer.addSegment3d(transform, v2, color, v6, color);
	lineRenderer.addSegment3d(transform, v3, color, v7, color);
}

void drawTransformedFrustum(GlLineRenderer& lineRenderer, const glm::mat4& transform, const float hfovDegrees,
							const float vfovDegrees, const float zNear, const float zFar, const glm::vec3& color)
{
	const float HRatio= tanf(degrees_to_radians(hfovDegrees) / 2.f);
	const float VRatio= tanf(degrees_to_radians(vfovDegrees) / 2.f);

	const glm::vec3 cameraRight(1.f, 0.f, 0.f);
	const glm::vec3 cameraUp(0.f, 1.f, 0.f);
	const glm::vec3 cameraForward(0.f, 0.f, -1.f);
	const glm::vec3 cameraOrigin(0.f);

	const glm::vec3 nearX= cameraRight * zNear * HRatio;
	const glm::vec3 farX= cameraRight * zFar * HRatio;

	const glm::vec3 nearY= cameraUp * zNear * VRatio;
	const glm::vec3 farY= cameraUp * zFar * VRatio;

	const glm::vec3 nearZ= cameraForward * zNear;
	const glm::vec3 farZ= cameraForward * zFar;

	const glm::vec3 nearCenter= cameraOrigin + nearZ;
	const glm::vec3 near0= cameraOrigin + nearX + nearY + nearZ;
	const glm::vec3 near1= cameraOrigin - nearX + nearY + nearZ;
	const glm::vec3 near2= cameraOrigin - nearX - nearY + nearZ;
	const glm::vec3 near3= cameraOrigin + nearX - nearY + nearZ;

	const glm::vec3 far0= cameraOrigin + farX + farY + farZ;
	const glm::vec3 far1= cameraOrigin - farX + farY + farZ;
	const glm::vec3 far2= cameraOrigin - farX - farY + farZ;
	const glm::vec3 far3= cameraOrigin + farX - farY + farZ;

	lineRenderer.addSegment3d(transform, near0, color, near1, color);
	lineRenderer.addSegment3d(transform, near1, color, near2, color);
	lineRenderer.addSegment3d(transform, near2, color, near3, color);
	lineRenderer.addSegment3d(transform, near3, color, near0, color);

	lineRenderer.addSegment3d(transform, far0, color, far1, color);
	lineRenderer.addSegment3d(transform, far1, color, far2, color);
	lineRenderer.addSegment3d(transform, far2, color, far3, color);
	lineRenderer.addSegment3d(transform, far3, color, far0, color);

	lineRenderer.addSegment3d(transform, cameraOrigin, color, far0, color);
	lineRenderer.addSegment3d(transform, cameraOrigin, color, far1, color);
	lineRenderer.addSegment3d(transform, cameraOrigin, color, far2, color);
	lineRenderer.addSegment3d(transform, cameraOrigin, color, far3, color);

	lineRenderer.addSegment3d(transform, cameraOrigin, color, nearCenter, color);
}
