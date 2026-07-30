#include "App.h"
#include "Colors.h"
#include "MikanCamera.h"
#include "MkError.h"
#include "MikanLineRenderer.h"
#include "IMkGraphicsContext.h"
#include "IMkLineRenderer.h"
#include "IMkShader.h"
#include "MkStateStack.h"
#include "MikanShaderCache.h"
#include "MikanTextRenderer.h"
#include "IMkVertexDefinition.h"
#include "MikanViewport.h"
#include "Logger.h"
#include "MathGLM.h"
#include "TextStyle.h"

#include "glm/ext/matrix_clip_space.hpp"

void drawPoint(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& point,
			   const glm::vec3& color, const float size)
{
	graphicsContext->getLineRenderer()->addPoint3d(transform, point, color, size);
}

void drawSegment(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
				 const glm::vec3& end, const glm::vec3& color)
{
	graphicsContext->getLineRenderer()->addSegment3d(transform, start, color, end, color);
}

void drawSegment(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
				 const glm::vec3& end, const glm::vec3& colorStart, const glm::vec3& colorEnd)
{
	graphicsContext->getLineRenderer()->addSegment3d(transform, start, colorStart, end, colorEnd);
}

void drawArrow(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& start,
			   const glm::vec3& end, const float headFraction, const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	const glm::vec3 headAxis= end - start;
	const float headSize= glm::length(headAxis) * headFraction;
	const glm::vec3 headOrigin= glm_vec3_lerp(end, start, headFraction);

	const glm::vec3 worldUp= glm::vec3(0, 1, 0);
	const glm::vec3 headForward= glm::normalize(headAxis);
	const glm::vec3 headLeft= glm::normalize(glm::cross(worldUp, headForward));
	const glm::vec3 headUp= glm::normalize(glm::cross(headForward, headLeft));

	const glm::vec3 headXPos= headOrigin - headLeft * headSize;
	const glm::vec3 headXNeg= headOrigin + headLeft * headSize;
	const glm::vec3 headYPos= headOrigin + headUp * headSize;
	const glm::vec3 headYNeg= headOrigin - headUp * headSize;

	lineRenderer->addSegment3d(transform, start, color, end, color);

	lineRenderer->addSegment3d(transform, headXPos, color, headYPos, color);
	lineRenderer->addSegment3d(transform, headYPos, color, headXNeg, color);
	lineRenderer->addSegment3d(transform, headXNeg, color, headYNeg, color);
	lineRenderer->addSegment3d(transform, headYNeg, color, headXPos, color);

	lineRenderer->addSegment3d(transform, headXPos, color, end, color);
	lineRenderer->addSegment3d(transform, headYPos, color, end, color);
	lineRenderer->addSegment3d(transform, headXNeg, color, end, color);
	lineRenderer->addSegment3d(transform, headYNeg, color, end, color);

	lineRenderer->addSegment3d(transform, headXPos, color, headXNeg, color);
	lineRenderer->addSegment3d(transform, headYPos, color, headYNeg, color);
}

void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float scale, bool drawLabels)
{
	drawTransformedAxes(graphicsContext, transform, scale, scale, scale, drawLabels);
}

void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, bool drawLabels)
{
	drawTransformedAxes(graphicsContext, transform, xScale, yScale, zScale, Colors::Red, Colors::Green, Colors::Blue,
						drawLabels);
}

void drawTransformedAxes(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xScale, float yScale,
						 float zScale, const glm::vec3& xColor, const glm::vec3& yColor, const glm::vec3& zColor,
						 bool drawLabels)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	glm::vec3 origin(0.f, 0.f, 0.f);
	glm::vec3 xAxis(xScale, 0.f, 0.f);
	glm::vec3 yAxis(0.f, yScale, 0.f);
	glm::vec3 zAxis(0.f, 0.f, zScale);

	lineRenderer->addSegment3d(transform, origin, Colors::Red, xAxis, xColor);
	lineRenderer->addSegment3d(transform, origin, Colors::Green, yAxis, yColor);
	lineRenderer->addSegment3d(transform, origin, Colors::Blue, zAxis, zColor);

	if (drawLabels)
	{
		TextStyle style= getDefaultTextStyle();
		drawTextAtWorldPosition(graphicsContext, style, glm::vec3(transform * glm::vec4(xAxis, 1.0f)), L"X");
		drawTextAtWorldPosition(graphicsContext, style, glm::vec3(transform * glm::vec4(yAxis, 1.0f)), L"Y");
		drawTextAtWorldPosition(graphicsContext, style, glm::vec3(transform * glm::vec4(zAxis, 1.0f)), L"Z");
	}
}

void drawTransformedCircle(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float radius,
						   const glm::vec3& color, int segmentCount)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

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

		lineRenderer->addSegment3d(transform, prevPoint, color, nextPoint, color);
		prevPoint= nextPoint;
	}
}

void drawTransformedSpiralArc(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float radius,
							  float radiusFractionPerCircle, float totalAngle, const glm::vec3& color, int segmentCount)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	float angleStep;
	if (segmentCount > 0)
	{
		angleStep= (k_real_two_pi / (float)segmentCount) * sgn(totalAngle);
	}
	else
	{
		static const float k_segmentMaxLength= 0.01f;
		static const float k_maxAngleStep= k_real_quarter_pi;
		angleStep= fminf(k_segmentMaxLength / radius, k_maxAngleStep) * sgn(totalAngle);
	}
	const float radiusStepPerFullSegment= -radius * radiusFractionPerCircle * fabsf(angleStep) / k_real_two_pi;
	const float absTotal= fabsf(totalAngle);
	const float absStep= fabsf(angleStep);

	float currentAngle= 0.f;
	float spiralRadius= radius;
	glm::vec3 prevPoint= glm::vec3(spiralRadius, 0.f, 0.f);

	// Draw start radial line
	lineRenderer->addSegment3d(transform, glm::vec3(0.f), color, prevPoint, color);

	// Draw the spiral arc, clamping the last segment to exactly totalAngle
	while (fabsf(currentAngle) < absTotal - 1e-6f)
	{
		const float remaining= absTotal - fabsf(currentAngle);
		const float stepFrac= (remaining < absStep) ? (remaining / absStep) : 1.f;
		const float nextAngle= currentAngle + angleStep * stepFrac;
		const glm::vec3 nextPoint= glm::vec3(cosf(nextAngle), 0.f, sinf(nextAngle)) * spiralRadius;

		lineRenderer->addSegment3d(transform, prevPoint, color, nextPoint, color);
		prevPoint= nextPoint;
		currentAngle= nextAngle;
		spiralRadius+= radiusStepPerFullSegment * stepFrac;
	}

	// Draw the end radial line
	lineRenderer->addSegment3d(transform, glm::vec3(0.f), color, prevPoint, color);
}

void drawGrid(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xSize, float zSize, int xSubDiv,
			  int zSubDiv, const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	int x0= -xSize / 2.f;
	int x1= xSize / 2.f;
	for (float z= -zSize / 2.f; z <= zSize / 2.f; z+= (zSize / (float)zSubDiv))
	{
		lineRenderer->addSegment3d(transform, glm::vec3(x0, 0.f, z), color, glm::vec3(x1, 0.f, z), color);
	}

	int z0= -zSize / 2.f;
	int z1= zSize / 2.f;
	for (float x= -xSize / 2.f; x <= xSize / 2.f; x+= (xSize / (float)xSubDiv))
	{
		lineRenderer->addSegment3d(transform, glm::vec3(x, 0.f, z0), color, glm::vec3(x, 0.f, z1), color);
	}
}

void drawTransformedQuad(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, float xSize, float ySize,
						 const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	const glm::vec3 p0(xSize / 2.f, ySize / 2.f, 0.f);
	const glm::vec3 p1(xSize / 2.f, -ySize / 2.f, 0.f);
	const glm::vec3 p2(-xSize / 2.f, -ySize / 2.f, 0.f);
	const glm::vec3 p3(-xSize / 2.f, ySize / 2.f, 0.f);

	lineRenderer->addSegment3d(transform, p0, color, p1, color);
	lineRenderer->addSegment3d(transform, p1, color, p2, color);
	lineRenderer->addSegment3d(transform, p2, color, p3, color);
	lineRenderer->addSegment3d(transform, p3, color, p0, color);
}

void drawTransformedTriangle(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const GlmTriangle& tri,
							 const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	lineRenderer->addSegment3d(transform, tri.v0, color, tri.v1, color);
	lineRenderer->addSegment3d(transform, tri.v1, color, tri.v2, color);
	lineRenderer->addSegment3d(transform, tri.v2, color, tri.v0, color);
}

void drawTransformedBox(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& half_extents,
						const glm::vec3& color)
{
	drawTransformedBox(graphicsContext, transform, -half_extents, half_extents, color);
}

void drawTransformedBox(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const glm::vec3& box_min,
						const glm::vec3& box_max, const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	const glm::vec3 v0(box_max.x, box_max.y, box_max.z);
	const glm::vec3 v1(box_min.x, box_max.y, box_max.z);
	const glm::vec3 v2(box_min.x, box_max.y, box_min.z);
	const glm::vec3 v3(box_max.x, box_max.y, box_min.z);
	const glm::vec3 v4(box_max.x, box_min.y, box_max.z);
	const glm::vec3 v5(box_min.x, box_min.y, box_max.z);
	const glm::vec3 v6(box_min.x, box_min.y, box_min.z);
	const glm::vec3 v7(box_max.x, box_min.y, box_min.z);

	lineRenderer->addSegment3d(transform, v0, color, v1, color);
	lineRenderer->addSegment3d(transform, v1, color, v2, color);
	lineRenderer->addSegment3d(transform, v2, color, v3, color);
	lineRenderer->addSegment3d(transform, v3, color, v0, color);

	lineRenderer->addSegment3d(transform, v4, color, v5, color);
	lineRenderer->addSegment3d(transform, v5, color, v6, color);
	lineRenderer->addSegment3d(transform, v6, color, v7, color);
	lineRenderer->addSegment3d(transform, v7, color, v4, color);

	lineRenderer->addSegment3d(transform, v0, color, v4, color);
	lineRenderer->addSegment3d(transform, v1, color, v5, color);
	lineRenderer->addSegment3d(transform, v2, color, v6, color);
	lineRenderer->addSegment3d(transform, v3, color, v7, color);
}

void drawTransformedFrustum(IMkGraphicsContext* graphicsContext, const glm::mat4& transform, const float hfov_radians,
							const float vfov_radians, const float zNear, const float zFar, const glm::vec3& color)
{
	IMkLineRenderer* lineRenderer= graphicsContext->getLineRenderer();

	const float HRatio= tanf(hfov_radians / 2.f);
	const float VRatio= tanf(vfov_radians / 2.f);

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

	lineRenderer->addSegment3d(transform, near0, color, near1, color);
	lineRenderer->addSegment3d(transform, near1, color, near2, color);
	lineRenderer->addSegment3d(transform, near2, color, near3, color);
	lineRenderer->addSegment3d(transform, near3, color, near0, color);

	lineRenderer->addSegment3d(transform, far0, color, far1, color);
	lineRenderer->addSegment3d(transform, far1, color, far2, color);
	lineRenderer->addSegment3d(transform, far2, color, far3, color);
	lineRenderer->addSegment3d(transform, far3, color, far0, color);

	lineRenderer->addSegment3d(transform, cameraOrigin, color, far0, color);
	lineRenderer->addSegment3d(transform, cameraOrigin, color, far1, color);
	lineRenderer->addSegment3d(transform, cameraOrigin, color, far2, color);
	lineRenderer->addSegment3d(transform, cameraOrigin, color, far3, color);

	lineRenderer->addSegment3d(transform, cameraOrigin, color, nearCenter, color);
}