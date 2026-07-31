#include "Scene3dPanel.h"

#include "GL/glew.h"

#include "imgui.h"

#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"

#include "Colors.h"
#include "DebugDraw.h"
#include "GlFrameBuffer.h"
#include "GlLineRenderer.h"
#include "MikanVideoSourceTypes.h"
#include "OrbitCamera.h"

// World is Z-up (marker plane = XY); the renderer/orbit camera are Y-up.
// displayFromWorld rotates world +Z to display +Y: (x,y,z) -> (x, z, -y)
static const glm::mat4 k_displayFromWorld=
	glm::rotate(glm::mat4(1.f), -glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f));

// OpenCV camera convention (+Y down, +Z forward) <-> GL camera convention
// (+Y up, -Z forward): 180-degree rotation about X. Its own inverse.
static const glm::mat4 k_glFromCvFlip(
	glm::vec4(1, 0, 0, 0),
	glm::vec4(0, -1, 0, 0),
	glm::vec4(0, 0, -1, 0),
	glm::vec4(0, 0, 0, 1));

// Distinct tint per camera for frustums + dimmed per-camera skeletons
static const glm::vec3 k_cameraColors[4]= {
	glm::vec3(1.f, 0.9f, 0.3f),  // yellow
	glm::vec3(0.4f, 0.9f, 1.f),  // cyan
	glm::vec3(1.f, 0.5f, 0.9f),  // magenta
	glm::vec3(0.6f, 1.f, 0.5f),  // green
};

Scene3dPanel::Scene3dPanel()
	: m_frameBuffer(std::make_unique<GlFrameBuffer>())
	, m_lineRenderer(std::make_unique<GlLineRenderer>())
	, m_camera(std::make_unique<OrbitCamera>())
{
	m_camera->setOrbitLocation(30.f, -40.f, 1.8f);
}

Scene3dPanel::~Scene3dPanel()= default;

void Scene3dPanel::draw(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
						const std::vector<const TrackingFrameResult*>& perCameraResults)
{
	if (!ImGui::Begin("3D Scene"))
	{
		ImGui::End();
		return;
	}

	const ImVec2 panelSize= ImGui::GetContentRegionAvail();
	const uint16_t fbWidth= (uint16_t)std::max(64.f, panelSize.x);
	const uint16_t fbHeight= (uint16_t)std::max(64.f, panelSize.y);

	if (!m_bRenderInitialized)
	{
		m_bRenderInitialized= m_frameBuffer->init(fbWidth, fbHeight) && m_lineRenderer->startup();
		if (!m_bRenderInitialized)
		{
			ImGui::TextDisabled("3D renderer failed to initialize (see log)");
			ImGui::End();
			return;
		}
	}
	m_frameBuffer->resize(fbWidth, fbHeight);

	renderScene(fusedResult, cameras, perCameraResults, (float)fbWidth / (float)fbHeight);

	// FBO textures are bottom-up; flip V
	ImGui::Image(
		(ImTextureID)(intptr_t)m_frameBuffer->getColorTextureId(),
		ImVec2((float)fbWidth, (float)fbHeight),
		ImVec2(0, 1), ImVec2(1, 0));

	// Orbit interaction on the image item
	if (ImGui::IsItemHovered())
	{
		ImGuiIO& io= ImGui::GetIO();

		if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			m_camera->adjustOrbitAngles(io.MouseDelta.x * 0.4f, -io.MouseDelta.y * 0.4f);

		if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle))
		{
			// Pan in the display-space horizontal plane
			const float panScale= m_camera->getOrbitRadius() * 0.002f;
			m_camera->adjustOrbitTargetPosition(glm::vec3(-io.MouseDelta.x * panScale, io.MouseDelta.y * panScale, 0.f));
		}

		if (io.MouseWheel != 0.f)
			m_camera->adjustOrbitRadius(-io.MouseWheel * 0.15f);
	}

	ImGui::End();
}

void Scene3dPanel::drawSkeleton(const TrackingFrameResult& result, float brightness, const glm::vec3* colorOverride)
{
	// Hand skeletons (world space only in this panel)
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedHand& hand= result.hands[sideIndex];
		if (!hand.tracked || !hand.hasWorldSpace)
			continue;

		const glm::vec3 baseColor=
			colorOverride != nullptr ? *colorOverride
									 : (hand.side == eHandSide::Left ? Colors::CornflowerBlue : Colors::Red);
		const glm::vec3 color= baseColor * brightness;

		for (int i= 0; i < HAND_CONNECTION_COUNT; ++i)
		{
			drawSegment(*m_lineRenderer, k_displayFromWorld,
						hand.worldPoints[HAND_CONNECTIONS[i][0]],
						hand.worldPoints[HAND_CONNECTIONS[i][1]],
						color);
		}
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			drawPoint(*m_lineRenderer, k_displayFromWorld, hand.worldPoints[i], Colors::White * brightness, 4.f);
		}
	}

	// Forearms
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedArm& arm= result.arms[sideIndex];
		if (!arm.valid || !arm.hasWorldSpace)
			continue;

		const glm::vec3 baseColor=
			colorOverride != nullptr
				? *colorOverride
				: ((eHandSide)sideIndex == eHandSide::Left ? Colors::CornflowerBlue : Colors::Red);
		const glm::vec3 color= baseColor * brightness;

		drawSegment(*m_lineRenderer, k_displayFromWorld, arm.elbowWorld, arm.wristWorld, color);
		drawPoint(*m_lineRenderer, k_displayFromWorld, arm.elbowWorld, color, 6.f);
	}
}

void Scene3dPanel::renderScene(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
							   const std::vector<const TrackingFrameResult*>& perCameraResults, float aspect)
{
	m_camera->setPerspectiveProjection(50.f, aspect, 0.01f, 100.f);

	m_frameBuffer->bindFrameBuffer();
	glClearColor(0.05f, 0.05f, 0.07f, 1.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	// Marker plane grid (world XY plane -> display XZ plane, which is what drawGrid expects)
	drawGrid(*m_lineRenderer, glm::mat4(1.f), 2.f, 2.f, 20, 20, Colors::DarkGray);

	// Marker axes at the world origin
	drawTransformedAxes(*m_lineRenderer, k_displayFromWorld, 0.1f);

	// One frustum per calibrated camera (cameraToWorld maps CV-convention
	// camera space to world; the frustum helper draws in GL convention)
	for (size_t cameraIndex= 0; cameraIndex < cameras.size(); ++cameraIndex)
	{
		const SceneCameraView& view= cameras[cameraIndex];
		if (!view.bHasExtrinsics || view.intrinsics == nullptr)
			continue;

		const glm::vec3& color= k_cameraColors[cameraIndex % 4];
		const glm::mat4 cameraXform= k_displayFromWorld * view.cameraToWorld * k_glFromCvFlip;
		drawTransformedFrustum(
			*m_lineRenderer, cameraXform,
			(float)view.intrinsics->hfov, (float)view.intrinsics->vfov,
			0.05f, 0.5f,
			color);
		drawTransformedAxes(*m_lineRenderer, cameraXform, 0.05f);
	}

	// Dimmed per-camera skeletons (world-space agreement check across cameras)
	if (m_bShowPerCameraSkeletons)
	{
		for (size_t cameraIndex= 0; cameraIndex < perCameraResults.size(); ++cameraIndex)
		{
			if (perCameraResults[cameraIndex] != nullptr)
			{
				const glm::vec3 color= k_cameraColors[cameraIndex % 4];
				drawSkeleton(*perCameraResults[cameraIndex], 0.45f, &color);
			}
		}
	}

	// Fused skeleton, full brightness
	drawSkeleton(fusedResult, 1.f, nullptr);

	m_lineRenderer->render3d(m_camera->getViewProjection());

	m_frameBuffer->unbindFrameBuffer();
}
