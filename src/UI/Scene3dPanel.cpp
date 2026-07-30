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

Scene3dPanel::Scene3dPanel()
	: m_frameBuffer(std::make_unique<GlFrameBuffer>())
	, m_lineRenderer(std::make_unique<GlLineRenderer>())
	, m_camera(std::make_unique<OrbitCamera>())
{
	m_camera->setOrbitLocation(30.f, -40.f, 1.8f);
}

Scene3dPanel::~Scene3dPanel()= default;

void Scene3dPanel::draw(const TrackingFrameResult& result, const glm::mat4& cameraToWorld, bool bHasExtrinsics,
						const MikanMonoIntrinsics* intrinsics)
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

	renderScene(result, cameraToWorld, bHasExtrinsics, intrinsics, (float)fbWidth / (float)fbHeight);

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

void Scene3dPanel::renderScene(const TrackingFrameResult& result, const glm::mat4& cameraToWorld, bool bHasExtrinsics,
							   const MikanMonoIntrinsics* intrinsics, float aspect)
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

	// Camera frustum (cameraToWorld maps CV-convention camera space to world;
	// the frustum helper draws in GL convention looking down -Z)
	if (bHasExtrinsics && intrinsics != nullptr)
	{
		const glm::mat4 cameraXform= k_displayFromWorld * cameraToWorld * k_glFromCvFlip;
		drawTransformedFrustum(
			*m_lineRenderer, cameraXform,
			(float)intrinsics->hfov, (float)intrinsics->vfov,
			0.05f, 1.5f,
			Colors::Yellow);
		drawTransformedAxes(*m_lineRenderer, cameraXform, 0.05f);
	}

	// Hand skeletons
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedHand& hand= result.hands[sideIndex];
		if (!hand.tracked)
			continue;

		const bool bUseWorld= hand.hasWorldSpace;
		if (!bUseWorld && !hand.hasCameraSpace)
			continue;

		// World-space points draw under the display rotation. Camera-space
		// points are OpenCV convention: with extrinsics map them to world,
		// otherwise just flip to GL orientation so they display upright.
		const glm::mat4 xform= bUseWorld
			? k_displayFromWorld
			: (k_displayFromWorld * (bHasExtrinsics ? cameraToWorld : k_glFromCvFlip));
		const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points= bUseWorld ? hand.worldPoints : hand.cameraPoints;
		const glm::vec3 color= hand.side == eHandSide::Left ? Colors::CornflowerBlue : Colors::Red;

		for (int i= 0; i < HAND_CONNECTION_COUNT; ++i)
		{
			drawSegment(*m_lineRenderer, xform,
						points[HAND_CONNECTIONS[i][0]],
						points[HAND_CONNECTIONS[i][1]],
						color);
		}
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
		{
			drawPoint(*m_lineRenderer, xform, points[i], Colors::White, 4.f);
		}
	}

	// Forearms
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const TrackedArm& arm= result.arms[sideIndex];
		if (!arm.valid)
			continue;

		const bool bUseWorld= arm.hasWorldSpace;
		if (!bUseWorld && !arm.hasCameraSpace)
			continue;

		const glm::mat4 xform= bUseWorld
			? k_displayFromWorld
			: (k_displayFromWorld * (bHasExtrinsics ? cameraToWorld : k_glFromCvFlip));
		const glm::vec3& elbow= bUseWorld ? arm.elbowWorld : arm.elbowCamera;
		const glm::vec3& wrist= bUseWorld ? arm.wristWorld : arm.wristCamera;
		const glm::vec3 color= (eHandSide)sideIndex == eHandSide::Left ? Colors::CornflowerBlue : Colors::Red;

		drawSegment(*m_lineRenderer, xform, elbow, wrist, color);
		drawPoint(*m_lineRenderer, xform, elbow, color, 6.f);
	}

	m_lineRenderer->render3d(m_camera->getViewProjection());

	m_frameBuffer->unbindFrameBuffer();
}
