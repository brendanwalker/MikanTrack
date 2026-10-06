#include "Scene3dPanel.h"

#include "GL/glew.h"

#include "imgui.h"

#include "LocText.h"

#include "glm/ext/matrix_transform.hpp"
#include "glm/gtc/constants.hpp"

#include "AppConfig.h"
#include "AvatarSkeleton.h"
#include "Colors.h"
#include "DebugDraw.h"
#include "GlFrameBuffer.h"
#include "GlLineRenderer.h"
#include "GlSkinnedMeshRenderer.h"
#include "HandPoseModel.h"
#include "MikanVideoSourceTypes.h"
#include "OrbitCamera.h"

#include "glm/gtc/quaternion.hpp"

static const glm::mat4 k_displayFromWorld=
	glm::rotate(glm::mat4(1.f), -glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f));

const glm::mat4& displayFromWorld()
{
	return k_displayFromWorld;
}

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
	, m_meshRenderer(std::make_unique<GlSkinnedMeshRenderer>())
{
	m_camera->setOrbitLocation(30.f, -40.f, 1.8f);
}

void Scene3dPanel::setAvatar(std::shared_ptr<const AvatarModel> model, std::shared_ptr<const AvatarSkeleton> skeleton,
							 uint32_t generation)
{
	// The face map follows the model, so it is rebuilt only when that changes
	if (model != m_avatarModel || generation != m_avatarGeneration)
	{
		m_faceMorphs= model != nullptr ? std::make_unique<AvatarFaceMorphs>(*model, AvatarFaceMap::build(*model))
									   : nullptr;
	}
	m_avatarModel= std::move(model);
	m_avatarSkeleton= std::move(skeleton);
	m_avatarGeneration= generation;
}

void Scene3dPanel::setAvatarFace(const TrackingFrameResult::FacePose* face)
{
	m_bHasAvatarFace= face != nullptr && face->present;
	if (m_bHasAvatarFace)
		m_avatarFace= *face;
}

void Scene3dPanel::setAvatarPose(const AvatarPose* pose)
{
	m_bHasAvatarPose= pose != nullptr && pose->valid;
	if (m_bHasAvatarPose)
		m_avatarPose= *pose;
}

Scene3dPanel::~Scene3dPanel()= default;

void Scene3dPanel::draw(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
						const std::vector<const TrackingFrameResult*>& perCameraResults, SceneGizmos& gizmos)
{
	for (SceneGizmos::ElbowHint& hint : gizmos.elbowHints)
		hint.bDragging= false;

	if (!ImGui::Begin(locWindowTitle("windows.scene3d")))
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
			ImGui::TextDisabled("%s", locText("scene3dPanel.rendererInitFailedText"));
			ImGui::End();
			return;
		}
	}
	m_frameBuffer->resize(fbWidth, fbHeight);

	renderScene(fusedResult, cameras, perCameraResults, gizmos, (float)fbWidth / (float)fbHeight);

	// FBO textures are bottom-up; flip V
	ImGui::Image(
		(ImTextureID)(intptr_t)m_frameBuffer->getColorTextureId(),
		ImVec2((float)fbWidth, (float)fbHeight),
		ImVec2(0, 1), ImVec2(1, 0));
	const ImVec2 imageMin= ImGui::GetItemRectMin();
	const bool bGizmoOwnsMouse= updateGizmoDrag(gizmos, glm::vec2(imageMin.x, imageMin.y),
												glm::vec2((float)fbWidth, (float)fbHeight));

	// Orbit interaction on the image item
	if (ImGui::IsItemHovered())
	{
		ImGuiIO& io= ImGui::GetIO();

		if (!bGizmoOwnsMouse && ImGui::IsMouseDragging(ImGuiMouseButton_Left))
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
	// Forward-kinematics render from the parametric pose: palm frame axes +
	// finger chains rebuilt from skeleton geometry and bend angles. This is
	// the same forward kinematics the VMC finger bones are built from, so
	// what you see here is what the stream carries.
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const HandPose& pose= result.poses[sideIndex];
		if (!pose.tracked || !pose.hasWorldPose)
			continue;

		const glm::vec3 baseColor=
			colorOverride != nullptr ? *colorOverride
									 : (pose.side == eHandSide::Left ? Colors::CornflowerBlue : Colors::Red);
		const glm::vec3 color= baseColor * brightness;

		// Palm transform in world space
		glm::mat4 palmTransform= glm::mat4_cast(pose.palmOrientationWorld);
		palmTransform[3]= glm::vec4(pose.palmPositionWorld, 1.f);

		// Palm frame axes (small: X toward fingers, Z out of the palm)
		drawTransformedAxes(*m_lineRenderer, k_displayFromWorld * palmTransform, 0.03f);

		// Palm outline: wrist -> each finger base
		const glm::vec3& middleBase= pose.skeleton.baseInPalm[(int)eFinger::Middle];
		const glm::vec3 wristInPalm(-middleBase.x, 0.f, 0.f);
		const glm::vec3 wristWorld= glm::vec3(palmTransform * glm::vec4(wristInPalm, 1.f));

		std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
		HandPoseModel::buildFingerJoints(palmTransform, pose.skeleton, pose.fingers, joints);

		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			drawSegment(*m_lineRenderer, k_displayFromWorld, wristWorld, joints[finger][0], color);
			for (int joint= 0; joint < 3; ++joint)
				drawSegment(*m_lineRenderer, k_displayFromWorld, joints[finger][joint], joints[finger][joint + 1], color);
			for (int joint= 0; joint < 4; ++joint)
				drawPoint(*m_lineRenderer, k_displayFromWorld, joints[finger][joint], Colors::White * brightness, 4.f);
		}

		// Forearm from the wrist IMU: wrist -> elbow. The DIRECTION is
		// measured, so this is a rigid extrapolation rather than the guess
		// the old geometric elbow estimate was; only the length is assumed.
		if (pose.hasForearmPose)
		{
			const glm::vec3 elbowWorld= pose.getElbowPositionWorld(m_forearmLengthMeters);
			drawSegment(*m_lineRenderer, k_displayFromWorld, wristWorld, elbowWorld, color);
			drawPoint(*m_lineRenderer, k_displayFromWorld, elbowWorld, Colors::Yellow * brightness, 7.f);

			// Forearm frame axes at the elbow, so a twisted forearm
			// (pronation/supination, which the IMU does observe) is visible
			glm::mat4 forearmTransform= glm::mat4_cast(pose.forearmOrientationWorld);
			forearmTransform[3]= glm::vec4(elbowWorld, 1.f);
			drawTransformedAxes(*m_lineRenderer, k_displayFromWorld * forearmTransform, 0.04f);

			// Vision body-pose shoulder: elbow -> shoulder completes the arm
			if (pose.hasShoulder)
			{
				drawSegment(*m_lineRenderer, k_displayFromWorld, elbowWorld, pose.shoulderPositionWorld, color);
				drawPoint(*m_lineRenderer, k_displayFromWorld, pose.shoulderPositionWorld,
						  Colors::Yellow * brightness, 7.f);
			}
		}
		else if (pose.hasShoulder)
		{
			drawPoint(*m_lineRenderer, k_displayFromWorld, pose.shoulderPositionWorld,
					  Colors::Yellow * brightness, 7.f);
		}
	}

	// Head pose from the vision body-pose solver: +X facing, +Z up
	if (result.head.valid)
	{
		glm::mat4 headTransform= glm::mat4_cast(result.head.orientationWorld);
		headTransform[3]= glm::vec4(result.head.positionWorld, 1.f);
		drawTransformedAxes(*m_lineRenderer, k_displayFromWorld * headTransform, 0.06f);
		drawPoint(*m_lineRenderer, k_displayFromWorld, result.head.positionWorld,
				  Colors::White * brightness, 6.f);
	}
}

bool Scene3dPanel::drawAvatar()
{
	if (m_avatarModel == nullptr || m_avatarSkeleton == nullptr)
	{
		// Unloaded: drop the GPU copy rather than keeping a 12 MB model resident
		if (m_meshRenderer->hasModel())
			m_meshRenderer->clear();
		return false;
	}
	if (!m_bShowAvatar)
		return false;

	if (!m_bMeshRendererInitialized)
	{
		m_bMeshRendererInitialized= m_meshRenderer->startup();
		if (!m_bMeshRendererInitialized)
			return false;
	}
	// Rebuild the GPU copy only when the model changed, not per frame
	if (m_uploadedAvatarGeneration != m_avatarGeneration || !m_meshRenderer->hasModel())
	{
		if (!m_meshRenderer->upload(*m_avatarModel))
			return false;
		m_uploadedAvatarGeneration= m_avatarGeneration;
	}

	// The pose carries the root placement, so the model matrix is only the
	// avatar's own axis convention and the display rotation every
	// world-space drawing takes
	if (m_bHasAvatarPose)
		computePosedGlobals(*m_avatarModel, *m_avatarSkeleton, m_avatarPose, m_posedGlobals);
	else
		m_posedGlobals= m_avatarSkeleton->getRestGlobalsAvatar();
	const glm::mat4 modelMatrix= k_displayFromWorld * m_avatarSkeleton->getWorldFromAvatar();

	// One key light from above and in front of the avatar (display space:
	// +Y up, +X the avatar's facing direction)
	const glm::vec3 lightDirection= glm::normalize(glm::vec3(0.6f, 1.f, 0.4f));
	if (m_faceMorphs != nullptr)
	{
		m_faceMorphs->evaluate(m_bHasAvatarFace ? &m_avatarFace.blendshapes : nullptr, m_morphWeights);
		m_meshRenderer->setMorphWeights(m_morphWeights);
	}
	m_meshRenderer->draw(m_camera->getViewProjection(), modelMatrix, m_posedGlobals, lightDirection);
	return true;
}

bool Scene3dPanel::projectToImage(const glm::vec3& world, const glm::vec2& imageMin, const glm::vec2& imageSize,
								  glm::vec2& outPixel) const
{
	const glm::vec4 clip= m_camera->getViewProjection() * k_displayFromWorld * glm::vec4(world, 1.f);
	if (clip.w <= 1e-6f)
		return false;
	const glm::vec2 ndc= glm::vec2(clip) / clip.w;
	// The image shows the framebuffer flipped, so NDC +Y is the image top
	outPixel= imageMin + glm::vec2((ndc.x * 0.5f + 0.5f) * imageSize.x, (0.5f - ndc.y * 0.5f) * imageSize.y);
	return true;
}

bool Scene3dPanel::updateGizmoDrag(SceneGizmos& gizmos, const glm::vec2& imageMin, const glm::vec2& imageSize)
{
	constexpr float kPickRadiusPixels= 8.f;
	const ImVec2 mouse= ImGui::GetIO().MousePos;
	const glm::vec2 mousePixel(mouse.x, mouse.y);

	if (m_draggedElbowHint >= 0 &&
		(!ImGui::IsMouseDown(ImGuiMouseButton_Left) || !gizmos.elbowHints[m_draggedElbowHint].bEnabled))
	{
		m_draggedElbowHint= -1;
	}

	// A press on a hint starts its drag, in the camera-facing plane through it
	if (m_draggedElbowHint < 0 && ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
	{
		float bestDistance= kPickRadiusPixels;
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			const SceneGizmos::ElbowHint& hint= gizmos.elbowHints[sideIndex];
			glm::vec2 pixel;
			if (!hint.bEnabled || !projectToImage(hint.hintWorld, imageMin, imageSize, pixel))
				continue;
			const float distance= glm::length(pixel - mousePixel);
			if (distance <= bestDistance)
			{
				bestDistance= distance;
				m_draggedElbowHint= sideIndex;
			}
		}
		if (m_draggedElbowHint >= 0)
		{
			m_dragPlanePoint= glm::vec3(k_displayFromWorld * glm::vec4(gizmos.elbowHints[m_draggedElbowHint].hintWorld, 1.f));
			m_dragPlaneNormal= glm::normalize(m_camera->getCameraPosition() - m_dragPlanePoint);
		}
	}
	if (m_draggedElbowHint < 0)
		return false;

	// Move the hint to where the mouse ray meets the drag plane
	SceneGizmos::ElbowHint& hint= gizmos.elbowHints[m_draggedElbowHint];
	hint.bDragging= true;
	hint.draggedHintWorld= hint.hintWorld;
	const float ndcX= (mousePixel.x - imageMin.x) / imageSize.x * 2.f - 1.f;
	const float ndcY= 1.f - (mousePixel.y - imageMin.y) / imageSize.y * 2.f;
	glm::vec3 rayOrigin, rayDirection;
	m_camera->unprojectRay(ndcX, ndcY, rayOrigin, rayDirection);
	const float denominator= glm::dot(rayDirection, m_dragPlaneNormal);
	if (std::fabs(denominator) > 1e-6f)
	{
		const float distance= glm::dot(m_dragPlanePoint - rayOrigin, m_dragPlaneNormal) / denominator;
		if (distance > 0.f)
		{
			const glm::vec3 displayPoint= rayOrigin + rayDirection * distance;
			hint.draggedHintWorld= glm::vec3(glm::inverse(k_displayFromWorld) * glm::vec4(displayPoint, 1.f));
		}
	}
	return true;
}

void Scene3dPanel::drawGizmos(const SceneGizmos& gizmos)
{
	// Rings face the camera: the circle helper draws in its local XZ plane,
	// so local Y goes along the view direction
	const glm::mat3 cameraRotation= glm::transpose(glm::mat3(m_camera->getViewMatrix()));
	const glm::mat3 facing(cameraRotation[0], cameraRotation[2], cameraRotation[1]);
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const SceneGizmos::ElbowHint& hint= gizmos.elbowHints[sideIndex];
		if (!hint.bEnabled)
			continue;
		const glm::vec3 color= sideIndex == 0 ? Colors::CornflowerBlue : Colors::Red;
		const glm::vec3 ringColor= sideIndex == m_draggedElbowHint ? Colors::White : Colors::Yellow;
		drawSegment(*m_lineRenderer, k_displayFromWorld, hint.shoulderWorld, hint.hintWorld, color);
		glm::mat4 ring(facing);
		ring[3]= k_displayFromWorld * glm::vec4(hint.hintWorld, 1.f);
		drawTransformedCircle(*m_lineRenderer, ring, 0.025f, ringColor, 24);
		drawPoint(*m_lineRenderer, k_displayFromWorld, hint.hintWorld, ringColor, 6.f);
	}
}

void Scene3dPanel::renderScene(const TrackingFrameResult& fusedResult, const std::vector<SceneCameraView>& cameras,
							   const std::vector<const TrackingFrameResult*>& perCameraResults,
							   const SceneGizmos& gizmos, float aspect)
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
	drawGizmos(gizmos);

	// The avatar mesh first, then the lines over it with depth off when it is
	// shown, so the tracked skeleton stays visible inside the character
	const bool bAvatarDrawn= drawAvatar();
	m_lineRenderer->render3d(m_camera->getViewProjection(), bAvatarDrawn);

	m_frameBuffer->unbindFrameBuffer();
}

std::vector<SceneCameraView> makeSceneCameraViews(const AppConfig& config)
{
	std::vector<SceneCameraView> cameras;
	for (size_t cameraIndex= 0; cameraIndex < config.cameraCount(); ++cameraIndex)
	{
		const CameraProfile& profile= config.camera(cameraIndex);
		SceneCameraView view;
		view.cameraToWorld= glm::mat4(profile.extrinsics.markerFromCamera);
		view.bHasExtrinsics= profile.extrinsics.present;
		view.intrinsics= profile.intrinsics.present ? &profile.intrinsics.intrinsics : nullptr;
		cameras.push_back(view);
	}
	return cameras;
}
