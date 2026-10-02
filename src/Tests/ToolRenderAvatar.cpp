#include "TestCommon.h"

#include "GL/glew.h"
#include "SDL.h"

#include "glm/ext/matrix_transform.hpp"

#include "AvatarRetarget.h"
#include "AvatarSkeleton.h"
#include "Colors.h"
#include "DebugDraw.h"
#include "GlFrameBuffer.h"
#include "GlLineRenderer.h"
#include "GlSkinnedMeshRenderer.h"
#include "MathGLM.h"
#include "OrbitCamera.h"
#include "PathUtils.h"
#include "Scene3dPanel.h"
#include "VrmLoader.h"

namespace
{
// A canned tracked frame for the posed render: hands up in front of the
// chest, the right hand curled and the left spread, the head turned, no
// measured elbows (so the default pole shows) and no shoulders (so the fixed
// root places the avatar). The user's lengths are the avatar's own.
void makeDemoFrame(const AvatarSkeleton& skeleton, TrackingFrameResult& outFrame, BodyDimensions& outUser)
{
	using B= eHumanoidBone;
	outUser.upperArmLengthMeters= skeleton.getUpperArmLength(eHandSide::Left);
	outUser.forearmLengthMeters= skeleton.getForearmLength(eHandSide::Left);
	outUser.shoulderWidthMeters= skeleton.getShoulderWidth();

	outFrame.timestampMs= 0.0;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const eHandSide side= (eHandSide)sideIndex;
		const AvatarSkeleton::HandRest& hand= skeleton.getHand(side);
		const float sign= sideIndex == 0 ? 1.f : -1.f;
		HandPose& pose= outFrame.poses[sideIndex];
		pose.tracked= true;
		pose.side= side;
		pose.presence= 1.f;
		pose.confidence= 1.f;
		pose.skeleton= hand.skeleton;
		pose.hasWorldPose= true;

		// Palm facing inward toward the body's midline, fingers forward
		const glm::quat restPalm= glm::quat_cast(glm::mat3(hand.palmFrameWorld));
		const glm::quat fingersForward= glm_shortest_arc(glm::vec3(0.f, sign, 0.f), glm::vec3(1.f, 0.f, 0.f));
		const glm::quat palmInward=
			glm::angleAxis(sign * glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f)) * fingersForward;
		pose.palmOrientationWorld= palmInward * restPalm;

		const glm::vec3 shoulder= skeleton.getBone(sideIndex == 0 ? B::LeftUpperArm : B::RightUpperArm).restPositionWorld;
		const glm::vec3 wrist= shoulder + glm::vec3(0.28f, -sign * 0.08f, -0.12f);
		const float halfPalm= hand.skeleton.baseInPalm[(int)eFinger::Middle].x;
		pose.palmPositionWorld= wrist + pose.palmOrientationWorld * glm::vec3(halfPalm, 0.f, 0.f);

		if (sideIndex == 1)
		{
			for (int finger= 1; finger < FINGER_COUNT; ++finger)
				pose.fingers[finger]= {0.f, 1.3f, 1.2f, 0.7f};
			pose.fingers[(int)eFinger::Thumb]= {0.3f, 0.6f, 0.8f, 0.4f};
		}
		else
		{
			pose.fingers[(int)eFinger::Index].lateral= 0.25f;
			pose.fingers[(int)eFinger::Pinky].lateral= -0.3f;
			pose.fingers[(int)eFinger::Thumb].lateral= 0.6f;
		}
	}
	outFrame.head.valid= true;
	outFrame.head.orientationWorld= glm::angleAxis(glm::radians(25.f), glm::vec3(0.f, 0.f, 1.f)) *
		glm::angleAxis(glm::radians(10.f), glm::vec3(0.f, 1.f, 0.f));
	outFrame.head.confidence= 1.f;
}
} // namespace

// Renders a VRM through the same renderer, framebuffer and display convention
// the 3D scene panel uses, into a PNG, from a hidden window. The way to look
// at what the avatar renderer and the retarget produce without driving the
// UI: a shader, sorting or retarget regression shows up in the image.
//
//   --render-avatar <file.vrm> [out.png] [yawDegrees] [pitchDegrees] [rest|demo]
static int runRenderAvatarTool(const TestArgs& args)
{
	if (args.empty())
	{
		MIKAN_LOG_ERROR("render-avatar")
			<< "Usage: --render-avatar <file.vrm> [out.png] [yawDegrees] [pitchDegrees] [rest|demo]";
		return 1;
	}
	const std::filesystem::path vrmPath= PathUtils::utf8ToPath(args[0]);
	const std::filesystem::path outPath= args.size() > 1 ? PathUtils::utf8ToPath(args[1]) : "render-avatar.png";
	const float yawDegrees= args.size() > 2 ? (float)std::atof(args[2].c_str()) : 35.f;
	const float pitchDegrees= args.size() > 3 ? (float)std::atof(args[3].c_str()) : -10.f;
	const bool bDemoPose= args.size() > 4 && args[4] == "demo";

	const VrmLoader::LoadResult loaded= VrmLoader::loadFile(vrmPath);
	if (loaded.model == nullptr)
	{
		MIKAN_LOG_ERROR("render-avatar") << "Load failed: " << loaded.error;
		return 1;
	}
	const AvatarSkeleton skeleton(*loaded.model);

	// The same context the app creates, on a window that never shows
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		MIKAN_LOG_ERROR("render-avatar") << "SDL_Init failed: " << SDL_GetError();
		return 1;
	}
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
	SDL_Window* window= SDL_CreateWindow("render-avatar", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 64, 64,
										 SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
	SDL_GLContext context= window != nullptr ? SDL_GL_CreateContext(window) : nullptr;
	if (context == nullptr)
	{
		MIKAN_LOG_ERROR("render-avatar") << "GL context failed: " << SDL_GetError();
		SDL_Quit();
		return 1;
	}
	SDL_GL_MakeCurrent(window, context);
	glewExperimental= GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		MIKAN_LOG_ERROR("render-avatar") << "glewInit failed";
		SDL_Quit();
		return 1;
	}
	MIKAN_LOG_INFO("render-avatar") << "OpenGL " << (const char*)glGetString(GL_VERSION) << " on "
									<< (const char*)glGetString(GL_RENDERER);

	int exitCode= 1;
	{
		constexpr uint16_t kWidth= 800;
		constexpr uint16_t kHeight= 1000;
		GlFrameBuffer frameBuffer;
		GlLineRenderer lineRenderer;
		GlSkinnedMeshRenderer meshRenderer;
		if (!frameBuffer.init(kWidth, kHeight) || !lineRenderer.startup() || !meshRenderer.startup() ||
			!meshRenderer.upload(*loaded.model))
		{
			MIKAN_LOG_ERROR("render-avatar") << "Renderer setup failed (see GL errors above)";
		}
		else
		{
			// Frame the avatar: orbit about its mid height, far enough back
			// to fit it, from the front-left by default
			const float hipsHeight= skeleton.getBone(eHumanoidBone::Hips).restPositionWorld.z;
			const float fullHeight= hipsHeight + skeleton.getHeightAboveHips();
			OrbitCamera camera;
			camera.setOrbitTargetPosition(glm::vec3(0.f, fullHeight * 0.5f, 0.f));
			camera.setOrbitLocation(yawDegrees, pitchDegrees, std::max(fullHeight * 1.6f, 0.5f));
			camera.setPerspectiveProjection(50.f, (float)kWidth / (float)kHeight, 0.01f, 100.f);

			frameBuffer.bindFrameBuffer();
			glClearColor(0.05f, 0.05f, 0.07f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			glEnable(GL_DEPTH_TEST);

			drawGrid(lineRenderer, glm::mat4(1.f), 2.f, 2.f, 20, 20, Colors::DarkGray);
			drawTransformedAxes(lineRenderer, displayFromWorld(), 0.1f);

			std::vector<glm::mat4> globals= skeleton.getRestGlobalsAvatar();
			if (bDemoPose)
			{
				TrackingFrameResult frame;
				BodyDimensions user;
				makeDemoFrame(skeleton, frame, user);
				AvatarRetargetConfig retargetConfig;
				retargetConfig.followShoulders= false;
				AvatarRetarget retarget;
				AvatarPose pose;
				retarget.solve(frame, user, skeleton, retargetConfig, pose);
				computePosedGlobals(*loaded.model, skeleton, pose, globals);
			}

			const glm::mat4 modelMatrix= displayFromWorld() * skeleton.getWorldFromAvatar();
			const glm::vec3 lightDirection= glm::normalize(glm::vec3(0.6f, 1.f, 0.4f));
			meshRenderer.draw(camera.getViewProjection(), modelMatrix, globals, lightDirection);
			lineRenderer.render3d(camera.getViewProjection());

			// Read back and flip: GL rows run bottom-up
			cv::Mat rgba(kHeight, kWidth, CV_8UC4);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glReadPixels(0, 0, kWidth, kHeight, GL_RGBA, GL_UNSIGNED_BYTE, rgba.data);
			frameBuffer.unbindFrameBuffer();

			cv::Mat bgr;
			cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
			cv::flip(bgr, bgr, 0);
			if (cv::imwrite(outPath.string(), bgr))
			{
				MIKAN_LOG_INFO("render-avatar") << "Wrote " << outPath;
				exitCode= 0;
			}
			else
			{
				MIKAN_LOG_ERROR("render-avatar") << "Could not write " << outPath;
			}
		}
		meshRenderer.shutdown();
		lineRenderer.shutdown();
		frameBuffer.dispose();
	}

	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return exitCode;
}
MIKAN_REGISTER_TEST("--render-avatar", "Render a VRM at rest to a PNG through the scene renderer (hidden window)",
					eTestCategory::Tool, runRenderAvatarTool);
