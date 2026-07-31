#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include "App.h"
#include "ArucoMarkerPoseSampler.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "ExtrinsicsWizard.h"
#include "HandFusion.h"
#include "HandTrackingPipeline.h"
#include "Logger.h"
#include "MonoLensDistortionCalibrator.h"
#include "OscWriterTest.h"

#ifdef _WIN32
#include <windows.h>
#endif

static int runApp(int argc, char** argv)
{
	for (int i= 1; i < argc; ++i)
	{
		if (std::string(argv[i]) == "--selftest")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			const bool bSuccess= runOscWriterSelfTest();

			log_dispose();
			return bSuccess ? 0 : 1;
		}

		if (std::string(argv[i]) == "--test-charuco")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-charuco.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;
			try
			{
				MIKAN_LOG_INFO("test-charuco") << "1: bare finder ctor";
				{
					CalibrationPatternFinder_Charuco finder(1280, 720, 11, 8, 16.f, 12.f,
															eCharucoDictionaryType::DICT_6X6);
				}

				MIKAN_LOG_INFO("test-charuco") << "2: ORT pipeline startup (mirrors app state)";
				HandTrackingPipeline pipeline;
				HandTrackingPipelineConfig pipelineConfig;
				pipeline.startup(pipelineConfig);

				MIKAN_LOG_INFO("test-charuco") << "3: MonoLensDistortionCalibrator ctor (wizard path)";
				MonoLensDistortionCalibrator calibrator(1280, 720, 11, 8, 16.f, 12.f,
														eCharucoDictionaryType::DICT_6X6, 12);

				MIKAN_LOG_INFO("test-charuco") << "4: update() with synthetic frames";
				cv::Mat grayFrame(720, 1280, CV_8UC1, cv::Scalar(128));
				for (int frame= 0; frame < 5; ++frame)
					calibrator.update(0.033f, &grayFrame);

				// 5: Regression test for the capture stall: render a synthetic
				// charuco board and run the wizard loop (update + overlay read
				// each frame). The overlay read must not commit capture state,
				// so the stability timer must fire and progress must advance.
				MIKAN_LOG_INFO("test-charuco") << "5: capture progress with a rendered board + overlay reads";
				{
					const cv::aruco::Dictionary dictionary=
						cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
					const cv::aruco::CharucoBoard board(cv::Size(11, 8), 0.016f, 0.012f, dictionary);
					cv::Mat boardImage;
					board.generateImage(cv::Size(1100, 800), boardImage, 40, 1);

					MonoLensDistortionCalibrator boardCalibrator(1100, 800, 11, 8, 16.f, 12.f,
																 eCharucoDictionaryType::DICT_6X6, 12);

					t_opencv_point2d_list overlayPoints;
					cv::Point2f overlayQuad[4];
					for (int frame= 0; frame < 40; ++frame) // 2s at 50ms > 1s stability window
					{
						boardCalibrator.update(0.05f, &boardImage);
						// Mimic the wizard overlay read every frame
						boardCalibrator.getPatternFinder()->getCurrentCalibrationPattern(overlayPoints, overlayQuad);
					}

					if (boardCalibrator.computeCalibrationProgress() <= 0.f)
					{
						MIKAN_LOG_ERROR("test-charuco")
							<< "REGRESSION: capture progress stuck at 0 with overlay reads active";
						result= 1;
					}
					else
					{
						MIKAN_LOG_INFO("test-charuco")
							<< "Captured " << boardCalibrator.computeCalibrationProgress() * 12.f
							<< " samples from static board (expected 1)";
					}

					// 6: Regression test for the stuck-green stall: right after a
					// capture, present the board >= 100px away with NO invalid
					// frame in between (fast board move). The hold cycle must
					// restart and produce a second capture.
					MIKAN_LOG_INFO("test-charuco") << "6: second capture after an immediate fast board move";
					{
						cv::Mat shiftedImage;
						const cv::Mat translation= (cv::Mat_<double>(2, 3) << 1, 0, 150, 0, 1, 0);
						cv::warpAffine(boardImage, shiftedImage, translation, boardImage.size(),
									   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255));

						for (int frame= 0; frame < 40; ++frame)
						{
							boardCalibrator.update(0.05f, &shiftedImage);
							boardCalibrator.getPatternFinder()->getCurrentCalibrationPattern(overlayPoints,
																							 overlayQuad);
						}

						const float capturedSamples= boardCalibrator.computeCalibrationProgress() * 12.f;
						if (capturedSamples < 2.f)
						{
							MIKAN_LOG_ERROR("test-charuco")
								<< "REGRESSION: no capture after fast board move (stuck hold state), samples="
								<< capturedSamples;
							result= 1;
						}
						else
						{
							MIKAN_LOG_INFO("test-charuco")
								<< "Captured " << capturedSamples << " samples after fast move (expected 2)";
						}
					}
				}

				MIKAN_LOG_INFO("test-charuco") << "All steps passed";
			}
			catch (const cv::Exception& e)
			{
				MIKAN_LOG_ERROR("test-charuco") << "cv::Exception: " << e.what();
				result= 1;
			}
			catch (const std::exception& e)
			{
				MIKAN_LOG_ERROR("test-charuco") << "std::exception: " << e.what();
				result= 1;
			}

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-extrinsics")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-extrinsics.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;
			try
			{
				// Ground truth: 1280x720 pinhole camera 0.8m directly above a
				// 100mm aruco marker at the world origin, looking straight down.
				// OpenCV camera axes: x=world X, y=-world Y, z=-world Z (down).
				const double fx= 800.0, fy= 800.0, cx= 640.0, cy= 360.0;
				const double cameraHeight= 0.8;

				MikanMonoIntrinsics intrinsics;
				intrinsics.pixel_width= 1280;
				intrinsics.pixel_height= 720;
				intrinsics.undistorted_camera_matrix.x0= fx;
				intrinsics.undistorted_camera_matrix.y1= fy;
				intrinsics.undistorted_camera_matrix.z0= cx;
				intrinsics.undistorted_camera_matrix.z1= cy;
				intrinsics.distorted_camera_matrix= intrinsics.undistorted_camera_matrix;

				// Project a world point on the table into the image
				auto projectTablePoint= [&](double worldX, double worldY) -> cv::Point2f {
					const double camX= worldX, camY= -worldY, camZ= cameraHeight;
					return cv::Point2f((float)(fx * camX / camZ + cx), (float)(fy * camY / camZ + cy));
				};

				// Render the marker into the synthetic camera image.
				// generateImageMarker corner order matches detection order
				// (top-left, top-right, bottom-right, bottom-left); lay the
				// marker's top edge along -worldY so it appears upright.
				const int markerImageSize= 400;
				cv::Mat markerImage;
				cv::aruco::generateImageMarker(
					cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250), 0, markerImageSize, markerImage, 1);

				const double halfLenM= 0.05; // 100mm marker
				const std::vector<cv::Point2f> sourceCorners= {
					{0.f, 0.f}, {(float)markerImageSize, 0.f},
					{(float)markerImageSize, (float)markerImageSize}, {0.f, (float)markerImageSize}};
				const std::vector<cv::Point2f> imageCorners= {
					projectTablePoint(-halfLenM, halfLenM), projectTablePoint(halfLenM, halfLenM),
					projectTablePoint(halfLenM, -halfLenM), projectTablePoint(-halfLenM, -halfLenM)};

				cv::Mat cameraImage(720, 1280, CV_8UC1, cv::Scalar(255));
				const cv::Mat homography= cv::getPerspectiveTransform(sourceCorners, imageCorners);
				cv::warpPerspective(markerImage, cameraImage, homography, cameraImage.size(), cv::INTER_LINEAR,
									cv::BORDER_TRANSPARENT);

				// Run the real sampler on the synthetic image
				ArucoMarkerPoseSampler sampler(intrinsics, 1280, 720, 100.f, 0, eCharucoDictionaryType::DICT_6X6, 12);
				sampler.setGrayscaleFrame(&cameraImage);
				for (int sample= 0; sample < 12; ++sample)
				{
					if (sampler.computeApertureRelativeMarkerXform())
						sampler.sampleLastApertureRelativeMarkerXform();
				}

				glm::dmat4 cameraFromMarker;
				if (!sampler.hasFinishedSampling() || !sampler.computeCalibratedMarkerPose(cameraFromMarker))
				{
					MIKAN_LOG_ERROR("test-extrinsics") << "Sampler failed to detect the synthetic marker";
					result= 1;
				}
				else
				{
					const double markerDistance= glm::length(glm::dvec3(cameraFromMarker[3]));
					MIKAN_LOG_INFO("test-extrinsics") << "Marker distance: " << markerDistance << " (expected 0.8)";

					// Wizard math: recovered camera height must be positive ~0.8
					const glm::dmat4 worldFromCamera= ExtrinsicsWizard::computeWorldFromCameraPose(cameraFromMarker);
					const double recoveredHeight= worldFromCamera[3].z;
					MIKAN_LOG_INFO("test-extrinsics") << "Camera height: " << recoveredHeight << " (expected +0.8)";

					if (fabs(markerDistance - cameraHeight) > 0.02 || fabs(recoveredHeight - cameraHeight) > 0.02)
					{
						MIKAN_LOG_ERROR("test-extrinsics") << "REGRESSION: pose/height mismatch";
						result= 1;
					}

					// Ray-cast a known table point through the wizard math and
					// verify the recovered world position (hand-scale path)
					const cv::Point2f testPixel= projectTablePoint(0.10, 0.05);
					glm::dvec3 hitCameraSpace;
					if (!ExtrinsicsWizard::raycastPixelOntoPlane(intrinsics, cameraFromMarker,
																 glm::vec2(testPixel.x, testPixel.y), hitCameraSpace))
					{
						MIKAN_LOG_ERROR("test-extrinsics") << "REGRESSION: table-plane raycast failed";
						result= 1;
					}
					else
					{
						const glm::dvec3 hitWorld= glm::dvec3(worldFromCamera * glm::dvec4(hitCameraSpace, 1.0));
						MIKAN_LOG_INFO("test-extrinsics")
							<< "Raycast hit world (" << hitWorld.x << ", " << hitWorld.y << ", " << hitWorld.z
							<< ") (expected 0.10, 0.05, 0)";
						if (fabs(hitWorld.x - 0.10) > 0.01 || fabs(hitWorld.y - 0.05) > 0.01 ||
							fabs(hitWorld.z) > 0.01)
						{
							MIKAN_LOG_ERROR("test-extrinsics") << "REGRESSION: raycast world position mismatch";
							result= 1;
						}
					}
				}

				if (result == 0)
					MIKAN_LOG_INFO("test-extrinsics") << "All extrinsics checks passed";
			}
			catch (const cv::Exception& e)
			{
				MIKAN_LOG_ERROR("test-extrinsics") << "cv::Exception: " << e.what();
				result= 1;
			}

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-fusion")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-fusion.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// Ground-truth left hand: palm plane spanned by basis (u, v),
			// hovering 10cm over the table around world (0.1, 0.05)
			auto makeHand= [](const glm::vec3& wrist, const glm::vec3& u, const glm::vec3& v,
							  const glm::vec3& noiseSeed, float noiseAmp) {
				TrackedHand hand;
				hand.tracked= true;
				hand.hasWorldSpace= true;
				hand.side= eHandSide::Left;
				hand.presence= 0.9f;
				for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
				{
					// crude but plausible layout: fingers fan out along +v with
					// spread along u, scaled to a ~17cm hand
					const float along= 0.02f + 0.15f * (float)(lm % 4) / 4.f * (lm >= 1 ? 1.f : 0.f);
					const float spread= ((float)(lm / 4) - 2.f) * 0.02f;
					glm::vec3 point= wrist + v * along + u * spread;

					// deterministic per-landmark pseudo-noise
					const float phase= (float)lm * 1.7f;
					point+= noiseAmp * glm::vec3(sinf(noiseSeed.x + phase), cosf(noiseSeed.y + phase * 1.3f),
												 sinf(noiseSeed.z + phase * 0.7f));
					hand.worldPoints[lm]= point;
				}
				return hand;
			};

			auto rmsError= [](const TrackedHand& a, const TrackedHand& b) {
				float sum= 0.f;
				for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
					sum+= glm::dot(a.worldPoints[lm] - b.worldPoints[lm], a.worldPoints[lm] - b.worldPoints[lm]);
				return sqrtf(sum / (float)HAND_LANDMARK_COUNT);
			};

			auto makeCameraResult= [](int cameraIndex, const glm::vec3& cameraPosWorld, double timestampMs,
									  const TrackedHand& hand) {
				CameraFrameResult camera;
				camera.cameraIndex= cameraIndex;
				camera.valid= true;
				camera.timestampMs= timestampMs;
				camera.hasExtrinsics= true;
				camera.markerFromCamera= glm::dmat4(1.0);
				camera.markerFromCamera[3]= glm::dvec4(cameraPosWorld, 1.0);
				camera.result.hands[(int)eHandSide::Left]= hand;
				return camera;
			};

			HandFusionConfig fusionConfig;
			fusionConfig.smoothingEnabled= false; // exactness for the pass-through checks
			HandFusion fusion;
			fusion.configure(fusionConfig);

			const glm::vec3 wristTruth(0.10f, 0.05f, 0.10f);
			const glm::vec3 cam1Pos(0.f, 0.f, 0.8f);   // overhead
			const glm::vec3 cam2Pos(0.f, -0.6f, 0.6f); // 45 degrees
			const double now= 10'000.0;

			// (a) Palm face-up (visible to both): fused should beat both noisy inputs
			{
				const glm::vec3 u(1, 0, 0), v(0, 1, 0); // palm normal = +Z (toward overhead cam)
				const TrackedHand truth= makeHand(wristTruth, u, v, glm::vec3(0), 0.f);
				const TrackedHand noisyA= makeHand(wristTruth, u, v, glm::vec3(1.f, 2.f, 3.f), 0.004f);
				const TrackedHand noisyB= makeHand(wristTruth, u, v, glm::vec3(7.f, 5.f, 9.f), 0.004f);

				const CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, noisyA);
				const CameraFrameResult camB= makeCameraResult(1, cam2Pos, now - 5.0, noisyB);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const float errA= rmsError(noisyA, truth);
				const float errB= rmsError(noisyB, truth);
				const float errFused= rmsError(fused.hands[(int)eHandSide::Left], truth);
				MIKAN_LOG_INFO("test-fusion") << "(a) rms error mm: camA=" << errA * 1000.f
					<< " camB=" << errB * 1000.f << " fused=" << errFused * 1000.f;
				if (!fused.hands[(int)eHandSide::Left].tracked || errFused > std::min(errA, errB) * 1.05f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(a) FAILED: fused should beat both noisy inputs";
					result= 1;
				}
			}

			// (b) Palm edge-on to the overhead camera, facing camera 2:
			// weights must shift to camera 2
			{
				const glm::vec3 u(1, 0, 0), v(0, 0, 1); // palm normal = -Y (toward cam2, edge-on to cam1)
				const TrackedHand handA= makeHand(wristTruth, u, v, glm::vec3(1.f, 2.f, 3.f), 0.004f);
				const TrackedHand handB= makeHand(wristTruth + glm::vec3(0.01f, 0.f, 0.f), u, v, glm::vec3(0), 0.f);

				const CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, handA);
				const CameraFrameResult camB= makeCameraResult(1, cam2Pos, now, handB);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const float distToB= rmsError(fused.hands[(int)eHandSide::Left], handB);
				const float distToA= rmsError(fused.hands[(int)eHandSide::Left], handA);
				MIKAN_LOG_INFO("test-fusion") << "(b) dominant camera=" << fusion.getDominantCamera(eHandSide::Left)
					<< " distToEdgeOnCam mm=" << distToA * 1000.f << " distToFaceOnCam mm=" << distToB * 1000.f;
				if (fusion.getDominantCamera(eHandSide::Left) != 1 || distToB >= distToA)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(b) FAILED: edge-on view should lose to the face-on camera";
					result= 1;
				}
			}

			// (c) Staleness: camera 2's result is 200ms old -> exact passthrough of camera 1
			{
				const glm::vec3 u(1, 0, 0), v(0, 1, 0);
				const TrackedHand handA= makeHand(wristTruth, u, v, glm::vec3(0), 0.f);
				const TrackedHand handB= makeHand(wristTruth + glm::vec3(0.3f, 0.f, 0.f), u, v, glm::vec3(0), 0.f);

				const CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, handA);
				const CameraFrameResult camB= makeCameraResult(1, cam2Pos, now - 200.0, handB);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const float distToA= rmsError(fused.hands[(int)eHandSide::Left], handA);
				MIKAN_LOG_INFO("test-fusion") << "(c) stale exclusion: distToFreshCam mm=" << distToA * 1000.f;
				if (distToA > 1e-6f || fusion.getDominantCamera(eHandSide::Left) != 0)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(c) FAILED: stale camera should be excluded";
					result= 1;
				}
			}

			// (d) Handedness conflict: same side, wrists 0.5m apart -> keep the
			// higher-scoring candidate only (no frankenhand averaging)
			{
				const glm::vec3 u(1, 0, 0), v(0, 1, 0);
				TrackedHand handA= makeHand(wristTruth, u, v, glm::vec3(0), 0.f); // palm normal +Z: face-on to cam1
				handA.presence= 0.95f;
				TrackedHand handB= makeHand(wristTruth + glm::vec3(0.5f, 0.f, 0.f), glm::vec3(1, 0, 0),
											glm::vec3(0, 0, 1), glm::vec3(0), 0.f); // edge-on to its camera
				handB.presence= 0.6f;

				const CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, handA);
				const CameraFrameResult camB= makeCameraResult(1, cam2Pos, now, handB);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const float distToA= rmsError(fused.hands[(int)eHandSide::Left], handA);
				MIKAN_LOG_INFO("test-fusion") << "(d) conflict gate: distToBetterCam mm=" << distToA * 1000.f;
				if (distToA > 1e-6f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(d) FAILED: conflicting candidates must not be averaged";
					result= 1;
				}
			}

			// (e) Handedness-mislabel recovery: camera 1 sees only the LEFT hand
			// (decisively labeled), camera 2 sees only the RIGHT hand but its
			// classifier MISLABELS it Left (weakly). Fusion must output two
			// separate hands, not collapse them into one side slot.
			{
				const glm::vec3 u(1, 0, 0), v(0, 1, 0);
				TrackedHand leftHand= makeHand(wristTruth, u, v, glm::vec3(0), 0.f);
				leftHand.side= eHandSide::Left;
				leftHand.presence= 0.9f;
				leftHand.handednessScore= 0.05f; // decisive Left

				TrackedHand rightHandMislabeled= makeHand(wristTruth + glm::vec3(0.3f, 0.f, 0.f), u, v, glm::vec3(0), 0.f);
				rightHandMislabeled.side= eHandSide::Left; // WRONG label from camera 2
				rightHandMislabeled.presence= 0.7f;
				rightHandMislabeled.handednessScore= 0.52f; // indecisive

				CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, leftHand);
				CameraFrameResult camB= makeCameraResult(1, cam2Pos, now, rightHandMislabeled);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const bool bLeftTracked= fused.hands[(int)eHandSide::Left].tracked;
				const bool bRightTracked= fused.hands[(int)eHandSide::Right].tracked;
				float leftDist= 1e9f, rightDist= 1e9f;
				if (bLeftTracked)
					leftDist= rmsError(fused.hands[(int)eHandSide::Left], leftHand);
				if (bRightTracked)
					rightDist= rmsError(fused.hands[(int)eHandSide::Right], rightHandMislabeled);

				MIKAN_LOG_INFO("test-fusion") << "(e) mislabel recovery: L tracked=" << bLeftTracked
					<< " R tracked=" << bRightTracked << " Lerr mm=" << leftDist * 1000.f
					<< " Rerr mm=" << rightDist * 1000.f;
				if (!bLeftTracked || !bRightTracked || leftDist > 0.01f || rightDist > 0.01f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(e) FAILED: two physical hands must fuse to two sides despite a mislabel";
					result= 1;
				}
			}

			// (f) Stereo hand-scale: both cameras observe the same hand with a
			// 20% depth overestimate (configured hand scale too large). The
			// wrist-ray triangulation must recover correction ~1/1.2.
			{
				const glm::vec3 u(1, 0, 0), v(0, 1, 0);
				const float depthError= 1.2f;
				const glm::vec3 wristA= cam1Pos + depthError * (wristTruth - cam1Pos);
				const glm::vec3 wristB= cam2Pos + depthError * (wristTruth - cam2Pos);

				TrackedHand handA= makeHand(wristA, u, v, glm::vec3(0), 0.f);
				TrackedHand handB= makeHand(wristB, u, v, glm::vec3(0), 0.f);

				CameraFrameResult camA= makeCameraResult(0, cam1Pos, now, handA);
				CameraFrameResult camB= makeCameraResult(1, cam2Pos, now, handB);

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				float correction= 0.f;
				const bool bHasSample= fusion.getStereoScaleSample(correction);
				const float expected= 1.f / depthError;
				MIKAN_LOG_INFO("test-fusion") << "(f) stereo scale: correction=" << correction
					<< " (expected " << expected << ")";
				if (!bHasSample || fabsf(correction - expected) > 0.01f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(f) FAILED: triangulated scale correction mismatch";
					result= 1;
				}
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-fusion") << "All fusion checks passed";

			log_dispose();
			return result;
		}
	}

	App app;
	return app.exec(argc, argv);
}

#if defined(_WIN32) && !defined(_CONSOLE)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	return runApp(__argc, __argv);
}
#else
int main(int argc, char** argv)
{
	return runApp(argc, argv);
}
#endif
