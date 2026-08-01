#include <string>

#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include <filesystem>
#include <fstream>

#include "App.h"
#include "AppConfig.h"
#include "ArucoMarkerPoseSampler.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "DiagnosticDump.h"
#include "ExtrinsicsWizard.h"
#include "HandFusion.h"
#include "HandPoseModel.h"
#include "HandTrackingPipeline.h"
#include "LandmarkTo3D.h"
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

			// Synthetic parametric hand observation for one camera
			auto makeObservation= [](const glm::vec3& palmPos, const glm::quat& palmOrient, float presence,
									 eHandSide labeledSide, float handednessScore, float bendAngle) {
				TrackingFrameResult frame;
				HandPose& pose= frame.poses[(int)labeledSide];
				pose.tracked= true;
				pose.side= labeledSide;
				pose.presence= presence;
				pose.hasWorldPose= true;
				pose.palmPositionWorld= palmPos;
				pose.palmOrientationWorld= palmOrient;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					pose.fingers[finger].proximal= bendAngle;
					pose.skeleton.phalanxLengths[finger]= {0.04f, 0.025f, 0.02f};
				}
				TrackedHand& hand= frame.hands[(int)labeledSide];
				hand.tracked= true;
				hand.side= labeledSide;
				hand.presence= presence;
				hand.handednessScore= handednessScore;
				// tests treat the score as already flip-adjusted
				hand.rightProb= handednessScore;
				return frame;
			};

			auto makeCameraResult= [](int cameraIndex, const glm::vec3& cameraPosWorld, double timestampMs,
									  const TrackingFrameResult& frame) {
				CameraFrameResult camera;
				camera.cameraIndex= cameraIndex;
				camera.valid= true;
				camera.timestampMs= timestampMs;
				camera.hasExtrinsics= true;
				camera.markerFromCamera= glm::dmat4(1.0);
				camera.markerFromCamera[3]= glm::dvec4(cameraPosWorld, 1.0);
				camera.result= frame;
				return camera;
			};

			HandFusionConfig fusionConfig;
			fusionConfig.smoothingEnabled= false; // exactness for pass-through checks
			HandFusion fusion;
			fusion.configure(fusionConfig);

			const glm::vec3 palmTruth(0.10f, 0.05f, 0.10f);
			const glm::vec3 cam1Pos(0.f, 0.f, 0.8f);   // overhead
			const glm::vec3 cam2Pos(0.f, -0.6f, 0.6f); // 45 degrees
			const double now= 10'000.0;
			// Palm normal (+Z of palm frame) pointing up at the overhead camera
			const glm::quat faceUpToCam1= glm::quat(1.f, 0.f, 0.f, 0.f);
			// Palm rotated 90 deg about X: normal points along -Y toward cam2,
			// edge-on to the overhead camera
			const glm::quat faceCam2= glm::angleAxis(glm::half_pi<float>(), glm::vec3(1.f, 0.f, 0.f));

			// (a) Both cameras face-on-ish: fused position beats both noisy inputs;
			// fused bend angle is between the two observations
			{
				const glm::vec3 noiseA(0.004f, -0.002f, 0.005f);
				const glm::vec3 noiseB(-0.003f, 0.004f, -0.004f);
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmTruth + noiseA, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.5f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now - 5.0, makeObservation(palmTruth + noiseB, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.7f));

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const HandPose& pose= fused.poses[(int)eHandSide::Left];
				const float errA= glm::length(noiseA);
				const float errB= glm::length(noiseB);
				const float errFused= glm::length(pose.palmPositionWorld - palmTruth);
				const float bend= pose.fingers[0].proximal;
				MIKAN_LOG_INFO("test-fusion") << "(a) palm err mm: A=" << errA * 1000.f << " B=" << errB * 1000.f
					<< " fused=" << errFused * 1000.f << " bend=" << bend;
				if (!pose.tracked || errFused > std::min(errA, errB) * 1.05f || bend < 0.5f || bend > 0.7f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(a) FAILED";
					result= 1;
				}
			}

			// (b) Palm edge-on to the overhead camera, facing camera 2:
			// camera 2 must dominate
			{
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmTruth + glm::vec3(0.01f, 0.f, 0.f), faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(palmTruth, faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				MIKAN_LOG_INFO("test-fusion") << "(b) dominant camera=" << fusion.getDominantCamera(eHandSide::Left);
				if (fusion.getDominantCamera(eHandSide::Left) != 1)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(b) FAILED: edge-on view should lose to the face-on camera";
					result= 1;
				}
			}

			// (c) Staleness: camera 2's result is 200ms old -> exact passthrough of camera 1
			{
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.3f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now - 200.0,
					makeObservation(palmTruth + glm::vec3(0.3f, 0.f, 0.f), faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.9f));

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const HandPose& pose= fused.poses[(int)eHandSide::Left];
				const float dist= glm::length(pose.palmPositionWorld - palmTruth);
				MIKAN_LOG_INFO("test-fusion") << "(c) stale exclusion: distToFreshCam mm=" << dist * 1000.f
					<< " bend=" << pose.fingers[0].proximal;
				if (dist > 1e-6f || fusion.getDominantCamera(eHandSide::Left) != 0 ||
					fabsf(pose.fingers[0].proximal - 0.3f) > 1e-6f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(c) FAILED: stale camera should be excluded";
					result= 1;
				}
			}

			// (d) Handedness-mislabel recovery: camera 1 sees only the LEFT hand
			// (decisively labeled), camera 2 sees only the RIGHT hand but its
			// classifier MISLABELS it Left (weakly). Fusion must output two
			// separate hands at their own positions, not collapse them.
			{
				const glm::vec3 rightPalmTruth= palmTruth + glm::vec3(0.3f, 0.f, 0.f);
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.05f, 0.2f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(rightPalmTruth, faceUpToCam1, 0.7f, eHandSide::Left, 0.52f, 0.6f));

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				const HandPose& left= fused.poses[(int)eHandSide::Left];
				const HandPose& right= fused.poses[(int)eHandSide::Right];
				const float leftDist= left.tracked ? glm::length(left.palmPositionWorld - palmTruth) : 1e9f;
				const float rightDist= right.tracked ? glm::length(right.palmPositionWorld - rightPalmTruth) : 1e9f;
				MIKAN_LOG_INFO("test-fusion") << "(d) mislabel recovery: L tracked=" << left.tracked
					<< " R tracked=" << right.tracked << " Lerr mm=" << leftDist * 1000.f
					<< " Rerr mm=" << rightDist * 1000.f;
				if (!left.tracked || !right.tracked || leftDist > 0.001f || rightDist > 0.001f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(d) FAILED: two physical hands must fuse to two sides despite a mislabel";
					result= 1;
				}
			}

			// (e) Stereo hand-scale: both cameras observe the same palm with a
			// 20% depth overestimate; triangulation must recover ~1/1.2
			{
				const float depthError= 1.2f;
				const glm::vec3 palmA= cam1Pos + depthError * (palmTruth - cam1Pos);
				const glm::vec3 palmB= cam2Pos + depthError * (palmTruth - cam2Pos);

				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmA, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(palmB, faceUpToCam1, 0.9f, eHandSide::Left, 0.1f, 0.f));

				TrackingFrameResult fused;
				fusion.fuse({&camA, &camB}, now, fused);

				float correction= 0.f;
				const bool bHasSample= fusion.getStereoScaleSample(correction);
				const float expected= 1.f / depthError;
				MIKAN_LOG_INFO("test-fusion") << "(e) stereo scale: correction=" << correction
					<< " (expected " << expected << ")";
				if (!bHasSample || fabsf(correction - expected) > 0.01f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(e) FAILED: triangulated scale correction mismatch";
					result= 1;
				}
			}

			// (f) Ray-aware clustering: camera 1 sees ONLY the left hand, but
			// with a 45% depth overestimate along its view ray (bad hand scale)
			// AND mislabels it Right (weakly - a DECISIVE opposite score would
			// trigger the vote-coherence veto, which is test (i)'s subject).
			// Euclidean clustering would split it into a phantom "right hand"
			// fighting camera 2's real right hand; ray-aware clustering must
			// merge it into the left cluster.
			{
				HandFusion freshFusion; // no temporal prior from earlier tests
				freshFusion.configure(fusionConfig);

				const glm::vec3 rightPalmTruth= palmTruth + glm::vec3(0.3f, 0.f, 0.f);
				// displaced ALONG cam1's view ray: euclidean offset ~0.32m (> the
				// 0.25m wrist gate), perpendicular offset 0
				const glm::vec3 leftPalmDisplaced= cam1Pos + 1.45f * (palmTruth - cam1Pos);

				const auto camA= makeCameraResult(
					0, cam1Pos, now,
					makeObservation(leftPalmDisplaced, faceUpToCam1, 0.7f, eHandSide::Right, 0.7f, 0.f));
				TrackingFrameResult camBFrame=
					makeObservation(palmTruth, faceUpToCam1, 0.9f, eHandSide::Left, 0.05f, 0.f);
				{
					// add camera 2's decisively-labeled right hand to the same frame
					const TrackingFrameResult rightFrame=
						makeObservation(rightPalmTruth, faceUpToCam1, 0.9f, eHandSide::Right, 0.95f, 0.f);
					camBFrame.poses[(int)eHandSide::Right]= rightFrame.poses[(int)eHandSide::Right];
					camBFrame.hands[(int)eHandSide::Right]= rightFrame.hands[(int)eHandSide::Right];
				}
				const auto camB= makeCameraResult(1, cam2Pos, now, camBFrame);

				TrackingFrameResult fused;
				freshFusion.fuse({&camA, &camB}, now, fused);

				const HandPose& left= fused.poses[(int)eHandSide::Left];
				const HandPose& right= fused.poses[(int)eHandSide::Right];
				const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalmTruth) : 1e9f;
				const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - palmTruth) : 1e9f;
				MIKAN_LOG_INFO("test-fusion") << "(f) ray clustering: L tracked=" << left.tracked
					<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
					<< " Rerr mm=" << rightErr * 1000.f;
				// Right must be the exact passthrough of camera 2's right hand
				// (no phantom competing for it); left may be pulled along cam1's
				// ray by the blend but must stay near the truth
				if (!left.tracked || !right.tracked || rightErr > 0.001f || leftErr > 0.25f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(f) FAILED: depth-displaced observation must merge into the left cluster";
					result= 1;
				}
			}

			// (g) Spatial side prior: two hands, both (mis)labeled Left - the
			// one on the +X side even decisively so. With the prior configured
			// (+X = right side), assignment must follow geometry, not the votes.
			{
				HandFusionConfig priorConfig= fusionConfig;
				priorConfig.spatialSidePriorAxis= 1; // +X toward the right hand
				HandFusion freshFusion;
				freshFusion.configure(priorConfig);

				const glm::vec3 leftPalm(-0.15f, 0.05f, 0.1f);
				const glm::vec3 rightPalm(0.15f, 0.05f, 0.1f);
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(leftPalm, faceUpToCam1, 0.9f, eHandSide::Left, 0.45f, 0.f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(rightPalm, faceUpToCam1, 0.8f, eHandSide::Left, 0.05f, 0.f));

				TrackingFrameResult fused;
				freshFusion.fuse({&camA, &camB}, now, fused);

				const HandPose& left= fused.poses[(int)eHandSide::Left];
				const HandPose& right= fused.poses[(int)eHandSide::Right];
				const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - leftPalm) : 1e9f;
				const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalm) : 1e9f;
				MIKAN_LOG_INFO("test-fusion") << "(g) spatial prior: L tracked=" << left.tracked
					<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
					<< " Rerr mm=" << rightErr * 1000.f;
				if (!left.tracked || !right.tracked || leftErr > 0.001f || rightErr > 0.001f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(g) FAILED: spatial prior must overrule a decisive mislabel";
					result= 1;
				}
			}

			// (h) Joint cluster pairing - regression from the 2026-08-01 clap
			// dump (real numbers, reacquisition frame i116). Both cameras'
			// slot LABELS are physically reversed after the clap, and greedy
			// nearest-position clustering paired the wrong hands cross-camera
			// (each camera's depth error put the wrong hand nearest). The
			// joint assignment + score-based votes must pair the physical
			// hands correctly and assign the true sides.
			{
				HandFusion freshFusion;
				freshFusion.configure(fusionConfig);

				const glm::vec3 cam0Pos(0.038f, 0.355f, 0.646f);  // real extrinsics
				const glm::vec3 cam1PosReal(0.073f, -0.284f, 0.650f);

				// cam0: physical RIGHT hand sits in its "Left" slot (hijack),
				// physical LEFT in its "Right" slot - scores tell the truth
				TrackingFrameResult cam0Frame=
					makeObservation(glm::vec3(0.018f, -0.023f, 0.164f), faceUpToCam1, 0.99f, eHandSide::Left, 0.34f, 0.f);
				{
					const TrackingFrameResult other= makeObservation(
						glm::vec3(0.018f, 0.139f, 0.164f), faceUpToCam1, 0.98f, eHandSide::Right, 0.04f, 0.f);
					cam0Frame.poses[(int)eHandSide::Right]= other.poses[(int)eHandSide::Right];
					cam0Frame.hands[(int)eHandSide::Right]= other.hands[(int)eHandSide::Right];
				}
				TrackingFrameResult cam1Frame=
					makeObservation(glm::vec3(0.011f, -0.039f, 0.110f), faceUpToCam1, 0.98f, eHandSide::Left, 0.97f, 0.f);
				{
					const TrackingFrameResult other= makeObservation(
						glm::vec3(0.023f, 0.066f, 0.201f), faceUpToCam1, 0.99f, eHandSide::Right, 0.78f, 0.f);
					cam1Frame.poses[(int)eHandSide::Right]= other.poses[(int)eHandSide::Right];
					cam1Frame.hands[(int)eHandSide::Right]= other.hands[(int)eHandSide::Right];
				}

				const auto camA= makeCameraResult(0, cam0Pos, now, cam0Frame);
				const auto camB= makeCameraResult(1, cam1PosReal, now, cam1Frame);

				TrackingFrameResult fused;
				freshFusion.fuse({&camA, &camB}, now, fused);

				// Physical right hand lives at y ~ -0.03, physical left at y ~ +0.10
				const HandPose& left= fused.poses[(int)eHandSide::Left];
				const HandPose& right= fused.poses[(int)eHandSide::Right];
				MIKAN_LOG_INFO("test-fusion") << "(h) clap-dump regression: L tracked=" << left.tracked
					<< " y=" << (left.tracked ? left.palmPositionWorld.y : 0.f)
					<< " R tracked=" << right.tracked
					<< " y=" << (right.tracked ? right.palmPositionWorld.y : 0.f);
				if (!left.tracked || !right.tracked ||
					left.palmPositionWorld.y < 0.05f || right.palmPositionWorld.y > 0.f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(h) FAILED: joint pairing must untangle the post-clap label reversal";
					result= 1;
				}
			}

			// (i) Decisive-disagreement veto: two cameras each track a
			// DIFFERENT physical hand only 8cm apart (post-clap separation),
			// with decisively opposite classifier scores. Position-only
			// clustering would merge them into one mixed cluster (cancelling
			// both votes); the veto must keep them separate so each side is
			// assigned correctly.
			{
				HandFusion freshFusion;
				freshFusion.configure(fusionConfig);

				const glm::vec3 leftPalm(0.10f, 0.05f, 0.10f);
				const glm::vec3 rightPalm(0.10f, 0.13f, 0.10f);
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(leftPalm, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(rightPalm, faceUpToCam1, 0.95f, eHandSide::Right, 0.95f, 0.f));

				TrackingFrameResult fused;
				freshFusion.fuse({&camA, &camB}, now, fused);

				const HandPose& left= fused.poses[(int)eHandSide::Left];
				const HandPose& right= fused.poses[(int)eHandSide::Right];
				const float leftErr= left.tracked ? glm::length(left.palmPositionWorld - leftPalm) : 1e9f;
				const float rightErr= right.tracked ? glm::length(right.palmPositionWorld - rightPalm) : 1e9f;
				MIKAN_LOG_INFO("test-fusion") << "(i) vote veto: L tracked=" << left.tracked
					<< " R tracked=" << right.tracked << " Lerr mm=" << leftErr * 1000.f
					<< " Rerr mm=" << rightErr * 1000.f;
				if (!left.tracked || !right.tracked || leftErr > 0.001f || rightErr > 0.001f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(i) FAILED: decisively opposed observations must not merge";
					result= 1;
				}
			}

			// (j) Pipeline slot side-collision ordering - regression from the
			// 2026-08-01_13-51-16 dump. Both slots' classifiers scored
			// "right"; the decisive one (0.98) must claim Right and displace
			// the weak one (0.64) to Left, regardless of the noise-level
			// presence difference that used to decide it.
			{
				// slot A = physical LEFT hand, weakly (wrongly) scored right,
				// marginally HIGHER presence; slot B = physical RIGHT hand,
				// decisively scored right
				const int order= HandTrackingPipeline::preferredSlotOrder(0.644f, 0.991f, 0.981f, 0.978f);
				MIKAN_LOG_INFO("test-fusion") << "(j) slot collision: first claim = slot " << order
					<< " (expected 1, the decisive one)";
				if (order != 1)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(j) FAILED: the decisive classifier must win the contested side";
					result= 1;
				}

				// Equal decisiveness falls back to presence
				if (HandTrackingPipeline::preferredSlotOrder(0.8f, 0.9f, 0.2f, 0.95f) != 1)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(j) FAILED: presence must break decisiveness ties";
					result= 1;
				}
			}

			// (k) Stability weighting - regression from the 2026-08-01_14-33-43
			// dump. Camera 0 sees the right hand edge-on: presence stays high
			// (0.87) but its palm estimate jitters by ~41mm frame to frame,
			// while camera 1 is rock steady at ~7mm. Presence cannot separate
			// these; measured jitter must, so the fused pose has to converge
			// on the steady camera.
			{
				HandFusion freshFusion;
				HandFusionConfig jitterConfig= fusionConfig;
				jitterConfig.jitterReferenceM= 0.015f;
				freshFusion.configure(jitterConfig);

				// Deterministic pseudo-noise (no Math.random in tests)
				auto noiseAt= [](int step) {
					const float phase= (float)step;
					return glm::vec3(0.041f * sinf(phase * 2.3f), 0.041f * sinf(phase * 3.7f),
									 0.041f * sinf(phase * 5.1f));
				};

				TrackingFrameResult fused;
				float lastConfidence= 0.f;
				for (int step= 0; step < 40; ++step)
				{
					const double stepTime= now + step * 33.0;
					// jittery camera: high presence, noisy position
					const auto camA= makeCameraResult(
						0, cam1Pos, stepTime,
						makeObservation(palmTruth + noiseAt(step), faceUpToCam1, 0.87f, eHandSide::Left, 0.05f, 0.f));
					// steady camera: exactly on truth
					const auto camB= makeCameraResult(
						1, cam2Pos, stepTime,
						makeObservation(palmTruth, faceUpToCam1, 0.97f, eHandSide::Left, 0.05f, 0.f));

					freshFusion.fuse({&camA, &camB}, stepTime, fused);
					lastConfidence= fused.poses[(int)eHandSide::Left].confidence;
				}

				const float err= glm::length(fused.poses[(int)eHandSide::Left].palmPositionWorld - palmTruth);
				MIKAN_LOG_INFO("test-fusion") << "(k) stability weighting: fused err mm=" << err * 1000.f
					<< " confidence=" << lastConfidence;
				// Without stability weighting the jittery camera would drag the
				// blend tens of mm off truth
				if (err > 0.008f || lastConfidence < 0.5f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(k) FAILED: the steady camera must dominate and confidence stay high";
					result= 1;
				}
			}

			// (l) Hard confidence gate: with minCameraConfidence above what a
			// jittery camera can reach, its observation is dropped entirely
			{
				HandFusion freshFusion;
				HandFusionConfig gateConfig= fusionConfig;
				gateConfig.jitterReferenceM= 0.015f;
				gateConfig.minCameraConfidence= 0.5f;
				freshFusion.configure(gateConfig);

				TrackingFrameResult fused;
				for (int step= 0; step < 40; ++step)
				{
					const double stepTime= now + step * 33.0;
					// Alternating 8cm displacement = sustained large jitter
					const glm::vec3 noise= (step % 2 == 0) ? glm::vec3(0.08f, 0.f, 0.f) : glm::vec3(0.f);
					const auto camA= makeCameraResult(
						0, cam1Pos, stepTime,
						makeObservation(palmTruth + noise, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
					freshFusion.fuse({&camA}, stepTime, fused);
				}

				const HandPose& pose= fused.poses[(int)eHandSide::Left];
				MIKAN_LOG_INFO("test-fusion") << "(l) confidence gate: tracked=" << pose.tracked;
				if (pose.tracked)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(l) FAILED: an observation below minCameraConfidence must be dropped";
					result= 1;
				}

				// ...and a steady observation at the same presence survives
				HandFusion steadyFusion;
				steadyFusion.configure(gateConfig);
				TrackingFrameResult steadyFused;
				for (int step= 0; step < 40; ++step)
				{
					const double stepTime= now + step * 33.0;
					const auto camA= makeCameraResult(
						0, cam1Pos, stepTime,
						makeObservation(palmTruth, faceUpToCam1, 0.95f, eHandSide::Left, 0.05f, 0.f));
					steadyFusion.fuse({&camA}, stepTime, steadyFused);
				}
				if (!steadyFused.poses[(int)eHandSide::Left].tracked)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(l) FAILED: a steady observation must pass the gate";
					result= 1;
				}
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-fusion") << "All fusion checks passed";

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-dump")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-dump.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// Synthetic camera result: one tracked left hand + a detection box
			CameraFrameResult cameraResult;
			cameraResult.cameraIndex= 0;
			cameraResult.valid= true;
			cameraResult.timestampMs= 1000.0;
			cameraResult.hasExtrinsics= true;

			TrackingFrameResult& frame= cameraResult.result;
			frame.frameIndex= 42;
			frame.timestampMs= 1000.0;
			frame.frameWidth= 128;
			frame.frameHeight= 128;

			TrackedHand& hand= frame.hands[(int)eHandSide::Left];
			hand.tracked= true;
			hand.side= eHandSide::Left;
			hand.presence= 0.9f;
			hand.handednessScore= 0.1f;
			hand.hasWorldSpace= true;
			for (int landmark= 0; landmark < HAND_LANDMARK_COUNT; ++landmark)
			{
				hand.imagePoints[landmark]= glm::vec3(20.f + landmark * 4.f, 30.f + landmark * 3.f, 0.f);
				hand.worldPoints[landmark]= glm::vec3(0.1f, 0.05f, 0.1f + landmark * 0.001f);
			}

			HandPose& pose= frame.poses[(int)eHandSide::Left];
			pose.tracked= true;
			pose.side= eHandSide::Left;
			pose.presence= 0.9f;
			pose.hasWorldPose= true;
			pose.palmPositionWorld= glm::vec3(0.1f, 0.05f, 0.1f);
			pose.fingers[1].proximal= 0.5f;

			DetectionBox box;
			box.corners= {glm::vec2(10, 10), glm::vec2(60, 10), glm::vec2(60, 60), glm::vec2(10, 60)};
			frame.palmDetections.push_back(box);

			FusionDiagnostics diagnostics;
			diagnostics.totalObservations= 1;
			{
				FusionDiagnostics::Cluster cluster;
				cluster.palmWorld= pose.palmPositionWorld;
				cluster.assignedSide= (int)eHandSide::Left;
				FusionDiagnostics::Observation observation;
				observation.cameraIndex= 0;
				observation.labeledSide= (int)eHandSide::Left;
				observation.weight= 0.9f;
				cluster.observations.push_back(observation);
				diagnostics.clusters.push_back(cluster);
			}

			DiagnosticDump dump;
			const int dominant[2]= {0, -1};
			for (int record= 0; record < 3; ++record)
				dump.record({&cameraResult}, frame, diagnostics, dominant, 1.02f);

			cv::Mat testFrame(128, 128, CV_8UC3, cv::Scalar(40, 40, 40));
			DiagCameraSnapshot snapshot;
			snapshot.lastResult= &cameraResult;
			snapshot.frame= &testFrame;
			snapshot.deviceFps= 30.f;
			snapshot.droppedFrames= 1;
			snapshot.activeEp= "test";
			snapshot.trackingEnabled= true;

			const std::filesystem::path dumpDir=
				std::filesystem::temp_directory_path() / "mikanmediapipe_test_dump";
			std::filesystem::remove_all(dumpDir);

			AppConfig config;
			bool bOk= dump.write(dumpDir.string(), {snapshot}, frame, config.toJsonString());
			bOk&= std::filesystem::exists(dumpDir / "dump.json");
			bOk&= std::filesystem::exists(dumpDir / "cam0_raw.png");
			bOk&= std::filesystem::exists(dumpDir / "cam0_annotated.png");

			// Sanity-check the JSON payload: all top-level sections present,
			// history depth matches, affinity table serialized
			if (bOk)
			{
				std::ifstream jsonFile(dumpDir / "dump.json");
				const std::string content(
					(std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());
				for (const char* needle :
					 {"\"config\"", "\"cameras\"", "\"fusedSnapshot\"", "\"history\"", "\"affinity\"",
					  "\"imagePoints\"", "\"assignedSide\""})
				{
					if (content.find(needle) == std::string::npos)
					{
						MIKAN_LOG_ERROR("test-dump") << "dump.json missing section " << needle;
						bOk= false;
					}
				}
			}

			MIKAN_LOG_INFO("test-dump") << "dump dir: " << dumpDir.string() << " ok=" << bOk;
			if (!bOk)
			{
				MIKAN_LOG_ERROR("test-dump") << "FAILED";
				result= 1;
			}
			else
			{
				std::filesystem::remove_all(dumpDir);
				MIKAN_LOG_INFO("test-dump") << "All dump checks passed";
			}

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-handpose")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-handpose.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// Build a synthetic RIGHT-hand landmark set with known articulation
			// by forward kinematics over a hand-authored skeleton, then verify
			// angle extraction + FK round-trips.
			HandSkeleton skeleton;
			// INVARIANT: the middle finger's base lies exactly on palm +X - the
			// palm frame defines +X as wrist -> middle MCP, so computeSkeleton
			// always emits (d, 0, 0) for it. Authoring a nonzero y here would
			// make the recovered palm frame differ from the frame this skeleton
			// is expressed in, which silently rotates every extracted angle.
			const float baseY[FINGER_COUNT]= {0.045f, 0.03f, 0.f, -0.01f, -0.03f};
			const float baseX[FINGER_COUNT]= {-0.01f, 0.035f, 0.04f, 0.035f, 0.03f};
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				skeleton.baseInPalm[finger]= glm::vec3(baseX[finger], baseY[finger], 0.f);
				skeleton.phalanxLengths[finger]= {0.045f, 0.027f, 0.022f};
			}
			skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
			// NOTE on chirality: computePalmFrame derives +Z from the landmark
			// layout; this skeleton (thumb/index at +Y) matches a RIGHT hand
			// viewed in its own palm frame.

			// Round-trip helper: FK with the given angles -> landmark set ->
			// re-extract angles -> max absolute error
			std::array<glm::vec3, HAND_LANDMARK_COUNT> points{};
			auto roundTrip= [&](const std::array<FingerAngles, FINGER_COUNT>& anglesIn, const char* label) {
				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
				HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesIn, joints);

				const glm::vec3 middleBase= skeleton.baseInPalm[(int)eFinger::Middle];
				points[(int)eHandLandmark::WRIST]= glm::vec3(-middleBase.x, 0.f, 0.f);
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
					for (int joint= 0; joint < 4; ++joint)
						points[FINGER_JOINTS[finger][joint]]= joints[finger][joint];

				std::array<FingerAngles, FINGER_COUNT> anglesOut{};
				HandPoseModel::computeFingerAngles(points, eHandSide::Right, skeleton.neutralDirInPalm, anglesOut);

				float maxError= 0.f;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					maxError= std::max(maxError, fabsf(anglesOut[finger].lateral - anglesIn[finger].lateral));
					maxError= std::max(maxError, fabsf(anglesOut[finger].proximal - anglesIn[finger].proximal));
					maxError=
						std::max(maxError, fabsf(anglesOut[finger].intermediate - anglesIn[finger].intermediate));
					maxError= std::max(maxError, fabsf(anglesOut[finger].distal - anglesIn[finger].distal));
				}
				MIKAN_LOG_INFO("test-handpose") << label << ": round-trip max error rad=" << maxError;
				return maxError;
			};

			// Moderate articulation
			std::array<FingerAngles, FINGER_COUNT> anglesModerate{};
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				anglesModerate[finger].lateral= 0.05f * (float)(finger - 2);
				anglesModerate[finger].proximal= 0.3f + 0.1f * (float)finger;
				anglesModerate[finger].intermediate= 0.4f;
				anglesModerate[finger].distal= 0.2f;
			}
			if (roundTrip(anglesModerate, "moderate curl") > 0.02f)
			{
				MIKAN_LOG_ERROR("test-handpose") << "FAILED: moderate-curl round-trip error too large";
				result= 1;
			}

			// Deep curl (fist): combined proximal+intermediate bend passes 90
			// degrees, where the old per-joint palmZ-cross sign disambiguation
			// degenerated and flipped the distal sign (Z-shaped fingers)
			std::array<FingerAngles, FINGER_COUNT> anglesFist{};
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
			{
				anglesFist[finger].lateral= 0.02f * (float)(finger - 2);
				anglesFist[finger].proximal= 1.2f;
				anglesFist[finger].intermediate= 1.4f;
				anglesFist[finger].distal= 1.0f;
			}
			if (roundTrip(anglesFist, "deep curl (fist)") > 0.02f)
			{
				MIKAN_LOG_ERROR("test-handpose") << "FAILED: deep-curl round-trip error too large "
													"(distal sign degeneracy regression)";
				result= 1;
			}

			// Thumb opposition: large MCP/IP flexion must round-trip AND must
			// actually sweep the thumb tip ACROSS the palm toward the pinky
			// (the pronated-hinge behavior; on the finger-style hinge this
			// motion was nearly unobservable)
			{
				std::array<FingerAngles, FINGER_COUNT> anglesOpposition{};
				anglesOpposition[(int)eFinger::Thumb].lateral= -0.3f;
				anglesOpposition[(int)eFinger::Thumb].proximal= 0.4f;
				anglesOpposition[(int)eFinger::Thumb].intermediate= 1.1f;
				anglesOpposition[(int)eFinger::Thumb].distal= 0.5f;
				if (roundTrip(anglesOpposition, "thumb opposition") > 0.02f)
				{
					MIKAN_LOG_ERROR("test-handpose") << "FAILED: thumb-opposition round-trip error too large";
					result= 1;
				}

				// Directional check: flexing the thumb must move its tip toward
				// the pinky side (-Y in this right-hand skeleton), not stay in
				// the palmar bend plane
				std::array<FingerAngles, FINGER_COUNT> anglesStraight{};
				anglesStraight[(int)eFinger::Thumb].lateral= -0.3f;
				anglesStraight[(int)eFinger::Thumb].proximal= 0.4f;

				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> jointsFlexed, jointsStraight;
				HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesOpposition, jointsFlexed);
				HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, anglesStraight, jointsStraight);

				const float tipShiftTowardPinky=
					jointsStraight[(int)eFinger::Thumb][3].y - jointsFlexed[(int)eFinger::Thumb][3].y;
				MIKAN_LOG_INFO("test-handpose")
					<< "thumb flexion tip shift toward pinky mm=" << tipShiftTowardPinky * 1000.f;
				if (tipShiftTowardPinky < 0.02f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: thumb flexion must sweep the tip across the palm toward the pinky";
					result= 1;
				}
			}

			// Wrong-label robustness: MediaPipe's handedness classifier flips
			// when the palm rotates away from the camera. The palm-frame
			// chirality is derived geometrically (curl + thumb evidence), so
			// extracting with the WRONG side label from a curled hand must
			// still produce the same angles (this was the palm-down mirroring
			// bug: label-based chirality inverted every lateral/proximal).
			{
				std::array<FingerAngles, FINGER_COUNT> anglesWrongLabel{};
				HandPoseModel::computeFingerAngles(points, eHandSide::Left, skeleton.neutralDirInPalm, anglesWrongLabel);
				std::array<FingerAngles, FINGER_COUNT> anglesRightLabel{};
				HandPoseModel::computeFingerAngles(points, eHandSide::Right, skeleton.neutralDirInPalm, anglesRightLabel);

				float maxLabelError= 0.f;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					maxLabelError=
						std::max(maxLabelError, fabsf(anglesWrongLabel[finger].lateral - anglesRightLabel[finger].lateral));
					maxLabelError= std::max(maxLabelError,
											fabsf(anglesWrongLabel[finger].proximal - anglesRightLabel[finger].proximal));
					maxLabelError= std::max(
						maxLabelError, fabsf(anglesWrongLabel[finger].intermediate - anglesRightLabel[finger].intermediate));
					maxLabelError=
						std::max(maxLabelError, fabsf(anglesWrongLabel[finger].distal - anglesRightLabel[finger].distal));
				}
				MIKAN_LOG_INFO("test-handpose") << "wrong-handedness-label max angle delta rad=" << maxLabelError;
				if (maxLabelError > 0.001f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: angles must be invariant to a mislabeled handedness (geometric chirality)";
					result= 1;
				}
			}

			// Skeleton round-trip: recompute from the (last) landmark set
			HandSkeleton skeletonOut;
			HandPoseModel::computeSkeleton(points, eHandSide::Right, skeletonOut);
			float maxLenError= 0.f;
			for (int finger= 0; finger < FINGER_COUNT; ++finger)
				for (int phalanx= 0; phalanx < 3; ++phalanx)
					maxLenError= std::max(maxLenError, fabsf(skeletonOut.phalanxLengths[finger][phalanx] -
															 skeleton.phalanxLengths[finger][phalanx]));
			MIKAN_LOG_INFO("test-handpose") << "skeleton round-trip max length error mm=" << maxLenError * 1000.f;
			if (maxLenError > 0.001f)
			{
				MIKAN_LOG_ERROR("test-handpose") << "FAILED: phalanx length round-trip mismatch";
				result= 1;
			}

			// Palm frame sanity: +X toward fingers, origin midway wrist<->middleMCP
			const glm::mat4 palmFrame= HandPoseModel::computePalmFrame(points, eHandSide::Right);
			const glm::vec3 xAxis= glm::vec3(palmFrame[0]);
			const glm::vec3 towardFingers=
				glm::normalize(points[(int)eHandLandmark::MIDDLE_MCP] - points[(int)eHandLandmark::WRIST]);
			MIKAN_LOG_INFO("test-handpose") << "palm X . towardFingers=" << glm::dot(xAxis, towardFingers);
			if (glm::dot(xAxis, towardFingers) < 0.99f)
			{
				MIKAN_LOG_ERROR("test-handpose") << "FAILED: palm frame X axis mismatch";
				result= 1;
			}

			// Convention checks (the four the OSC schema promises). Work in the
			// palm frame directly: build FK from a single nonzero angle and
			// verify the joint moves the promised way.
			{
				auto fkWith= [&](int finger, float lat, float prox, float inter, float dist) {
					std::array<FingerAngles, FINGER_COUNT> a{};
					a[finger].lateral= lat;
					a[finger].proximal= prox;
					a[finger].intermediate= inter;
					a[finger].distal= dist;
					std::array<std::array<glm::vec3, 4>, FINGER_COUNT> j;
					HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, a, j);
					return j[finger];
				};

				const int indexFinger= (int)eFinger::Index;
				const std::array<glm::vec3, 4> neutral= fkWith(indexFinger, 0.f, 0.f, 0.f, 0.f);

				// (i) zero angles reproduce the neutral direction
				const glm::vec3 neutralBone= glm::normalize(neutral[1] - neutral[0]);
				const glm::vec3 expectedNeutral= glm::normalize(skeleton.neutralDirInPalm[indexFinger]);
				MIKAN_LOG_INFO("test-handpose")
					<< "convention: zero-angle bone . neutralDir=" << glm::dot(neutralBone, expectedNeutral);
				if (glm::dot(neutralBone, expectedNeutral) < 0.999f)
				{
					MIKAN_LOG_ERROR("test-handpose") << "FAILED: zero angles must reproduce the neutral direction";
					result= 1;
				}

				// (ii) positive proximal bends TOWARD the palm (+Z)
				const std::array<glm::vec3, 4> bent= fkWith(indexFinger, 0.f, 0.6f, 0.f, 0.f);
				const float bendTowardPalm= glm::normalize(bent[1] - bent[0]).z;
				MIKAN_LOG_INFO("test-handpose") << "convention: +proximal bone z=" << bendTowardPalm;
				if (bendTowardPalm < 0.1f)
				{
					MIKAN_LOG_ERROR("test-handpose") << "FAILED: +proximal must curl toward the palm (+Z)";
					result= 1;
				}

				// (iii) positive lateral is counter-clockwise about palm +Z,
				// i.e. toward palm +Y = cross(palmZ, palmX)
				const std::array<glm::vec3, 4> splayed= fkWith(indexFinger, 0.4f, 0.f, 0.f, 0.f);
				const glm::vec3 splayDelta= glm::normalize(splayed[1] - splayed[0]) - neutralBone;
				MIKAN_LOG_INFO("test-handpose") << "convention: +lateral delta y=" << splayDelta.y;
				if (splayDelta.y < 0.05f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: +lateral must rotate CCW about palm +Z (toward palm +Y)";
					result= 1;
				}

				// (iv) intermediate/distal are relative to their PARENT bone:
				// with a nonzero proximal, an intermediate of 0 must leave the
				// middle bone collinear with the proximal bone
				const glm::vec3 proximalDir= glm::normalize(bent[1] - bent[0]);
				const glm::vec3 intermediateDir= glm::normalize(bent[2] - bent[1]);
				MIKAN_LOG_INFO("test-handpose")
					<< "convention: zero-inter collinearity=" << glm::dot(proximalDir, intermediateDir);
				if (glm::dot(proximalDir, intermediateDir) < 0.999f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: intermediate/distal must be relative to the parent bone, not the palm";
					result= 1;
				}
			}

			// Rest-pose capture: feeding a pose back as its own neutral must
			// make that exact pose read all-zero angles
			{
				std::array<FingerAngles, FINGER_COUNT> restAngles{};
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					restAngles[finger].lateral= 0.15f - 0.05f * finger;
					restAngles[finger].proximal= 0.5f - 0.08f * finger;
				}
				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> restJoints;
				HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, restAngles, restJoints);

				std::array<glm::vec3, HAND_LANDMARK_COUNT> restPoints{};
				const glm::vec3 middleBaseRest= skeleton.baseInPalm[(int)eFinger::Middle];
				restPoints[(int)eHandLandmark::WRIST]= glm::vec3(-middleBaseRest.x, 0.f, 0.f);
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
					for (int joint= 0; joint < 4; ++joint)
						restPoints[FINGER_JOINTS[finger][joint]]= restJoints[finger][joint];

				const HandPoseModel::NeutralDirections captured=
					HandPoseModel::captureRestPose(restPoints, eHandSide::Right);

				std::array<FingerAngles, FINGER_COUNT> zeroed{};
				HandPoseModel::computeFingerAngles(restPoints, eHandSide::Right, captured, zeroed);

				float maxRest= 0.f;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					maxRest= std::max(maxRest, fabsf(zeroed[finger].lateral));
					maxRest= std::max(maxRest, fabsf(zeroed[finger].proximal));
				}
				MIKAN_LOG_INFO("test-handpose") << "rest capture: max residual lateral/proximal rad=" << maxRest;
				if (maxRest > 1e-3f)
				{
					MIKAN_LOG_ERROR("test-handpose") << "FAILED: a captured rest pose must read zero angles";
					result= 1;
				}
			}

			// FK reprojection: rebuild the hand from ONLY palm pose + skeleton +
			// angles, project into a synthetic camera, and compare against the
			// projection of the original 3D landmarks. End-to-end check that the
			// parameterization loses nothing a client needs to redraw the hand.
			{
				std::array<FingerAngles, FINGER_COUNT> poseAngles{};
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					poseAngles[finger].lateral= 0.1f - 0.04f * finger;
					poseAngles[finger].proximal= 0.35f + 0.05f * finger;
					poseAngles[finger].intermediate= 0.4f;
					poseAngles[finger].distal= 0.25f;
				}

				// Place the hand in front of a camera (OpenCV convention)
				const glm::quat handRotation= glm::angleAxis(0.6f, glm::normalize(glm::vec3(0.3f, 1.f, 0.2f)));
				glm::mat4 handTransform= glm::mat4_cast(handRotation);
				handTransform[3]= glm::vec4(0.02f, -0.01f, 0.45f, 1.f);

				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> truthJoints;
				HandPoseModel::buildFingerJoints(handTransform, skeleton, poseAngles, truthJoints);

				std::array<glm::vec3, HAND_LANDMARK_COUNT> truthPoints{};
				const glm::vec3 middleBaseFk= skeleton.baseInPalm[(int)eFinger::Middle];
				truthPoints[(int)eHandLandmark::WRIST]=
					glm::vec3(handTransform * glm::vec4(-middleBaseFk.x, 0.f, 0.f, 1.f));
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
					for (int joint= 0; joint < 4; ++joint)
						truthPoints[FINGER_JOINTS[finger][joint]]= truthJoints[finger][joint];

				// Extract the parametric representation, exactly as the app does
				const glm::mat4 extractedPalm= HandPoseModel::computePalmFrame(truthPoints, eHandSide::Right);
				HandSkeleton extractedSkeleton;
				HandPoseModel::computeSkeleton(truthPoints, eHandSide::Right, extractedSkeleton);
				extractedSkeleton.neutralDirInPalm= skeleton.neutralDirInPalm;
				std::array<FingerAngles, FINGER_COUNT> extractedAngles{};
				HandPoseModel::computeFingerAngles(truthPoints, eHandSide::Right,
												   extractedSkeleton.neutralDirInPalm, extractedAngles);

				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> rebuiltJoints;
				HandPoseModel::buildFingerJoints(extractedPalm, extractedSkeleton, extractedAngles, rebuiltJoints);

				// Project both into a 1280x720 pinhole camera
				const float fxTest= 900.f, fyTest= 900.f, cxTest= 640.f, cyTest= 360.f;
				auto project= [&](const glm::vec3& p) {
					return glm::vec2(fxTest * p.x / p.z + cxTest, fyTest * p.y / p.z + cyTest);
				};

				float maxReprojection= 0.f;
				float sumReprojection= 0.f;
				int reprojectionCount= 0;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					for (int joint= 0; joint < 4; ++joint)
					{
						const float error= glm::length(project(rebuiltJoints[finger][joint]) -
													   project(truthJoints[finger][joint]));
						maxReprojection= std::max(maxReprojection, error);
						sumReprojection+= error;
						reprojectionCount++;
					}
				}
				const float meanReprojection= sumReprojection / (float)reprojectionCount;
				MIKAN_LOG_INFO("test-handpose")
					<< "FK reprojection: mean px=" << meanReprojection << " max px=" << maxReprojection;
				if (maxReprojection > 1.f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: the FK hand must reproject onto the source landmarks";
					result= 1;
				}
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-handpose") << "All hand-pose checks passed";

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-pnp")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-pnp.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// Pinhole camera
			const float fx= 800.f, fy= 800.f, cx= 640.f, cy= 360.f;
			MikanMonoIntrinsics intrinsics;
			intrinsics.pixel_width= 1280;
			intrinsics.pixel_height= 720;
			intrinsics.undistorted_camera_matrix.x0= fx;
			intrinsics.undistorted_camera_matrix.y1= fy;
			intrinsics.undistorted_camera_matrix.z0= cx;
			intrinsics.undistorted_camera_matrix.z1= cy;

			// Canonical hand model in hand-local meters, wrist at the origin,
			// slightly non-planar (finger curl) so the pose is well-determined
			std::array<glm::vec3, HAND_LANDMARK_COUNT> modelPoints;
			for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
			{
				const float along= 0.02f + 0.15f * (float)(lm % 4) / 4.f * (lm >= 1 ? 1.f : 0.f);
				const float spread= ((float)(lm / 4) - 2.f) * 0.02f;
				const float curl= 0.012f * (float)(lm % 4);
				modelPoints[lm]= glm::vec3(spread, along, curl);
			}
			modelPoints[0]= glm::vec3(0.f);
			const float boneLength= glm::length(modelPoints[(int)eHandLandmark::MIDDLE_MCP]);

			// Ground-truth rigid pose (OpenCV camera convention)
			const glm::mat3 rotationTruth=
				glm::mat3(glm::rotate(glm::mat4(1.f), 0.4f, glm::vec3(0, 1, 0)) *
						  glm::rotate(glm::mat4(1.f), -0.3f, glm::vec3(1, 0, 0)));
			const glm::vec3 translationTruth(0.05f, -0.03f, 0.7f);

			auto buildHand= [&](std::array<glm::vec3, HAND_LANDMARK_COUNT>& outTruthCamera) {
				TrackedHand hand;
				hand.tracked= true;
				hand.side= eHandSide::Left;
				hand.presence= 0.9f;
				hand.modelPoints= modelPoints;
				for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
				{
					const glm::vec3 cameraPoint= rotationTruth * modelPoints[lm] + translationTruth;
					outTruthCamera[lm]= cameraPoint;
					hand.imagePoints[lm]= glm::vec3(
						fx * cameraPoint.x / cameraPoint.z + cx,
						fy * cameraPoint.y / cameraPoint.z + cy,
						0.f);
				}
				return hand;
			};

			auto runEstimator= [&](bool bUsePnp, bool bPalmOnly, const char* label) {
				LandmarkTo3D landmarkTo3D;
				landmarkTo3D.configure(intrinsics, boneLength, false, 1.f, 0.05f);
				landmarkTo3D.setPnpConfig(bUsePnp, bPalmOnly);

				std::array<glm::vec3, HAND_LANDMARK_COUNT> truthCamera;
				TrackingFrameResult frame;
				frame.timestampMs= 1000.0;
				frame.hands[(int)eHandSide::Left]= buildHand(truthCamera);
				landmarkTo3D.process(frame);

				const TrackedHand& hand= frame.hands[(int)eHandSide::Left];
				float sum= 0.f;
				for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
					sum+= glm::dot(hand.cameraPoints[lm] - truthCamera[lm], hand.cameraPoints[lm] - truthCamera[lm]);
				const float rms= sqrtf(sum / (float)HAND_LANDMARK_COUNT);

				MIKAN_LOG_INFO("test-pnp") << label << ": tracked=" << hand.hasCameraSpace
					<< " rms error mm=" << rms * 1000.f;
				return hand.hasCameraSpace ? rms : 1e9f;
			};

			// PnP paths must recover the exact synthetic pose (sub-mm)
			const float rmsPnpAll= runEstimator(true, false, "PnP all-21");
			const float rmsPnpPalm= runEstimator(true, true, "PnP palm-only");
			// Legacy is informative only (expected worse on a rotated hand)
			const float rmsLegacy= runEstimator(false, false, "Legacy two-point");

			if (rmsPnpAll > 0.001f || rmsPnpPalm > 0.001f)
			{
				MIKAN_LOG_ERROR("test-pnp") << "FAILED: PnP must recover the exact synthetic pose";
				result= 1;
			}
			if (rmsLegacy < rmsPnpAll)
			{
				MIKAN_LOG_INFO("test-pnp") << "note: legacy beat PnP on this synthetic (unexpected but not fatal)";
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-pnp") << "All PnP checks passed";

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
