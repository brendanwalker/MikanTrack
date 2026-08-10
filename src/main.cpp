#include <string>

#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "opencv2/calib3d.hpp"
#include "nlohmann/json.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>

#include "App.h"
#include "AppConfig.h"
#include "ArucoMarkerPoseSampler.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "DiagnosticDump.h"
#include "ExtrinsicsWizard.h"
#include "HandFusion.h"
#include "HandPoseModel.h"
#include "HandRoiQuality.h"
#include "HandTrackingPipeline.h"
#include "LandmarkTo3D.h"
#include "Logger.h"
#include "MonoLensDistortionCalibrator.h"
#include "OscWriterTest.h"
#include "DepthFrameView.h"
#include "ImuOrientationFilter.h"
#include "ImuService.h"
#include "JoyconDevice.h"
#include "JoyconDeviceManager.h"

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
			loggerSettings.log_filename= "test-osc.log";
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
					// Generous margin: step 6 shifts the board 150px and detection
					// requires the ENTIRE board visible (no partial captures)
					board.generateImage(cv::Size(1100, 800), boardImage, 180, 1);

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

				// 7: End-to-end intrinsics recovery through a KNOWN synthetic
				// camera: project the board's corner geometry through ground-truth
				// intrinsics at a spread of tilted poses and verify the solve
				// recovers the true hfov. Regression guard for the free-k6
				// rational-model bug (hfov of 180/8.6 deg with sub-pixel
				// reprojection error) and the new plausibility gate.
				MIKAN_LOG_INFO("test-charuco") << "7: synthetic intrinsics recovery";
				{
					const int width= 1280, height= 720;
					const double fx= 800.0;
					const double truthHfov= 2.0 * glm::degrees(atan((width * 0.5) / fx)); // 77.32 deg

					// 10x7 corner grid, 16mm squares (matches an 11x8 board)
					OpenCVCalibrationGeometry geometry;
					for (int row= 0; row < 7; ++row)
						for (int col= 0; col < 10; ++col)
							geometry.points.push_back(cv::Point3f(col * 16.f, row * 16.f, 0.f));

					const cv::Matx33d cameraMatrix(fx, 0, width * 0.5, 0, fx, height * 0.5, 0, 0, 1);
					const cv::Mat noDistortion= cv::Mat::zeros(1, 8, CV_64F);

					std::vector<t_opencv_point2d_list> imagePointsList;
					std::vector<t_opencv_pointID_list> imagePointIDList;
					// 12 poses: varied tilt about X and Y, varied depth + offset
					for (int sample= 0; sample < 12; ++sample)
					{
						const double tiltX= glm::radians(-25.0 + 10.0 * (sample % 3));
						const double tiltY= glm::radians(-20.0 + 8.0 * (sample % 5));
						const cv::Vec3d rvecX(tiltX, 0, 0), rvecY(0, tiltY, 0);
						cv::Matx33d rotX, rotY;
						cv::Rodrigues(rvecX, rotX);
						cv::Rodrigues(rvecY, rotY);
						cv::Vec3d rvec;
						cv::Rodrigues(rotY * rotX, rvec);
						const cv::Vec3d tvec(-80.0 + 30.0 * (sample % 4), -60.0 + 25.0 * (sample % 3),
											 350.0 + 40.0 * sample);

						t_opencv_point2d_list imagePoints;
						cv::projectPoints(geometry.points, rvec, tvec, cameraMatrix, noDistortion, imagePoints);

						t_opencv_pointID_list pointIDs;
						for (int id= 0; id < (int)geometry.points.size(); ++id)
							pointIDs.push_back(id);

						imagePointsList.push_back(imagePoints);
						imagePointIDList.push_back(pointIDs);
					}

					MikanMonoIntrinsics recovered;
					double reprojectionError= 0.0;
					const bool bSolved= computeMonoLensCameraCalibration(
						width, height, geometry, imagePointsList, imagePointIDList, recovered, reprojectionError);

					MIKAN_LOG_INFO("test-charuco")
						<< "recovered hfov=" << recovered.hfov << " (truth " << truthHfov
						<< ") reprojection=" << reprojectionError << "px";
					if (!bSolved || fabs(recovered.hfov - truthHfov) > 2.0 || reprojectionError > 0.5)
					{
						MIKAN_LOG_ERROR("test-charuco")
							<< "REGRESSION: synthetic intrinsics recovery failed";
						result= 1;
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

					// THE MARKER DEFINES THE WORLD AXES, not just the origin.
					// The synthetic marker above was laid down with its own
					// pattern-right along +X and its pattern-top along +Y, so
					// recovering a point that way round is what fixes the
					// convention the printed sheet and its labels depend on.
					//
					// This is asserted separately from the raycast check
					// because the two fail for different reasons and only one
					// of them means "the world frame silently rotated".
					{
						// A point one marker-half toward the pattern's right
						// edge, and one toward its top edge
						const cv::Point2f rightPixel= projectTablePoint(halfLenM, 0.0);
						const cv::Point2f topPixel= projectTablePoint(0.0, halfLenM);

						glm::dvec3 rightCameraSpace, topCameraSpace;
						const bool bCast=
							ExtrinsicsWizard::raycastPixelOntoPlane(intrinsics, cameraFromMarker,
																	glm::vec2(rightPixel.x, rightPixel.y),
																	rightCameraSpace) &&
							ExtrinsicsWizard::raycastPixelOntoPlane(intrinsics, cameraFromMarker,
																	glm::vec2(topPixel.x, topPixel.y),
																	topCameraSpace);
						if (!bCast)
						{
							MIKAN_LOG_ERROR("test-extrinsics") << "REGRESSION: axis raycast failed";
							result= 1;
						}
						else
						{
							const glm::dvec3 rightWorld=
								glm::normalize(glm::dvec3(worldFromCamera * glm::dvec4(rightCameraSpace, 1.0)));
							const glm::dvec3 topWorld=
								glm::normalize(glm::dvec3(worldFromCamera * glm::dvec4(topCameraSpace, 1.0)));

							const double rightAlongX= glm::dot(rightWorld, glm::dvec3(1.0, 0.0, 0.0));
							const double topAlongY= glm::dot(topWorld, glm::dvec3(0.0, 1.0, 0.0));

							MIKAN_LOG_INFO("test-extrinsics")
								<< "Marker axes: pattern-right . +X = " << rightAlongX
								<< ", pattern-top . +Y = " << topAlongY << " (both expected +1)";

							if (rightAlongX < 0.99 || topAlongY < 0.99)
							{
								MIKAN_LOG_ERROR("test-extrinsics")
									<< "REGRESSION: the world frame no longer matches the printed marker's "
									   "labels - pattern-right must be world +X and pattern-top world +Y";
								result= 1;
							}
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

			// (a) Both cameras face-on-ish: fused position beats both noisy
			// inputs; the bend angle comes from the SELECTED source camera (the
			// face-on overhead one), not a blend of the two
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
				if (!pose.tracked || errFused > std::min(errA, errB) * 1.05f || fabsf(bend - 0.5f) > 1e-6f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(a) FAILED";
					result= 1;
				}
			}

			// (b) Palm edge-on to the overhead camera, facing camera 2:
			// camera 2 must dominate (fresh fusion state - this tests the
			// weighting, not the articulation-source hysteresis)
			{
				HandFusion freshFusion;
				freshFusion.configure(fusionConfig);
				const auto camA= makeCameraResult(
					0, cam1Pos, now, makeObservation(palmTruth + glm::vec3(0.01f, 0.f, 0.f), faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));
				const auto camB= makeCameraResult(
					1, cam2Pos, now, makeObservation(palmTruth, faceCam2, 0.9f, eHandSide::Left, 0.1f, 0.f));

				TrackingFrameResult fused;
				freshFusion.fuse({&camA, &camB}, now, fused);

				MIKAN_LOG_INFO("test-fusion") << "(b) dominant camera=" << freshFusion.getDominantCamera(eHandSide::Left);
				if (freshFusion.getDominantCamera(eHandSide::Left) != 1)
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

			// (m) Stereo landmark triangulation: two cameras with full projective
			// geometry observe one synthetic hand. The fused pose must come from
			// the triangulated landmarks (exact recovery), NOT from the
			// (deliberately corrupted) per-camera monocular poses. A second run
			// feeds one camera a DIFFERENT physical hand's pixels - the
			// reprojection residual must veto the pairing.
			{
				// Authored RIGHT-hand skeleton (same conventions as --test-handpose;
				// middle finger base exactly on palm +X)
				HandSkeleton skeleton;
				const float baseY[FINGER_COUNT]= {0.045f, 0.03f, 0.f, -0.01f, -0.03f};
				const float baseX[FINGER_COUNT]= {-0.01f, 0.035f, 0.04f, 0.035f, 0.03f};
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					skeleton.baseInPalm[finger]= glm::vec3(baseX[finger], baseY[finger], 0.f);
					skeleton.phalanxLengths[finger]= {0.045f, 0.027f, 0.022f};
				}
				skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);

				std::array<FingerAngles, FINGER_COUNT> anglesTruth{};
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					anglesTruth[finger].lateral= 0.04f * (float)(finger - 2);
					anglesTruth[finger].proximal= 0.25f + 0.1f * (float)finger;
					anglesTruth[finger].intermediate= 0.35f;
					anglesTruth[finger].distal= 0.15f;
				}

				// World-space hand at palmTruth (identity orientation: palm +Z up,
				// facing the overhead camera)
				auto buildWorldHand= [&](const glm::vec3& palmCenter,
										 std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints) {
					glm::mat4 palmTransform(1.f);
					palmTransform[3]= glm::vec4(palmCenter, 1.f);
					std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
					HandPoseModel::buildFingerJoints(palmTransform, skeleton, anglesTruth, joints);
					const glm::vec3 middleBase= skeleton.baseInPalm[(int)eFinger::Middle];
					outPoints[(int)eHandLandmark::WRIST]= palmCenter + glm::vec3(-middleBase.x, 0.f, 0.f);
					for (int finger= 0; finger < FINGER_COUNT; ++finger)
						for (int joint= 0; joint < 4; ++joint)
							outPoints[FINGER_JOINTS[finger][joint]]= joints[finger][joint];
				};

				// OpenCV-convention camera looking at a target (markerFromCamera
				// columns = camera axes in world; +Z toward the scene)
				auto makeLookAtCamera= [](const glm::vec3& cameraPos, const glm::vec3& target) {
					const glm::vec3 z= glm::normalize(target - cameraPos);
					const glm::vec3 x= glm::normalize(glm::cross(glm::vec3(0.f, 1.f, 0.f), z));
					const glm::vec3 y= glm::cross(z, x);
					glm::dmat4 markerFromCamera(1.0);
					markerFromCamera[0]= glm::dvec4(x, 0.0);
					markerFromCamera[1]= glm::dvec4(y, 0.0);
					markerFromCamera[2]= glm::dvec4(z, 0.0);
					markerFromCamera[3]= glm::dvec4(cameraPos, 1.0);
					return markerFromCamera;
				};

				const float fx= 600.f, fy= 600.f, cx= 640.f, cy= 360.f;
				auto projectTo= [&](const glm::dmat4& markerFromCamera, const glm::vec3& world) {
					const glm::dvec4 camPoint= glm::inverse(markerFromCamera) * glm::dvec4(glm::dvec3(world), 1.0);
					return glm::vec3((float)(fx * camPoint.x / camPoint.z + cx),
									 (float)(fy * camPoint.y / camPoint.z + cy), 0.f);
				};

				// A camera observation: real image points (projected from
				// worldHand), but a corrupted monocular pose - 3cm depth error
				// along the view ray and +0.2 rad on every proximal angle. If any
				// of that corruption reaches the fused output, the stereo path
				// didn't run.
				auto makeStereoResult= [&](int cameraIndex, const glm::vec3& cameraPos,
										   const std::array<glm::vec3, HAND_LANDMARK_COUNT>& imageHand,
										   const std::array<glm::vec3, HAND_LANDMARK_COUNT>& worldHand,
										   const glm::vec3& palmCenter) {
					const glm::dmat4 markerFromCamera= makeLookAtCamera(cameraPos, palmCenter);

					TrackingFrameResult frame;
					TrackedHand& hand= frame.hands[(int)eHandSide::Right];
					hand.tracked= true;
					hand.side= eHandSide::Right;
					hand.presence= 0.9f;
					hand.handednessScore= 0.9f;
					hand.rightProb= 0.9f;
					for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
						hand.imagePoints[i]= projectTo(markerFromCamera, imageHand[i]);
					hand.worldPoints= worldHand; // carries the assumed hand scale
					hand.hasWorldSpace= true;

					HandPose& pose= frame.poses[(int)eHandSide::Right];
					pose.tracked= true;
					pose.side= eHandSide::Right;
					pose.presence= 0.9f;
					pose.hasWorldPose= true;
					pose.palmPositionWorld= palmCenter + glm::normalize(palmCenter - cameraPos) * 0.03f;
					pose.palmOrientationWorld= glm::quat(1.f, 0.f, 0.f, 0.f);
					pose.fingers= anglesTruth;
					for (int finger= 0; finger < FINGER_COUNT; ++finger)
						pose.fingers[finger].proximal+= 0.2f; // monocular articulation error
					pose.skeleton= skeleton;

					CameraFrameResult camera;
					camera.cameraIndex= cameraIndex;
					camera.valid= true;
					camera.timestampMs= now;
					camera.hasExtrinsics= true;
					camera.markerFromCamera= markerFromCamera;
					camera.hasIntrinsics= true;
					camera.fx= fx;
					camera.fy= fy;
					camera.cx= cx;
					camera.cy= cy;
					camera.result= frame;
					return camera;
				};

				std::array<glm::vec3, HAND_LANDMARK_COUNT> worldHand;
				buildWorldHand(palmTruth, worldHand);
				const glm::vec3 camAPos= palmTruth + glm::vec3(0.f, 0.f, 0.8f);
				const glm::vec3 camBPos= palmTruth + glm::vec3(0.f, -0.55f, 0.4f);

				HandFusion triFusion;
				triFusion.configure(fusionConfig); // triangulation on by default

				const auto camA= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
				const auto camB= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
				TrackingFrameResult fused;
				triFusion.fuse({&camA, &camB}, now, fused);

				const HandPose& pose= fused.poses[(int)eHandSide::Right];
				float maxAngleError= 0.f;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].lateral - anglesTruth[finger].lateral));
					maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].proximal - anglesTruth[finger].proximal));
					maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].intermediate - anglesTruth[finger].intermediate));
					maxAngleError= std::max(maxAngleError, fabsf(pose.fingers[finger].distal - anglesTruth[finger].distal));
				}
				const float palmError= glm::length(pose.palmPositionWorld - palmTruth);
				const FusionDiagnostics& diagnostics= triFusion.getLastDiagnostics();
				const bool bDiagTriangulated=
					!diagnostics.clusters.empty() && diagnostics.clusters[0].triangulated;
				MIKAN_LOG_INFO("test-fusion") << "(m) triangulation: stereoTriangulated=" << pose.stereoTriangulated
					<< " palm err mm=" << palmError * 1000.f << " max angle err rad=" << maxAngleError
					<< " residual px=" << (diagnostics.clusters.empty() ? -1.f : diagnostics.clusters[0].triResidualRmsPx);
				// The corrupted mono poses had 30mm palm error and +0.2 rad on the
				// proximals - exact recovery proves the stereo geometry won
				if (!pose.tracked || !pose.stereoTriangulated || !bDiagTriangulated ||
					palmError > 0.002f || maxAngleError > 0.02f)
				{
					MIKAN_LOG_ERROR("test-fusion") << "(m) FAILED: triangulated pose must recover the true hand";
					result= 1;
				}

				// Residual gating: deterministic ~10px 2D noise on camera B makes
				// the triangulation survive (well under the veto) but the
				// residual factor must visibly reduce the fused confidence.
				// (The midpoint solve splits one view's error across both views'
				// residuals, so the RMS lands well below the injected amplitude.)
				{
					auto camBNoisy= camB;
					TrackedHand& noisyHand= camBNoisy.result.hands[(int)eHandSide::Right];
					for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
					{
						noisyHand.imagePoints[i].x+= (i % 2 == 0) ? 10.f : -10.f;
						noisyHand.imagePoints[i].y+= (float)((i % 3) - 1) * 10.f;
					}

					HandFusion noisyFusion;
					noisyFusion.configure(fusionConfig);
					TrackingFrameResult noisyFused;
					noisyFusion.fuse({&camA, &camBNoisy}, now, noisyFused);

					const HandPose& noisyPose= noisyFused.poses[(int)eHandSide::Right];
					const float cleanConfidence= pose.confidence;
					MIKAN_LOG_INFO("test-fusion") << "(m) residual gate: clean confidence=" << cleanConfidence
						<< " noisy confidence=" << noisyPose.confidence << " residual px="
						<< noisyFusion.getLastDiagnostics().clusters[0].triResidualRmsPx;
					if (!noisyPose.stereoTriangulated || cleanConfidence < 0.85f ||
						noisyPose.confidence > cleanConfidence - 0.05f || noisyPose.confidence < 0.1f)
					{
						MIKAN_LOG_ERROR("test-fusion")
							<< "(m) FAILED: 2D noise must reduce confidence via the residual factor";
						result= 1;
					}
				}

				// Mismatched pairing: camera B's pixels come from a DIFFERENT hand
				// 15cm away, while both monocular poses still cluster together.
				// The reprojection residual must veto, and the output falls back
				// to the best single observation (still tracked).
				std::array<glm::vec3, HAND_LANDMARK_COUNT> otherHand;
				buildWorldHand(palmTruth + glm::vec3(0.15f, 0.f, 0.f), otherHand);

				HandFusion vetoFusion;
				vetoFusion.configure(fusionConfig);
				const auto camBWrong= makeStereoResult(1, camBPos, otherHand, worldHand, palmTruth);
				TrackingFrameResult vetoFused;
				vetoFusion.fuse({&camA, &camBWrong}, now, vetoFused);

				const HandPose& vetoPose= vetoFused.poses[(int)eHandSide::Right];
				const FusionDiagnostics& vetoDiagnostics= vetoFusion.getLastDiagnostics();
				const bool bVetoed=
					!vetoDiagnostics.clusters.empty() && vetoDiagnostics.clusters[0].triVetoed;
				MIKAN_LOG_INFO("test-fusion") << "(m) veto: tracked=" << vetoPose.tracked
					<< " stereoTriangulated=" << vetoPose.stereoTriangulated << " vetoed=" << bVetoed
					<< " residual px="
					<< (vetoDiagnostics.clusters.empty() ? -1.f : vetoDiagnostics.clusters[0].triResidualRmsPx);
				if (!vetoPose.tracked || vetoPose.stereoTriangulated || !bVetoed)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(m) FAILED: a mismatched pairing must be vetoed by the reprojection residual";
					result= 1;
				}

				// (o) Solo-cluster rescue: camera B's MONO palm is 30cm off
				// laterally (as measured live during a pointing gesture), so
				// position clustering strands it - but its image points are
				// good, and the probe triangulation must pair it anyway
				{
					auto camBFar= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
					camBFar.result.poses[(int)eHandSide::Right].palmPositionWorld=
						palmTruth + glm::vec3(0.3f, 0.f, 0.f);

					HandFusion rescueFusion;
					rescueFusion.configure(fusionConfig);
					TrackingFrameResult rescueFused;
					rescueFusion.fuse({&camA, &camBFar}, now, rescueFused);
					const bool bClusterRescued= rescueFused.poses[(int)eHandSide::Right].stereoTriangulated;

					// ...and the same via the low-presence pool: camera B's
					// pose is too weak to be a clustering candidate at all
					auto camBWeak= makeStereoResult(1, camBPos, worldHand, worldHand, palmTruth);
					camBWeak.result.poses[(int)eHandSide::Right].presence= 0.3f;
					camBWeak.result.hands[(int)eHandSide::Right].presence= 0.3f;

					HandFusion poolFusion;
					poolFusion.configure(fusionConfig);
					TrackingFrameResult poolFused;
					poolFusion.fuse({&camA, &camBWeak}, now, poolFused);
					const bool bPoolRescued= poolFused.poses[(int)eHandSide::Right].stereoTriangulated;

					MIKAN_LOG_INFO("test-fusion") << "(o) rescue: displaced-mono=" << bClusterRescued
						<< " low-presence-pool=" << bPoolRescued;
					if (!bClusterRescued || !bPoolRescued)
					{
						MIKAN_LOG_ERROR("test-fusion")
							<< "(o) FAILED: solo clusters must pair via probe triangulation";
						result= 1;
					}
				}

				// (p) Triangulated-angle hold: after a stereo fuse, a brief
				// single-camera fallback must keep the triangulated angles
				// (the mono articulation carries pose-dependent bias, +0.2 rad
				// here); a sustained fallback adopts the mono angles
				{
					HandFusion holdFusion;
					holdFusion.configure(fusionConfig);

					TrackingFrameResult holdFused;
					holdFusion.fuse({&camA, &camB}, now, holdFused);
					const float triProx= holdFused.poses[(int)eHandSide::Right].fingers[1].proximal;

					auto camASolo= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
					camASolo.timestampMs= now + 100.0;
					holdFusion.fuse({&camASolo}, now + 100.0, holdFused);
					const HandPose& heldPose= holdFused.poses[(int)eHandSide::Right];
					const float heldProx= heldPose.fingers[1].proximal;

					auto camALate= makeStereoResult(0, camAPos, worldHand, worldHand, palmTruth);
					camALate.timestampMs= now + 500.0;
					holdFusion.fuse({&camALate}, now + 500.0, holdFused);
					const float lateProx= holdFused.poses[(int)eHandSide::Right].fingers[1].proximal;

					const float monoProx= anglesTruth[1].proximal + 0.2f;
					MIKAN_LOG_INFO("test-fusion") << "(p) angle hold: tri=" << triProx << " held=" << heldProx
						<< " late=" << lateProx << " (mono=" << monoProx << ")";
					if (heldPose.stereoTriangulated || fabsf(heldProx - triProx) > 1e-4f ||
						fabsf(lateProx - monoProx) > 1e-4f)
					{
						MIKAN_LOG_ERROR("test-fusion")
							<< "(p) FAILED: brief fallback must hold tri angles; sustained fallback goes mono";
						result= 1;
					}
				}
			}

			// (n) Articulation-source hysteresis (non-triangulated path): the
			// incumbent camera keeps supplying angles until a challenger beats
			// its weight decisively for several consecutive fuses - weight noise
			// alone must not flip the source
			{
				HandFusion selFusion;
				selFusion.configure(fusionConfig);

				auto fuseStep= [&](int step, float presenceA, float presenceB, TrackingFrameResult& outFused) {
					const double stepTime= now + step * 33.0;
					const auto camA= makeCameraResult(
						0, cam1Pos, stepTime,
						makeObservation(palmTruth, faceUpToCam1, presenceA, eHandSide::Left, 0.1f, 0.5f));
					const auto camB= makeCameraResult(
						1, cam2Pos, stepTime,
						makeObservation(palmTruth, faceUpToCam1, presenceB, eHandSide::Left, 0.1f, 0.9f));
					selFusion.fuse({&camA, &camB}, stepTime, outFused);
				};

				TrackingFrameResult fused;
				// Establish camera A as the incumbent
				for (int step= 0; step < 3; ++step)
					fuseStep(step, 0.9f, 0.7f, fused);
				const float bendIncumbent= fused.poses[(int)eHandSide::Left].fingers[0].proximal;

				// Challenger decisively ahead (incumbent presence stays at the
				// candidate threshold so it isn't dropped outright): the
				// incumbent must survive the first kArticulationSwitchFrames-1
				// fuses...
				float bendHolding= -1.f;
				for (int step= 3; step < 3 + 4; ++step)
				{
					fuseStep(step, 0.5f, 0.99f, fused);
					bendHolding= fused.poses[(int)eHandSide::Left].fingers[0].proximal;
				}
				// ...and lose the job on the 5th
				fuseStep(7, 0.5f, 0.99f, fused);
				const float bendSwitched= fused.poses[(int)eHandSide::Left].fingers[0].proximal;

				MIKAN_LOG_INFO("test-fusion") << "(n) articulation selection: incumbent bend=" << bendIncumbent
					<< " holding bend=" << bendHolding << " switched bend=" << bendSwitched;
				if (fabsf(bendIncumbent - 0.5f) > 1e-6f || fabsf(bendHolding - 0.5f) > 1e-6f ||
					fabsf(bendSwitched - 0.9f) > 1e-6f)
				{
					MIKAN_LOG_ERROR("test-fusion")
						<< "(n) FAILED: source must hold through the hysteresis window, then switch";
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
			// One side with a live IMU, one without - the writer must emit both
			DiagImuState imuStates[2];
			imuStates[0].deviceConnected= true;
			imuStates[0].streaming= true;
			imuStates[0].calibrated= true;
			imuStates[0].orientationValid= true;
			imuStates[0].sampleRateHz= 200.f;
			imuStates[0].millisecondsSinceLastSample= 4.0;
			imuStates[0].forearmAxisConsistency= 0.88f;
			imuStates[0].armAxisDominance= 0.94f;
			imuStates[0].gyroBiasDegreesPerSecond= glm::vec3(0.1f, -0.2f, 0.3f);
			imuStates[0].yawSigmaRadians= 0.02f;
			for (int record= 0; record < 3; ++record)
				dump.record({&cameraResult}, frame, diagnostics, dominant, 1.02f, imuStates);

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
			std::vector<DiagImuRawSample> rawImu[2];
			DiagImuRawSample rawSample;
			rawSample.timestampMs= 12.0;
			rawSample.acceleration= glm::vec3(0.f, 0.f, 9.8f);
			rawSample.angularVelocity= glm::vec3(0.1f, -0.2f, 0.3f);
			rawImu[0].push_back(rawSample);

			DiagImuCapture lastCapture[2];
			lastCapture[0].present= true;
			lastCapture[0].poseSamples= 240;
			lastCapture[0].poseSpreadDegrees= 12.5f;
			lastCapture[0].motionUsable= true;
			lastCapture[0].axisDominance= 0.93f;

			bool bOk= dump.write(dumpDir.string(), {snapshot}, frame, config.toJsonString(), rawImu,
								 lastCapture);
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
					  "\"imagePoints\"", "\"assignedSide\"", "\"imu\"", "\"imuRaw\"", "\"imuLastCapture\"", "\"filterOrientation\"",
					  "\"gravityAcceptRatio\"", "\"armAxisDominance\"",
					  "\"forearmAxisConsistency\""})
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

				// Capturing the rest angles and subtracting them must zero ALL
				// FOUR degrees of freedom, intermediate and distal included
				std::array<FingerAngles, FINGER_COUNT> capturedRest{};
				HandPoseModel::captureRestAngles(restPoints, eHandSide::Right, capturedRest);

				HandSkeleton restSkeleton;
				HandPoseModel::computeSkeleton(restPoints, eHandSide::Right, restSkeleton);
				std::array<FingerAngles, FINGER_COUNT> measured{};
				HandPoseModel::computeFingerAngles(restPoints, eHandSide::Right,
												   restSkeleton.neutralDirInPalm, measured);

				float maxRest= 0.f;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
				{
					maxRest= std::max(maxRest, fabsf(measured[finger].lateral - capturedRest[finger].lateral));
					maxRest= std::max(maxRest, fabsf(measured[finger].proximal - capturedRest[finger].proximal));
					maxRest= std::max(
						maxRest, fabsf(measured[finger].intermediate - capturedRest[finger].intermediate));
					maxRest= std::max(maxRest, fabsf(measured[finger].distal - capturedRest[finger].distal));
				}
				MIKAN_LOG_INFO("test-handpose") << "rest capture: max residual (all 4 DoF) rad=" << maxRest;
				if (maxRest > 1e-5f)
				{
					MIKAN_LOG_ERROR("test-handpose") << "FAILED: a captured rest pose must read zero angles";
					result= 1;
				}

				// ...and a DIFFERENT pose must still read its true deviation,
				// i.e. the offset shifts zero without distorting the scale
				std::array<FingerAngles, FINGER_COUNT> movedAngles= restAngles;
				movedAngles[(int)eFinger::Index].proximal+= 0.3f;
				movedAngles[(int)eFinger::Index].intermediate+= 0.2f;
				std::array<std::array<glm::vec3, 4>, FINGER_COUNT> movedJoints;
				HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, movedAngles, movedJoints);

				std::array<glm::vec3, HAND_LANDMARK_COUNT> movedPoints= restPoints;
				for (int finger= 0; finger < FINGER_COUNT; ++finger)
					for (int joint= 0; joint < 4; ++joint)
						movedPoints[FINGER_JOINTS[finger][joint]]= movedJoints[finger][joint];

				std::array<FingerAngles, FINGER_COUNT> movedMeasured{};
				HandPoseModel::computeFingerAngles(movedPoints, eHandSide::Right,
												   restSkeleton.neutralDirInPalm, movedMeasured);
				const float proximalDeviation= movedMeasured[(int)eFinger::Index].proximal -
					capturedRest[(int)eFinger::Index].proximal;
				const float intermediateDeviation= movedMeasured[(int)eFinger::Index].intermediate -
					capturedRest[(int)eFinger::Index].intermediate;
				MIKAN_LOG_INFO("test-handpose") << "rest capture: deviation prox=" << proximalDeviation
					<< " inter=" << intermediateDeviation << " (expected 0.3 / 0.2)";
				if (fabsf(proximalDeviation - 0.3f) > 1e-3f || fabsf(intermediateDeviation - 0.2f) > 1e-3f)
				{
					MIKAN_LOG_ERROR("test-handpose")
						<< "FAILED: rest-relative angles must still report true deviation";
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

		// WHAT THIS CANNOT TELL YOU: it compares the gyro against the
		// accelerometer WITHIN one device, so it is blind to any error in the
		// sensor frame as a whole. Acceleration is a vector and angular
		// velocity is a pseudovector, so under a reflection both pick up sign
		// flips that cancel in dg/dt = -w x g - a mirrored frame satisfies the
		// correct physical relation and passes cleanly.
		//
		// Settling a frame convention needs an EXTERNAL reference. The one
		// that worked was strapping both controllers rigidly together and
		// solving for the transform between their raw streams: identical
		// physical motion, no vision, no mounting, no wrist, and the
		// determinant answers rotation-vs-reflection outright. Chasing it
		// through vision instead produced three inconclusive analyses and two
		// confidently wrong corrections.
		// Solves the fixed transform between TWO rigidly coupled IMUs from a
		// dump's raw sample history, and reports whether it is a rotation or a
		// reflection.
		//
		// This is the measurement that settles a sensor frame convention, and
		// it is here as a tool because the alternative - inferring it from
		// vision - produced three inconclusive analyses and two confidently
		// wrong corrections before a five-minute rigid-body capture answered
		// it outright. Strap both controllers to one object, tumble it through
		// all three axes for a few seconds, press F9, and run this on the dump.
		//
		// det(Q) = +1 : the frames differ by a ROTATION, which the mounting
		//               calibration absorbs. No correction is needed.
		// det(Q) = -1 : a REFLECTION, which no quaternion mounting can
		//               represent, so it must be corrected in the raw samples.
		//
		// The scale ratio comes free and is worth reading: on one rigid body
		// both gyros must report the same rate, so a ratio far from 1 is a
		// scale-factor problem that no axis remapping would ever fix.
		if (std::string(argv[i]) == "--test-imupair")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-imupair.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			if (i + 1 >= argc)
			{
				MIKAN_LOG_ERROR("test-imupair") << "usage: --test-imupair <path to dump.json>";
				log_dispose();
				return 1;
			}

			int result= 0;
			try
			{
				std::ifstream dumpFile(argv[i + 1]);
				if (!dumpFile.is_open())
				{
					MIKAN_LOG_ERROR("test-imupair") << "Could not open " << argv[i + 1];
					log_dispose();
					return 1;
				}
				nlohmann::json dump;
				dumpFile >> dump;

				auto readSide= [&dump](const char* side, std::vector<double>& outTimes,
									   std::vector<cv::Vec3d>& outGyro, std::vector<cv::Vec3d>& outAccel) {
					for (const nlohmann::json& sample : dump["imuRaw"][side])
					{
						outTimes.push_back(sample["t"].get<double>());
						outGyro.push_back(cv::Vec3d(sample["gyro"][0], sample["gyro"][1], sample["gyro"][2]));
						outAccel.push_back(
							cv::Vec3d(sample["accel"][0], sample["accel"][1], sample["accel"][2]));
					}
				};

				std::vector<double> leftTimes, rightTimes;
				std::vector<cv::Vec3d> leftGyro, rightGyro, leftAccel, rightAccel;
				readSide("left", leftTimes, leftGyro, leftAccel);
				readSide("right", rightTimes, rightGyro, rightAccel);

				if (leftTimes.size() < 50 || rightTimes.size() < 50)
				{
					MIKAN_LOG_ERROR("test-imupair")
						<< "Need raw samples from BOTH controllers (left " << leftTimes.size() << ", right "
						<< rightTimes.size() << ")";
					log_dispose();
					return 1;
				}

				// Resample the right onto the left's clock; both are the same
				// steady_clock so this is interpolation, not alignment
				auto sampleAt= [](const std::vector<double>& times, const std::vector<cv::Vec3d>& values,
								  double t) {
					const size_t index= std::min(
						std::max<size_t>(
							(size_t)(std::lower_bound(times.begin(), times.end(), t) - times.begin()), 1),
						times.size() - 1);
					const double span= std::max(1e-9, times[index] - times[index - 1]);
					const double alpha= std::clamp((t - times[index - 1]) / span, 0.0, 1.0);
					return values[index - 1] * (1.0 - alpha) + values[index] * alpha;
				};

				struct PairStats
				{
					const char* name;
					double det= 0.0;
					double residualDegrees= 0.0;
					double scaleRatio= 0.0;
					int samples= 0;
					cv::Matx33d transform;
				};

				auto solve= [&](const char* name, const std::vector<cv::Vec3d>& leftValues,
								const std::vector<cv::Vec3d>& rightValues, double minMagnitude) {
					PairStats stats;
					stats.name= name;

					cv::Matx33d covariance= cv::Matx33d::zeros();
					std::vector<std::pair<cv::Vec3d, cv::Vec3d>> pairs;
					double ratioSum= 0.0;
					for (size_t index= 0; index < leftTimes.size(); ++index)
					{
						const double t= leftTimes[index];
						if (t < rightTimes.front() || t > rightTimes.back())
							continue;
						const cv::Vec3d a= leftValues[index];
						const cv::Vec3d b= sampleAt(rightTimes, rightValues, t);
						const double na= cv::norm(a), nb= cv::norm(b);
						if (na < minMagnitude || nb < minMagnitude)
							continue;
						pairs.emplace_back(a, b);
						ratioSum+= nb / na;
						// H = sum b a^T; the transform maps LEFT into RIGHT
						for (int row= 0; row < 3; ++row)
							for (int col= 0; col < 3; ++col)
								covariance(row, col)+= b[row] * a[col];
					}
					stats.samples= (int)pairs.size();
					if (stats.samples < 30)
						return stats;

					stats.scaleRatio= ratioSum / (double)stats.samples;

					cv::Mat w, u, vt;
					cv::SVD::compute(cv::Mat(covariance), w, u, vt, cv::SVD::FULL_UV);
					const cv::Matx33d transform= cv::Matx33d((cv::Mat)(u * vt));
					stats.transform= transform;
					stats.det= cv::determinant(transform);

					double errorSum= 0.0;
					for (const std::pair<cv::Vec3d, cv::Vec3d>& pair : pairs)
					{
						const cv::Vec3d mapped= transform * (pair.first / cv::norm(pair.first));
						const double dot= std::clamp(mapped.dot(pair.second / cv::norm(pair.second)), -1.0, 1.0);
						errorSum+= std::acos(dot);
					}
					stats.residualDegrees= errorSum / (double)pairs.size() * 180.0 / CV_PI;
					return stats;
				};

				const PairStats gyro= solve("gyro", leftGyro, rightGyro, 0.5);
				const PairStats accel= solve("accel", leftAccel, rightAccel, 1.0);

				for (const PairStats& stats : {gyro, accel})
				{
					if (stats.samples < 30)
					{
						MIKAN_LOG_ERROR("test-imupair")
							<< stats.name << ": only " << stats.samples
							<< " usable samples - tumble the pair through more motion";
						result= 1;
						continue;
					}
					MIKAN_LOG_INFO("test-imupair")
						<< stats.name << ": det=" << stats.det << " ("
						<< (stats.det > 0.0 ? "ROTATION - the mounting absorbs it, no correction needed"
											: "REFLECTION - must be corrected in the raw samples")
						<< "), residual=" << stats.residualDegrees << " deg, right/left magnitude ratio="
						<< stats.scaleRatio << ", n=" << stats.samples;
				}

				// The two sensors must agree about which it is; disagreeing
				// means one of the two streams is not what it claims to be
				if (gyro.samples >= 30 && accel.samples >= 30 &&
					(gyro.det > 0.0) != (accel.det > 0.0))
				{
					MIKAN_LOG_ERROR("test-imupair")
						<< "gyro and accelerometer disagree on handedness - one stream is mislabelled";
					result= 1;
				}
			}
			catch (const std::exception& e)
			{
				MIKAN_LOG_ERROR("test-imupair") << "Failed to analyze the dump: " << e.what();
				result= 1;
			}

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-imuaxes")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-imuaxes.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			// Are the gyro axes consistent with the accelerometer axes?
			//
			// A mounting calibration can absorb any fixed ROTATION between the
			// sensor and the body it rides, but it cannot absorb an axis
			// PERMUTATION or sign flip between the two sensors inside the
			// chip - that isn't a rotation, and it makes integrated motion go
			// the wrong way while static tilt still looks fine.
			//
			// The check needs no integration and no ground truth. "Up" is
			// fixed in the world, so its direction in the SENSOR frame must
			// obey dg/dt = -omega x g. Score every signed permutation of the
			// gyro axes against the gravity motion the accelerometer actually
			// measured, and the right one wins outright.
			JoyconDeviceManager manager;
			manager.startup();
			const size_t deviceCount= manager.getDeviceCount();
			if (deviceCount == 0)
			{
				MIKAN_LOG_ERROR("test-imuaxes") << "No Joy-Cons paired";
				log_dispose();
				return 1;
			}
			for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
				manager.getDeviceByIndex(deviceIndex)->open();

			// Collection time is a knob because it is the thing that decides
			// whether the result is conclusive: it takes a lot of varied,
			// slow rotation to accumulate enough usable windows. 12s was not
			// enough in practice; 30s was.
			int collectSeconds= 30;
			for (int argIndex= i + 1; argIndex < argc; ++argIndex)
			{
				const int parsed= atoi(argv[argIndex]);
				if (parsed > 0)
				{
					collectSeconds= parsed;
					break;
				}
			}

			MIKAN_LOG_INFO("test-imuaxes")
				<< "Collecting " << collectSeconds
				<< " seconds - SLOWLY rotate each controller through all three axes "
				   "(roll, pitch, yaw) in large sweeps, avoiding sharp shakes.";

			std::vector<std::vector<ImuSample>> collected(deviceCount);
			for (int second= 0; second < collectSeconds; ++second)
			{
				std::this_thread::sleep_for(std::chrono::seconds(1));
				for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
					manager.getDeviceByIndex(deviceIndex)->fetchSamples(collected[deviceIndex]);
				MIKAN_LOG_INFO("test-imuaxes") << "  " << (second + 1) << "/" << collectSeconds;
			}

			// All 48 signed permutations: which axis of the raw gyro feeds
			// each output axis, and with what sign
			static const int kPermutations[6][3]= {{0, 1, 2}, {0, 2, 1}, {1, 0, 2},
												   {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
			const char* kAxisNames[3]= {"X", "Y", "Z"};

			int result= 0;
			for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
			{
				IImuDevice* device= manager.getDeviceByIndex(deviceIndex);
				const std::vector<ImuSample>& samples= collected[deviceIndex];
				if (samples.size() < 200)
				{
					MIKAN_LOG_ERROR("test-imuaxes")
						<< device->getFriendlyName() << ": only " << samples.size() << " samples";
					result= 1;
					continue;
				}

				// Gyro bias from the quietest second (so a resting stretch
				// anywhere in the capture serves as the zero reference)
				glm::vec3 bias(0.f);
				{
					float quietest= 1e9f;
					for (size_t start= 0; start + 200 < samples.size(); start+= 200)
					{
						glm::vec3 sum(0.f);
						float motion= 0.f;
						for (size_t k= start; k < start + 200; ++k)
						{
							sum+= samples[k].angularVelocity;
							motion+= glm::length(samples[k].angularVelocity);
						}
						if (motion < quietest)
						{
							quietest= motion;
							bias= sum / 200.f;
						}
					}
				}

				// Score over WINDOWS, not consecutive samples. Between two
				// samples 5ms apart the gyro term is |w|*dt ~ 0.005, far below
				// accelerometer noise, so every candidate scores the same and
				// the winner is noise. Integrating across ~0.4s of real
				// rotation makes the gyro term ~1 rad - orders of magnitude
				// above the noise floor - so a wrong mapping cannot hide.
				constexpr float kWindowSeconds= 0.4f;
				constexpr float kMinWindowRotationRadians= 0.35f; // ~20 deg: below this a window says nothing

				struct Window
				{
					size_t startIndex= 0;
					size_t endIndex= 0;
					glm::vec3 gravityStart{0.f};
					glm::vec3 gravityEnd{0.f};
				};
				std::vector<Window> windows;
				{
					size_t startIndex= 0;
					for (size_t k= 1; k < samples.size(); ++k)
					{
						const double elapsedMs= samples[k].timestampMs - samples[startIndex].timestampMs;
						if (elapsedMs < kWindowSeconds * 1000.0)
							continue;

						// Both ends must be reading gravity alone, or the
						// direction we are predicting isn't gravity
						const float startMagnitude= glm::length(samples[startIndex].acceleration);
						const float endMagnitude= glm::length(samples[k].acceleration);
						const bool bEndsAreGravity= fabsf(startMagnitude - 9.80665f) < 0.6f &&
							fabsf(endMagnitude - 9.80665f) < 0.6f;

						if (bEndsAreGravity)
						{
							Window window;
							window.startIndex= startIndex;
							window.endIndex= k;
							window.gravityStart= samples[startIndex].acceleration / startMagnitude;
							window.gravityEnd= samples[k].acceleration / endMagnitude;
							// Only keep windows containing real rotation
							if (glm::length(window.gravityEnd - window.gravityStart) > 0.15f)
								windows.push_back(window);
						}
						startIndex= k;
					}
				}

				if (windows.size() < 5)
				{
					MIKAN_LOG_ERROR("test-imuaxes")
						<< device->getFriendlyName() << ": only " << windows.size()
						<< " usable rotation windows - rotate the controller more (and more slowly)";
					result= 1;
					continue;
				}

				float bestScore= 1e30f, identityScore= 0.f, runnerUpScore= 1e30f;
				int bestPermutation= 0, bestSigns= 0;
				for (int permutationIndex= 0; permutationIndex < 6; ++permutationIndex)
				{
					for (int signMask= 0; signMask < 8; ++signMask)
					{
						const glm::vec3 signs((signMask & 1) ? -1.f : 1.f, (signMask & 2) ? -1.f : 1.f,
											  (signMask & 4) ? -1.f : 1.f);

						float score= 0.f;
						int usedWindows= 0;
						for (const Window& window : windows)
						{
							// Accumulate the body-frame rotation across the window
							glm::quat deltaRotation(1.f, 0.f, 0.f, 0.f);
							float sweptRadians= 0.f;
							for (size_t k= window.startIndex; k < window.endIndex; ++k)
							{
								const float dt=
									(float)((samples[k + 1].timestampMs - samples[k].timestampMs) / 1000.0);
								if (dt <= 0.f || dt > 0.05f)
									continue;

								const glm::vec3 rawRate= samples[k].angularVelocity - bias;
								const glm::vec3 mappedRate(
									signs.x * rawRate[kPermutations[permutationIndex][0]],
									signs.y * rawRate[kPermutations[permutationIndex][1]],
									signs.z * rawRate[kPermutations[permutationIndex][2]]);

								const float rate= glm::length(mappedRate);
								if (rate > 1e-9f)
								{
									deltaRotation= glm::normalize(
										deltaRotation * glm::angleAxis(rate * dt, mappedRate / rate));
									sweptRadians+= rate * dt;
								}
							}
							if (sweptRadians < kMinWindowRotationRadians)
								continue;

							// A world-fixed direction in body coordinates
							// transforms by the INVERSE of the body rotation
							const glm::vec3 predictedGravityEnd=
								glm::inverse(deltaRotation) * window.gravityStart;
							score+= glm::length(predictedGravityEnd - window.gravityEnd);
							usedWindows++;
						}

						if (usedWindows < 5)
							continue;
						score/= (float)usedWindows;

						const bool bIsIdentity= permutationIndex == 0 && signMask == 0;
						if (bIsIdentity)
							identityScore= score;
						if (score < bestScore)
						{
							runnerUpScore= bestScore;
							bestScore= score;
							bestPermutation= permutationIndex;
							bestSigns= signMask;
						}
						else if (score < runnerUpScore)
						{
							runnerUpScore= score;
						}
					}
				}

				char mapping[64];
				snprintf(mapping, sizeof(mapping), "(%s%s, %s%s, %s%s)",
						 (bestSigns & 1) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][0]],
						 (bestSigns & 2) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][1]],
						 (bestSigns & 4) ? "-" : "+", kAxisNames[kPermutations[bestPermutation][2]]);
				const bool bIsIdentity= bestPermutation == 0 && bestSigns == 0;

				// A candidate only means something if it beats the
				// alternatives DECISIVELY. A near-tie means the measurement
				// carried no information about the mapping (too little
				// rotation, or too much accelerometer noise), and reporting a
				// winner then would be reporting noise.
				constexpr float kDecisiveRatio= 2.f;
				const float identityRatio= bestScore > 1e-9f ? identityScore / bestScore : 1.f;
				const float runnerUpRatio= bestScore > 1e-9f ? runnerUpScore / bestScore : 1.f;

				MIKAN_LOG_INFO("test-imuaxes")
					<< device->getFriendlyName() << ": " << windows.size() << " windows | best " << mapping
					<< " residual " << bestScore << " | identity residual " << identityScore << " (x"
					<< identityRatio << ") | runner-up (x" << runnerUpRatio << ")";

				if (identityRatio < kDecisiveRatio && runnerUpRatio < kDecisiveRatio)
				{
					MIKAN_LOG_WARNING("test-imuaxes")
						<< "  -> INCONCLUSIVE: no mapping wins decisively (need >" << kDecisiveRatio
						<< "x). Rotate through larger, slower sweeps on all three axes and rerun.";
					result= 1;
				}
				else if (bIsIdentity)
				{
					MIKAN_LOG_INFO("test-imuaxes")
						<< "  -> gyro axes agree with the accelerometer (identity wins by x"
						<< runnerUpRatio << ")";
				}
				else
				{
					MIKAN_LOG_ERROR("test-imuaxes")
						<< "  -> MISMATCH: remap the gyro to " << mapping << " (beats identity by x"
						<< identityRatio << ")";
					result= 1;
				}
			}

			manager.shutdown();
			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-imufilter")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-imufilter.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			constexpr float kGravity= 9.80665f;
			const glm::vec3 worldUp(0.f, 0.f, 1.f);
			constexpr float kDt= 1.f / 200.f; // Joy-Con rate

			// Synthetic sensor: given a true sensor->world orientation and a
			// true body-frame rate, produce what the IMU would report
			// (accelerometer reads specific force = +1g along "up")
			auto simulateAccel= [&](const glm::quat& sensorToWorld) {
				return glm::transpose(glm::mat3_cast(sensorToWorld)) * worldUp * kGravity;
			};

			// (a) Static, gravity only. A stationary sensor with a real
			// Joy-Con bias must hold level and learn the bias components it
			// CAN observe. Gravity constrains only 2 DoF (tilt), so the bias
			// about the gravity axis is unobservable here - the same physics
			// that makes yaw unobservable. Asserting that explicitly keeps
			// the limitation documented instead of surprising us later.
			{
				const glm::vec3 trueBias(0.007f, -0.033f, -0.006f); // measured Joy-Con R
				const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f); // level: gravity axis = sensor Z

				ImuOrientationFilter filter;
				filter.configure(ImuOrientationFilterConfig());

				const glm::vec3 accel= simulateAccel(trueOrientation);
				for (int step= 0; step < 200 * 30; ++step) // 30 seconds
				{
					ImuSample sample;
					sample.angularVelocity= trueBias; // stationary: the reading IS the bias
					sample.acceleration= accel;
					filter.processSample(sample, kDt);
				}

				const glm::vec3 estimatedBias= filter.getGyroBias();
				const glm::vec2 tiltBiasError(estimatedBias.x - trueBias.x, estimatedBias.y - trueBias.y);
				const float tiltBiasErrorMagnitude= glm::length(tiltBiasError);
				const float gravityAxisBiasError= fabsf(estimatedBias.z - trueBias.z);
				const glm::vec3 estimatedUp= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);
				const float tiltErrorDegrees= glm::degrees(acosf(std::clamp(estimatedUp.z, -1.f, 1.f)));

				MIKAN_LOG_INFO("test-imufilter")
					<< "(a) static/gravity-only: tilt-axis bias err=" << tiltBiasErrorMagnitude
					<< " rad/s, gravity-axis bias err=" << gravityAxisBiasError
					<< " rad/s (expected ~unobservable), tilt err=" << tiltErrorDegrees << " deg";
				if (tiltBiasErrorMagnitude > 0.002f || tiltErrorDegrees > 0.5f || !filter.isTiltConverged())
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(a) FAILED: must learn the OBSERVABLE bias axes and hold level";
					result= 1;
				}
				if (gravityAxisBiasError < 0.5f * fabsf(trueBias.z))
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(a) FAILED: gravity-axis bias appears observable - the gravity Jacobian is wrong";
					result= 1;
				}
			}

			// (a2) Same scenario WITH vision. Absolute orientation makes the
			// third bias axis observable, so the full 3-axis bias converges -
			// this is the concrete payoff of anchoring the filter to vision.
			{
				const glm::vec3 trueBias(0.007f, -0.033f, -0.006f);
				const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);

				ImuOrientationFilter filter;
				filter.configure(ImuOrientationFilterConfig());

				const glm::vec3 accel= simulateAccel(trueOrientation);
				for (int step= 0; step < 200 * 30; ++step)
				{
					ImuSample sample;
					sample.angularVelocity= trueBias;
					sample.acceleration= accel;
					filter.processSample(sample, kDt);

					// Vision at ~30 Hz, as the tracker would supply it
					if (step % 7 == 0)
						filter.updateWithOrientation(trueOrientation);
				}

				const float biasError= glm::length(filter.getGyroBias() - trueBias);
				MIKAN_LOG_INFO("test-imufilter")
					<< "(a2) static/vision-aided: full bias err=" << biasError << " rad/s ("
					<< glm::degrees(biasError) << " deg/s)";
				if (biasError > 0.002f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(a2) FAILED: vision must make the full gyro bias observable";
					result= 1;
				}
			}

			// (b) Rotation tracking: sustained rotation about a tilted axis
			// with a bias present. Orientation must track the truth.
			{
				const glm::vec3 trueBias(0.01f, -0.02f, 0.005f);
				const glm::vec3 trueRate= glm::normalize(glm::vec3(0.3f, 0.6f, 0.2f)) * 1.2f; // rad/s

				ImuOrientationFilter filter;
				filter.configure(ImuOrientationFilterConfig());

				// Let it settle at rest first so the bias is known
				glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
				for (int step= 0; step < 200 * 20; ++step)
				{
					ImuSample sample;
					sample.angularVelocity= trueBias;
					sample.acceleration= simulateAccel(trueOrientation);
					filter.processSample(sample, kDt);
				}

				// Now rotate for 4 seconds
				for (int step= 0; step < 200 * 4; ++step)
				{
					const float angle= glm::length(trueRate) * kDt;
					trueOrientation= glm::normalize(
						trueOrientation * glm::angleAxis(angle, glm::normalize(trueRate)));

					ImuSample sample;
					sample.angularVelocity= trueRate + trueBias;
					// Rotating in place: the accelerometer still sees only
					// gravity, so the gravity update stays valid
					sample.acceleration= simulateAccel(trueOrientation);
					filter.processSample(sample, kDt);
				}

				const glm::quat error= glm::inverse(trueOrientation) * filter.getOrientation();
				const float errorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(error.x, error.y, error.z)), 0.f, 1.f)));
				MIKAN_LOG_INFO("test-imufilter") << "(b) rotation: orientation err=" << errorDegrees << " deg";
				if (errorDegrees > 2.f)
				{
					MIKAN_LOG_ERROR("test-imufilter") << "(b) FAILED: must track sustained rotation";
					result= 1;
				}
			}

			// (c) Motion gating: a hard linear acceleration must NOT be
			// mistaken for gravity and tilt the estimate
			{
				ImuOrientationFilter filter;
				filter.configure(ImuOrientationFilterConfig());

				const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
				for (int step= 0; step < 200 * 10; ++step)
				{
					ImuSample sample;
					sample.acceleration= simulateAccel(trueOrientation);
					filter.processSample(sample, kDt);
				}
				const glm::vec3 upBefore= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);

				// 5 m/s^2 sideways shove for half a second
				int rejectedCount= 0;
				for (int step= 0; step < 100; ++step)
				{
					filter.predict(glm::vec3(0.f), kDt);
					const glm::vec3 shoved= simulateAccel(trueOrientation) + glm::vec3(5.f, 0.f, 0.f);
					if (!filter.updateWithGravity(shoved))
						rejectedCount++;
				}
				const glm::vec3 upAfter= glm::mat3_cast(filter.getOrientation()) * glm::vec3(0, 0, 1);
				const float tiltChangeDegrees=
					glm::degrees(acosf(std::clamp(glm::dot(upBefore, upAfter), -1.f, 1.f)));

				MIKAN_LOG_INFO("test-imufilter") << "(c) gating: rejected " << rejectedCount
					<< "/100 accelerated samples, tilt moved " << tiltChangeDegrees << " deg";
				if (rejectedCount != 100 || tiltChangeDegrees > 0.1f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(c) FAILED: accelerated samples must be gated out of the gravity update";
					result= 1;
				}
			}

			// (d) Yaw is unobservable from inertial data alone, and vision
			// must be able to fix it. Sanity-checks the whole reason the
			// filter is vision-anchored.
			{
				ImuOrientationFilter filter;
				filter.configure(ImuOrientationFilterConfig());

				const glm::quat trueOrientation(1.f, 0.f, 0.f, 0.f);
				for (int step= 0; step < 200 * 10; ++step)
				{
					ImuSample sample;
					sample.acceleration= simulateAccel(trueOrientation);
					filter.processSample(sample, kDt);
				}

				const glm::vec3 sigmaBefore= filter.getOrientationSigma();

				// Vision says the sensor is yawed 40 degrees
				const glm::quat visionOrientation=
					glm::angleAxis(glm::radians(40.f), glm::vec3(0.f, 0.f, 1.f));
				for (int update= 0; update < 20; ++update)
					filter.updateWithOrientation(visionOrientation);

				const glm::vec3 sigmaAfter= filter.getOrientationSigma();
				const glm::quat error= glm::inverse(visionOrientation) * filter.getOrientation();
				const float errorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(error.x, error.y, error.z)), 0.f, 1.f)));

				MIKAN_LOG_INFO("test-imufilter")
					<< "(d) yaw: sigma z before=" << sigmaBefore.z << " after=" << sigmaAfter.z
					<< " rad, orientation err after vision=" << errorDegrees << " deg";
				// Before vision, yaw uncertainty must still be large (gravity
				// can't see it); after vision it must be small and correct
				if (sigmaBefore.z < 0.5f || sigmaAfter.z > 0.1f || errorDegrees > 3.f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(d) FAILED: yaw must be unobservable inertially and fixable by vision";
					result= 1;
				}
			}

			// (e) Mounting calibration round trip. The whole chain has three
			// places a transpose/inverse could hide (mounting capture,
			// forearm reconstruction, wrist rotation) and a mistake in any
			// one produces plausible-looking but wrong wrist angles, so
			// verify it end to end against a known truth.
			{
				// Arbitrary "how the strap happens to sit" rotation
				const glm::quat trueForearmToSensor= glm::normalize(
					glm::angleAxis(glm::radians(63.f), glm::normalize(glm::vec3(0.3f, -0.8f, 0.5f))));

				// CALIBRATION: wrist straight, so forearm frame == palm frame
				const glm::quat forearmAtCapture=
					glm::angleAxis(glm::radians(21.f), glm::vec3(0.f, 0.f, 1.f));
				const glm::quat palmAtCapture= forearmAtCapture; // straight wrist
				// The sensor reports q_sw where q_fw = q_sw * q_fs
				const glm::quat sensorAtCapture= forearmAtCapture * glm::inverse(trueForearmToSensor);

				// captureMounting computes inverse(q_sw) * q_palm
				const glm::quat capturedMounting=
					glm::normalize(glm::inverse(sensorAtCapture) * palmAtCapture);
				const glm::quat mountingError= glm::inverse(trueForearmToSensor) * capturedMounting;
				const float mountingErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(mountingError.x, mountingError.y, mountingError.z)), 0.f, 1.f)));

				// RUNTIME: forearm moved AND the wrist is now bent 35 deg
				const glm::quat forearmNow= glm::normalize(
					glm::angleAxis(glm::radians(-40.f), glm::normalize(glm::vec3(0.2f, 0.9f, 0.1f))));
				const glm::quat trueWristLocal=
					glm::angleAxis(glm::radians(35.f), glm::vec3(1.f, 0.f, 0.f));
				const glm::quat palmNow= forearmNow * trueWristLocal;
				const glm::quat sensorNow= forearmNow * glm::inverse(trueForearmToSensor);

				// getForearmOrientation computes q_sw * q_fs
				const glm::quat reconstructedForearm= glm::normalize(sensorNow * capturedMounting);
				const glm::quat forearmError= glm::inverse(forearmNow) * reconstructedForearm;
				const float forearmErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(forearmError.x, forearmError.y, forearmError.z)), 0.f, 1.f)));

				// HandPose::getWristRotation computes inverse(forearm) * palm
				HandPose pose;
				pose.hasWorldPose= true;
				pose.hasForearmPose= true;
				pose.palmOrientationWorld= palmNow;
				pose.forearmOrientationWorld= reconstructedForearm;
				const glm::quat wristError= glm::inverse(trueWristLocal) * pose.getWristRotation();
				const float wristErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(wristError.x, wristError.y, wristError.z)), 0.f, 1.f)));

				// And a straight wrist must read identity, whatever the mounting
				HandPose straightPose;
				straightPose.hasWorldPose= true;
				straightPose.hasForearmPose= true;
				straightPose.palmOrientationWorld= forearmNow;
				straightPose.forearmOrientationWorld= reconstructedForearm;
				const float straightDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(straightPose.getWristRotation().x,
										  straightPose.getWristRotation().y,
										  straightPose.getWristRotation().z)), 0.f, 1.f)));

				MIKAN_LOG_INFO("test-imufilter")
					<< "(e) mounting round trip: mounting err=" << mountingErrorDegrees
					<< " deg, forearm err=" << forearmErrorDegrees << " deg, wrist err="
					<< wristErrorDegrees << " deg, straight-wrist reads " << straightDegrees << " deg";
				if (mountingErrorDegrees > 0.01f || forearmErrorDegrees > 0.01f ||
					wristErrorDegrees > 0.01f || straightDegrees > 0.01f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(e) FAILED: mounting calibration / wrist rotation chain is inconsistent";
					result= 1;
				}
			}

			// (f) Motion-based arm-axis recovery. A held pose alone has to get
			// all three mounting DoF right at one instant, and in practice got
			// the ARM AXIS wrong by ~60 deg - which is the one the elbow rides
			// on. Twisting the forearm measures that axis directly: pronation
			// rotates about the arm and nothing else, so the dominant axis of
			// the sensor-frame angular-velocity scatter IS the arm axis.
			{
				const glm::quat trueMounting= glm::normalize(
					glm::angleAxis(glm::radians(115.f), glm::normalize(glm::vec3(0.3f, -0.7f, 0.6f))));
				// Arm axis (forearm +X) as the SENSOR sees it
				const glm::vec3 trueSensorArmAxis=
					glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));

				auto accumulate= [](glm::mat3& scatter, const glm::vec3& rate, float weight) {
					for (int col= 0; col < 3; ++col)
						for (int row= 0; row < 3; ++row)
							scatter[col][row]+= rate[col] * rate[row] * weight;
				};

				// Simulated twisting: mostly about the arm axis, with a little
				// off-axis wobble because no one twists perfectly
				glm::mat3 twistScatter(0.f);
				const glm::vec3 offAxis= glm::normalize(glm::cross(trueSensorArmAxis, glm::vec3(0.f, 0.f, 1.f)));
				for (int sampleIndex= 0; sampleIndex < 400; ++sampleIndex)
				{
					const float phase= (float)sampleIndex * 0.05f;
					const glm::vec3 rate=
						trueSensorArmAxis * (3.f * sinf(phase)) + offAxis * (0.3f * sinf(phase * 2.7f));
					accumulate(twistScatter, rate, 0.005f);
				}

				float twistDominance= 0.f;
				const glm::vec3 measuredAxis= imuDominantRotationAxis(twistScatter, twistDominance);
				const float axisErrorDegrees= glm::degrees(
					acosf(std::clamp(fabsf(glm::dot(measuredAxis, trueSensorArmAxis)), 0.f, 1.f)));

				// Isotropic motion (waving the arm around, no real twist) must
				// score as uninformative so the UI can refuse the capture
				glm::mat3 isotropicScatter(0.f);
				accumulate(isotropicScatter, glm::vec3(1.f, 0.f, 0.f), 1.f);
				accumulate(isotropicScatter, glm::vec3(0.f, 1.f, 0.f), 1.f);
				accumulate(isotropicScatter, glm::vec3(0.f, 0.f, 1.f), 1.f);
				float isotropicDominance= 0.f;
				imuDominantRotationAxis(isotropicScatter, isotropicDominance);

				MIKAN_LOG_INFO("test-imufilter")
					<< "(f) motion axis: measured axis err=" << axisErrorDegrees
					<< " deg (dominance=" << twistDominance
					<< "), isotropic dominance=" << isotropicDominance;
				if (axisErrorDegrees > 2.f || twistDominance < 0.9f || isotropicDominance > 0.4f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(f) FAILED: motion-based arm axis recovery is wrong";
					result= 1;
				}
			}

			// (n) TWO-MOTION MOUNTING SOLVE. A twist fixes the forearm's long
			// axis, a curl about the elbow hinge fixes the roll that a twist
			// cannot see, and the accelerometer's centripetal term says which
			// end of the axis is the hand. No vision enters the geometry at
			// all, which is the entire point: the vision-averaged mounting it
			// replaces was corrupted by a wrist that would not hold still, and
			// averaging harder could not fix that.
			{
				const glm::quat trueMounting= glm::normalize(
					glm::angleAxis(glm::radians(115.f), glm::normalize(glm::vec3(0.3f, -0.7f, 0.6f))));
				const glm::vec3 armAxisSensor= glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));
				const glm::vec3 hingeAxisSensor= glm::normalize(trueMounting * glm::vec3(0.f, 1.f, 0.f));
				const glm::vec3 wobbleAxis=
					glm::normalize(glm::cross(armAxisSensor, glm::vec3(0.f, 0.f, 1.f)));
				constexpr float k_trueRadiusMeters= 0.22f;

				auto mountingErrorDegrees= [](const glm::quat& a, const glm::quat& b) {
					glm::quat delta= glm::inverse(a) * b;
					if (delta.w < 0.f) // double cover: measure the short way round
						delta= -delta;
					return glm::degrees(2.f * asinf(std::clamp(
						glm::length(glm::vec3(delta.x, delta.y, delta.z)), 0.f, 1.f)));
				};

				// TWIST: back-and-forth about the long axis, with the off-axis
				// wobble nobody actually avoids
				std::vector<MotionSample> twist;
				for (int sampleIndex= 0; sampleIndex < 1200; ++sampleIndex)
				{
					const float phase= (float)sampleIndex * 0.02f;
					MotionSample sample;
					sample.rate= armAxisSensor * (4.f * sinf(phase)) +
						wobbleAxis * (0.3f * sinf(phase * 2.7f));
					sample.acceleration= glm::vec3(0.f, 0.f, 9.81f);
					sample.dtSeconds= 0.005f;
					twist.push_back(sample);
				}

				// CURL: back-and-forth about the hinge, with the accelerometer
				// carrying what a real one would - gravity as seen from a
				// rotating frame, the centripetal term pointing at the elbow,
				// and the tangential term perpendicular to both
				auto buildCurl= [&](const glm::vec3& axis) {
					std::vector<MotionSample> curl;
					glm::vec3 up= glm::normalize(glm::vec3(0.2f, 0.3f, 1.f));
					float previousRate= 0.f;
					for (int sampleIndex= 0; sampleIndex < 1600; ++sampleIndex)
					{
						constexpr float dt= 0.005f;
						const float phase= (float)sampleIndex * 0.02f;
						const float hingeRate= 3.5f * sinf(phase);
						const glm::vec3 rate=
							axis * hingeRate + wobbleAxis * (0.15f * sinf(phase * 3.3f));

						up= glm::normalize(up - glm::cross(rate, up) * dt);
						const float angularAcceleration= (hingeRate - previousRate) / dt;
						previousRate= hingeRate;

						MotionSample sample;
						sample.rate= rate;
						sample.acceleration= up * 9.81f -
							armAxisSensor * (hingeRate * hingeRate * k_trueRadiusMeters) +
							glm::cross(axis * angularAcceleration, armAxisSensor * k_trueRadiusMeters);
						sample.dtSeconds= dt;
						curl.push_back(sample);
					}
					return curl;
				};
				const std::vector<MotionSample> curl= buildCurl(hingeAxisSensor);

				// A deliberately TERRIBLE palmar hint: 50 degrees off, which is
				// worse than the 54 degree pose spread that made the old solve
				// unusable. It only has to choose between two candidates a half
				// turn apart, so it must still land on the right one.
				const glm::quat badHint= glm::normalize(
					glm::angleAxis(glm::radians(50.f), glm::normalize(glm::vec3(0.5f, 0.6f, -0.6f))) *
					trueMounting);

				MountingCaptureResult solved;
				imuSolveMountingFromMotions(twist, curl, &badHint, ePalmarSource::Vision, solved);
				const float solvedErrorDegrees= mountingErrorDegrees(trueMounting, solved.forearmToSensor);

				// The hint decides the palmar bit and nothing else, so a hint on
				// the OTHER side must move the answer by exactly a half turn -
				// proving the bit is driven by the reference rather than baked in
				const glm::quat flippedTruth= glm::normalize(
					trueMounting * glm::angleAxis(glm::radians(180.f), glm::vec3(1.f, 0.f, 0.f)));
				MountingCaptureResult flippedSolved;
				imuSolveMountingFromMotions(twist, curl, &flippedTruth, ePalmarSource::Vision, flippedSolved);
				const float flippedErrorDegrees=
					mountingErrorDegrees(flippedTruth, flippedSolved.forearmToSensor);

				// A curl that turns about the SAME axis as the twist (the
				// shoulder rotating instead of the elbow bending) leaves roll
				// unobservable, so it has to be refused rather than extrapolated
				MountingCaptureResult degenerate;
				imuSolveMountingFromMotions(twist, buildCurl(armAxisSensor), &badHint,
											ePalmarSource::Vision, degenerate);

				// No reference at all: the palmar side is genuinely unknown, and
				// guessing it is a coin flip that silently mirrors the hand
				MountingCaptureResult unhinted;
				imuSolveMountingFromMotions(twist, curl, nullptr, ePalmarSource::None, unhinted);

				MIKAN_LOG_INFO("test-imufilter")
					<< "(n) two-motion solve: err=" << solvedErrorDegrees << " deg, inter-axis="
					<< solved.interAxisAngleDegrees << " deg, hinge spread=" << solved.hingeSpreadDegrees
					<< " deg over " << solved.curlStrokes << " strokes, radius="
					<< solved.forearmLengthMeters << " m (r=" << solved.lengthFitCorrelation
					<< "), flipped-hint err=" << flippedErrorDegrees
					<< " deg, parallel-curl usable=" << degenerate.bMotionUsable
					<< " (inter-axis " << degenerate.interAxisAngleDegrees
					<< " deg), unhinted usable=" << unhinted.bMotionUsable;

				if (!solved.bMotionUsable || solvedErrorDegrees > 3.f ||
					solved.interAxisAngleDegrees < 85.f || solved.hingeSpreadDegrees > 5.f ||
					solved.curlStrokes < 3 || !solved.bLengthMeasured ||
					fabsf(solved.forearmLengthMeters - k_trueRadiusMeters) > 0.02f ||
					flippedErrorDegrees > 3.f || degenerate.bMotionUsable ||
					degenerate.interAxisAngleDegrees > 20.f || unhinted.bMotionUsable ||
					unhinted.palmarSource != ePalmarSource::None)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(n) FAILED: two-motion mounting solve is wrong";
					result= 1;
				}
			}

			// (g) Twist gating. Both live failures came down to the same thing:
			// dominance alone accepts anything. A scatter is rank-1 - so
			// dominance ~1.0 - for a moment's motion, and ALSO for a constant
			// rate offset. A Joy-Con whose bias had run away to -5371 deg/s
			// scored 0.9999 while sitting on a desk, and the wizard's twist
			// stage completed instantly.
			{
				auto accumulate= [](glm::mat3& scatter, float& path, glm::vec3& net, const glm::vec3& rate,
									float dt) {
					const float magnitude= glm::length(rate);
					for (int col= 0; col < 3; ++col)
						for (int row= 0; row < 3; ++row)
							scatter[col][row]+= rate[col] * rate[row] * dt;
					path+= magnitude * dt;
					net+= rate * dt;
				};

				const glm::vec3 armAxis= glm::normalize(glm::vec3(0.4f, -0.8f, 0.45f));

				// A brief flick: single-axis by construction, but nowhere near
				// enough rotation to locate anything
				glm::mat3 flickScatter(0.f);
				float flickPath= 0.f;
				glm::vec3 flickNet(0.f);
				for (int i= 0; i < 20; ++i)
					accumulate(flickScatter, flickPath, flickNet, armAxis * 0.6f, 0.005f);

				// A constant rate - what a runaway gyro bias looks like. Lots of
				// "rotation", perfectly single-axis, and completely meaningless.
				glm::mat3 driftScatter(0.f);
				float driftPath= 0.f;
				glm::vec3 driftNet(0.f);
				for (int i= 0; i < 2000; ++i)
					accumulate(driftScatter, driftPath, driftNet, armAxis * 94.f, 0.005f);

				// Real pronation/supination: back and forth about the arm axis
				glm::mat3 twistScatter(0.f);
				float twistPath= 0.f;
				glm::vec3 twistNet(0.f);
				const glm::vec3 wobbleAxis= glm::normalize(glm::cross(armAxis, glm::vec3(0.f, 0.f, 1.f)));
				for (int i= 0; i < 800; ++i)
				{
					const float phase= (float)i * 0.05f;
					accumulate(twistScatter, twistPath, twistNet,
							   armAxis * (3.f * sinf(phase)) + wobbleAxis * (0.25f * sinf(phase * 2.7f)),
							   0.005f);
				}

				struct TwistCase
				{
					const char* name;
					const glm::mat3* scatter;
					float path;
					const glm::vec3* net;
					bool bExpectUsable;
				};
				const TwistCase cases[]= {
					{"flick", &flickScatter, flickPath, &flickNet, false},
					{"constant-rate", &driftScatter, driftPath, &driftNet, false},
					{"back-and-forth", &twistScatter, twistPath, &twistNet, true},
				};

				for (const TwistCase& twistCase : cases)
				{
					float dominance= 0.f, progress= 0.f, reversal= 0.f;
					imuEvaluateTwist(*twistCase.scatter, twistCase.path, *twistCase.net, dominance, progress,
									 reversal);
					const bool bUsable= imuIsTwistUsable(dominance, progress, reversal);
					MIKAN_LOG_INFO("test-imufilter")
						<< "(g) twist " << twistCase.name << ": dominance=" << dominance
						<< " progress=" << progress << " reversal=" << reversal << " usable=" << bUsable;
					if (bUsable != twistCase.bExpectUsable)
					{
						MIKAN_LOG_ERROR("test-imufilter")
							<< "(g) FAILED: '" << twistCase.name << "' should"
							<< (twistCase.bExpectUsable ? " " : " NOT ") << "be usable";
						result= 1;
					}
				}

				// The two rejected cases must still LOOK single-axis - that is
				// the whole point of the extra gates
				float dominance= 0.f, progress= 0.f, reversal= 0.f;
				imuEvaluateTwist(driftScatter, driftPath, driftNet, dominance, progress, reversal);
				if (dominance < 0.99f || reversal > 0.01f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(g) FAILED: a constant rate should score dominance ~1 and reversal ~0, got "
						<< dominance << " / " << reversal;
					result= 1;
				}
			}

			// (h) Gyro bias bound. The bias feeds back into predict(), so an
			// unbounded estimate is self-amplifying rather than self-correcting.
			{
				ImuOrientationFilter filter;
				ImuOrientationFilterConfig config;
				filter.configure(config);
				filter.initializeFromGravity(glm::vec3(0.f, 0.f, 9.80665f));

				filter.setGyroBias(glm::vec3(94.f, -50.f, 3.f)); // absurd, as observed live
				const glm::vec3 bounded= filter.getGyroBias();
				const bool bClamped= fabsf(bounded.x - config.maxGyroBias) < 1e-5f &&
									 fabsf(bounded.y + config.maxGyroBias) < 1e-5f &&
									 fabsf(bounded.z - config.maxGyroBias) < 1e-5f;

				// A sane measured bias must survive untouched
				filter.setGyroBias(glm::vec3(0.01f, -0.02f, 0.005f));
				const glm::vec3 kept= filter.getGyroBias();
				const bool bKept= fabsf(kept.x - 0.01f) < 1e-6f && fabsf(kept.y + 0.02f) < 1e-6f &&
								  fabsf(kept.z - 0.005f) < 1e-6f;

				MIKAN_LOG_INFO("test-imufilter")
					<< "(h) bias bound: clamped to " << bounded.x << ", " << bounded.y << ", " << bounded.z
					<< " rad/s; measured bias preserved=" << bKept;
				if (!bClamped || !bKept)
				{
					MIKAN_LOG_ERROR("test-imufilter") << "(h) FAILED: gyro bias bound is not enforced";
					result= 1;
				}
			}

			// (i) Wrist axial residual as a mounting-roll health check. The
			// wrist has no axial degree of freedom, so any axial rotation in
			// the measured joint is mounting roll error. Nothing acts on this
			// automatically - the curl measures roll properly now, and an
			// earlier version that DID correct it was steering a good roll
			// with the noisiest signal in the system - but it still has to
			// SEE the error, or a bad roll has no symptom.
			{
				const glm::quat trueMounting= glm::normalize(
					glm::angleAxis(glm::radians(174.f), glm::normalize(glm::vec3(-0.14f, 0.94f, 0.31f))));
				const glm::quat rollError= glm::angleAxis(glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
				const glm::quat badMounting= glm::normalize(trueMounting * rollError);

				auto axialDegrees= [](const glm::quat& rotation) {
					glm::quat q= rotation;
					if (q.w < 0.f)
						q= -q;
					const glm::vec3 axisPart(q.x, q.y, q.z);
					const float axisLength= glm::length(axisPart);
					if (axisLength < 1e-6f)
						return 0.f;
					const float angle= 2.f * asinf(std::min(1.f, axisLength));
					return glm::degrees(angle * (axisPart.x / axisLength));
				};

				// The arm moves and the wrist genuinely flexes, but never
				// twists - so a correct mounting must stay near zero while the
				// rolled one reads its error, both under real motion.
				//
				// "Near zero" rather than zero: composing flexion with
				// deviation produces a couple of degrees of genuine axial
				// component, which is the wrist's own slack and is why the UI
				// only calls this bad above 5 degrees.
				float worstGoodResidual= 0.f;
				float meanBadResidual= 0.f;
				constexpr int k_steps= 4000;
				for (int step= 0; step < k_steps; ++step)
				{
					const float t= (float)step * 0.01f;
					const glm::quat sensorToWorld= glm::normalize(
						glm::angleAxis(0.7f * sinf(t), glm::normalize(glm::vec3(0.2f, 0.3f, 0.93f))));
					// Real wrist motion: flexion and deviation only, no twist
					const glm::quat trueWrist=
						glm::normalize(glm::angleAxis(0.4f * sinf(t * 1.7f), glm::vec3(0.f, 1.f, 0.f)) *
									   glm::angleAxis(0.2f * sinf(t * 2.3f), glm::vec3(0.f, 0.f, 1.f)));
					const glm::quat palm= glm::normalize(sensorToWorld * trueMounting * trueWrist);

					const glm::quat goodForearm= glm::normalize(sensorToWorld * trueMounting);
					worstGoodResidual= std::max(worstGoodResidual,
						fabsf(axialDegrees(glm::normalize(glm::inverse(goodForearm) * palm))));

					const glm::quat badForearm= glm::normalize(sensorToWorld * badMounting);
					meanBadResidual+=
						fabsf(axialDegrees(glm::normalize(glm::inverse(badForearm) * palm))) / (float)k_steps;
				}

				MIKAN_LOG_INFO("test-imufilter")
					<< "(i) axial residual: correct mounting reads at most " << worstGoodResidual
					<< " deg, a 45 deg rolled one averages " << meanBadResidual << " deg";

				if (worstGoodResidual > 3.f || meanBadResidual < 40.f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(i) FAILED: the wrist axial residual does not report mounting roll error";
					result= 1;
				}
			}

			// (k) Averaging the pose mounting. inverse(q_sensor) * q_palm is
			// CONSTANT under the correct model - both terms rotate with the
			// arm - so wrist bend is the only thing perturbing it, and the
			// mean recovers the mounting. This is what replaced sampling one
			// held pose, where a wrist 90+ deg off flipped the arm-axis
			// eigenvector and locked in a reproducible 180 deg error.
			{
				const glm::quat trueMounting= glm::normalize(
					glm::angleAxis(glm::radians(167.f), glm::normalize(glm::vec3(-0.02f, -0.09f, 0.99f))));

				glm::vec4 sum(0.f);
				int samples= 0;
				float worstSingleSampleDegrees= 0.f;
				for (int step= 0; step < 600; ++step)
				{
					const float t= (float)step * 0.02f;
					// The arm sweeps through many orientations as it twists
					const glm::quat sensorToWorld= glm::normalize(
						glm::angleAxis(2.2f * sinf(t), glm::normalize(glm::vec3(0.1f, -0.2f, 0.97f))) *
						glm::angleAxis(0.5f * sinf(t * 0.7f), glm::vec3(1.f, 0.f, 0.f)));
					// Wrist bend: bounded, zero-mean, and large enough that a
					// single unlucky sample is badly wrong
					const glm::quat wristBend= glm::normalize(
						glm::angleAxis(0.9f * sinf(t * 1.9f), glm::vec3(0.f, 1.f, 0.f)) *
						glm::angleAxis(0.6f * sinf(t * 2.6f), glm::vec3(0.f, 0.f, 1.f)));
					const glm::quat palm= glm::normalize(sensorToWorld * trueMounting * wristBend);

					const glm::quat sample= glm::normalize(glm::inverse(sensorToWorld) * palm);
					const glm::quat singleError= glm::inverse(trueMounting) * sample;
					worstSingleSampleDegrees= std::max(worstSingleSampleDegrees,
						glm::degrees(2.f * asinf(std::clamp(
							glm::length(glm::vec3(singleError.x, singleError.y, singleError.z)), 0.f, 1.f))));

					// Same hemisphere alignment the service applies
					glm::quat aligned= sample;
					if (samples > 0)
					{
						const glm::quat mean= glm::normalize(glm::quat(sum.w, sum.x, sum.y, sum.z));
						if (glm::dot(aligned, mean) < 0.f)
							aligned= -aligned;
					}
					sum+= glm::vec4(aligned.x, aligned.y, aligned.z, aligned.w);
					samples++;
				}

				const glm::quat mean= glm::normalize(glm::quat(sum.w, sum.x, sum.y, sum.z));
				const glm::quat meanError= glm::inverse(trueMounting) * mean;
				const float meanErrorDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(meanError.x, meanError.y, meanError.z)), 0.f, 1.f)));

				// The arm axis sign is resolved against the pose mounting, so
				// the mean's +X must agree with the truth's - that is the
				// specific failure this replaced
				const glm::vec3 trueAxis= glm::normalize(trueMounting * glm::vec3(1.f, 0.f, 0.f));
				const glm::vec3 meanAxis= glm::normalize(mean * glm::vec3(1.f, 0.f, 0.f));
				const float axisAgreement= glm::dot(trueAxis, meanAxis);

				MIKAN_LOG_INFO("test-imufilter")
					<< "(k) pose averaging: worst single sample " << worstSingleSampleDegrees
					<< " deg off, mean " << meanErrorDegrees << " deg off, arm-axis agreement "
					<< axisAgreement;

				// The point of the test: single samples are badly wrong, the
				// mean is not, and the sign never flips
				if (worstSingleSampleDegrees < 45.f || meanErrorDegrees > 15.f || axisAgreement < 0.9f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(k) FAILED: averaging did not recover the mounting from noisy samples";
					result= 1;
				}
			}

			// (l) Yaw covariance must RE-INFLATE. Yaw is not observable from
			// inertial data, so its uncertainty has to grow between vision
			// corrections. Without that it collapses on the first correction
			// and never recovers, and since the Kalman gain is P/(P+R) a tiny
			// prior against a loose vision measurement switches the anchor off
			// in all but name - measured live as yaw sigma pinned at 0.010 rad
			// with the applied correction reading 0.00 deg.
			{
				ImuOrientationFilter filter;
				ImuOrientationFilterConfig config;
				filter.configure(config);
				filter.initializeFromGravity(glm::vec3(0.f, 0.f, 9.80665f));

				// Settle, then let vision pin yaw hard
				for (int step= 0; step < 400; ++step)
				{
					filter.predict(glm::vec3(0.f), 1.f / 200.f);
					filter.updateWithGravity(glm::vec3(0.f, 0.f, 9.80665f));
				}
				for (int step= 0; step < 200; ++step)
					filter.updateWithYawReference(glm::quat(1.f, 0.f, 0.f, 0.f), 0.05f);

				const float pinnedYaw= filter.getOrientationSigma().z;

				// Now coast on the gyro alone for two seconds
				for (int step= 0; step < 400; ++step)
				{
					filter.predict(glm::vec3(0.f), 1.f / 200.f);
					filter.updateWithGravity(glm::vec3(0.f, 0.f, 9.80665f));
				}
				const float coastedYaw= filter.getOrientationSigma().z;
				const float coastedTilt= filter.getTiltSigma();

				// A vision correction must now actually move the estimate
				const glm::quat reference= glm::angleAxis(glm::radians(20.f), glm::vec3(0.f, 0.f, 1.f));
				const glm::quat before= filter.getOrientation();
				for (int step= 0; step < 30; ++step)
					filter.updateWithYawReference(reference, 0.31f);
				const glm::quat moved= glm::inverse(before) * filter.getOrientation();
				const float movedDegrees= glm::degrees(2.f * asinf(std::clamp(
					glm::length(glm::vec3(moved.x, moved.y, moved.z)), 0.f, 1.f)));

				MIKAN_LOG_INFO("test-imufilter")
					<< "(l) yaw covariance: pinned " << pinnedYaw << " rad, re-inflated to " << coastedYaw
					<< " after 2s coasting (tilt stayed " << coastedTilt << "), vision then moved yaw "
					<< movedDegrees << " deg";

				// Yaw must grow, tilt must NOT (gravity observes it), and the
				// anchor must have real authority again
				if (coastedYaw <= pinnedYaw * 1.5f || coastedTilt > 0.05f || movedDegrees < 1.f)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(l) FAILED: yaw uncertainty is not recovering, so the vision anchor is inert";
					result= 1;
				}
			}

			// (m) Palmar side must be temporally stable.
			//
			// The sign is decided per frame from geometry. On a nearly flat
			// hand the evidence is weak and its SIGN follows landmark noise,
			// so the palm frame inverts between frames - measured live as the
			// left palm frame flipping twice in 240 frames while the right
			// never did, which poisoned the mounting average with two clusters
			// 180 deg apart (70 deg pose spread against the right's 13).
			//
			// The test asserts the CONTRAST: the unprotected path must flip,
			// or the fix is being credited for nothing.
			{
				// A flat hand whose fingers carry a barely-there curl. The
				// curl sign alternates, which is what landmark noise does to a
				// hand held flat, and it keeps the score inside the weak band
				// where the old code had no stable answer.
				auto buildHand= [](float curlSign) {
					std::array<glm::vec3, HAND_LANDMARK_COUNT> points{};
					points[(int)eHandLandmark::WRIST]= glm::vec3(0.f, 0.f, 0.f);
					const float spread[FINGER_COUNT]= {0.035f, 0.02f, 0.f, -0.018f, -0.032f};
					for (int finger= 0; finger < FINGER_COUNT; ++finger)
					{
						const int* joints= FINGER_JOINTS[finger];
						const glm::vec3 base(0.075f, spread[finger], 0.f);
						points[joints[0]]= base;
						for (int joint= 1; joint < 4; ++joint)
						{
							const float along= 0.022f * (float)joint;
							// Quadratic droop = a gentle curl; ~0.5 deg per joint
							const float curl= curlSign * 0.00018f * (float)(joint * joint);
							points[joints[joint]]= base + glm::vec3(along, 0.f, curl);
						}
					}
					return points;
				};

				auto runSequence= [&buildHand](HandPoseModel::PalmarSideMemory* memory) {
					int flips= 0;
					glm::vec3 previousZ(0.f);
					for (int step= 0; step < 120; ++step)
					{
						// Sign alternates every few frames, as noise would
						const float curlSign= sinf((float)step * 1.1f) > 0.f ? 1.f : -1.f;
						const std::array<glm::vec3, HAND_LANDMARK_COUNT> points= buildHand(curlSign);
						const glm::vec3 z=
							glm::vec3(HandPoseModel::computePalmFrame(points, eHandSide::Left, memory)[2]);
						if (step > 0 && glm::dot(z, previousZ) < 0.f)
							flips++;
						previousZ= z;
					}
					return flips;
				};

				HandPoseModel::PalmarSideMemory memory;
				const int flipsWithMemory= runSequence(&memory);
				const int flipsWithoutMemory= runSequence(nullptr);

				// A reacquired hand must not inherit the old side
				HandPoseModel::PalmarSideMemory cleared= memory;
				cleared.reset();
				const bool bClearedForgets= glm::dot(cleared.palmarNormal, cleared.palmarNormal) < 1e-6f;

				MIKAN_LOG_INFO("test-imufilter")
					<< "(m) palmar side: " << flipsWithMemory << " flips with memory, "
					<< flipsWithoutMemory << " without, reset clears=" << bClearedForgets;

				// flipsWithoutMemory > 0 is the teeth: without it this test
				// passes whether or not the memory does anything at all
				if (flipsWithoutMemory == 0)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(m) FAILED: the unprotected path did not flip, so this proves nothing";
					result= 1;
				}
				else if (flipsWithMemory != 0 || !bClearedForgets)
				{
					MIKAN_LOG_ERROR("test-imufilter")
						<< "(m) FAILED: the palmar side is not temporally stable";
					result= 1;
				}
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-imufilter") << "All IMU filter checks passed";

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-joycon")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-joycon.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// (a) Decode unit test: hand-built report bytes with known values
			{
				unsigned char report[64]= {};
				report[0]= 0x30;
				// sample 0 at offset 13: accel (1000, -2000, 4096), gyro (100, -100, 0)
				auto writeInt16= [&report](int offset, short value) {
					report[offset]= (unsigned char)(value & 0xFF);
					report[offset + 1]= (unsigned char)((value >> 8) & 0xFF);
				};
				writeInt16(13, 1000);
				writeInt16(15, -2000);
				writeInt16(17, 4096);
				writeInt16(19, 100);
				writeInt16(21, -100);
				writeInt16(23, 0);

				const ImuSample sample= JoyconDevice::decodeSample(report, 13, 1234.0);
				// 4096 raw * 0.000244 g * 9.80665 = ~9.80 m/s^2 (i.e. ~1g)
				const float expectedZ= 4096.f * 0.000244f * 9.80665f;
				const float expectedGyroX= 100.f * 0.070f * 0.01745329252f;
				MIKAN_LOG_INFO("test-joycon")
					<< "(a) decode: accel=(" << sample.acceleration.x << "," << sample.acceleration.y << ","
					<< sample.acceleration.z << ") gyro=(" << sample.angularVelocity.x << ","
					<< sample.angularVelocity.y << "," << sample.angularVelocity.z << ")";
				if (fabsf(sample.acceleration.z - expectedZ) > 1e-4f ||
					fabsf(sample.angularVelocity.x - expectedGyroX) > 1e-6f ||
					sample.acceleration.y >= 0.f || sample.angularVelocity.y >= 0.f ||
					sample.timestampMs != 1234.0)
				{
					MIKAN_LOG_ERROR("test-joycon") << "(a) FAILED: sample decode mismatch";
					result= 1;
				}
			}

			// (b) Live readout: enumerate, open, stream for a few seconds.
			// This is the hardware gate for the whole IMU feature - if a
			// Joy-Con won't stream here, nothing downstream can work.
			{
				JoyconDeviceManager manager;
				manager.startup();

				const size_t deviceCount= manager.getDeviceCount();
				MIKAN_LOG_INFO("test-joycon") << "(b) devices found: " << deviceCount;
				if (deviceCount == 0)
				{
					MIKAN_LOG_WARNING("test-joycon")
						<< "No Joy-Cons paired. Pair them in Windows Bluetooth settings first "
						   "(hold the small sync button on the rail until the lights run), then rerun.";
				}

				for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
				{
					IImuDevice* device= manager.getDeviceByIndex(deviceIndex);
					MIKAN_LOG_INFO("test-joycon")
						<< "  [" << deviceIndex << "] " << device->getFriendlyName() << " side="
						<< (device->getSide() == eImuSide::Left ? "Left"
															   : (device->getSide() == eImuSide::Right ? "Right"
																								      : "Unassigned"));
					if (!device->open())
					{
						MIKAN_LOG_ERROR("test-joycon") << "  open FAILED";
						result= 1;
					}
				}

				// Stream for 5 seconds, reporting once a second
				std::vector<ImuSample> samples;
				for (int second= 0; second < 5 && deviceCount > 0; ++second)
				{
					std::this_thread::sleep_for(std::chrono::seconds(1));
					for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
					{
						IImuDevice* device= manager.getDeviceByIndex(deviceIndex);
						samples.clear();
						device->fetchSamples(samples);
						if (samples.empty())
						{
							MIKAN_LOG_WARNING("test-joycon")
								<< "  " << device->getFriendlyName() << ": no samples this second";
							continue;
						}

						const ImuSample& newest= samples.back();
						const float accelMagnitude= glm::length(newest.acceleration);
						MIKAN_LOG_INFO("test-joycon")
							<< "  " << device->getFriendlyName() << ": " << samples.size() << " samples, "
							<< device->getSampleRateHz() << " Hz, battery "
							<< (int)(device->getBatteryLevel() * 100.f) << "%"
							<< " | accel (" << newest.acceleration.x << ", " << newest.acceleration.y << ", "
							<< newest.acceleration.z << ") |a|=" << accelMagnitude
							<< " | gyro (" << newest.angularVelocity.x << ", " << newest.angularVelocity.y
							<< ", " << newest.angularVelocity.z << ")";

						// Held still, |accel| must read ~9.81 - the single best
						// sanity check that scaling and decode are right
						if (second == 4 && fabsf(accelMagnitude - 9.80665f) > 2.5f)
						{
							MIKAN_LOG_WARNING("test-joycon")
								<< "  |accel| is " << accelMagnitude
								<< " m/s^2, expected ~9.81 when held still - check the accel scale";
						}
					}
				}

				bool bAnyStreamed= false;
				for (size_t deviceIndex= 0; deviceIndex < deviceCount; ++deviceIndex)
					bAnyStreamed|= manager.getDeviceByIndex(deviceIndex)->isStreaming();
				if (deviceCount > 0 && !bAnyStreamed)
				{
					MIKAN_LOG_ERROR("test-joycon")
						<< "(b) FAILED: devices opened but never streamed IMU samples";
					result= 1;
				}

				manager.shutdown();
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-joycon") << "Joy-Con checks passed";

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-depthview")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-depthview.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// Synthetic RealSense frame: flat plane at 0.6m, pinhole streams,
			// depth camera offset 15mm to the left of color (D455-ish layout)
			constexpr int kW= 848, kH= 480;
			std::vector<uint16_t> depthImage((size_t)kW * kH, 600); // 600 * 1mm = 0.6m

			DepthFrameView view;
			view.valid= true;
			view.depthData= depthImage.data();
			view.depthWidth= kW;
			view.depthHeight= kH;
			view.depthUnitsMeters= 0.001f;
			view.colorIntrinsics= {1280, 720, 900.f, 900.f, 640.f, 360.f, 0, {}};
			view.depthIntrinsics= {kW, kH, 420.f, 420.f, 424.f, 240.f, 0, {}};
			view.depthToColorTranslation[0]= 0.015f;

			// (a) center of the color image -> straight-ahead point at 0.6m
			glm::vec3 point(0.f);
			bool bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
			MIKAN_LOG_INFO("test-depthview") << "(a) center: ok=" << bOk << " p=(" << point.x << "," << point.y
				<< "," << point.z << ")";
			if (!bOk || fabsf(point.x) > 0.003f || fabsf(point.y) > 0.003f || fabsf(point.z - 0.6f) > 0.005f)
			{
				MIKAN_LOG_ERROR("test-depthview") << "(a) FAILED: expected ~(0,0,0.6)";
				result= 1;
			}

			// (b) off-center pixel: the deprojected ray must hit the plane at
			// the right lateral offset (100px right of center at fx=900, 0.6m
			// -> x = 100/900*0.6 = 66.7mm)
			bOk= view.sampleCameraPointAtColorPixel(740.f, 360.f, 0.15f, 1.5f, point);
			MIKAN_LOG_INFO("test-depthview") << "(b) offset: ok=" << bOk << " x=" << point.x;
			if (!bOk || fabsf(point.x - 0.0667f) > 0.004f || fabsf(point.z - 0.6f) > 0.005f)
			{
				MIKAN_LOG_ERROR("test-depthview") << "(b) FAILED: lateral offset wrong";
				result= 1;
			}

			// (c) hole rejection: zero out a patch -> sampling inside must fail
			for (int y= 200; y < 280; ++y)
				for (int x= 380; x < 470; ++x)
					depthImage[(size_t)y * kW + x]= 0;
			bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
			MIKAN_LOG_INFO("test-depthview") << "(c) hole: ok=" << bOk << " (expected 0)";
			if (bOk)
			{
				MIKAN_LOG_ERROR("test-depthview") << "(c) FAILED: hole must not resolve";
				result= 1;
			}

			// (d) out-of-range rejection: plane beyond maxDepth reads as no hand
			std::fill(depthImage.begin(), depthImage.end(), (uint16_t)2000); // 2m
			bOk= view.sampleCameraPointAtColorPixel(640.f, 360.f, 0.15f, 1.5f, point);
			MIKAN_LOG_INFO("test-depthview") << "(d) far plane: ok=" << bOk << " (expected 0)";
			if (bOk)
			{
				MIKAN_LOG_ERROR("test-depthview") << "(d) FAILED: 2m surface must be rejected as non-hand";
				result= 1;
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-depthview") << "All depth-view checks passed";

			log_dispose();
			return result;
		}

		if (std::string(argv[i]) == "--test-roiquality")
		{
			LoggerSettings loggerSettings= {};
			loggerSettings.min_log_level= LogSeverityLevel::info;
			loggerSettings.log_filename= "test-roiquality.log";
			loggerSettings.enable_console= true;
			log_init(loggerSettings);

			int result= 0;

			// A tracked hand whose 21 landmarks tile the given box, so the
			// analyzer's expanded ROI lands where the test wants it
			auto makeHand= [](const cv::Rect2f& landmarkBox) {
				TrackedHand hand;
				hand.tracked= true;
				for (int lm= 0; lm < HAND_LANDMARK_COUNT; ++lm)
				{
					hand.imagePoints[lm]= glm::vec3(
						landmarkBox.x + landmarkBox.width * (float)(lm % 5) / 4.f,
						landmarkBox.y + landmarkBox.height * (float)(lm / 5) / 4.f,
						0.f);
				}
				return hand;
			};

			// Landmarks span 100x100 at the frame center; the analyzer expands
			// that by 1.5 -> a 150x150 ROI, with the ring reaching out to 210
			const cv::Size frameSize(640, 480);
			const cv::Point2f center(320.f, 240.f);
			const cv::Rect2f landmarkBox(center.x - 50.f, center.y - 50.f, 100.f, 100.f);
			const cv::Rect roiRect((int)center.x - 75, (int)center.y - 75, 150, 150);

			// (a) uniform hand patch on a dark background: exact mean, no
			// clipping, no contrast, strong separation
			{
				cv::Mat frame(frameSize, CV_8UC3, cv::Scalar(40, 40, 40));
				frame(roiRect).setTo(cv::Scalar(140, 140, 140));

				TrackedHand hand= makeHand(landmarkBox);
				HandRoiQuality::analyzeHand(frame, hand);
				const HandImageQuality& quality= hand.imageQuality;

				MIKAN_LOG_INFO("test-roiquality")
					<< "(a) uniform: valid=" << quality.valid << " mean=" << quality.meanLuma
					<< " contrast=" << quality.contrast << " separation=" << quality.backgroundSeparation
					<< " clipS=" << quality.shadowClipRatio << " clipH=" << quality.highlightClipRatio;
				if (!quality.valid || fabsf(quality.meanLuma - 140.f) > 3.f || quality.contrast > 2.f ||
					quality.backgroundSeparation < 90.f || quality.shadowClipRatio > 0.001f ||
					quality.highlightClipRatio > 0.001f || quality.noise > 0.5f)
				{
					MIKAN_LOG_ERROR("test-roiquality") << "(a) FAILED";
					result= 1;
				}
			}

			// (b) half blown out, half pitch black: both clip ratios ~50%
			{
				cv::Mat frame(frameSize, CV_8UC3, cv::Scalar(40, 40, 40));
				cv::Rect leftHalf(roiRect.x, roiRect.y, roiRect.width / 2, roiRect.height);
				cv::Rect rightHalf(roiRect.x + roiRect.width / 2, roiRect.y, roiRect.width / 2, roiRect.height);
				frame(leftHalf).setTo(cv::Scalar(255, 255, 255));
				frame(rightHalf).setTo(cv::Scalar(0, 0, 0));

				TrackedHand hand= makeHand(landmarkBox);
				HandRoiQuality::analyzeHand(frame, hand);
				const HandImageQuality& quality= hand.imageQuality;

				MIKAN_LOG_INFO("test-roiquality")
					<< "(b) clipping: clipH=" << quality.highlightClipRatio
					<< " clipS=" << quality.shadowClipRatio << " contrast=" << quality.contrast;
				if (!quality.valid ||
					fabsf(quality.highlightClipRatio - 0.5f) > 0.05f ||
					fabsf(quality.shadowClipRatio - 0.5f) > 0.05f ||
					quality.contrast < 100.f)
				{
					MIKAN_LOG_ERROR("test-roiquality") << "(b) FAILED";
					result= 1;
				}
			}

			// (c) noise and sharpness must separate: raw Laplacian variance
			// scores a noisy flat patch as "sharp"; the median split must not
			{
				auto measure= [&](const cv::Mat& frame) {
					TrackedHand hand= makeHand(landmarkBox);
					HandRoiQuality::analyzeHand(frame, hand);
					return hand.imageQuality;
				};

				// Clean flat patch
				cv::Mat flatFrame(frameSize, CV_8UC3, cv::Scalar(128, 128, 128));
				const HandImageQuality flatQuality= measure(flatFrame);

				// Same patch with heavy deterministic sensor noise
				cv::Mat noisyFrame= flatFrame.clone();
				{
					cv::RNG rng(12345);
					cv::Mat noise(frameSize, CV_16SC3);
					rng.fill(noise, cv::RNG::NORMAL, 0.0, 8.0);
					cv::Mat wide;
					noisyFrame.convertTo(wide, CV_16SC3);
					wide+= noise;
					wide.convertTo(noisyFrame, CV_8UC3);
				}
				const HandImageQuality noisyQuality= measure(noisyFrame);

				// Crisp checkerboard (real edges) and its blurred twin
				cv::Mat checkerFrame(frameSize, CV_8UC3, cv::Scalar(40, 40, 40));
				for (int row= 0; row < roiRect.height / 8; ++row)
					for (int col= 0; col < roiRect.width / 8; ++col)
					{
						const uint8_t value= ((row + col) % 2 == 0) ? 200 : 60;
						checkerFrame(cv::Rect(roiRect.x + col * 8, roiRect.y + row * 8, 8, 8))
							.setTo(cv::Scalar(value, value, value));
					}
				const HandImageQuality checkerQuality= measure(checkerFrame);

				cv::Mat blurredFrame;
				cv::GaussianBlur(checkerFrame, blurredFrame, cv::Size(9, 9), 3.0);
				const HandImageQuality blurredQuality= measure(blurredFrame);

				MIKAN_LOG_INFO("test-roiquality")
					<< "(c) noise: flat=" << flatQuality.noise << " noisy=" << noisyQuality.noise
					<< " | sharpness: checker=" << checkerQuality.sharpness
					<< " blurred=" << blurredQuality.sharpness << " noisyFlat=" << noisyQuality.sharpness;
				if (noisyQuality.noise < flatQuality.noise + 3.f ||
					checkerQuality.sharpness < blurredQuality.sharpness * 2.f ||
					noisyQuality.sharpness > checkerQuality.sharpness * 0.5f)
				{
					MIKAN_LOG_ERROR("test-roiquality")
						<< "(c) FAILED: noise/sharpness must separate (noisy flat must not read as sharp)";
					result= 1;
				}
			}

			// (d) resolution invariance: the same scene at 2x resolution must
			// produce the same luminance statistics (the decimation contract)
			{
				cv::Mat frame(frameSize, CV_8UC3, cv::Scalar(40, 40, 40));
				frame(roiRect).setTo(cv::Scalar(140, 140, 140));
				TrackedHand hand= makeHand(landmarkBox);
				HandRoiQuality::analyzeHand(frame, hand);

				cv::Mat doubleFrame;
				cv::resize(frame, doubleFrame, cv::Size(), 2.0, 2.0, cv::INTER_NEAREST);
				cv::Rect2f doubleBox(landmarkBox.x * 2.f, landmarkBox.y * 2.f,
									 landmarkBox.width * 2.f, landmarkBox.height * 2.f);
				TrackedHand doubleHand= makeHand(doubleBox);
				HandRoiQuality::analyzeHand(doubleFrame, doubleHand);

				MIKAN_LOG_INFO("test-roiquality")
					<< "(d) scale: mean " << hand.imageQuality.meanLuma << " vs "
					<< doubleHand.imageQuality.meanLuma << ", separation " << hand.imageQuality.backgroundSeparation
					<< " vs " << doubleHand.imageQuality.backgroundSeparation;
				if (!doubleHand.imageQuality.valid ||
					fabsf(hand.imageQuality.meanLuma - doubleHand.imageQuality.meanLuma) > 5.f ||
					fabsf(hand.imageQuality.backgroundSeparation -
						  doubleHand.imageQuality.backgroundSeparation) > 10.f)
				{
					MIKAN_LOG_ERROR("test-roiquality") << "(d) FAILED: statistics must survive decimation";
					result= 1;
				}
			}

			// (e) degenerate ROI: landmarks collapsed to a point stay invalid
			{
				cv::Mat frame(frameSize, CV_8UC3, cv::Scalar(128, 128, 128));
				TrackedHand hand= makeHand(cv::Rect2f(320.f, 240.f, 0.f, 0.f));
				HandRoiQuality::analyzeHand(frame, hand);
				if (hand.imageQuality.valid)
				{
					MIKAN_LOG_ERROR("test-roiquality") << "(e) FAILED: degenerate ROI must stay invalid";
					result= 1;
				}
				else
				{
					MIKAN_LOG_INFO("test-roiquality") << "(e) degenerate ROI rejected";
				}
			}

			// (f) flicker tracker: an 8 Hz / 5% amplitude ripple on the frame
			// mean must be found (instability ~ amplitude/sqrt(2)), and a
			// steady signal must read as stable with no dominant frequency
			{
				LumaFlickerTracker tracker;
				const double fps= 60.0;
				for (int frameIndex= 0; frameIndex < 300; ++frameIndex)
				{
					const double t= frameIndex / fps;
					const double luma= 120.0 * (1.0 + 0.05 * sin(2.0 * CV_PI * 8.0 * t));
					cv::Mat frame(8, 8, CV_8UC3,
								  cv::Scalar(cvRound(luma), cvRound(luma), cvRound(luma)));
					tracker.addFrame(frame, t * 1000.0);
				}
				const float rippleInstability= tracker.getInstability();
				const float rippleHz= tracker.getDominantHz();

				tracker.reset();
				for (int frameIndex= 0; frameIndex < 300; ++frameIndex)
				{
					cv::Mat frame(8, 8, CV_8UC3, cv::Scalar(120, 120, 120));
					tracker.addFrame(frame, frameIndex / fps * 1000.0);
				}
				const float steadyInstability= tracker.getInstability();
				const float steadyHz= tracker.getDominantHz();

				MIKAN_LOG_INFO("test-roiquality")
					<< "(f) flicker: ripple instability=" << rippleInstability << " @" << rippleHz
					<< "Hz, steady instability=" << steadyInstability << " @" << steadyHz << "Hz";
				if (rippleInstability < 0.02f || rippleInstability > 0.05f ||
					fabsf(rippleHz - 8.f) > 0.5f ||
					steadyInstability > 0.005f || steadyHz != 0.f)
				{
					MIKAN_LOG_ERROR("test-roiquality") << "(f) FAILED";
					result= 1;
				}
			}

			if (result == 0)
				MIKAN_LOG_INFO("test-roiquality") << "All ROI-quality checks passed";

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

			auto runEstimator= [&](const char* label) {
				LandmarkTo3D landmarkTo3D;
				landmarkTo3D.configure(intrinsics, boneLength);

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

			// PnP must recover the exact synthetic pose (sub-mm)
			const float rmsPnp= runEstimator("PnP all-21");
			if (rmsPnp > 0.001f)
			{
				MIKAN_LOG_ERROR("test-pnp") << "FAILED: PnP must recover the exact synthetic pose";
				result= 1;
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
