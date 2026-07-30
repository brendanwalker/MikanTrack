#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include "App.h"
#include "ArucoMarkerPoseSampler.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "ExtrinsicsWizard.h"
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
