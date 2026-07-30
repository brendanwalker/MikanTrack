#include <string>

#include "opencv2/core.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include "App.h"
#include "CalibrationPatternFinder_Charuco.h"
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
