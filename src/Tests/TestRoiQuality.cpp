#include "TestCommon.h"

static int runRoiQualityTest(const TestArgs& args)
{
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

	return result;
}

MIKAN_REGISTER_TEST("--test-roiquality", "Hand ROI image-quality metrics + flicker tracker", eTestCategory::SelfTest, runRoiQualityTest);
