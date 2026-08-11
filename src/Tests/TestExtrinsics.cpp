#include "TestCommon.h"

static int runExtrinsicsTest(const TestArgs& args)
{
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
		auto arucoFinder= std::make_shared<CalibrationPatternFinder_Aruco>(
			1280, 720, 100.f, 0, eCharucoDictionaryType::DICT_6X6);
		PatternPoseSampler sampler(intrinsics, arucoFinder, 12);
		sampler.setGrayscaleFrame(&cameraImage);
		for (int sampleIndex= 0; sampleIndex < 12; ++sampleIndex)
			sampler.captureSample();

		glm::dmat4 cameraFromMarker;

		if (!sampler.hasFinishedSampling() || !sampler.computeCalibratedPatternPose(cameraFromMarker))
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

		// (pair) Cross-camera validation math: perfect synthetic
		// observations must score near-zero error, and a 1-degree rotation
		// error on one camera must be flagged
		{
			const int gridCols= 10, gridRows= 7;
			const float spacingMM= 16.f;

			// A camera pose in CV convention (maps CV camera space -> world)
			auto makeLookAtCv= [](const glm::dvec3& eye, const glm::dvec3& target) {
				const glm::dvec3 zAxis= glm::normalize(target - eye); // CV +Z = forward
				const glm::dvec3 xAxis= glm::normalize(glm::cross(zAxis, glm::dvec3(0, 0, 1)));
				const glm::dvec3 yAxis= glm::cross(zAxis, xAxis);
				glm::dmat4 markerFromCamera(1.0);
				markerFromCamera[0]= glm::dvec4(xAxis, 0.0);
				markerFromCamera[1]= glm::dvec4(yAxis, 0.0);
				markerFromCamera[2]= glm::dvec4(zAxis, 0.0);
				markerFromCamera[3]= glm::dvec4(eye, 1.0);
				return markerFromCamera;
			};

			auto makeObservations= [&](int cameraIndex, const glm::dmat4& markerFromCamera) {
				CameraBoardObservations observations;
				observations.cameraIndex= cameraIndex;
				observations.intrinsics= intrinsics;
				observations.markerFromCamera= markerFromCamera;
				for (int row= 0; row < gridRows; ++row)
				{
					for (int col= 0; col < gridCols; ++col)
					{
						// Flat grid on the table, centered on the origin
						const glm::dvec3 pointWorld(
							(col - (gridCols - 1) * 0.5) * spacingMM * 0.001,
							(row - (gridRows - 1) * 0.5) * spacingMM * 0.001, 0.0);
						cv::Point2f imagePoint;
						if (!projectWorldPoint(intrinsics, markerFromCamera, pointWorld, imagePoint))
							continue;
						observations.pointIDs.push_back(row * gridCols + col);
						observations.imagePoints.push_back(imagePoint);
						observations.objectPointsMM.push_back(
							cv::Point3f(col * spacingMM, 0.f, -row * spacingMM));
					}
				}
				return observations;
			};

			const glm::dmat4 cameraA= makeLookAtCv(glm::dvec3(0.02, 0.01, 0.5), glm::dvec3(0, 0, 0));
			const glm::dmat4 cameraB= makeLookAtCv(glm::dvec3(0.4, -0.05, 0.45), glm::dvec3(0, 0, 0));

			ExtrinsicsPairQuality cleanQuality;
			evaluateExtrinsicsPair(makeObservations(0, cameraA), makeObservations(1, cameraB),
								   cleanQuality);

			// The same observations judged against a camera-B pose whose
			// ORIENTATION is off by 1 degree (rotated about its own center -
			// a world-side rotation through the origin would nearly fix every
			// ray through the board and hide the error)
			const glm::dmat4 cameraBBad=
				cameraB * glm::rotate(glm::dmat4(1.0), glm::radians(1.0), glm::dvec3(1, 0, 0));
			CameraBoardObservations observationsBBad= makeObservations(1, cameraB);
			observationsBBad.markerFromCamera= cameraBBad;
			ExtrinsicsPairQuality badQuality;
			evaluateExtrinsicsPair(makeObservations(0, cameraA), observationsBBad, badQuality);

			MIKAN_LOG_INFO("test-extrinsics")
				<< "(pair) clean: rms px=" << cleanQuality.reprojectionRmsPx
				<< " spacing mm=" << cleanQuality.spacingErrorMm << " scale=" << cleanQuality.spacingScale
				<< " flat mm=" << cleanQuality.planarityRmsMm << " | 1-deg error: rms px="
				<< badQuality.reprojectionRmsPx << " spacing mm=" << badQuality.spacingErrorMm;
			if (!cleanQuality.valid || cleanQuality.reprojectionRmsPx > 0.05f ||
				cleanQuality.spacingErrorMm > 0.05f || fabs(cleanQuality.spacingScale - 1.f) > 0.001f ||
				cleanQuality.planarityRmsMm > 0.05f)
			{
				MIKAN_LOG_ERROR("test-extrinsics")
					<< "(pair) FAILED: perfect observations must validate cleanly";
				result= 1;
			}
			// The signature of a small orientation error with only two views:
			// the rays still nearly intersect (modest reprojection growth) but
			// the RECONSTRUCTION warps - the spacing check exists precisely
			// because reprojection alone can hide this
			if (!badQuality.valid || badQuality.spacingErrorMm < 1.f ||
				badQuality.reprojectionRmsPx < 0.1f)
			{
				MIKAN_LOG_ERROR("test-extrinsics")
					<< "(pair) FAILED: a 1-degree extrinsics error must be flagged";
				result= 1;
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

	return result;
}

MIKAN_REGISTER_TEST("--test-extrinsics", "Aruco marker pose + table-plane raycast", eTestCategory::SelfTest, runExtrinsicsTest);
