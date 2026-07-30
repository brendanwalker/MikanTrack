#include "CameraMath.h"
#include "Logger.h"
#include "MikanMathTypes.h"
#include "MikanVideoSourceTypes.h"
#include "MikanCameraTypes.h"
#include "MathGLM.h"
#include "MathUtility.h"
#include "MathTypeConversion.h"

#include "opencv2/opencv.hpp"

glm::mat4 computeGLMCameraViewMatrix(const glm::mat4& poseXform)
{
	// Convert the camera pose transform into a modelview matrix
	// This is adapted from glm::lookAt
	const glm::vec3 right= glm::vec3(poseXform[0]);
	const glm::vec3 up= glm::vec3(poseXform[1]);
	const glm::vec3 forward= glm::vec3(poseXform[2]);
	const glm::vec3 eye= glm::vec3(poseXform[3]);

	glm::mat4 modelView(1.f);
	modelView[0][0]= right.x;
	modelView[1][0]= right.y;
	modelView[2][0]= right.z;
	modelView[0][1]= up.x;
	modelView[1][1]= up.y;
	modelView[2][1]= up.z;
	modelView[0][2]= forward.x;
	modelView[1][2]= forward.y;
	modelView[2][2]= forward.z;
	modelView[3][0]= -glm::dot(right, eye);
	modelView[3][1]= -glm::dot(up, eye);
	modelView[3][2]= -glm::dot(forward, eye);

	return modelView;
}

bool computeMonoLensCameraCalibration(const int frameWidth, const int frameHeight,
									  const OpenCVCalibrationGeometry& opencvLensCalibrationGeometry,
									  const std::vector<t_opencv_point2d_list>& cvImagePointsList,
									  const std::vector<t_opencv_pointID_list>& cvImagePointIDs,
									  MikanMonoIntrinsics& outIntrinsics, double& outReprojectionError)
{
	bool bSuccess= true;

	// Initialize the output intrinsics with default values
	outIntrinsics= MikanMonoIntrinsics();

	// Each 2d image point set needs a corresponding 3d object point set.
	// The calibration geometry point list is indexed by corner id (charuco corner ids
	// are assigned in row-major board order), so build each sample's object point list
	// by looking up the ids of the corners that were actually detected.
	// This allows partial-board captures instead of assuming every corner was seen.
	const t_opencv_point3d_list& geometryPoints= opencvLensCalibrationGeometry.points;
	std::vector<t_opencv_point2d_list> cvValidImagePointsList;
	std::vector<t_opencv_point3d_list> cvObjectPointsList;
	for (size_t sampleIndex= 0; sampleIndex < cvImagePointsList.size() && sampleIndex < cvImagePointIDs.size();
		 ++sampleIndex)
	{
		const t_opencv_point2d_list& imagePoints= cvImagePointsList[sampleIndex];
		const t_opencv_pointID_list& imagePointIDs= cvImagePointIDs[sampleIndex];

		// A sample needs at least 6 point correspondences to constrain the solve
		// (relaxed from requiring every corner on the board)
		if (imagePoints.size() < 6 || imagePoints.size() != imagePointIDs.size())
			continue;

		t_opencv_point3d_list objectPoints;
		objectPoints.reserve(imagePoints.size());
		bool bValidSample= true;
		for (const int pointID : imagePointIDs)
		{
			if (pointID < 0 || pointID >= (int)geometryPoints.size())
			{
				bValidSample= false;
				break;
			}

			objectPoints.push_back(geometryPoints[pointID]);
		}

		if (!bValidSample)
			continue;

		cvValidImagePointsList.push_back(imagePoints);
		cvObjectPointsList.push_back(objectPoints);
	}

	if (cvValidImagePointsList.size() == 0)
	{
		MIKAN_MT_LOG_ERROR("computeCameraCalibration") << "No valid image point samples to calibrate with";
		return false;
	}

	// Compute an initial guess for the distorted camera intrinsic matrix
	// using correspondence between the object points and the image points
	cv::Size cvImageSize(frameWidth, frameHeight);
	cv::Matx33d cvDistortedIntrinsicMatrix=
		cv::initCameraMatrix2D(cvObjectPointsList, cvValidImagePointsList, cvImageSize);

	// Refine the camera intrinsic matrix and compute distortion parameters
	cv::Mat cvDistCoeffsRowVector;
	try
	{
		outReprojectionError= cv::calibrateCamera(
			cvObjectPointsList, cvValidImagePointsList, cvImageSize,
			cvDistortedIntrinsicMatrix,         // Input/Output camera intrinsic matrix
			cvDistCoeffsRowVector,              // Output distortion coefficients
			cv::noArray(), cv::noArray(),       // best fit board poses as rvec/tvec pairs
			cv::CALIB_FIX_ASPECT_RATIO +        // The functions considers only fy as a free parameter
				cv::CALIB_FIX_PRINCIPAL_POINT + // The principal point is not changed during the global optimization
				cv::CALIB_ZERO_TANGENT_DIST
				+                          // Tangential distortion coefficients (p1,p2) are set to zeros and stay zero
				cv::CALIB_RATIONAL_MODEL + // Coefficients k4, k5, and k6 are enabled
				cv::CALIB_FIX_K3 + cv::CALIB_FIX_K4 + cv::CALIB_FIX_K5, // radial distortion coefficients k3, k4, & k5
																		// are not changed during the optimization
			cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON));

		bSuccess= is_valid_float(outReprojectionError);
	}
	catch (cv::Exception* e)
	{
		MIKAN_MT_LOG_ERROR("computeCameraCalibration") << "Error computing lens calibration: " << e->msg;
		bSuccess= false;
	}

	if (bSuccess)
	{
		// Create a modified camera intrinsic matrix to crop out the unwanted border
		cv::Mat cvUndistortedIntrinsicMatrix= cv::getOptimalNewCameraMatrix(
			cvDistortedIntrinsicMatrix, cvDistCoeffsRowVector, cv::Size(frameWidth, frameHeight),
			0.f); // We want 0% of the garbage border

		// Derive the FoV angles from the image size and the newly computed undistorted intrinsic matrix
		double hfov, vfov;
		double unusedFocalLength;
		cv::Point2d ununsedPrincipalPoint;
		double aspectRatio;
		cv::calibrationMatrixValues(cvUndistortedIntrinsicMatrix, cvImageSize, 0.0,
									0.0, // Don't know (and don't need) the physical aperture size of the lens
									hfov, vfov, unusedFocalLength, ununsedPrincipalPoint, aspectRatio);

		// cv::calibrateCamera() will return all 14 distortion parameters, but we only want the first 8
		// Also convert from a row vector (OpenCV format) back to a column vector (Mikan format)
		cv::Mat cvDistCoeffsColVector;
		cv::transpose(cvDistCoeffsRowVector.colRange(cv::Range(0, 8)), cvDistCoeffsColVector);

		// Write the calibration output state
		outIntrinsics.pixel_width= (double)frameWidth;
		outIntrinsics.pixel_height= (double)frameHeight;
		outIntrinsics.hfov= hfov;
		outIntrinsics.vfov= vfov;
		outIntrinsics.aspect_ratio= aspectRatio;
		outIntrinsics.znear= DEFAULT_MONO_ZNEAR;
		outIntrinsics.zfar= DEFAULT_MONO_ZFAR;
		outIntrinsics.distortion_coefficients= cv_vec8_to_Mikan_distortion(cvDistCoeffsColVector);
		outIntrinsics.distorted_camera_matrix= cv_mat33d_to_MikanMatrix3d(cvDistortedIntrinsicMatrix);
		outIntrinsics.undistorted_camera_matrix= cv_mat33d_to_MikanMatrix3d(cvUndistortedIntrinsicMatrix);
	}

	return bSuccess;
}

bool computeOpenCVCameraRelativePatternTransform(const MikanMonoIntrinsics& intrinsics,
												 const t_opencv_point2d_list& undistortedImagePoints,
												 const t_opencv_point3d_list& objectPointsMM, cv::Quatd& outOrientation,
												 cv::Vec3d& outPositionMM, double* outMeanError)
{
	// Bail if there isn't a corresponding world point for every image point
	if (undistortedImagePoints.size() != objectPointsMM.size() || undistortedImagePoints.size() == 0)
	{
		return false;
	}

	// Fetch the 3x3 GLM camera undistorted intrinsic matrix and store into a 3x3 openCV matrix
	const MikanMatrix3d& glmIntrinsicMatrix= intrinsics.undistorted_camera_matrix;
	cv::Matx33d cvIntrinsicMatrix= MikanMatrix3d_to_cv_mat33d(glmIntrinsicMatrix);

	// Zero out the distortion coefficients since we are working with undistorted image points
	cv::Matx<double, 1, 8> cvDistCoeffsRowVector= cv::Matx<double, 1, 8>::zeros();

	// Use AP3P solver if we have exactly 4 points, otherwise use the iterative solver
	// since the iterative solver needs at least 6 points
	int solver= undistortedImagePoints.size() == 4 ? cv::SOLVEPNP_AP3P : cv::SOLVEPNP_ITERATIVE;

	// Given an object model and the image points samples we could be able to compute
	// a position and orientation of the mat relative to the camera
	cv::Mat rvec;
	cv::Mat tvecMM; // Mat position in millimeters
	if (!cv::solvePnP(objectPointsMM, undistortedImagePoints, cvIntrinsicMatrix, cvDistCoeffsRowVector, rvec, tvecMM,
					  false, // useExtrinsicGuess
					  solver))
	{
		return false;
	}

	if (outMeanError != nullptr)
	{
		// Project the 3D points to 2D using the obtained transformation
		std::vector<cv::Point2f> projectedPoints;
		cv::projectPoints(objectPointsMM, rvec, tvecMM, cvIntrinsicMatrix, cvDistCoeffsRowVector, projectedPoints);

		// Calculate the reprojection error
		double totalError= 0.0;
		for (size_t i= 0; i < undistortedImagePoints.size(); ++i)
		{
			double error= cv::norm(undistortedImagePoints[i] - projectedPoints[i]);
			totalError+= error;
		}

		*outMeanError= totalError / undistortedImagePoints.size();
	}

	outOrientation= cv::Quatd::createFromRvec(rvec);
	outPositionMM= tvecMM;

	return true;
}

void convertOpenCVCameraRelativePoseToGLMMat(const cv::Quatd& orientation, const cv::Vec3d& positionMM,
											 glm::dmat4& outXform)
{
	// OpenCV camera relative locations in millimeters (by convention from object points passed into solvePnP),
	// but the output GLM transform is in meters (by convention as defined by our renderer)
	cv::Vec3d tvec= positionMM * k_millimeters_to_meters;

	// Convert the OpenCV Rodrigues vector used to store orientation into a 3x3 rotation matrix
	cv::Matx33d rmat= orientation.toRotMat3x3();

	// Compose the openCV  rotation and translation together,
	// then convert OpenCV coordinates to OpenGL coordinates
	// (x, y, z) -> (x, -y, -z)
	cv::Matx44d cv_RTMat=
		cv::Matx44d(rmat(0, 0), rmat(0, 1), rmat(0, 2), tvec(0), -rmat(1, 0), -rmat(1, 1), -rmat(1, 2), -tvec(1),
					-rmat(2, 0), -rmat(2, 1), -rmat(2, 2), -tvec(2), 0.0F, 0.0F, 0.0F, 1.0F);

	// Convert OpenCV matrix to GLM matrix
	glm::mat4 RTMat;
	for (int row= 0; row < 4; ++row)
	{
		for (int col= 0; col < 4; ++col)
		{
			// GLM indexed by column first
			// OpenCV indexed by row first
			RTMat[col][row]= cv_RTMat(row, col);
		}
	}

	// Optionally, rotate result about the x-axis by specified amount.
	glm::mat4 flipAboutX= glm::rotate(glm::mat4(1.f), glm::radians(180.0f), glm::vec3(1.0f, 0.0f, 0.0f));
	outXform= RTMat * flipAboutX;
}

void extractCameraIntrinsicMatrixParameters(const MikanMatrix3d& intrinsic_matrix, float& out_focal_length_x,
											float& out_focal_length_y, float& out_principal_point_x,
											float& out_principal_point_y, float& out_skew)
{
	out_focal_length_x= intrinsic_matrix.x0;
	out_focal_length_y= intrinsic_matrix.y1;
	out_principal_point_x= intrinsic_matrix.z0;
	out_principal_point_y= intrinsic_matrix.z1;
	out_skew= intrinsic_matrix.y0;
}

void extractCameraIntrinsicMatrixParameters(const cv::Matx33f& intrinsic_matrix, float& out_focal_length_x,
											float& out_focal_length_y, float& out_principal_point_x,
											float& out_principal_point_y, float& out_skew)
{
	out_focal_length_x= intrinsic_matrix(0, 0);
	out_focal_length_y= intrinsic_matrix(1, 1);
	out_principal_point_x= intrinsic_matrix(0, 2);
	out_principal_point_y= intrinsic_matrix(1, 2);
	out_skew= intrinsic_matrix(0, 1);
}

void createDefautMonoIntrinsics(int pixelWidth, int pixelHeight, MikanMonoIntrinsics& outIntrinsics)
{
	outIntrinsics= MikanMonoIntrinsics();

	double aspectRatio= (double)pixelHeight / (double)pixelWidth;
	double c_x= (double)pixelWidth / 2.0;
	double c_y= (double)pixelHeight / 2.0;
	double hfov= DEFAULT_MONO_HFOV;
	double vfov= hfov * aspectRatio;
	double f_x= c_x / tan(glm::radians(hfov / 2.0));
	double f_y= c_y / tan(glm::radians(vfov / 2.0));

	outIntrinsics.pixel_width= (double)pixelWidth;
	outIntrinsics.pixel_height= (double)pixelHeight;
	outIntrinsics.hfov= hfov;
	outIntrinsics.vfov= vfov;
	outIntrinsics.znear= DEFAULT_MONO_ZNEAR;
	outIntrinsics.zfar= DEFAULT_MONO_ZFAR;
	outIntrinsics.distorted_camera_matrix= {f_x, 0.0, c_x, 0.0, f_y, c_y, 0.0, 0.0, 1.0};
	outIntrinsics.undistorted_camera_matrix= outIntrinsics.distorted_camera_matrix;
}
// Adapted from https://jamesgregson.blogspot.com/2011/11/matching-calibrated-cameras-with-opengl.html
// Great articles on the subject:
//     https://amytabb.com/tips/tutorials/2019/06/28/OpenCV-to-OpenGL-tutorial-essentials/
//     https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
static void computeOpenGLProjMatFromCameraMatrix(const double pixelWidth, const double pixelHeight,
												 const MikanMatrix3d& cameraMatrix, const float zNear, const float zFar,
												 glm::mat4& outProjection, int* outViewport)
{
	// Extract the 5 OpenCV camera intrinsic parameters
	float alpha, beta, skew, u0, v0;
	extractCameraIntrinsicMatrixParameters(cameraMatrix,
										   alpha, // focal length x
										   beta,  // focal length y
										   u0,    // principal point x (usually image center x)
										   v0,    // principal point y (usually image center y)
										   skew); // skew (usually 0)

	// These parameters define the final viewport that is rendered into by
	// the camera.
	const float& L= 0;
	const float& R= pixelWidth;
	const float& B= 0;
	const float& T= pixelHeight;

	// near and far clipping planes, these only matter for the mapping from
	// world-space z-coordinate into the depth coordinate for OpenGL
	const float& N= zNear;
	const float& F= zFar;

	// Construct an orthographic matrix which maps from projected
	// coordinates to normalized device coordinates in the range [-1, 1].
	// OpenGL then maps coordinates in NDC to the current viewport
	glm::mat4 NDC= glm::mat4(0);
	NDC[0][0]= 2.0 / (R - L);
	NDC[3][0]= -(R + L) / (R - L);
	NDC[1][1]= 2.0 / (T - B);
	NDC[3][1]= -(T + B) / (T - B);
	NDC[2][2]= -2.0 / (F - N);
	NDC[3][2]= -(F + N) / (F - N);
	NDC[3][3]= 1.0;

	// construct a projection matrix, this is identical to the
	// projection matrix computed for the intrinsics, except an
	// additional row is inserted to map the z-coordinate to OpenGL.
	glm::mat4 persp= glm::mat4(0);
	persp[0][0]= alpha;
	persp[1][0]= skew;
	persp[2][0]= -u0;
	persp[1][1]= beta;
	persp[2][1]= -v0;
	persp[2][2]= N + F;
	persp[3][2]= N * F;
	persp[2][3]= -1.0;

	// resulting OpenGL frustum is the product of the orthographic
	// mapping to normalized device coordinates and the augmented
	// camera intrinsic matrix
	outProjection= NDC * persp;

	// set the viewport parameters
	if (outViewport != nullptr)
	{
		outViewport[0]= L;
		outViewport[1]= B;
		outViewport[2]= R - L;
		outViewport[3]= T - B;
	}
}

void computeOpenGLProjMatFromCameraIntrinsics(const MikanMonoIntrinsics& intrinsics, glm::mat4& outProjection,
											  int* outViewport)
{
	computeOpenGLProjMatFromCameraMatrix(intrinsics.pixel_width, intrinsics.pixel_height,
										 intrinsics.undistorted_camera_matrix, intrinsics.znear, intrinsics.zfar,
										 outProjection, outViewport);
}

void computeOpenGLProjMatFromCameraIntrinsics(const MikanStereoIntrinsics& intrinsics, eStereoIntrinsicsSide side,
											  glm::mat4& outProjection, int* outViewport)
{
	computeOpenGLProjMatFromCameraMatrix(intrinsics.pixel_width, intrinsics.pixel_height,
										 (side == eStereoIntrinsicsSide::left) ? intrinsics.left_camera_matrix
																			   : intrinsics.right_camera_matrix,
										 intrinsics.znear, intrinsics.zfar, outProjection, outViewport);
}

void computeCameraRayAtPixel(const struct MikanMonoIntrinsics& intrinsics, const glm::mat4& cameraXform,
							 const glm::vec2& imagePoint, glm::vec3& outRayStart, glm::vec3& outRayDirection)
{
	const glm::vec3 glmCameraRight= cameraXform[0];
	const glm::vec3 glmCameraUp= cameraXform[1];
	const glm::vec3 glmCameraForward= cameraXform[2] * -1.f; // -Z is forward
	const glm::vec3 glmCameraCenter= cameraXform[3];

	float focal_length_x, focal_length_y, principal_point_x, principal_point_y, skew;
	extractCameraIntrinsicMatrixParameters(intrinsics.undistorted_camera_matrix, focal_length_x, focal_length_y,
										   principal_point_x, principal_point_y, skew);

	const float local_x= (imagePoint.x - principal_point_x) / focal_length_x;
	const float local_y= (principal_point_y - imagePoint.y) / focal_length_y; // flip y

	outRayStart= glmCameraCenter;
	outRayDirection= glm::normalize(local_x * glmCameraRight + local_y * glmCameraUp + glmCameraForward);
}

void computeCameraPuckToApertureXform(const glm::dmat4& cameraPuckXform_VRSpace, const glm::dmat4& matPuckXform_VRSpace,
									  const glm::dmat4& apertureToPatternXform_CameraSpace,
									  const glm::dvec3& matPuckOffsetMM, CameraPuckToApertureResults& outResults)
{
	// Convert mm offset to meters.
	// NOTE: Y and Z are intentionally swapped here to convert from the
	// physical measurement coordinate space to VR tracking space convention.
	const double puckToPatternX= matPuckOffsetMM.x * k_millimeters_to_meters;
	const double puckToPatternY= matPuckOffsetMM.y * k_millimeters_to_meters;
	const double puckToPatternZ= matPuckOffsetMM.z * k_millimeters_to_meters;

	// The calibration pattern faces the camera, so it is rotated 180 degrees about Y
	// relative to the mat tracking puck's forward direction.
	const glm::dmat4 puckYawRot180= glm::rotate(glm::dmat4(1.0), k_real64_pi, glm::dvec3(0.0, 1.0, 0.0));

	// Physical translation from mat puck origin to pattern origin (Y/Z swapped)
	const glm::dmat4 translateToPatternXform=
		glm::translate(glm::dmat4(1.0), glm::dvec3(puckToPatternX, puckToPatternZ, puckToPatternY));

	// Combined mat-puck-to-pattern-origin transform: rotate first, then translate
	const glm::dmat4 matPuckToPatternXform= glm_composite_xform(puckYawRot180, translateToPatternXform);

	// Place the pattern in VR tracking space using the mat puck's pose
	const glm::dmat4 patternXform_VRSpace= glm_composite_xform(matPuckToPatternXform, matPuckXform_VRSpace);

	// Invert the aperture-to-pattern transform to get pattern-to-aperture
	const glm::dmat4 patternToApertureXform= glm::inverse(apertureToPatternXform_CameraSpace);

	// Derive the aperture's pose in VR tracking space
	const glm::dmat4 apertureXform_VRSpace= glm_composite_xform(patternToApertureXform, patternXform_VRSpace);

	// Compute the relative offset from the camera tracking puck to the aperture
	const glm::dmat4 cameraPuckToApertureXform= glm_relative_xform(cameraPuckXform_VRSpace, apertureXform_VRSpace);

	outResults.apertureToPatternXform_CameraSpace= apertureToPatternXform_CameraSpace;
	outResults.apertureToMatPuckXform_CameraSpace=
		glm_composite_xform(glm::inverse(matPuckToPatternXform), apertureToPatternXform_CameraSpace);

	outResults.patternXform_VRSpace= patternXform_VRSpace;
	outResults.apertureXform_VRSpace= apertureXform_VRSpace;
	outResults.cameraPuckToApertureXform= cameraPuckToApertureXform;

	outResults.bIsValid= true;
}