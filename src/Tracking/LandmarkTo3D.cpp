#include "LandmarkTo3D.h"

#include <algorithm>
#include <cmath>

#include "glm/geometric.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"

#include "glm/gtc/quaternion.hpp"

#include "HandPoseModel.h"
#include "Logger.h"

static constexpr float kMinDepthMeters= 0.05f;
static constexpr float kDefaultDtSeconds= 1.f / 60.f;

void LandmarkTo3D::configure(
	const MikanMonoIntrinsics& intrinsics,
	double refLengthMeters)
{
	// MikanMatrix3d is column-major: fx= x0, fy= y1, cx= z0, cy= z1.
	// Landmarks arrive in undistorted image space, so use the undistorted matrix.
	const MikanMatrix3d& cameraMatrix= intrinsics.undistorted_camera_matrix;
	m_fx= (float)cameraMatrix.x0;
	m_fy= (float)cameraMatrix.y1;
	m_cx= (float)cameraMatrix.z0;
	m_cy= (float)cameraMatrix.z1;
	m_refLengthMeters= (float)refLengthMeters;

	m_bSideWasTracked[0]= false;
	m_bSideWasTracked[1]= false;
	m_bPnpPoseValid[0]= false;
	m_bPnpPoseValid[1]= false;

	m_bConfigured= m_fx > 1e-3f && m_fy > 1e-3f && m_refLengthMeters > 1e-4f;
	if (!m_bConfigured)
	{
		MIKAN_MT_LOG_WARNING("LandmarkTo3D::configure")
			<< "Invalid intrinsics/hand scale (fx=" << m_fx
			<< " fy=" << m_fy << " refLength=" << m_refLengthMeters << ")";
	}
}

glm::vec3 LandmarkTo3D::backProject(float u, float v, float z) const
{
	// P= z * K^-1 * [u v 1]^T, OpenCV convention (+X right, +Y down, +Z forward)
	return glm::vec3(
		(u - m_cx) * z / m_fx,
		(v - m_cy) * z / m_fy,
		z);
}

void LandmarkTo3D::process(TrackingFrameResult& ioResult,
						   const std::array<HandDepthMeasurement, 2>* depthMeasurements)
{
	if (!m_bConfigured)
		return;

	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		TrackedHand& hand= ioResult.hands[sideIndex];

		if (hand.tracked)
		{
			// fresh acquisition: drop the stale PnP warm start
			if (!m_bSideWasTracked[sideIndex])
				m_bPnpPoseValid[sideIndex]= false;

			// Hardware depth (RealSense) beats any monocular estimate when the
			// palm is sufficiently resolved; otherwise the PnP/legacy path runs
			const bool bDepthResolved=
				depthMeasurements != nullptr &&
				processHandDepth(hand, (*depthMeasurements)[sideIndex]);
			if (!bDepthResolved)
				processHand(hand);

			if (hand.hasCameraSpace)
				fillHandPose(hand, ioResult.poses[sideIndex]);
		}
		m_bSideWasTracked[sideIndex]= hand.tracked && hand.hasCameraSpace;
	}
}

bool LandmarkTo3D::processHandDepth(TrackedHand& hand, const HandDepthMeasurement& measurement)
{
	// Depth measures the SKIN SURFACE; the joint center sits roughly this far
	// deeper along the view ray (back-of-hand to knuckle center, palm-down)
	constexpr float kSurfaceToJointOffsetM= 0.009f;
	// The 6 quasi-rigid palm landmarks anchor the palm transform
	constexpr int kPalmLandmarks[6]= {
		(int)eHandLandmark::WRIST,      (int)eHandLandmark::THUMB_CMC,
		(int)eHandLandmark::INDEX_MCP,  (int)eHandLandmark::MIDDLE_MCP,
		(int)eHandLandmark::RING_MCP,   (int)eHandLandmark::PINKY_MCP};

	int resolvedPalmCount= 0;
	float palmDepths[6];
	for (const int landmark : kPalmLandmarks)
	{
		if (measurement.bValid[landmark])
			palmDepths[resolvedPalmCount++]= measurement.cameraPoints[landmark].z;
	}
	// The palm transform needs a well-conditioned rigid anchor; with fewer
	// than 4 of 6 palm points the monocular estimate is more trustworthy
	if (resolvedPalmCount < 4)
		return false;

	std::nth_element(palmDepths, palmDepths + resolvedPalmCount / 2, palmDepths + resolvedPalmCount);
	const float palmMedianDepth= palmDepths[resolvedPalmCount / 2] + kSurfaceToJointOffsetM;

	// Resolved landmarks: measured metric point, pushed from skin surface to
	// joint center along the view ray
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		if (measurement.bValid[i])
		{
			const glm::vec3& surfacePoint= measurement.cameraPoints[i];
			const float range= glm::length(surfacePoint);
			hand.cameraPoints[i]= range > 1e-4f
				? surfacePoint * ((range + kSurfaceToJointOffsetM) / range)
				: surfacePoint;
		}
	}

	// Unresolved landmarks (depth holes - typically fingertips): back-project
	// the 2D landmark at the parent joint's depth (finger segments are short,
	// so parent depth bounds the error by ~a phalanx length); unresolved palm
	// points use the palm median depth
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		float chainDepth= palmMedianDepth;
		for (int joint= 0; joint < 4; ++joint)
		{
			const int landmark= FINGER_JOINTS[finger][joint];
			if (measurement.bValid[landmark])
				chainDepth= hand.cameraPoints[landmark].z;
			else
				hand.cameraPoints[landmark]=
					backProject(hand.imagePoints[landmark].x, hand.imagePoints[landmark].y, chainDepth);
		}
	}
	if (!measurement.bValid[(int)eHandLandmark::WRIST])
		hand.cameraPoints[(int)eHandLandmark::WRIST]= backProject(
			hand.imagePoints[(int)eHandLandmark::WRIST].x, hand.imagePoints[(int)eHandLandmark::WRIST].y,
			palmMedianDepth);

	hand.hasCameraSpace= true;
	return true;
}

void LandmarkTo3D::fillHandPose(const TrackedHand& hand, HandPose& outPose)
{
	outPose.tracked= true;
	outPose.side= hand.side;
	outPose.presence= hand.presence;

	// Palm transform from the (rigid, PnP-consistent) camera-space landmarks
	const glm::mat4 palmFrame=
		HandPoseModel::computePalmFrame(hand.cameraPoints, hand.side, &m_cameraPalmarMemory[(int)hand.side]);
	outPose.palmPositionCamera= glm::vec3(palmFrame[3]);
	outPose.palmOrientationCamera= glm::quat_cast(glm::mat3(palmFrame));
	outPose.hasCameraPose= true;

	// Skeleton: the user's measured bones when they have been calibrated,
	// which is also what clients rebuild the hand from. Without a calibration
	// it falls back to the model's LOCAL articulation (scale/depth invariant -
	// all depth noise stays in the palm transform above), rescaled to metric
	// by the calibrated hand scale. Either way the flat-hand rest pose is
	// filled in by computeSkeleton/makeDefaultNeutralDirections.
	if (m_bHasCalibratedSkeleton[(int)hand.side])
	{
		outPose.skeleton= m_calibratedSkeleton[(int)hand.side];
	}
	else
	{
		std::array<glm::vec3, HAND_LANDMARK_COUNT> metricModel;
		const glm::vec3 modelWrist= hand.modelPoints[(int)eHandLandmark::WRIST];
		const float modelBone=
			glm::length(hand.modelPoints[(int)eHandLandmark::MIDDLE_MCP] - modelWrist);
		const float modelScale= modelBone > 1e-4f ? m_refLengthMeters / modelBone : 1.f;
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			metricModel[i]= hand.modelPoints[i] * modelScale;
		HandPoseModel::computeSkeleton(metricModel, hand.side, outPose.skeleton,
									   &m_modelPalmarMemory[(int)hand.side]);
	}

	HandPoseModel::computeFingerAngles(hand.modelPoints, hand.side, outPose.skeleton.neutralDirInPalm,
									   outPose.fingers, &m_modelPalmarMemory[(int)hand.side]);

	// Fidelity of the parameterization itself, so it must be measured on the
	// RAW angles - before the rest offset reinterprets what zero means
	outPose.fkReprojectionPx= computeFkReprojectionError(hand, outPose);

	// Rest calibration: report deviation from this camera's captured rest
	// pose. Removing each camera's own view-dependent bias is also what makes
	// the two cameras' angles comparable for fusion.
	if (m_bHasRestAngles[(int)hand.side])
	{
		const std::array<FingerAngles, FINGER_COUNT>& rest= m_restAngles[(int)hand.side];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			outPose.fingers[finger].lateral-= rest[finger].lateral;
			outPose.fingers[finger].proximal-= rest[finger].proximal;
			outPose.fingers[finger].intermediate-= rest[finger].intermediate;
			outPose.fingers[finger].distal-= rest[finger].distal;
		}
	}
}

float LandmarkTo3D::computeFkReprojectionError(const TrackedHand& hand, const HandPose& pose) const
{
	if (!m_bConfigured || !pose.hasCameraPose)
		return 0.f;

	// Rebuild the hand exactly as a client would - from the palm transform,
	// skeleton and angles alone - and project it back into the image. Any
	// error the parameterization introduces (wrong neutral reference, hinge
	// convention, dropped degree of freedom) shows up here as pixels.
	glm::mat4 palmTransform= glm::mat4_cast(pose.palmOrientationCamera);
	palmTransform[3]= glm::vec4(pose.palmPositionCamera, 1.f);

	std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
	HandPoseModel::buildFingerJoints(palmTransform, pose.skeleton, pose.fingers, joints);

	float errorSum= 0.f;
	int count= 0;
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
	{
		for (int joint= 0; joint < 4; ++joint)
		{
			const glm::vec3& p= joints[finger][joint];
			if (p.z < 1e-3f)
				continue;

			const glm::vec2 projected(m_fx * p.x / p.z + m_cx, m_fy * p.y / p.z + m_cy);
			const glm::vec2 observed= glm::vec2(hand.imagePoints[FINGER_JOINTS[finger][joint]]);
			errorSum+= glm::length(projected - observed);
			count++;
		}
	}

	return count > 0 ? errorSum / (float)count : 0.f;
}

void LandmarkTo3D::processHand(TrackedHand& hand)
{
	// solvePnP over the per-frame metric hand model is the sole monocular
	// depth estimator (a rare solve failure just skips camera space for the
	// frame - fusion's staleness window absorbs it). The old two-point
	// wrist-bone estimator lost every A/B and was removed.
	processHandPnp(hand);
}

bool LandmarkTo3D::buildCalibratedObjectPoints(const TrackedHand& hand,
											   std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints)
{
	const int sideIndex= (int)hand.side;
	if (!m_bHasCalibratedSkeleton[sideIndex])
		return false;

	const HandSkeleton& skeleton= m_calibratedSkeleton[sideIndex];

	// The model contributes articulation only. Angles are read against the
	// calibrated neutral directions so the forward kinematics below inverts
	// them exactly.
	std::array<FingerAngles, FINGER_COUNT> angles{};
	HandPoseModel::computeFingerAngles(hand.modelPoints, hand.side, skeleton.neutralDirInPalm, angles,
									   &m_modelPalmarMemory[sideIndex]);

	std::array<std::array<glm::vec3, 4>, FINGER_COUNT> joints;
	HandPoseModel::buildFingerJoints(glm::mat4(1.f), skeleton, angles, joints);

	// Palm +X is wrist -> middle MCP about the palm center, so the wrist sits
	// opposite that base (the same reconstruction makeDefaultNeutralDirections
	// uses). FK covers the other 20 landmarks.
	outPoints[(int)eHandLandmark::WRIST]=
		glm::vec3(-skeleton.baseInPalm[(int)eFinger::Middle].x, 0.f, 0.f);
	for (int finger= 0; finger < FINGER_COUNT; ++finger)
		for (int joint= 0; joint < 4; ++joint)
			outPoints[FINGER_JOINTS[finger][joint]]= joints[finger][joint];

	return true;
}

bool LandmarkTo3D::processHandPnp(TrackedHand& hand)
{
	const int wristIndex= (int)eHandLandmark::WRIST;
	const int middleMcpIndex= (int)eHandLandmark::MIDDLE_MCP;

	// The PnP object model is a per-frame articulated hand, WRIST-RELATIVE so
	// the recovered translation lands on the wrist. Preferred: the user's
	// measured skeleton posed by the model's angles. Fallback (no bone
	// calibration yet): the landmark model's own metric hand, rescaled so the
	// wrist->middleMCP bone matches the calibrated hand scale.
	//
	// The fallback keeps its subtract-then-scale form deliberately. Scaling
	// first is algebraically identical and bit-wise is not, and an
	// uncalibrated session has to replay against older recordings exactly.
	std::array<glm::vec3, HAND_LANDMARK_COUNT> objectModel{};
	if (buildCalibratedObjectPoints(hand, objectModel))
	{
		const glm::vec3 wristInPalm= objectModel[wristIndex];
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			objectModel[i]-= wristInPalm;
	}
	else
	{
		const glm::vec3 modelWrist= hand.modelPoints[wristIndex];
		const float modelBoneLength= glm::length(hand.modelPoints[middleMcpIndex] - modelWrist);
		if (modelBoneLength < 1e-4f)
			return false;
		const float modelScale= m_refLengthMeters / modelBoneLength;
		for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
			objectModel[i]= (hand.modelPoints[i] - modelWrist) * modelScale;
	}

	// All 21 correspondences: landmark jitter averages across the hand
	std::vector<cv::Point3f> objectPoints;
	std::vector<cv::Point2f> imagePoints;
	objectPoints.reserve(HAND_LANDMARK_COUNT);
	imagePoints.reserve(HAND_LANDMARK_COUNT);
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		objectPoints.emplace_back(objectModel[i].x, objectModel[i].y, objectModel[i].z);
		imagePoints.emplace_back(hand.imagePoints[i].x, hand.imagePoints[i].y);
	}

	// Undistorted pinhole camera; image points arrive pre-undistorted
	const cv::Matx33d cameraMatrix(m_fx, 0, m_cx, 0, m_fy, m_cy, 0, 0, 1);

	// Warm-start from the previous frame's pose: faster convergence and no
	// planar-ambiguity flips when the hand is held flat
	const int sideIndex= (int)hand.side;
	const bool bUseGuess= m_bPnpPoseValid[sideIndex];
	cv::Vec3d rvec(m_pnpRvec[sideIndex][0], m_pnpRvec[sideIndex][1], m_pnpRvec[sideIndex][2]);
	cv::Vec3d tvec(m_pnpTvec[sideIndex][0], m_pnpTvec[sideIndex][1], m_pnpTvec[sideIndex][2]);

	try
	{
		if (!cv::solvePnP(objectPoints, imagePoints, cameraMatrix, cv::noArray(), rvec, tvec, bUseGuess,
						  cv::SOLVEPNP_ITERATIVE))
		{
			return false;
		}
	}
	catch (const cv::Exception&)
	{
		return false;
	}

	// Sanity: the hand must be in front of the camera at a plausible distance
	const double depth= tvec(2);
	if (!std::isfinite(depth) || depth < kMinDepthMeters || depth > 5.0 ||
		!std::isfinite(tvec(0)) || !std::isfinite(tvec(1)))
	{
		m_bPnpPoseValid[sideIndex]= false;
		return false;
	}

	for (int axis= 0; axis < 3; ++axis)
	{
		m_pnpRvec[sideIndex][axis]= rvec(axis);
		m_pnpTvec[sideIndex][axis]= tvec(axis);
	}
	m_bPnpPoseValid[sideIndex]= true;

	// cameraPoints= R * obj + t for all 21 landmarks (articulation comes from
	// the per-frame object model)
	cv::Matx33d rotation;
	cv::Rodrigues(rvec, rotation);
	for (int i= 0; i < HAND_LANDMARK_COUNT; ++i)
	{
		const cv::Vec3d rotated=
			rotation * cv::Vec3d(objectModel[i].x, objectModel[i].y, objectModel[i].z) + tvec;
		hand.cameraPoints[i]= glm::vec3((float)rotated(0), (float)rotated(1), (float)rotated(2));
	}

	hand.hasCameraSpace= true;
	return true;
}

