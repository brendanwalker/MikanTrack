#pragma once

// Shared include block for the commands in this folder, so each test file is
// its own function plus one registration line. Deliberately generous rather
// than minimal: a test tree's compile time is not worth the per-file
// bookkeeping, and every command here already reaches across most of the app.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "glm/gtc/constants.hpp"
#include "glm/gtc/quaternion.hpp"

#include "nlohmann/json.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect/charuco_detector.hpp"

#include "App.h"
#include "AppConfig.h"
#include "CalibrationPatternFinder_Aruco.h"
#include "CalibrationPatternFinder_Charuco.h"
#include "DepthFrameView.h"
#include "DiagnosticDump.h"
#include "ExtrinsicsValidation.h"
#include "ExtrinsicsWizard.h"
#include "PatternPoseSampler.h"
#include "HandBoneCalibrator.h"
#include "BodyPoseSolver.h"
#include "BodyPoseTracker.h"
#include "HandFusion.h"
#include "HandPoseModel.h"
#include "HandRoiQuality.h"
#include "HandTrackingPipeline.h"
#include "ImuOrientationFilter.h"
#include "ImuService.h"
#include "JoyconDevice.h"
#include "JoyconDeviceManager.h"
#include "LandmarkTo3D.h"
#include "Logger.h"
#include "MonoLensDistortionCalibrator.h"
#include "OscWriterTest.h"
#include "PatternExport.h"
#include "SpaceTransforms.h"
#include "TrackingJson.h"
#include "TrackingRecorder.h"
#include "TrackingRecording.h"
#include "TrackingReplay.h"
#include "VmcRetarget.h"

#include "TestRegistry.h"
