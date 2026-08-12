#include "AppConfig.h"

#include <ctime>
#include <filesystem>
#include <fstream>

#include <windows.h>
#include <shlobj.h>

#include "nlohmann/json.hpp"

#include "HandPoseModel.h"
#include "Logger.h"

using json= nlohmann::json;

static constexpr int k_configVersion= 2;
static constexpr float k_autoSaveCooldownSeconds= 3.f;

// -- json helpers ----
static json matrix3dToJson(const MikanMatrix3d& m)
{
	return json::array({m.x0, m.x1, m.x2, m.y0, m.y1, m.y2, m.z0, m.z1, m.z2});
}

static void matrix3dFromJson(const json& j, MikanMatrix3d& m)
{
	if (!j.is_array() || j.size() != 9)
		return;
	m.x0= j[0]; m.x1= j[1]; m.x2= j[2];
	m.y0= j[3]; m.y1= j[4]; m.y2= j[5];
	m.z0= j[6]; m.z1= j[7]; m.z2= j[8];
}

static json dmat4ToJson(const glm::dmat4& m)
{
	json arr= json::array();
	for (int c= 0; c < 4; ++c)
		for (int r= 0; r < 4; ++r)
			arr.push_back(m[c][r]);
	return arr;
}

static void dmat4FromJson(const json& j, glm::dmat4& m)
{
	if (!j.is_array() || j.size() != 16)
		return;
	int i= 0;
	for (int c= 0; c < 4; ++c)
		for (int r= 0; r < 4; ++r)
			m[c][r]= j[i++];
}

static json distortionToJson(const MikanDistortionCoefficients& d)
{
	return json::array({d.k1, d.k2, d.k3, d.k4, d.k5, d.k6, d.p1, d.p2});
}

static void distortionFromJson(const json& j, MikanDistortionCoefficients& d)
{
	if (!j.is_array() || j.size() != 8)
		return;
	d.k1= j[0]; d.k2= j[1]; d.k3= j[2]; d.k4= j[3];
	d.k5= j[4]; d.k6= j[5]; d.p1= j[6]; d.p2= j[7];
}

static json restAnglesToJson(const RestAnglesConfig& restAngles)
{
	json out= json::object();
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!restAngles.present[sideIndex])
			continue;
		json values= json::array();
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			const FingerAngles& angles= restAngles.angles[sideIndex][finger];
			values.push_back(angles.lateral);
			values.push_back(angles.proximal);
			values.push_back(angles.intermediate);
			values.push_back(angles.distal);
		}
		out[sideIndex == 0 ? "left" : "right"]= values;
	}
	return out;
}

// Per side: 5 finger bases (xyz) then 5 phalanx triples, so 30 floats.
// neutralDirInPalm is rebuilt from the bases rather than stored.
static json handSkeletonToJson(const HandSkeletonConfig& handSkeleton)
{
	json out= json::object();
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		if (!handSkeleton.present[sideIndex])
			continue;
		const HandSkeleton& skeleton= handSkeleton.skeleton[sideIndex];
		json values= json::array();
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			const glm::vec3& base= skeleton.baseInPalm[finger];
			values.push_back(base.x);
			values.push_back(base.y);
			values.push_back(base.z);
		}
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				values.push_back(skeleton.phalanxLengths[finger][phalanx]);
		out[sideIndex == 0 ? "left" : "right"]= values;
	}
	return out;
}

static void handSkeletonFromJson(const json& hs, HandSkeletonConfig& outHandSkeleton)
{
	constexpr size_t kValueCount= FINGER_COUNT * 3 + FINGER_COUNT * 3;
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const char* key= sideIndex == 0 ? "left" : "right";
		outHandSkeleton.present[sideIndex]= false;
		if (!hs.contains(key) || !hs[key].is_array() || hs[key].size() != kValueCount)
			continue;

		HandSkeleton& skeleton= outHandSkeleton.skeleton[sideIndex];
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			skeleton.baseInPalm[finger]= glm::vec3(hs[key][finger * 3 + 0], hs[key][finger * 3 + 1],
												   hs[key][finger * 3 + 2]);
		}
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
			for (int phalanx= 0; phalanx < 3; ++phalanx)
				skeleton.phalanxLengths[finger][phalanx]= hs[key][FINGER_COUNT * 3 + finger * 3 + phalanx];

		skeleton.neutralDirInPalm= HandPoseModel::makeDefaultNeutralDirections(skeleton);
		outHandSkeleton.present[sideIndex]= true;
	}
}

static void restAnglesFromJson(const json& ra, RestAnglesConfig& outRestAngles)
{
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const char* key= sideIndex == 0 ? "left" : "right";
		outRestAngles.present[sideIndex]= false;
		if (!ra.contains(key) || !ra[key].is_array() || ra[key].size() != FINGER_COUNT * 4)
			continue;
		for (int finger= 0; finger < FINGER_COUNT; ++finger)
		{
			FingerAngles& angles= outRestAngles.angles[sideIndex][finger];
			angles.lateral= ra[key][finger * 4 + 0];
			angles.proximal= ra[key][finger * 4 + 1];
			angles.intermediate= ra[key][finger * 4 + 2];
			angles.distal= ra[key][finger * 4 + 3];
		}
		outRestAngles.present[sideIndex]= true;
	}
}

// -- camera profile (de)serialization ----
static void cameraProfileFromJson(const json& j, CameraProfile& profile)
{
	const json& v= j.value("video", json::object());
	profile.video.deviceName= v.value("deviceName", "");
	profile.video.devicePath= v.value("devicePath", "");
	profile.video.modeName= v.value("modeName", "");

	const json& in= j.value("intrinsics", json::object());
	profile.intrinsics.present= in.value("present", false);
	profile.intrinsics.reprojectionError= in.value("reprojectionError", 0.0);
	MikanMonoIntrinsics& mono= profile.intrinsics.intrinsics;
	mono.pixel_width= in.value("width", 0.0);
	mono.pixel_height= in.value("height", 0.0);
	mono.aspect_ratio= in.value("aspectRatio", 0.0);
	mono.hfov= in.value("hfov", 0.0);
	mono.vfov= in.value("vfov", 0.0);
	mono.znear= in.value("znear", 0.1);
	mono.zfar= in.value("zfar", 20.0);
	if (in.contains("distortedCameraMatrix"))
		matrix3dFromJson(in["distortedCameraMatrix"], mono.distorted_camera_matrix);
	if (in.contains("undistortedCameraMatrix"))
		matrix3dFromJson(in["undistortedCameraMatrix"], mono.undistorted_camera_matrix);
	if (in.contains("distortion"))
		distortionFromJson(in["distortion"], mono.distortion_coefficients);

	restAnglesFromJson(j.value("restAngles", json::object()), profile.restAngles);

	const json& ex= j.value("extrinsics", json::object());
	profile.extrinsics.present= ex.value("present", false);
	profile.extrinsics.patternReprojectionErrorPx= ex.value("patternReprojectionErrorPx", 0.0);
	profile.extrinsics.patternCornerCount= ex.value("patternCornerCount", 0);
	profile.extrinsics.markerId= ex.value("markerId", 0);
	profile.extrinsics.markerLengthMM= ex.value("markerLengthMm", 100.0);
	if (ex.contains("markerFromCamera"))
		dmat4FromJson(ex["markerFromCamera"], profile.extrinsics.markerFromCamera);
}

static json cameraProfileToJson(const CameraProfile& profile)
{
	const MikanMonoIntrinsics& mono= profile.intrinsics.intrinsics;
	return {
		{"video",
		 {
			 {"deviceName", profile.video.deviceName},
			 {"devicePath", profile.video.devicePath},
			 {"modeName", profile.video.modeName},
		 }},
		{"intrinsics",
		 {
			 {"present", profile.intrinsics.present},
			 {"reprojectionError", profile.intrinsics.reprojectionError},
			 {"width", mono.pixel_width},
			 {"height", mono.pixel_height},
			 {"aspectRatio", mono.aspect_ratio},
			 {"hfov", mono.hfov},
			 {"vfov", mono.vfov},
			 {"znear", mono.znear},
			 {"zfar", mono.zfar},
			 {"distortedCameraMatrix", matrix3dToJson(mono.distorted_camera_matrix)},
			 {"undistortedCameraMatrix", matrix3dToJson(mono.undistorted_camera_matrix)},
			 {"distortion", distortionToJson(mono.distortion_coefficients)},
		 }},
		{"restAngles", restAnglesToJson(profile.restAngles)},
		{"extrinsics",
		 {
			 {"present", profile.extrinsics.present},
			 {"patternReprojectionErrorPx", profile.extrinsics.patternReprojectionErrorPx},
			 {"patternCornerCount", profile.extrinsics.patternCornerCount},
			 {"markerId", profile.extrinsics.markerId},
			 {"markerLengthMm", profile.extrinsics.markerLengthMM},
			 {"markerFromCamera", dmat4ToJson(profile.extrinsics.markerFromCamera)},
		 }},
	};
}

// -- AppConfig ----
std::string AppConfig::getConfigFilePath()
{
	PWSTR appDataPath= nullptr;
	std::filesystem::path configDir;
	if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &appDataPath)))
	{
		configDir= std::filesystem::path(appDataPath) / "MikanMediaPipe";
		CoTaskMemFree(appDataPath);
	}
	else
	{
		configDir= std::filesystem::current_path();
	}

	std::error_code ec;
	std::filesystem::create_directories(configDir, ec);

	return (configDir / "config.json").string();
}

std::string AppConfig::makeDumpDirectoryPath()
{
	const std::filesystem::path configDir= std::filesystem::path(getConfigFilePath()).parent_path();

	const std::time_t now= std::time(nullptr);
	std::tm local{};
	localtime_s(&local, &now);
	char stamp[32];
	std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &local);

	return (configDir / "dumps" / stamp).string();
}

std::string AppConfig::getRecordingsDirectoryPath()
{
	const std::filesystem::path configDir= std::filesystem::path(getConfigFilePath()).parent_path();
	return (configDir / "recordings").string();
}

std::string AppConfig::makeRecordingFilePath()
{
	const std::filesystem::path recordingsDir(getRecordingsDirectoryPath());
	std::error_code ec;
	std::filesystem::create_directories(recordingsDir, ec);

	const std::time_t now= std::time(nullptr);
	std::tm local{};
	localtime_s(&local, &now);
	char stamp[32];
	std::strftime(stamp, sizeof(stamp), "%Y-%m-%d_%H-%M-%S", &local);

	return (recordingsDir / (std::string(stamp) + ".jsonl")).string();
}

// Applies a parsed config JSON onto the config object (shared by load() and
// loadFromJsonString()). Throws on malformed structure; the callers own the
// error handling and camera-list guarantee.
static void applyConfigJson(AppConfig& config, const json& j);

bool AppConfig::load()
{
	const std::string path= getConfigFilePath();
	std::ifstream file(path);
	if (!file.is_open())
	{
		MIKAN_LOG_INFO("AppConfig::load") << "No config found at " << path << ", using defaults";
		return false;
	}

	try
	{
		json j;
		file >> j;
		applyConfigJson(*this, j);
	}
	catch (const std::exception& e)
	{
		MIKAN_LOG_ERROR("AppConfig::load") << "Failed to parse " << path << ": " << e.what();
		if (cameras.empty())
			cameras.emplace_back();
		return false;
	}

	MIKAN_LOG_INFO("AppConfig::load") << "Loaded config from " << path;
	return true;
}

bool AppConfig::loadFromJsonString(const std::string& jsonText)
{
	try
	{
		applyConfigJson(*this, json::parse(jsonText));
	}
	catch (const std::exception& e)
	{
		MIKAN_LOG_ERROR("AppConfig::loadFromJsonString") << "Failed to parse: " << e.what();
		if (cameras.empty())
			cameras.emplace_back();
		return false;
	}
	return true;
}

static void applyConfigJson(AppConfig& config, const json& j)
{
	config.cameras.clear();
	if (j.contains("cameras") && j["cameras"].is_array())
	{
		for (const json& cameraJson : j["cameras"])
		{
			CameraProfile profile;
			cameraProfileFromJson(cameraJson, profile);
			config.cameras.push_back(profile);
		}
	}
	else
	{
		// v1 migration: the old schema kept a single camera's video/
		// intrinsics/extrinsics at the top level - fold into cameras[0]
		CameraProfile profile;
		cameraProfileFromJson(j, profile);
		config.cameras.push_back(profile);
		MIKAN_LOG_INFO("AppConfig::load") << "Migrated v1 single-camera config to camera list";
	}
	if (config.cameras.empty())
		config.cameras.emplace_back();

	const json& fu= j.value("fusion", json::object());
	config.fusion.stalenessWindowMs= fu.value("stalenessWindowMs", 66.0);
	config.fusion.wristMatchMaxDistM= fu.value("wristMatchMaxDistM", 0.25f);
	config.fusion.minCameraConfidence= fu.value("minCameraConfidence", 0.f);
	config.fusion.jitterReferenceMm= fu.value("jitterReferenceMm", 15.f);
	config.fusion.triangulationEnabled= fu.value("triangulationEnabled", true);
	config.fusion.triangulationMaxResidualPx= fu.value("triangulationMaxResidualPx", 25.f);
	config.fusion.residualReferencePx= fu.value("residualReferencePx", 8.f);

	restAnglesFromJson(j.value("fusedRestAngles", json::object()), config.fusedRestAngles);
	handSkeletonFromJson(j.value("handSkeleton", json::object()), config.handSkeleton);

	const json& eq= j.value("extrinsicsQuality", json::object());
	config.extrinsicsQuality.present= eq.value("present", false);
	config.extrinsicsQuality.pairCount= eq.value("pairCount", 0);
	config.extrinsicsQuality.worstPairCameraA= eq.value("worstPairCameraA", -1);
	config.extrinsicsQuality.worstPairCameraB= eq.value("worstPairCameraB", -1);
	config.extrinsicsQuality.worstPairSharedCorners= eq.value("worstPairSharedCorners", 0);
	config.extrinsicsQuality.worstPairReprojectionRmsPx= eq.value("worstPairReprojectionRmsPx", 0.0);
	config.extrinsicsQuality.worstPairSpacingErrorMm= eq.value("worstPairSpacingErrorMm", 0.0);
	config.extrinsicsQuality.worstPairSpacingScale= eq.value("worstPairSpacingScale", 1.0);
	config.extrinsicsQuality.worstPairPlanarityRmsMm= eq.value("worstPairPlanarityRmsMm", 0.0);

	const json& im= j.value("imu", json::object());
	config.imu.enabled= im.value("enabled", true);
	config.imu.visionYawSigma= im.value("visionYawSigma", 0.35f);
	config.imu.swapSides= im.value("swapSides", false);
	config.imu.forearmLengthMeters= im.value("forearmLengthMeters", 0.25f);
	for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
	{
		const char* key= sideIndex == 0 ? "mountingLeft" : "mountingRight";
		config.imu.mountingPresent[sideIndex]= false;
		if (!im.contains(key) || !im[key].is_array() || im[key].size() != 4)
			continue;
		config.imu.forearmToSensor[sideIndex]=
			glm::quat((float)im[key][3], (float)im[key][0], (float)im[key][1], (float)im[key][2]);
		config.imu.mountingPresent[sideIndex]= true;
	}

	const json& hs= j.value("handScale", json::object());
	config.handScale.present= hs.value("present", false);
	config.handScale.refLengthMeters= hs.value("refLengthMeters", 0.08);

	const json& cb= j.value("charucoBoard", json::object());
	config.charucoBoard.cols= cb.value("cols", 11);
	config.charucoBoard.rows= cb.value("rows", 8);
	config.charucoBoard.squareLengthMM= cb.value("squareMm", 16.0);
	config.charucoBoard.markerLengthMM= cb.value("markerMm", 12.0);

	const json& tr= j.value("tracking", json::object());
	config.tracking.flipHandedness= tr.value("flipHandedness", true);
	config.tracking.detectorIntervalFrames= tr.value("detectorIntervalFrames", 30);
	config.tracking.palmScoreThresholdRelaxed= tr.value("palmScoreThresholdRelaxed", 0.25f);
	config.tracking.useRealSenseDepth= tr.value("useRealSenseDepth", true);
	config.tracking.palmMinCutoff= tr.value("palmMinCutoff", 3.0f);
	config.tracking.palmBeta= tr.value("palmBeta", 0.1f);
	config.tracking.angleMinCutoff= tr.value("angleMinCutoff", 0.75f);
	config.tracking.angleBeta= tr.value("angleBeta", 0.02f);
	config.tracking.smoothingEnabled= tr.value("smoothingEnabled", true);
	config.tracking.onnxEp= tr.value("onnxEp", "directml");

	const json& os= j.value("osc", json::object());
	config.osc.enabled= os.value("enabled", true);
	config.osc.targetIp= os.value("ip", "127.0.0.1");
	config.osc.targetPort= os.value("port", 8000);
	config.osc.maxRateHz= os.value("maxRateHz", 60);
	config.osc.minConfidence= os.value("minConfidence", 0.f);
	config.osc.holdOnDropoutMs= os.value("holdOnDropoutMs", 250.f);
	config.osc.logPalmFrames= os.value("logPalmFrames", false);
}

std::string AppConfig::toJsonString() const
{
	json j;
	j["configVersion"]= k_configVersion;

	json camerasJson= json::array();
	for (const CameraProfile& profile : cameras)
		camerasJson.push_back(cameraProfileToJson(profile));
	j["cameras"]= camerasJson;

	j["fusion"]= {
		{"stalenessWindowMs", fusion.stalenessWindowMs},
		{"wristMatchMaxDistM", fusion.wristMatchMaxDistM},
		{"minCameraConfidence", fusion.minCameraConfidence},
		{"jitterReferenceMm", fusion.jitterReferenceMm},
		{"triangulationEnabled", fusion.triangulationEnabled},
		{"triangulationMaxResidualPx", fusion.triangulationMaxResidualPx},
		{"residualReferencePx", fusion.residualReferencePx},
	};

	j["fusedRestAngles"]= restAnglesToJson(fusedRestAngles);
	j["handSkeleton"]= handSkeletonToJson(handSkeleton);

	j["extrinsicsQuality"]= {
		{"present", extrinsicsQuality.present},
		{"pairCount", extrinsicsQuality.pairCount},
		{"worstPairCameraA", extrinsicsQuality.worstPairCameraA},
		{"worstPairCameraB", extrinsicsQuality.worstPairCameraB},
		{"worstPairSharedCorners", extrinsicsQuality.worstPairSharedCorners},
		{"worstPairReprojectionRmsPx", extrinsicsQuality.worstPairReprojectionRmsPx},
		{"worstPairSpacingErrorMm", extrinsicsQuality.worstPairSpacingErrorMm},
		{"worstPairSpacingScale", extrinsicsQuality.worstPairSpacingScale},
		{"worstPairPlanarityRmsMm", extrinsicsQuality.worstPairPlanarityRmsMm},
	};

	{
		json imuJson= {
			{"enabled", imu.enabled},
			{"visionYawSigma", imu.visionYawSigma},
			{"swapSides", imu.swapSides},
			{"forearmLengthMeters", imu.forearmLengthMeters},
		};
		for (int sideIndex= 0; sideIndex < 2; ++sideIndex)
		{
			if (!imu.mountingPresent[sideIndex])
				continue;
			const glm::quat& q= imu.forearmToSensor[sideIndex];
			imuJson[sideIndex == 0 ? "mountingLeft" : "mountingRight"]= json::array({q.x, q.y, q.z, q.w});
		}
		j["imu"]= imuJson;
	}

	j["handScale"]= {
		{"present", handScale.present},
		{"refLengthMeters", handScale.refLengthMeters},
	};

	j["charucoBoard"]= {
		{"cols", charucoBoard.cols},
		{"rows", charucoBoard.rows},
		{"squareMm", charucoBoard.squareLengthMM},
		{"markerMm", charucoBoard.markerLengthMM},
	};

	j["tracking"]= {
		{"flipHandedness", tracking.flipHandedness},
		{"detectorIntervalFrames", tracking.detectorIntervalFrames},
		{"palmScoreThresholdRelaxed", tracking.palmScoreThresholdRelaxed},
		{"useRealSenseDepth", tracking.useRealSenseDepth},
		{"palmMinCutoff", tracking.palmMinCutoff},
		{"palmBeta", tracking.palmBeta},
		{"angleMinCutoff", tracking.angleMinCutoff},
		{"angleBeta", tracking.angleBeta},
		{"smoothingEnabled", tracking.smoothingEnabled},
		{"onnxEp", tracking.onnxEp},
	};

	j["osc"]= {
		{"enabled", osc.enabled},
		{"ip", osc.targetIp},
		{"port", osc.targetPort},
		{"maxRateHz", osc.maxRateHz},
		{"minConfidence", osc.minConfidence},
		{"holdOnDropoutMs", osc.holdOnDropoutMs},
		{"logPalmFrames", osc.logPalmFrames},
	};

	return j.dump(2);
}

bool AppConfig::save() const
{
	const std::string path= getConfigFilePath();
	std::ofstream file(path);
	if (!file.is_open())
	{
		MIKAN_LOG_ERROR("AppConfig::save") << "Failed to open " << path << " for writing";
		return false;
	}

	file << toJsonString();
	return true;
}

void AppConfig::updateAutoSave(float deltaSeconds)
{
	if (!m_bDirty)
		return;

	m_secondsSinceDirty+= deltaSeconds;
	if (m_secondsSinceDirty >= k_autoSaveCooldownSeconds)
	{
		save();
		m_bDirty= false;
		m_secondsSinceDirty= 0.f;
	}
}
