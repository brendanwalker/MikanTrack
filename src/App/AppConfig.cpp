#include "AppConfig.h"

#include <filesystem>
#include <fstream>

#include <windows.h>
#include <shlobj.h>

#include "nlohmann/json.hpp"

#include "Logger.h"

using json= nlohmann::json;

static constexpr int k_configVersion= 1;
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

		const json& v= j.value("video", json::object());
		video.deviceName= v.value("deviceName", "");
		video.devicePath= v.value("devicePath", "");
		video.modeName= v.value("modeName", "");

		const json& in= j.value("intrinsics", json::object());
		intrinsics.present= in.value("present", false);
		intrinsics.reprojectionError= in.value("reprojectionError", 0.0);
		MikanMonoIntrinsics& mono= intrinsics.intrinsics;
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

		const json& ex= j.value("extrinsics", json::object());
		extrinsics.present= ex.value("present", false);
		extrinsics.markerId= ex.value("markerId", 0);
		extrinsics.markerLengthMM= ex.value("markerLengthMm", 100.0);
		if (ex.contains("markerFromCamera"))
			dmat4FromJson(ex["markerFromCamera"], extrinsics.markerFromCamera);

		const json& hs= j.value("handScale", json::object());
		handScale.present= hs.value("present", false);
		handScale.refLengthMeters= hs.value("refLengthMeters", 0.08);

		const json& cb= j.value("charucoBoard", json::object());
		charucoBoard.cols= cb.value("cols", 11);
		charucoBoard.rows= cb.value("rows", 8);
		charucoBoard.squareLengthMM= cb.value("squareMm", 16.0);
		charucoBoard.markerLengthMM= cb.value("markerMm", 12.0);

		const json& tr= j.value("tracking", json::object());
		tracking.flipHandedness= tr.value("flipHandedness", true);
		tracking.usePoseModel= tr.value("usePoseModel", true);
		tracking.detectorIntervalFrames= tr.value("detectorIntervalFrames", 30);
		tracking.poseFrameDivider= tr.value("poseFrameDivider", 2);
		tracking.smoothingMinCutoff= tr.value("smoothingMinCutoff", 1.0f);
		tracking.smoothingBeta= tr.value("smoothingBeta", 0.05f);
		tracking.smoothingEnabled= tr.value("smoothingEnabled", true);
		tracking.onnxEp= tr.value("onnxEp", "directml");

		const json& os= j.value("osc", json::object());
		osc.enabled= os.value("enabled", true);
		osc.targetIp= os.value("ip", "127.0.0.1");
		osc.targetPort= os.value("port", 8000);
		osc.maxRateHz= os.value("maxRateHz", 60);
	}
	catch (const std::exception& e)
	{
		MIKAN_LOG_ERROR("AppConfig::load") << "Failed to parse " << path << ": " << e.what();
		return false;
	}

	MIKAN_LOG_INFO("AppConfig::load") << "Loaded config from " << path;
	return true;
}

bool AppConfig::save() const
{
	json j;
	j["configVersion"]= k_configVersion;

	j["video"]= {
		{"deviceName", video.deviceName},
		{"devicePath", video.devicePath},
		{"modeName", video.modeName},
	};

	const MikanMonoIntrinsics& mono= intrinsics.intrinsics;
	j["intrinsics"]= {
		{"present", intrinsics.present},
		{"reprojectionError", intrinsics.reprojectionError},
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
	};

	j["extrinsics"]= {
		{"present", extrinsics.present},
		{"markerId", extrinsics.markerId},
		{"markerLengthMm", extrinsics.markerLengthMM},
		{"markerFromCamera", dmat4ToJson(extrinsics.markerFromCamera)},
	};

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
		{"usePoseModel", tracking.usePoseModel},
		{"detectorIntervalFrames", tracking.detectorIntervalFrames},
		{"poseFrameDivider", tracking.poseFrameDivider},
		{"smoothingMinCutoff", tracking.smoothingMinCutoff},
		{"smoothingBeta", tracking.smoothingBeta},
		{"smoothingEnabled", tracking.smoothingEnabled},
		{"onnxEp", tracking.onnxEp},
	};

	j["osc"]= {
		{"enabled", osc.enabled},
		{"ip", osc.targetIp},
		{"port", osc.targetPort},
		{"maxRateHz", osc.maxRateHz},
	};

	const std::string path= getConfigFilePath();
	std::ofstream file(path);
	if (!file.is_open())
	{
		MIKAN_LOG_ERROR("AppConfig::save") << "Failed to open " << path << " for writing";
		return false;
	}

	file << j.dump(2);
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
