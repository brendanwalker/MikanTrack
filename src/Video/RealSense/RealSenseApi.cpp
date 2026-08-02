#include "RealSenseApi.h"

#include <windows.h>

#include <string>

#include "Logger.h"

#include "librealsense2/rs.h" // RS2_API_VERSION

static HMODULE loadRealSenseDll()
{
	// 1) next to the exe (a copy dropped by the user/build)
	HMODULE module= ::LoadLibraryA("realsense2.dll");
	if (module != nullptr)
		return module;

	// 2) standard SDK install locations
	const char* candidates[]= {
		"C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64\\realsense2.dll",
		"C:\\Program Files\\Intel RealSense SDK 2.0\\bin\\x64\\realsense2.dll",
	};
	for (const char* path : candidates)
	{
		module= ::LoadLibraryA(path);
		if (module != nullptr)
			return module;
	}

	// 3) user-documents install (the SDK installer offers this as a target);
	// %USERPROFILE% keeps it user-agnostic
	char userProfile[MAX_PATH]= {};
	DWORD length= ::GetEnvironmentVariableA("USERPROFILE", userProfile, sizeof(userProfile));
	if (length > 0 && length < sizeof(userProfile))
	{
		const std::string documentsPath=
			std::string(userProfile) + "\\Documents\\RealSense SDK 2.0\\bin\\x64\\realsense2.dll";
		module= ::LoadLibraryA(documentsPath.c_str());
		if (module != nullptr)
			return module;
	}

	return nullptr;
}

bool RealSenseApi::loadEntryPoints()
{
	HMODULE module= loadRealSenseDll();
	if (module == nullptr)
	{
		MIKAN_LOG_INFO("RealSenseApi") << "realsense2.dll not found - RealSense backend disabled";
		return false;
	}

	bool bAllResolved= true;
	auto resolve= [&](auto& fn, const char* name) {
		fn= reinterpret_cast<std::remove_reference_t<decltype(fn)>>(::GetProcAddress(module, name));
		if (fn == nullptr)
		{
			MIKAN_LOG_ERROR("RealSenseApi") << "Missing entry point: " << name;
			bAllResolved= false;
		}
	};

	resolve(create_context, "rs2_create_context");
	resolve(delete_context, "rs2_delete_context");
	resolve(query_devices, "rs2_query_devices");
	resolve(delete_device_list, "rs2_delete_device_list");
	resolve(get_device_count, "rs2_get_device_count");
	resolve(create_device, "rs2_create_device");
	resolve(delete_device, "rs2_delete_device");
	resolve(get_device_info, "rs2_get_device_info");
	resolve(create_pipeline, "rs2_create_pipeline");
	resolve(delete_pipeline, "rs2_delete_pipeline");
	resolve(create_config, "rs2_create_config");
	resolve(delete_config, "rs2_delete_config");
	resolve(config_enable_device, "rs2_config_enable_device");
	resolve(config_enable_stream, "rs2_config_enable_stream");
	resolve(pipeline_start_with_config, "rs2_pipeline_start_with_config");
	resolve(pipeline_stop, "rs2_pipeline_stop");
	resolve(delete_pipeline_profile, "rs2_delete_pipeline_profile");
	resolve(pipeline_wait_for_frames, "rs2_pipeline_wait_for_frames");
	resolve(embedded_frames_count, "rs2_embedded_frames_count");
	resolve(extract_frame, "rs2_extract_frame");
	resolve(get_frame_data, "rs2_get_frame_data");
	resolve(get_frame_timestamp, "rs2_get_frame_timestamp");
	resolve(get_frame_stream_profile, "rs2_get_frame_stream_profile");
	resolve(get_frame_width, "rs2_get_frame_width");
	resolve(get_frame_height, "rs2_get_frame_height");
	resolve(depth_frame_get_units, "rs2_depth_frame_get_units");
	resolve(release_frame, "rs2_release_frame");
	resolve(get_stream_profile_data, "rs2_get_stream_profile_data");
	resolve(get_video_stream_intrinsics, "rs2_get_video_stream_intrinsics");
	resolve(get_extrinsics, "rs2_get_extrinsics");
	resolve(get_error_message, "rs2_get_error_message");
	resolve(free_error, "rs2_free_error");

	if (!bAllResolved)
		return false;

	// Sanity: create a context at the header's API version (mismatched
	// dll/header majors fail here, loudly, instead of corrupting later)
	rs2_error* error= nullptr;
	rs2_context* context= create_context(RS2_API_VERSION, &error);
	if (context == nullptr)
	{
		if (error != nullptr)
		{
			MIKAN_LOG_ERROR("RealSenseApi")
				<< "rs2_create_context failed (dll/header version mismatch?): " << get_error_message(error);
			free_error(error);
		}
		return false;
	}
	delete_context(context);

	MIKAN_LOG_INFO("RealSenseApi") << "librealsense2 loaded (API version " << RS2_API_VERSION_STR << ")";
	return true;
}

const RealSenseApi* RealSenseApi::get()
{
	static RealSenseApi s_api;
	static bool s_bInitialized= false;
	if (!s_bInitialized)
	{
		s_bInitialized= true;
		s_api.m_bLoaded= s_api.loadEntryPoints();
	}
	return &s_api;
}

bool RealSenseApi::checkError(rs2_error* error, const char* context) const
{
	if (error == nullptr)
		return false;

	MIKAN_LOG_ERROR("RealSenseApi") << context << ": " << get_error_message(error);
	free_error(const_cast<rs2_error*>(error));
	return true;
}
