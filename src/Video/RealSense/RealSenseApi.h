#pragma once

// Dynamic loader for the librealsense2 C API (stable ABI). The app has NO
// link-time dependency on the RealSense SDK: realsense2.dll is located at
// runtime (exe directory first, then the standard SDK install locations) and
// the needed rs2_* entry points are resolved with GetProcAddress. When the
// DLL is absent, isAvailable() is false and the RealSense backend simply
// contributes no devices.
//
// Headers come from deps/librealsense/include (vendored from the official
// IntelRealSense/librealsense repo, tag v2.58.3 - matching the SDK the DLL
// ships from). Only POD structs and enums are used from them.

#include "librealsense2/h/rs_types.h"
#include "librealsense2/h/rs_sensor.h"

struct rs2_context;
struct rs2_device_list;
struct rs2_device;
struct rs2_pipeline;
struct rs2_pipeline_profile;
struct rs2_config;
struct rs2_frame;
struct rs2_stream_profile;

// Resolved rs2_* entry points (subset the backend needs)
struct RealSenseApi
{
	// context / enumeration
	rs2_context* (*create_context)(int api_version, rs2_error** error)= nullptr;
	void (*delete_context)(rs2_context* context)= nullptr;
	rs2_device_list* (*query_devices)(const rs2_context* context, rs2_error** error)= nullptr;
	void (*delete_device_list)(rs2_device_list* list)= nullptr;
	int (*get_device_count)(const rs2_device_list* list, rs2_error** error)= nullptr;
	rs2_device* (*create_device)(const rs2_device_list* list, int index, rs2_error** error)= nullptr;
	void (*delete_device)(rs2_device* device)= nullptr;
	const char* (*get_device_info)(const rs2_device* device, rs2_camera_info info, rs2_error** error)= nullptr;

	// pipeline / config
	rs2_pipeline* (*create_pipeline)(rs2_context* context, rs2_error** error)= nullptr;
	void (*delete_pipeline)(rs2_pipeline* pipeline)= nullptr;
	rs2_config* (*create_config)(rs2_error** error)= nullptr;
	void (*delete_config)(rs2_config* config)= nullptr;
	void (*config_enable_device)(rs2_config* config, const char* serial, rs2_error** error)= nullptr;
	void (*config_enable_stream)(rs2_config* config, rs2_stream stream, int index, int width, int height,
								 rs2_format format, int framerate, rs2_error** error)= nullptr;
	rs2_pipeline_profile* (*pipeline_start_with_config)(rs2_pipeline* pipeline, rs2_config* config,
														rs2_error** error)= nullptr;
	void (*pipeline_stop)(rs2_pipeline* pipeline, rs2_error** error)= nullptr;
	void (*delete_pipeline_profile)(rs2_pipeline_profile* profile)= nullptr;
	rs2_frame* (*pipeline_wait_for_frames)(rs2_pipeline* pipeline, unsigned int timeout_ms,
										   rs2_error** error)= nullptr;

	// frames
	int (*embedded_frames_count)(rs2_frame* composite, rs2_error** error)= nullptr;
	rs2_frame* (*extract_frame)(rs2_frame* composite, int index, rs2_error** error)= nullptr;
	const void* (*get_frame_data)(const rs2_frame* frame, rs2_error** error)= nullptr;
	double (*get_frame_timestamp)(const rs2_frame* frame, rs2_error** error)= nullptr;
	const rs2_stream_profile* (*get_frame_stream_profile)(const rs2_frame* frame, rs2_error** error)= nullptr;
	int (*get_frame_width)(const rs2_frame* frame, rs2_error** error)= nullptr;
	int (*get_frame_height)(const rs2_frame* frame, rs2_error** error)= nullptr;
	float (*depth_frame_get_units)(const rs2_frame* frame, rs2_error** error)= nullptr;
	void (*release_frame)(rs2_frame* frame)= nullptr;

	// stream profiles / calibration
	void (*get_stream_profile_data)(const rs2_stream_profile* profile, rs2_stream* stream, rs2_format* format,
									int* index, int* unique_id, int* framerate, rs2_error** error)= nullptr;
	void (*get_video_stream_intrinsics)(const rs2_stream_profile* profile, rs2_intrinsics* intrinsics,
										rs2_error** error)= nullptr;
	void (*get_extrinsics)(const rs2_stream_profile* from, const rs2_stream_profile* to,
						   rs2_extrinsics* extrinsics, rs2_error** error)= nullptr;

	// errors
	const char* (*get_error_message)(const rs2_error* error)= nullptr;
	void (*free_error)(rs2_error* error)= nullptr;

	// Loads realsense2.dll and resolves the table. Idempotent; thread-safe
	// for the app's usage (called from the main thread at startup).
	// Returns false (and stays unavailable) when the DLL or any entry point
	// is missing.
	static const RealSenseApi* get();

	bool isAvailable() const { return m_bLoaded; }

	// Logs + frees a non-null rs2_error; returns true if there WAS an error
	bool checkError(rs2_error* error, const char* context) const;

private:
	bool loadEntryPoints();
	bool m_bLoaded= false;
};
