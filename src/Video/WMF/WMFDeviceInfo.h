#pragma once

// -- includes -----
#include <vector>
#include <string>

//-- constants -----
#define INVALID_DEVICE_FORMAT_INDEX -1
#define UNSPECIFIED_CAMERA_WIDTH 0xFFFFFFFF
#define UNSPECIFIED_CAMERA_HEIGHT 0xFFFFFFFF
#define UNSPECIFIED_CAMERA_FPS 0xFFFFFFFF

// -- definitions -----
struct WMFDeviceFormatInfo
{
	int device_format_index= 0;

	unsigned int frame_size= 0;
	unsigned int height= 0;
	unsigned int width= 0;

	unsigned int yuv_matrix=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideotransfermatrix
	unsigned int transfer_function=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideotransferfunction
	unsigned int video_nominal_range=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfnominalrange
	unsigned int video_chroma_subsampling=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideochromasubsampling
	unsigned int video_lighting=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideolighting

	int default_stride= 0; // Stride is positive for top-down images, and negative for bottom-up images.
	unsigned int fixed_size_samples= 0;
	unsigned int frame_rate_numerator= 0;
	unsigned int frame_rate_denominator= 0;
	unsigned int pixel_aspect_ratio_horizontal= 0;
	unsigned int pixel_aspect_ratio_vertical= 0;
	unsigned int all_samples_independant= 0;
	unsigned int frame_rate_range_min_numerator= 0;
	unsigned int frame_rate_range_min_denominator= 0;
	unsigned int sample_size= 0;
	unsigned int video_primaries=
		0; // https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideoprimaries
	unsigned int interlace_mode= 0;
	unsigned int frame_rate_range_max_numerator= 0;
	unsigned int frame_rate_range_max_denominator= 0;

	std::string am_format_type_name;
	std::string major_type_name;
	std::string sub_type_name;
	std::string format_friendly_name;

	bool isCompressedFormat() const;
};

struct WMFDeviceInfo
{
	int wmfDeviceIndex;
	std::string deviceFriendlyName; // utf-8 string
	std::string deviceSymbolicLink; // can be passed in as the value of the DevicePath argument of the
									// SetupDiOpenDeviceInterface function.
	std::string uniqueIdentifier;
	unsigned int usbVendorId;
	unsigned int usbProductId;
	std::vector<WMFDeviceFormatInfo> deviceAvailableFormats;

	WMFDeviceInfo()
		: wmfDeviceIndex(-1)
		, deviceFriendlyName()
		, deviceSymbolicLink()
		, uniqueIdentifier()
		, usbVendorId(0)
		, usbProductId(0)
		, deviceAvailableFormats()
	{
	}

	int findDeviceFormatByName(const std::string& format_name) const;
	int findBestDeviceFormatIndex(unsigned int width, unsigned int height, unsigned int frameRate,
								  const char* buffer_format) const;
};