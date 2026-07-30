// -- includes -----
#include "WMFDeviceInfo.h"

// -- WMF Device Format Info -----
bool WMFDeviceFormatInfo::isCompressedFormat() const
{
	return sub_type_name == "H264" || sub_type_name == "MJPG" || sub_type_name == "H265" || sub_type_name == "VP80"
		   || sub_type_name == "VP90";
}

// -- WMF Device Info -----
int WMFDeviceInfo::findDeviceFormatByName(const std::string& format_name) const
{
	for (const WMFDeviceFormatInfo& info : deviceAvailableFormats)
	{
		if (info.format_friendly_name == format_name)
		{
			return info.device_format_index;
		}
	}
	return INVALID_DEVICE_FORMAT_INDEX;
}

int WMFDeviceInfo::findBestDeviceFormatIndex(unsigned int w, unsigned int h, unsigned int frameRate,
											 const char* buffer_format) const
{
	int result_id= INVALID_DEVICE_FORMAT_INDEX;
	for (int attempt= 0; attempt < 2; ++attempt)
	{
		for (const WMFDeviceFormatInfo& info : deviceAvailableFormats)
		{
			unsigned int rounded_frame_rate= info.frame_rate_numerator / info.frame_rate_denominator;

			if ((w == UNSPECIFIED_CAMERA_WIDTH || info.width == w)
				&& (h == UNSPECIFIED_CAMERA_HEIGHT || info.height == h) && info.sub_type_name == buffer_format
				&& (frameRate == UNSPECIFIED_CAMERA_FPS || rounded_frame_rate == frameRate))
			{
				result_id= info.device_format_index;
				break;
			}
		}

		if (result_id != INVALID_DEVICE_FORMAT_INDEX)
		{
			break;
		}
		else if (attempt == 0)
		{
			// Fallback to no FPS restriction on second pass
			frameRate= UNSPECIFIED_CAMERA_FPS;
		}
	}

	return result_id;
}