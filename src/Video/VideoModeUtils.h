#pragma once

#include "IUsbVideoDevice.h"

#include <string>
#include <vector>

// Free helper functions for working with IUsbVideoDevice video mode names.
// Mode names are the strings built by the static makeVideoModeName() helper in
// WMFDeviceList.cpp, e.g. "1920x1080@30fps (NV12)".
namespace VideoModeUtils
{
// Builds a video mode name in the same format as the WMF device list:
// "<width>x<height>[i]@<fps>fps (<format>)"
// (equivalent to the static makeVideoModeName() in WMFDeviceList.cpp)
std::string makeVideoModeName(int width, int height, unsigned int frameRateNumerator,
							  unsigned int frameRateDenominator, const std::string& format, bool isInterlaced= false);

// Decomposes a mode name like "1920x1080@30fps (NV12)" into
// resolution ("1920x1080"), frame rate ("30fps") and format ("NV12") strings.
// Returns false if the string doesn't look like a video mode name.
bool parseVideoModeName(const std::string& modeName, std::string& outResolution, std::string& outFrameRate,
						std::string& outFormat);

// Frame rate display string for a video mode
// ("30" for whole rates, "29.97" for fractional rates)
std::string formatFrameRate(const UsbVideoModeProperties& modeProperties);

// -- Current mode accessors -----
bool getVideoModeResolutionName(const IUsbVideoDevice* device, std::string& outResolution);
bool getVideoModeFrameRateName(const IUsbVideoDevice* device, std::string& outFrameRate);
bool getVideoModeFormatName(const IUsbVideoDevice* device, std::string& outFormat);

// -- UI combo option lists -----
// All mode names on the device, in device order
bool getVideoModeNames(const IUsbVideoDevice* device, std::vector<std::string>& outVideoModeNames);

// Distinct option lists relative to the device's current video mode:
// - resolutions: all distinct resolutions, sorted by area (descending)
// - frame rates: distinct frame rates available at the current resolution (descending)
// - formats: distinct formats available at the current resolution + frame rate (alphabetical)
bool getVideoModeOptionLists(const IUsbVideoDevice* device, std::vector<std::string>& outResolutionNames,
							 std::vector<std::string>& outFrameRateNames, std::vector<std::string>& outFormatNames);
bool getDistinctResolutionNames(const IUsbVideoDevice* device, std::vector<std::string>& outResolutionNames);
bool getDistinctFrameRateNames(const IUsbVideoDevice* device, std::vector<std::string>& outFrameRateNames);
bool getDistinctFormatNames(const IUsbVideoDevice* device, std::vector<std::string>& outFormatNames);

// -- Best-match selection -----
// Finds the video mode matching a resolution ("1920x1080"), frame rate ("30" or "30fps")
// and format ("NV12") selection. Returns -1 / empty string if no mode matches.
int findBestVideoModeIndex(const IUsbVideoDevice* device, const std::string& resolution, const std::string& frameRate,
						   const std::string& format);
std::string findBestVideoModeName(const IUsbVideoDevice* device, const std::string& resolution,
								  const std::string& frameRate, const std::string& format);
bool setVideoModeToBestMatch(IUsbVideoDevice* device, const std::string& resolution, const std::string& frameRate,
							 const std::string& format);

// Finds the video mode closest to the desired width/height/frame rate
// (pass -1 for any). Falls back to ignoring the frame rate, then to the first
// available mode. Returns -1 only if the device has no modes at all.
int findBestVideoModeIndex(const IUsbVideoDevice* device, int width, int height, int frameRate);
}; // namespace VideoModeUtils
