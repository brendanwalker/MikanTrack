// Free-function helpers extracted from MikanXR:
// - makeVideoModeName mirrors the static helper in WMFDeviceList.cpp
// - the option list / best match logic is ported from
//   MikanXR src/Editor/ECS/VideoSource/USBVideoSourceComponent.cpp
//   (rebuildVideoModeOptionLists / setVideoModeToBestMatch / findBestVideoModeIndex),
//   operating directly on IUsbVideoDevice with no editor dependencies.
#include "VideoModeUtils.h"

#include <iomanip>
#include <set>
#include <sstream>

namespace VideoModeUtils
{
std::string makeVideoModeName(int width, int height, unsigned int frameRateNumerator,
							  unsigned int frameRateDenominator, const std::string& format, bool isInterlaced)
{
	std::ostringstream ss;

	ss << width << "x" << height;

	// Add interlace info if relevant
	if (isInterlaced)
	{
		ss << "i";
	}

	if (frameRateDenominator != 0)
	{
		const double fps= static_cast<double>(frameRateNumerator) / frameRateDenominator;

		ss << "@" << std::fixed << std::setprecision(0) << fps << "fps";
	}

	if (!format.empty() && format.find('{') == std::string::npos)
	{
		ss << " (" << format << ")";
	}

	return ss.str();
}

bool parseVideoModeName(const std::string& modeName, std::string& outResolution, std::string& outFrameRate,
						std::string& outFormat)
{
	outResolution.clear();
	outFrameRate.clear();
	outFormat.clear();

	// "1920x1080@30fps (NV12)" -> "1920x1080" + "30fps" + "NV12"
	const size_t atPos= modeName.find('@');
	if (atPos == std::string::npos)
		return false;

	outResolution= modeName.substr(0, atPos);
	if (outResolution.find('x') == std::string::npos)
		return false;

	const size_t frameRateEnd= modeName.find(' ', atPos + 1);
	if (frameRateEnd == std::string::npos)
	{
		// No format suffix (e.g. raw GUID formats get omitted from the name)
		outFrameRate= modeName.substr(atPos + 1);
	}
	else
	{
		outFrameRate= modeName.substr(atPos + 1, frameRateEnd - (atPos + 1));

		const size_t openParen= modeName.find('(', frameRateEnd);
		if (openParen != std::string::npos)
		{
			const size_t closeParen= modeName.find(')', openParen + 1);
			if (closeParen != std::string::npos)
			{
				outFormat= modeName.substr(openParen + 1, closeParen - (openParen + 1));
			}
		}
	}

	return !outResolution.empty() && !outFrameRate.empty();
}

std::string formatFrameRate(const UsbVideoModeProperties& modeProperties)
{
	if (modeProperties.frame_rate_demonenator == 0)
		return "0";
	if (modeProperties.frame_rate_demonenator == 1)
		return std::to_string(modeProperties.frame_rate_numerator);

	const float fps= static_cast<float>(modeProperties.frame_rate_numerator)
					 / static_cast<float>(modeProperties.frame_rate_demonenator);

	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2) << fps;
	return oss.str();
}

// Strips an optional "fps" suffix so both "30fps" (parseVideoModeName output)
// and "30" (formatFrameRate output) are accepted as frame rate selections
static std::string stripFpsSuffix(const std::string& frameRate)
{
	if (frameRate.size() > 3 && frameRate.compare(frameRate.size() - 3, 3, "fps") == 0)
	{
		return frameRate.substr(0, frameRate.size() - 3);
	}

	return frameRate;
}

// -- Current mode accessors -----
bool getVideoModeResolutionName(const IUsbVideoDevice* device, std::string& outResolution)
{
	UsbVideoModeProperties modeProperties;
	if (device != nullptr && device->getVideoModeProperties(device->getVideoModeIndex(), modeProperties))
	{
		outResolution= std::to_string(modeProperties.width) + "x" + std::to_string(modeProperties.height);
		return true;
	}

	return false;
}

bool getVideoModeFrameRateName(const IUsbVideoDevice* device, std::string& outFrameRate)
{
	UsbVideoModeProperties modeProperties;
	if (device != nullptr && device->getVideoModeProperties(device->getVideoModeIndex(), modeProperties)
		&& modeProperties.frame_rate_demonenator > 0)
	{
		// Use formatFrameRate so the name matches the entries produced by
		// getDistinctFrameRateNames (the editor used integer division here,
		// which disagreed with its own combo list for fractional rates)
		outFrameRate= formatFrameRate(modeProperties);
		return true;
	}

	return false;
}

bool getVideoModeFormatName(const IUsbVideoDevice* device, std::string& outFormat)
{
	UsbVideoModeProperties modeProperties;
	if (device != nullptr && device->getVideoModeProperties(device->getVideoModeIndex(), modeProperties)
		&& modeProperties.format != nullptr)
	{
		outFormat= modeProperties.format;
		return true;
	}

	return false;
}

// -- UI combo option lists -----
bool getVideoModeNames(const IUsbVideoDevice* device, std::vector<std::string>& outVideoModeNames)
{
	if (device != nullptr)
	{
		size_t modeCount= device->getAvailableVideoModesCount();
		outVideoModeNames.clear();
		outVideoModeNames.reserve(modeCount);
		for (size_t i= 0; i < modeCount; ++i)
		{
			UsbVideoModeProperties modeProperties;
			if (device->getVideoModeProperties(i, modeProperties))
			{
				outVideoModeNames.push_back(modeProperties.name);
			}
		}

		return true;
	}

	return false;
}

bool getVideoModeOptionLists(const IUsbVideoDevice* device, std::vector<std::string>& outResolutionNames,
							 std::vector<std::string>& outFrameRateNames, std::vector<std::string>& outFormatNames)
{
	outResolutionNames.clear();
	outFrameRateNames.clear();
	outFormatNames.clear();

	if (device == nullptr)
		return false;

	UsbVideoModeProperties currentModeProperties;
	const int currentVideoModeIndex= device->getVideoModeIndex();
	if (!device->getVideoModeProperties(currentVideoModeIndex, currentModeProperties))
		return false;

	// Helper struct to store resolution with area for sorting
	struct ResolutionInfo
	{
		std::string name;
		int width;
		int height;
		int area;

		ResolutionInfo(const UsbVideoModeProperties& modeProperties)
			: width(modeProperties.width)
			, height(modeProperties.height)
			, area(modeProperties.width * modeProperties.height)
			, name(std::to_string(modeProperties.width) + "x" + std::to_string(modeProperties.height))
		{
		}

		bool operator<(const ResolutionInfo& other) const
		{
			// Sort by area (descending), then by width (descending)
			if (area != other.area)
				return area > other.area;
			return width > other.width;
		}
	};

	// Helper struct to store frame rate for sorting
	struct FrameRateInfo
	{
		std::string name;
		int numerator;
		int denomenator;
		float fps;

		FrameRateInfo(const UsbVideoModeProperties& modeProperties)
			: numerator(modeProperties.frame_rate_numerator)
			, denomenator(modeProperties.frame_rate_demonenator)
			, fps(static_cast<float>(numerator) / static_cast<float>(denomenator))
			, name(formatFrameRate(modeProperties))
		{
		}

		bool operator<(const FrameRateInfo& other) const { return fps > other.fps; }
	};

	std::set<ResolutionInfo> uniqueResolutions;
	std::set<FrameRateInfo> uniqueFrameRates;
	std::set<std::string> uniqueFormats;

	size_t modeCount= device->getAvailableVideoModesCount();
	for (size_t i= 0; i < modeCount; ++i)
	{
		UsbVideoModeProperties modeProperties;
		if (device->getVideoModeProperties(i, modeProperties))
		{
			// Collect unique resolutions
			uniqueResolutions.insert(ResolutionInfo(modeProperties));

			// If the mode's resolution matches the current mode, add the possible frame rates
			if (currentModeProperties.width == modeProperties.width
				&& currentModeProperties.height == modeProperties.height)
			{
				// Collect unique frame rates for the current resolution
				if (modeProperties.frame_rate_demonenator > 0)
				{
					uniqueFrameRates.insert(FrameRateInfo(modeProperties));

					// Collect unique formats for the current resolution and frame rate
					if (currentModeProperties.frame_rate_numerator == modeProperties.frame_rate_numerator
						&& currentModeProperties.frame_rate_demonenator == modeProperties.frame_rate_demonenator)
					{
						uniqueFormats.insert(modeProperties.format);
					}
				}
			}
		}
	}

	// Convert sorted resolutions to string vector
	for (const auto& resInfo : uniqueResolutions)
	{
		outResolutionNames.push_back(resInfo.name);
	}

	// Convert sorted frame rates to string vector (descending order)
	for (const auto& frameRateInfo : uniqueFrameRates)
	{
		outFrameRateNames.push_back(frameRateInfo.name);
	}

	// Convert formats to string vector (alphabetical)
	outFormatNames.assign(uniqueFormats.begin(), uniqueFormats.end());

	return true;
}

bool getDistinctResolutionNames(const IUsbVideoDevice* device, std::vector<std::string>& outResolutionNames)
{
	std::vector<std::string> frameRateNames;
	std::vector<std::string> formatNames;
	return getVideoModeOptionLists(device, outResolutionNames, frameRateNames, formatNames);
}

bool getDistinctFrameRateNames(const IUsbVideoDevice* device, std::vector<std::string>& outFrameRateNames)
{
	std::vector<std::string> resolutionNames;
	std::vector<std::string> formatNames;
	return getVideoModeOptionLists(device, resolutionNames, outFrameRateNames, formatNames);
}

bool getDistinctFormatNames(const IUsbVideoDevice* device, std::vector<std::string>& outFormatNames)
{
	std::vector<std::string> resolutionNames;
	std::vector<std::string> frameRateNames;
	return getVideoModeOptionLists(device, resolutionNames, frameRateNames, outFormatNames);
}

// -- Best-match selection -----
int findBestVideoModeIndex(const IUsbVideoDevice* device, const std::string& resolution, const std::string& frameRate,
						   const std::string& format)
{
	if (device == nullptr)
		return -1;

	const std::string desiredFrameRate= stripFpsSuffix(frameRate);

	// Parse resolution string (format: "WIDTHxHEIGHT")
	int desiredWidth= -1;
	int desiredHeight= -1;
	if (!resolution.empty())
	{
		size_t xPos= resolution.find('x');
		if (xPos != std::string::npos)
		{
			desiredWidth= std::stoi(resolution.substr(0, xPos));
			desiredHeight= std::stoi(resolution.substr(xPos + 1));
		}
	}

	// Find best matching video mode
	size_t modeCount= device->getAvailableVideoModesCount();
	for (size_t modeIndex= 0; modeIndex < modeCount; ++modeIndex)
	{
		UsbVideoModeProperties modeProperties;
		if (device->getVideoModeProperties(modeIndex, modeProperties) && modeProperties.width == desiredWidth
			&& modeProperties.height == desiredHeight && modeProperties.format != nullptr
			&& modeProperties.format == format && formatFrameRate(modeProperties) == desiredFrameRate)
		{
			return (int)modeIndex;
		}
	}

	return -1;
}

std::string findBestVideoModeName(const IUsbVideoDevice* device, const std::string& resolution,
								  const std::string& frameRate, const std::string& format)
{
	const int modeIndex= findBestVideoModeIndex(device, resolution, frameRate, format);
	if (modeIndex != -1)
	{
		UsbVideoModeProperties modeProperties;
		if (device->getVideoModeProperties(modeIndex, modeProperties) && modeProperties.name != nullptr)
		{
			return modeProperties.name;
		}
	}

	return std::string();
}

bool setVideoModeToBestMatch(IUsbVideoDevice* device, const std::string& resolution, const std::string& frameRate,
							 const std::string& format)
{
	const int modeIndex= findBestVideoModeIndex(device, resolution, frameRate, format);
	if (modeIndex == -1)
		return false;

	// Already the current mode (setVideoModeByIndex would return false)
	if (device->getVideoModeIndex() == modeIndex)
		return true;

	return device->setVideoModeByIndex(modeIndex);
}

int findBestVideoModeIndex(const IUsbVideoDevice* device, int width, int height, int frameRate)
{
	if (device == nullptr)
		return -1;

	int result_id= -1;

	size_t numFormats= device->getAvailableVideoModesCount();
	if (numFormats > 0)
	{
		for (int attempt= 0; attempt < 2; ++attempt)
		{
			for (size_t testDeviceIndex= 0; testDeviceIndex < numFormats; ++testDeviceIndex)
			{
				if (UsbVideoModeProperties modeInfo; device->getVideoModeProperties(testDeviceIndex, modeInfo)
													 && modeInfo.frame_rate_demonenator > 0)
				{
					int rounded_frame_rate= modeInfo.frame_rate_numerator / modeInfo.frame_rate_demonenator;

					if ((width == -1 || modeInfo.width == width) && (height == -1 || modeInfo.height == height)
						&& (frameRate == -1 || rounded_frame_rate == frameRate))
					{
						result_id= (int)testDeviceIndex;
						break;
					}
				}
			}

			if (result_id != -1)
			{
				break;
			}
			else if (attempt == 0)
			{
				// Fallback to no FPS restriction on second pass
				frameRate= -1;
			}
		}

		if (result_id == -1)
		{
			// If we didn't find an exact match, just return the first available mode
			result_id= 0;
		}
	}

	return result_id;
}
}; // namespace VideoModeUtils
