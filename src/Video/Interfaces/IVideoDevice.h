#pragma once

// -- includes -----
#include <memory>

// -- definitions -----
enum class eVideoOpeningStatus : int
{
	failed= -1,
	closed= 0,
	opening= 1,
	open= 2,
};

enum class eVideoStreamingStatus : int
{
	failed= -1,
	stopped= 0,
	pendingStart= 1,
	started= 2,
};

/// The list of possible camera drivers used by Mikan
enum class eVideoSettingType : int
{
	INVALID= -1,

	Brightness,
	Contrast,
	Hue,
	Saturation,
	Sharpness,
	Gamma,
	WhiteBalance,
	RedBalance,
	GreenBalance,
	BlueBalance,
	Gain,
	Pan,
	Tilt,
	Roll,
	Zoom,
	Exposure,
	Iris,
	Focus,

	COUNT
};

/// Constraints on the values for a single tracker property
struct VideoSettingConstraint
{
	int min_value= 0;
	int max_value= 0;
	int stepping_delta= 0;
	int default_value= 0;
	bool is_automatic= false;
};

// The color matrix is used to convert between Y'PbPr video source and non-linear RGB for CG
// See https://en.wikipedia.org/wiki/Y%E2%80%B2UV
// These mirror https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideotransfermatrix
enum class eVideoColorMatrix : int
{
	INVALID= -1,

	BT709, // ITU-R BT.709 color matrix (Common HD color matrix)
	BT601, // ITU-R BT.601 color matrix (Common SD color matrix)
	SMPTE240M,
	BT2020_10, // ITU-R BT.2020 color matrix (Common UHD color matrix)
	BT2020_12,
	Identity,     // Identity. IEC 61966-2-1 (sRGB), SMPTE ST.428-1.
	FCC47,        // FCC Title 47. The exact formula is defined in ISO/IEC 23091-2.
	YCgCo,        // YCbCr pixels are actually YCoCg pixels.
	SMPTE2085,    // The matrix defined for High Dynamic Range (HDR) video, defined by SMPTE ST 2085
	Chroma,       // Chromacity-derived non-constant luminance system, as defined in IEC 23091-2.
	Chroma_const, // Chromacity-derived constant luminance system, as defined in IEC 23091-2.
	ICtCp,        // High Dynamic Range (HDR) and Wide Color Gamut video, as defined by ITU-R BT.2100

	COUNT
};

// Formula to use for converting from non-linear RGB video source to linear RGB for CG
// These mirror the WMF constants here
// https://learn.microsoft.com/en-us/windows/win32/api/mfobjects/ne-mfobjects-mfvideotransferfunction
enum class eVideoTransferFunction : int
{
	INVALID= -1,

	NoGamma,      // No gamma, display referred (relative).
	Gamma_1_0,    // Linear RGB (gamma = 1.0)
	Gamma_1_8,    // True 1.8 gamma, L' = L^1/1.8
	Gamma_2_0,    // True 2.0 gamma, L' = L^1/2.0
	Gamma_2_2,    // True 2.2 gamma, L' = L^1/2.2
	Gamma_2_6,    // True 2.2 gamma, L' = L^1/2.6
	Gamma_2_8,    // True 2.8 gamma, L' = L^1/2.8
	BT1361_ECG,   // The transfer function Extended Color Gamut video, as defined by ITU-R BT.1361.
	BT709,        // ITU-R BT.709 transfer function. Gamma 2.2 curve with a linear segment in the lower range
	BT709_SYM,    // Symmetric ITU-R BT.709.
	HLG,          // Hybrid Log-Gamma, ARIB STD-B67
	SPMTE_240M,   // Gamma 2.2 curve with a linear segment in the lower range
	SMPTE_2084,   // SMPTE ST.2084 also known as PQ. Also defined in ITU-R BT.2100
	SMPTE_428,    // Video transfer function defined in SMPTE ST 428-1
	SRGB,         // Gamma 2.4 curve with a linear segment in the lower range
	LOG_100,      // Logarithmic transfer (100:1 range); for example, as used in H.264 video.
	LOG_316,      // Logarithmic transfer (316.22777:1 range); for example, as used in H.264 video.
	BT2020,       // Non-constant luminance ITU-R BT.2020
	BT2020_CONST, // Constant luminance ITU-R BT.2020

	COUNT
};

struct VideoColorimetry
{
	eVideoColorMatrix colorMatrix;
	eVideoTransferFunction transferFunction;

	// Full color range values (0-255) or partial (16-235 for 8-bit components, 16-240 for chroma)
	bool isFullRange;
};

// VideoDevice interface
class IVideoDevice
{
public:
	IVideoDevice()= default;
	virtual ~IVideoDevice() {}

	// -- Device Properties
	virtual const char* getDevicePath() const= 0;
	virtual const char* getFriendlyName() const= 0;

	// -- Video Settings
	virtual bool isVideoSettingSupported(const eVideoSettingType property_type) const= 0;
	virtual bool getVideoSettingConstraint(const eVideoSettingType property_type,
										   VideoSettingConstraint& outConstraint) const= 0;
	virtual void setVideoSetting(const eVideoSettingType property_type, int desired_value)= 0;
	virtual int getVideoSetting(const eVideoSettingType property_type) const= 0;
};

using IVideoDevicePtr= std::shared_ptr<IVideoDevice>;
using IVideoDeviceConstPtr= std::shared_ptr<const IVideoDevice>;
using IVideoDeviceWeakPtr= std::weak_ptr<IVideoDevice>;
