#include "Logger.h"
#include "MemoryUtils.h"
#include "WMFDeviceInfo.h"
#include "WMFUtility.h"

#include <Mfapi.h>
#include <Mferror.h>
#include <Strmif.h>
#include <Shlwapi.h>

#include <sstream>

namespace WMFUtility
{
const char* getWMFVideoFormatName(const GUID& format)
{
	if (format == MFVideoFormat_H264)
		return "H264";
	else if (format == MFVideoFormat_H265)
		return "H265";
	else if (format == MFVideoFormat_YUY2)
		return "YUY2";
	else if (format == MFVideoFormat_NV12)
		return "NV12";
	else if (format == MFVideoFormat_MJPG)
		return "MJPG";
	else if (format == MFVideoFormat_RGB24)
		return "RGB24";
	else
		return "Unknown";
}

const char* getUSBVideoFormatName(eUSBVideoFrameBufferFormat format)
{
	switch (format)
	{
	case eUSBVideoFrameBufferFormat::USBVideo_NV12:
		return "NV12";
	case eUSBVideoFrameBufferFormat::USBVideo_YUY2:
		return "YUY2";
	case eUSBVideoFrameBufferFormat::USBVideo_RGB24:
		return "RGB24";
	default:
		return "Unknown";
	}
}

std::string getHresultMessage(HRESULT hr)
{
	// Check if it's a facility_win32 HRESULT
	if (SUCCEEDED(hr))
	{
		return "Operation successful";
	}

	DWORD flags= FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS;

	// Handle specific facilities if needed (e.g., FACILITY_WIN32)
	if ((hr & 0xFFFF0000) == MAKE_HRESULT(1, FACILITY_WIN32, 0))
	{
		flags|= FORMAT_MESSAGE_FROM_SYSTEM; // Already included above, but good for clarity
		hr= HRESULT_CODE(hr);               // Extract the Win32 error code
	}

	LPSTR messageBuffer= nullptr;
	size_t size= FormatMessageA(flags, NULL,
								hr, // Use the HRESULT or the extracted code
								MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

	std::stringstream stream;
	stream << "Unknown error 0x" << std::hex << std::uppercase << hr;
	std::string message= stream.str();

	if (size > 0)
	{
		message= messageBuffer;
		LocalFree(messageBuffer); // Free the buffer allocated by FormatMessage
	}

	return message;
}

IMFMediaType* makeWMFMediaTypeFromSourceReader(IMFSourceReader* pSourceReader, DWORD deviceFormatIndex,
											   GUID wmfOutputFormat)
{
	IMFMediaType* pClonedType= nullptr;

	// Get the complete native media type by index
	IMFMediaType* pNativeType= nullptr;
	HRESULT hr=
		pSourceReader->GetNativeMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, deviceFormatIndex, &pNativeType);

	if (SUCCEEDED(hr))
	{
		// Clone the native type to preserve all attributes
		hr= MFCreateMediaType(&pClonedType);
		if (SUCCEEDED(hr))
		{
			hr= pNativeType->CopyAllItems(pClonedType);

			// Override with the desired subtype
			if (SUCCEEDED(hr))
			{
				hr= pClonedType->SetGUID(MF_MT_SUBTYPE, wmfOutputFormat);
			}

			if (!SUCCEEDED(hr))
			{
				MemoryUtils::safeRelease(&pClonedType);
			}
		}

		MemoryUtils::safeRelease(&pNativeType);
	}

	return pClonedType;
}

IMFMediaType* makeWMFMediaTypeFromDeviceFormatInfo(const WMFDeviceFormatInfo& m_deviceFormat, GUID wmfOutputFormat)
{
	IMFMediaType* pClonedType= nullptr;

	HRESULT hr= MFCreateMediaType(&pClonedType);

	if (SUCCEEDED(hr))
		hr= pClonedType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);

	if (SUCCEEDED(hr))
		hr= pClonedType->SetGUID(MF_MT_SUBTYPE, wmfOutputFormat);

	if (SUCCEEDED(hr))
	{
		hr= MFSetAttributeSize(pClonedType, MF_MT_FRAME_SIZE, m_deviceFormat.width, m_deviceFormat.height);
	}
	if (SUCCEEDED(hr) && m_deviceFormat.frame_rate_numerator > 0 && m_deviceFormat.frame_rate_denominator > 0)
	{
		hr= MFSetAttributeRatio(pClonedType, MF_MT_FRAME_RATE, m_deviceFormat.frame_rate_numerator,
								m_deviceFormat.frame_rate_denominator);
	}

	return pClonedType;
}

std::string GUIDToString(const GUID& guid)
{
	std::stringstream guidStr;
	guidStr << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << guid.Data1 << "-" << std::setw(4)
			<< guid.Data2 << "-" << std::setw(4) << guid.Data3 << "-" << std::setw(2) << (int)guid.Data4[0]
			<< std::setw(2) << (int)guid.Data4[1] << "-" << std::setw(2) << (int)guid.Data4[2] << std::setw(2)
			<< (int)guid.Data4[3] << std::setw(2) << (int)guid.Data4[4] << std::setw(2) << (int)guid.Data4[5]
			<< std::setw(2) << (int)guid.Data4[6] << std::setw(2) << (int)guid.Data4[7];

	return guidStr.str();
}

void logNativeMediaType(const std::string& function, IMFMediaType* pMediaType)
{
	GUID nativeSubtype;
	pMediaType->GetGUID(MF_MT_SUBTYPE, &nativeSubtype);

	UINT32 nativeWidth= 0, nativeHeight= 0;
	MFGetAttributeSize(pMediaType, MF_MT_FRAME_SIZE, &nativeWidth, &nativeHeight);

	UINT32 codecDataSize= 0;
	pMediaType->GetBlobSize(MF_MT_USER_DATA, &codecDataSize);

	std::string guidStr= GUIDToString(nativeSubtype);

	MIKAN_LOG_INFO(function) << "Native input type - Subtype GUID: " << guidStr << " ("
							 << getWMFVideoFormatName(nativeSubtype) << ")"
							 << ", Resolution: " << nativeWidth << "x" << nativeHeight
							 << ", Codec data size: " << codecDataSize << " bytes";
}

HRESULT findBestH264Decoder(CLSID* pDecoderCLSID)
{
	if (!pDecoderCLSID)
	{
		return E_POINTER;
	}

	// Known problematic vendor MFT CLSIDs to avoid
	// AMD Advanced Media Framework (AMF) Decoders
	static const CLSID CLSID_AMD_H264_DECODER= {
		0x82CE8B14, 0xF24E, 0x4F2E, {0x80, 0x93, 0xDB, 0x8C, 0x63, 0x09, 0x87, 0x72}};

	// NVIDIA Video Decoder
	static const CLSID CLSID_NVIDIA_H264_DECODER= {
		0x56AF0A3E, 0x47A9, 0x4F49, {0x9F, 0x7A, 0x6C, 0x1A, 0x6E, 0x72, 0x0B, 0x5A}};

	// Configure input/output types for MFT enumeration
	MFT_REGISTER_TYPE_INFO inputType= {MFMediaType_Video, MFVideoFormat_H264};
	MFT_REGISTER_TYPE_INFO outputType= {MFMediaType_Video, MFVideoFormat_NV12};

	UINT32 flags= MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_ASYNCMFT | MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER;

	IMFActivate** ppActivate= nullptr;
	UINT32 count= 0;

	// Enumerate H.264 decoders
	HRESULT hr= MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER, flags, &inputType, &outputType, &ppActivate, &count);

	if (FAILED(hr) || count == 0)
	{
		MIKAN_LOG_WARNING("WMFVideoFrameProcessor::findBestH264Decoder")
			<< "No H.264 decoders found: " << getHresultMessage(hr);
		return E_FAIL;
	}

	MIKAN_LOG_INFO("WMFVideoFrameProcessor::findBestH264Decoder") << "Found " << count << " H.264 decoder(s)";

	// Find the best decoder (prefer Microsoft, avoid AMD/NVIDIA)
	CLSID selectedCLSID= GUID_NULL;
	CLSID microsoftCLSID= GUID_NULL;
	bool foundMicrosoft= false;

	for (UINT32 i= 0; i < count; i++)
	{
		CLSID clsid;
		hr= ppActivate[i]->GetGUID(MFT_TRANSFORM_CLSID_Attribute, &clsid);

		if (SUCCEEDED(hr))
		{
			// Get friendly name for logging
			WCHAR* pName= nullptr;
			UINT32 nameLen= 0;
			ppActivate[i]->GetAllocatedString(MFT_FRIENDLY_NAME_Attribute, &pName, &nameLen);

			std::wstring wname= pName ? pName : L"Unknown";
			std::string name(wname.begin(), wname.end());

			// Check if this is a problematic vendor MFT
			bool isProblematic= (clsid == CLSID_AMD_H264_DECODER || clsid == CLSID_NVIDIA_H264_DECODER);

			// Check if this is Microsoft's decoder (best guess by name)
			bool isMicrosoft= (name.find("Microsoft") != std::string::npos);

			MIKAN_LOG_INFO("WMFVideoFrameProcessor::findBestH264Decoder")
				<< "  [" << i << "] " << name << (isMicrosoft ? " (Microsoft)" : "")
				<< (isProblematic ? " (PROBLEMATIC vendor - may cause hangs)" : "");

			if (pName)
			{
				CoTaskMemFree(pName);
			}

			// Prefer Microsoft's decoder (don't skip problematic ones, just prioritize Microsoft)
			if (isMicrosoft && !foundMicrosoft)
			{
				microsoftCLSID= clsid;
				foundMicrosoft= true;
			}

			// Track any non-Microsoft decoder as fallback
			if (!isMicrosoft && selectedCLSID == GUID_NULL)
			{
				selectedCLSID= clsid;
			}
		}
	}

	// Cleanup
	for (UINT32 i= 0; i < count; i++)
	{
		ppActivate[i]->Release();
	}
	CoTaskMemFree(ppActivate);

	// Prefer Microsoft, fallback to first non-problematic
	if (foundMicrosoft)
	{
		*pDecoderCLSID= microsoftCLSID;
		MIKAN_LOG_INFO("WMFVideoFrameProcessor::findBestH264Decoder") << "Selected Microsoft H.264 decoder";
		return S_OK;
	}
	else if (selectedCLSID != GUID_NULL)
	{
		*pDecoderCLSID= selectedCLSID;
		MIKAN_LOG_INFO("WMFVideoFrameProcessor::findBestH264Decoder")
			<< "Selected first available non-problematic decoder";
		return S_OK;
	}

	MIKAN_LOG_ERROR("WMFVideoFrameProcessor::findBestH264Decoder") << "No suitable H.264 decoder found";
	return E_FAIL;
}

} // namespace WMFUtility