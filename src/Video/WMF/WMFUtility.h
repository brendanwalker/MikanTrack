#pragma once

#include "IUsbVideoDevice.h"

#include <guiddef.h>

#include <Mfidl.h>
#include <mfreadwrite.h>

#include <string>

typedef long HRESULT;

namespace WMFUtility
{
bool isCompressedFormat(const WMFDeviceFormatInfo& deviceFormat);
const char* getWMFVideoFormatName(const GUID& format);
const char* getUSBVideoFormatName(eUSBVideoFrameBufferFormat format);
std::string getHresultMessage(HRESULT hr);
IMFMediaType* makeWMFMediaTypeFromSourceReader(IMFSourceReader* pSourceReader, DWORD deviceFormatIndex,
											   GUID wmfOutputFormat);
IMFMediaType* makeWMFMediaTypeFromDeviceFormatInfo(const WMFDeviceFormatInfo& deviceFormat, GUID wmfOutputFormat);
void logNativeMediaType(const std::string& function, IMFMediaType* pMediaType);
HRESULT findBestH264Decoder(CLSID* pDecoderCLSID);
}; // namespace WMFUtility