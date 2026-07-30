#pragma once

#include "IUsbVideoDevice.h"

#include <mfreadwrite.h>
#include <stdint.h>

class WMFVideoFrameProcessor : public IMFSourceReaderCallback
{
public:
	enum class State
	{
		Stopped,
		Starting,
		Running,
		Stopping,
		Failed
	};

	WMFVideoFrameProcessor(int deviceIndex, const struct WMFDeviceFormatInfo& deviceFormat,
						   class MikanWMFVideoDevice* listener);
	~WMFVideoFrameProcessor();

	HRESULT init(IMFMediaSource* pSource);
	void dispose();

	void startVideoFrameStream();
	void stopVideoFrameStream();

	inline State getState() const { return m_state; }
	inline bool getIsRunning() const { return m_state == State::Running; }

protected:
	// IUnknown methods
	STDMETHODIMP QueryInterface(REFIID iid, void** ppv);
	STDMETHODIMP_(ULONG)
	AddRef();
	STDMETHODIMP_(ULONG)
	Release();

	// IMFSourceReaderCallback methods
	STDMETHODIMP OnReadSample(HRESULT hrStatus, DWORD dwStreamIndex, DWORD dwStreamFlags, LONGLONG llTimestamp,
							  IMFSample* pSample);
	STDMETHODIMP OnFlush(DWORD dwStreamIndex) { return S_OK; }
	STDMETHODIMP OnEvent(DWORD dwStreamIndex, IMFMediaEvent* pEvent) { return S_OK; }

private:
	HRESULT processCompressedSample(IMFSample* pCompressedSample, IMFSample** ppDecodedSample);

	int m_deviceIndex;
	const WMFDeviceFormatInfo& m_deviceFormat;
	class MikanWMFVideoDevice* m_videoSourceListener;

	long m_referenceCount;

	IMFSourceReader* m_pSourceReader;
	IMFTransform* m_pDecoderTransform;
	IMFMediaType* m_pNativeInputType; // Saved native H.264 media type from camera
	bool m_bNeedsDecoder;
	MFT_OUTPUT_STREAM_INFO m_decoderOutputInfo;

	State m_state;
	int64_t m_sampleIndex;
	eUSBVideoFrameBufferFormat m_outputFormat;
	GUID m_wmfOutputFormat;

	// NV12 plane offset tracking
	bool m_nv12_offsets_detected;
	size_t m_nv12_uv_plane_offset;
};
