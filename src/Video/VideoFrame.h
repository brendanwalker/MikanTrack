#pragma once

#include <cstdint>
#include <vector>

#include "IUsbVideoDevice.h"

// A heap-owned copy of one raw camera frame, made inside the Media Foundation
// callback (the callback's UsbVideoFrameBuffer is only valid during the call).
// Blocks are recycled through a freelist to avoid per-frame allocation.
struct VideoFrameBlock
{
	int64_t frameIndex= -1;
	double timestampMs= 0.0;

	eUSBVideoFrameBufferFormat format= eUSBVideoFrameBufferFormat::USBVideo_UNKNOWN;
	UsbVideoFrameSection sections[2];
	int sectionCount= 0;

	std::vector<uint8_t> data;

	void copyFrom(const UsbVideoFrameBuffer& buffer, int64_t inFrameIndex, double inTimestampMs)
	{
		frameIndex= inFrameIndex;
		timestampMs= inTimestampMs;
		format= buffer.data_format;
		sectionCount= buffer.section_count;
		for (int i= 0; i < sectionCount; ++i)
			sections[i]= buffer.sections[i];
		data.assign(buffer.data, buffer.data + buffer.byte_count);
	}
};
