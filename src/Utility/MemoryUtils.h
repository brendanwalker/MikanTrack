#pragma once

#include "stdlib.h" // size_t

//-- macros -----
#ifndef ARRAY_SIZE
#define ARRAY_SIZE(_A) (sizeof(_A) / sizeof((_A)[0]))
#endif

//-- utility methods -----
namespace MemoryUtils
{
template <typename T>
void safeRelease(T** ppT)
{
	if (*ppT)
	{
		(*ppT)->Release();
		*ppT= nullptr;
	}
}

template <typename T>
void safeReleaseAllCount(T** ppT)
{
	if (*ppT)
	{
		unsigned long e= (*ppT)->Release();

		while (e)
		{
			e= (*ppT)->Release();
		}

		*ppT= nullptr;
	}
}
}; // namespace MemoryUtils