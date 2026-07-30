// -- includes -----
#include "StringUtils.h"

#include <wchar.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>

#include <codecvt>
#include <locale>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

// -- public methods -----
namespace StringUtils
{
bool convertWcsToMbs(const wchar_t* wc_string, char* out_mb_string, const size_t mb_buffer_size)
{
	bool success= false;

	if (wc_string != nullptr)
	{
#if defined WIN32 || defined _WIN32 || defined WINCE
		size_t countConverted;
		const wchar_t* wcsIndirectString= wc_string;
		mbstate_t mbstate;

		success=
			wcsrtombs_s(&countConverted, out_mb_string, mb_buffer_size, &wcsIndirectString, _TRUNCATE, &mbstate) == 0;
#else
		success= wcstombs(out_mb_string, wc_string, mb_buffer_size) != static_cast<size_t>(-1);
#endif
	}

	return success;
}

bool convertMbsToWcs(const char* mb_string, wchar_t* out_wc_string, const size_t wc_buffer_size)
{
	bool success= false;

	if (mb_string != nullptr)
	{
		const char* mbsIndirectString= mb_string;
		mbstate_t mbstate;

		success= mbsrtowcs(out_wc_string, &mbsIndirectString, wc_buffer_size, &mbstate) != static_cast<size_t>(-1);
	}

	return success;
}

std::wstring convertUTF8StringToWString(const std::string& str)
{
	using convert_typeX= std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.from_bytes(str);
}

std::string convertWStringToUTF8String(const std::wstring& wstr)
{
	using convert_typeX= std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.to_bytes(wstr);
}

int formatString(char* buffer, size_t buffer_size, const char* format, ...)
{
	// Bake out the text string
	va_list args;
	va_start(args, format);
	int chars_written= vsnprintf(buffer, buffer_size, format, args);
	buffer[buffer_size - 1]= 0;
	va_end(args);

	return chars_written;
}

std::vector<std::string> splitString(const std::string& s, char seperator)
{
	std::vector<std::string> output;

	std::string::size_type prev_pos= 0, pos= 0;

	while ((pos= s.find(seperator, pos)) != std::string::npos)
	{
		std::string substring(s.substr(prev_pos, pos - prev_pos));

		output.push_back(substring);

		prev_pos= ++pos;
	}

	output.push_back(s.substr(prev_pos, pos - prev_pos)); // Last word

	return output;
}

std::string joinString(const std::vector<std::string>& elems, char delim)
{
	std::string s;

	for (auto it= elems.begin(); it != elems.end(); ++it)
	{
		s+= (*it);
		if (it + 1 != elems.end())
		{
			s+= delim;
		}
	}

	return s;
}

int formatWString(wchar_t* buffer, size_t buffer_size, const wchar_t* format, ...)
{
	// Bake out the text string
	va_list args;
	va_start(args, format);
	int chars_written= vswprintf(buffer, buffer_size, format, args);
	buffer[buffer_size - 1]= L'\0';
	va_end(args);

	return chars_written;
}

std::string base64Encode(const std::vector<uint8_t>& data)
{
	static const char kBase64Chars[]= "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	std::string result;
	const size_t size= data.size();
	result.reserve(((size + 2) / 3) * 4);

	for (size_t i= 0; i < size; i+= 3)
	{
		uint32_t group= static_cast<uint32_t>(data[i]) << 16;
		if (i + 1 < size)
			group|= static_cast<uint32_t>(data[i + 1]) << 8;
		if (i + 2 < size)
			group|= static_cast<uint32_t>(data[i + 2]);

		result+= kBase64Chars[(group >> 18) & 0x3F];
		result+= kBase64Chars[(group >> 12) & 0x3F];
		result+= (i + 1 < size) ? kBase64Chars[(group >> 6) & 0x3F] : '=';
		result+= (i + 2 < size) ? kBase64Chars[(group >> 0) & 0x3F] : '=';
	}
	return result;
}
}; // namespace StringUtils