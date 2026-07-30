#pragma once

#include "MikanUtilityExport.h"

#include "stdlib.h" // size_t
#include <string>
#include <sstream>
#include <vector>

//-- utility methods -----
namespace StringUtils
{
template <typename t_enum_class>
static t_enum_class FindEnumValue(const std::string& stringValue, const std::string enumStrings[])
{
	for (int enumIntValue= 0; enumIntValue < (int)t_enum_class::COUNT; ++enumIntValue)
	{
		if (enumStrings[enumIntValue] == stringValue)
		{
			return (t_enum_class)enumIntValue;
		}
	}

	return t_enum_class::INVALID;
}

template <typename t_enum_class>
static t_enum_class FindEnumValue(const std::string& stringValue, const char** enumStrings)
{
	for (int enumIntValue= 0; enumIntValue < (int)t_enum_class::COUNT; ++enumIntValue)
	{
		std::string enumStringValue= enumStrings[enumIntValue];

		if (enumStringValue == stringValue)
		{
			return (t_enum_class)enumIntValue;
		}
	}

	return t_enum_class::INVALID;
}

MIKAN_UTILITY_FUNC(bool) convertWcsToMbs(const wchar_t* wc_string, char* out_mb_string, const size_t mb_buffer_size);
MIKAN_UTILITY_FUNC(bool) convertMbsToWcs(const char* mb_string, wchar_t* out_wc_string, const size_t wc_buffer_size);

MIKAN_UTILITY_FUNC(std::wstring) convertUTF8StringToWString(const std::string& str);
MIKAN_UTILITY_FUNC(std::string) convertWStringToUTF8String(const std::wstring& wstr);

// https://codereview.stackexchange.com/questions/46596/format-string-inline
template <typename t_arg_type>
std::string stringify(t_arg_type const& arg)
{
	std::stringstream stringStream;
	stringStream << arg;
	return stringStream.str();
}

template <typename t_arg_type, typename... Args>
std::string stringify(t_arg_type arg, const Args&... args)
{
	return stringify(arg) + stringify(args...);
}

/// Formats a string into the given target buffer
/// \param buffer The target buffer to write in to
/// \param buffer_size The max number of bytes that can be written to the buffer
/// \param format The formatting string that will be written to the buffer
/// \return The number of characters successfully written
MIKAN_UTILITY_FUNC(int) formatString(char* buffer, size_t buffer_size, const char* format, ...);

// Split a string on a single character seperator
MIKAN_UTILITY_FUNC(std::vector<std::string>) splitString(const std::string& s, char seperator);

// Join a set of strings together with the given delimeter
MIKAN_UTILITY_FUNC(std::string) joinString(const std::vector<std::string>& elems, char delim);

/// Formats a wide character string into the given target buffer
/// \param buffer The target buffer to write in to
/// \param buffer_size The max number of bytes that can be written to the buffer
/// \param format The formatting string that will be written to the buffer
/// \return The number of characters successfully written
MIKAN_UTILITY_FUNC(int) formatWString(wchar_t* buffer, size_t buffer_size, const wchar_t* format, ...);

/// Encodes a vector of bytes into a Base64 string.
/// \param data The vector of bytes to encode.
/// \return A Base64 encoded string.
MIKAN_UTILITY_FUNC(std::string) base64Encode(const std::vector<uint8_t>& data);
}; // namespace StringUtils