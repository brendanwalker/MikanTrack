#include "TestCommon.h"

#include "ImGuiTheme.h"
#include "LocalizationManager.h"

// Validates the localization tables without a GL context. Load-time
// validation warnings (key parity in both directions, printf specifier
// mismatches, embedded ##, _meta problems) are hard failures here, window
// stable IDs must be unique, the unknown-key fallback must pass the key
// through, and every displayed codepoint must sit inside the glyph ranges
// the font atlas actually bakes.

static bool decodeNextCodepoint(const std::string& text, size_t& inOutIndex,
								unsigned int& outCodepoint)
{
	const unsigned char lead= (unsigned char)text[inOutIndex];
	size_t extraBytes= 0;
	if (lead < 0x80)
	{
		outCodepoint= lead;
	}
	else if ((lead & 0xE0) == 0xC0)
	{
		outCodepoint= lead & 0x1F;
		extraBytes= 1;
	}
	else if ((lead & 0xF0) == 0xE0)
	{
		outCodepoint= lead & 0x0F;
		extraBytes= 2;
	}
	else if ((lead & 0xF8) == 0xF0)
	{
		outCodepoint= lead & 0x07;
		extraBytes= 3;
	}
	else
	{
		return false;
	}

	for (size_t byteIndex= 0; byteIndex < extraBytes; ++byteIndex)
	{
		++inOutIndex;
		if (inOutIndex >= text.size() || ((unsigned char)text[inOutIndex] & 0xC0) != 0x80)
			return false;
		outCodepoint= (outCodepoint << 6) | ((unsigned char)text[inOutIndex] & 0x3F);
	}
	++inOutIndex;
	return true;
}

static bool isCodepointInRanges(unsigned int codepoint, const ImWchar* ranges)
{
	for (const ImWchar* range= ranges; range[0] != 0; range+= 2)
	{
		if (codepoint >= range[0] && codepoint <= range[1])
			return true;
	}
	return false;
}

static int runLocalizationTest(const TestArgs& args)
{
	LocalizationManager manager;
	if (!manager.startup(nullptr))
	{
		MIKAN_LOG_ERROR("loc-test") << "Manager startup failed (missing or corrupt en.json)";
		return 1;
	}

	int errorCount= 0;

	// Load-time validation: anything the loader had to warn about and
	// backfill is a table bug
	for (const std::string& warning : manager.getLoadWarnings())
	{
		MIKAN_LOG_ERROR("loc-test") << "Load warning: " << warning;
		++errorCount;
	}

	// Window stable IDs: the English text of every windows.* key IS the
	// window's ImGui ID, so empties or duplicates would merge windows
	const std::map<std::string, std::string>* englishStrings= manager.getRawStrings("en");
	if (englishStrings == nullptr)
	{
		MIKAN_LOG_ERROR("loc-test") << "No English table loaded";
		manager.shutdown();
		return 1;
	}
	{
		std::map<std::string, std::string> titleToKey;
		for (const auto& [key, text] : *englishStrings)
		{
			if (key.rfind("windows.", 0) != 0)
				continue;
			if (text.empty())
			{
				MIKAN_LOG_ERROR("loc-test") << "Empty window title: " << key;
				++errorCount;
				continue;
			}
			const auto [existingIt, bInserted]= titleToKey.emplace(text, key);
			if (!bInserted)
			{
				MIKAN_LOG_ERROR("loc-test") << "Duplicate window title '" << text << "' ("
											<< key << " vs " << existingIt->second << ")";
				++errorCount;
			}
		}
	}

	// Unknown-key fallback: the key itself, never a sentinel
	{
		const char* bogusKey= "bogus.key";
		if (manager.fetchText(bogusKey) != bogusKey)
		{
			MIKAN_LOG_ERROR("loc-test") << "Unknown key did not pass through";
			++errorCount;
		}
	}

	// Glyph coverage: every string of every language (plus the native
	// language names shown in the selector) must render with the baked atlas
	const ImWchar* glyphRanges= ImGuiTheme::getJapaneseGlyphRanges();
	size_t checkedStringCount= 0;
	for (const LocalizationManager::LanguageInfo& info : manager.getSupportedLanguages())
	{
		const std::map<std::string, std::string>* rawStrings= manager.getRawStrings(info.code);
		if (rawStrings == nullptr)
			continue;

		std::vector<std::pair<std::string, std::string>> checkStrings(rawStrings->begin(),
																	  rawStrings->end());
		checkStrings.emplace_back("_meta.nativeName", info.nativeName);

		for (const auto& [key, text] : checkStrings)
		{
			++checkedStringCount;
			size_t byteIndex= 0;
			while (byteIndex < text.size())
			{
				unsigned int codepoint= 0;
				if (!decodeNextCodepoint(text, byteIndex, codepoint))
				{
					MIKAN_LOG_ERROR("loc-test")
						<< info.code << ": '" << key << "' has malformed UTF-8";
					++errorCount;
					break;
				}
				if (codepoint < 0x20)
					continue; // control characters (\n) are not glyphs
				if (!isCodepointInRanges(codepoint, glyphRanges))
				{
					char codepointHex[16];
					snprintf(codepointHex, sizeof(codepointHex), "U+%04X", codepoint);
					MIKAN_LOG_ERROR("loc-test")
						<< info.code << ": '" << key << "' uses codepoint " << codepointHex
						<< " outside the baked glyph ranges";
					++errorCount;
				}
			}
		}
	}

	const size_t languageCount= manager.getSupportedLanguages().size();
	const size_t keyCount= englishStrings->size();
	manager.shutdown();

	if (errorCount > 0)
	{
		MIKAN_LOG_ERROR("loc-test") << errorCount << " error(s)";
		return 1;
	}

	MIKAN_LOG_INFO("loc-test") << "Localization tables OK: " << languageCount << " language(s), "
							   << keyCount << " key(s), " << checkedStringCount
							   << " string(s) glyph-checked";
	return 0;
}

MIKAN_REGISTER_TEST("--loc-test",
					"Validates the localization tables: key parity, printf specifiers, window IDs, glyph coverage",
					eTestCategory::SelfTest, runLocalizationTest);
