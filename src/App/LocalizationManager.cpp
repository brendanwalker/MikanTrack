#include "LocalizationManager.h"

#include <cstdarg>
#include <cstdio>
#include <fstream>

#include <windows.h>

#include "nlohmann/json.hpp"

#include "GlobalSettings.h"
#include "Logger.h"
#include "PathUtils.h"
#include "StringUtils.h"

using json= nlohmann::json;

static const char* k_defaultLanguageCode= "en";

LocalizationManager* LocalizationManager::s_instance= nullptr;

// Extracts the ordered printf conversion sequence of a format string as
// normalized tokens ("d", "ld", "f", "s", ...; '*' width/precision emit an
// extra "d"). Returns false on a malformed specifier.
static bool parsePrintfSpecs(const std::string& text, std::vector<std::string>& outSpecs)
{
	const std::string flagChars= "-+ #0";
	const std::string convChars= "diouxXeEfFgGaAcsp";

	for (size_t i= 0; i < text.size(); ++i)
	{
		if (text[i] != '%')
			continue;

		++i;
		if (i >= text.size())
			return false;
		if (text[i] == '%')
			continue;

		while (i < text.size() && flagChars.find(text[i]) != std::string::npos)
			++i;
		if (i < text.size() && text[i] == '*')
		{
			outSpecs.push_back("d");
			++i;
		}
		else
		{
			while (i < text.size() && isdigit((unsigned char)text[i]))
				++i;
		}
		if (i < text.size() && text[i] == '.')
		{
			++i;
			if (i < text.size() && text[i] == '*')
			{
				outSpecs.push_back("d");
				++i;
			}
			else
			{
				while (i < text.size() && isdigit((unsigned char)text[i]))
					++i;
			}
		}

		std::string lengthModifier;
		while (i < text.size() && std::string("hljztL").find(text[i]) != std::string::npos)
		{
			lengthModifier+= text[i];
			++i;
		}

		if (i >= text.size() || convChars.find(text[i]) == std::string::npos)
			return false;
		outSpecs.push_back(lengthModifier + text[i]);
	}

	return true;
}

bool LocalizationManager::startup(GlobalSettings* globalSettings)
{
	m_globalSettings= globalSettings;
	m_loadWarnings.clear();
	m_languages.clear();
	m_currentLanguage= nullptr;
	m_english= nullptr;

	const std::filesystem::path localizationDir=
		PathUtils::getResourceDirectory() / "localization";

	// English first: it is the fallback source every other language is
	// validated against
	{
		Language english;
		if (!loadLanguageFile(localizationDir / "en.json", english))
		{
			MIKAN_LOG_ERROR("LocalizationManager::startup")
				<< "No usable en.json under " << localizationDir
				<< "; every fetch will return its key";
			s_instance= this;
			return false;
		}
		m_languages["en"]= std::move(english);
		m_english= &m_languages["en"];
	}

	if (std::filesystem::is_directory(localizationDir))
	{
		for (const std::string& filename :
			 PathUtils::listFilenamesInDirectory(localizationDir, ".json"))
		{
			const std::string code= std::filesystem::path(filename).stem().string();
			if (code == "en")
				continue;

			Language language;
			if (loadLanguageFile(localizationDir / filename, language))
				m_languages[code]= std::move(language);
		}
	}

	for (auto& [code, language] : m_languages)
		finalizeLanguage(language);

	m_currentLanguageCode= resolveStartupLanguage();
	m_currentLanguage= &m_languages[m_currentLanguageCode];

	MIKAN_LOG_INFO("LocalizationManager::startup")
		<< m_languages.size() << " language(s) loaded, active: " << m_currentLanguageCode;

	s_instance= this;
	return true;
}

void LocalizationManager::shutdown()
{
	if (s_instance == this)
		s_instance= nullptr;
	m_languages.clear();
	m_currentLanguage= nullptr;
	m_english= nullptr;
}

bool LocalizationManager::loadLanguageFile(const std::filesystem::path& path, Language& outLanguage)
{
	std::ifstream file(path);
	if (!file.is_open())
		return false;

	json j;
	try
	{
		file >> j;
	}
	catch (const std::exception& e)
	{
		addLoadWarning("Failed to parse " + path.filename().string() + ": " + e.what());
		return false;
	}

	const std::string code= path.stem().string();
	outLanguage.info.code= code;
	outLanguage.info.nativeName= code;
	if (j.contains("_meta") && j["_meta"].is_object())
	{
		const json& meta= j["_meta"];
		outLanguage.info.nativeName= meta.value("nativeName", code);
		if (meta.value("code", code) != code)
			addLoadWarning(path.filename().string() + ": _meta.code does not match the filename");
	}
	else
	{
		addLoadWarning(path.filename().string() + ": missing _meta section");
	}

	for (const auto& [sectionName, section] : j.items())
	{
		if (!sectionName.empty() && sectionName[0] == '_')
			continue;
		if (!section.is_object())
		{
			addLoadWarning(path.filename().string() + ": section '" + sectionName +
						   "' is not an object");
			continue;
		}

		for (const auto& [stringKey, value] : section.items())
		{
			if (!value.is_string())
			{
				addLoadWarning(path.filename().string() + ": " + sectionName + "." + stringKey +
							   " is not a string");
				continue;
			}
			outLanguage.rawStrings[sectionName + "." + stringKey]= value.get<std::string>();
		}
	}

	return true;
}

void LocalizationManager::finalizeLanguage(Language& language)
{
	const bool bIsEnglish= &language == m_english;

	// Orphans: keys English does not define never made it into code, so they
	// are dead weight (and usually typos)
	if (!bIsEnglish)
	{
		for (const auto& [key, text] : language.rawStrings)
		{
			if (m_english->rawStrings.find(key) == m_english->rawStrings.end())
				addLoadWarning(language.info.code + ": orphan key '" + key + "' (not in en)");
		}
	}

	for (const auto& [key, englishText] : m_english->rawStrings)
	{
		std::string text= englishText;

		if (!bIsEnglish)
		{
			const auto it= language.rawStrings.find(key);
			if (it == language.rawStrings.end())
			{
				addLoadWarning(language.info.code + ": missing key '" + key + "'");
			}
			else
			{
				const std::string& translated= it->second;
				std::vector<std::string> englishSpecs, translatedSpecs;
				const bool bEnglishOk= parsePrintfSpecs(englishText, englishSpecs);
				const bool bTranslatedOk= parsePrintfSpecs(translated, translatedSpecs);

				if (translated.find("##") != std::string::npos)
				{
					addLoadWarning(language.info.code + ": '" + key +
								   "' contains ## (reserved for ImGui IDs)");
				}
				else if (!bTranslatedOk || (bEnglishOk && englishSpecs != translatedSpecs))
				{
					addLoadWarning(language.info.code + ": '" + key +
								   "' printf specifiers do not match en");
				}
				else
				{
					text= translated;
				}
			}
		}
		else
		{
			std::vector<std::string> specs;
			if (englishText.find("##") != std::string::npos)
				addLoadWarning("en: '" + key + "' contains ## (reserved for ImGui IDs)");
			if (!parsePrintfSpecs(englishText, specs))
				addLoadWarning("en: '" + key + "' has a malformed printf specifier");
		}

		StringEntry entry;
		entry.text= text;
		entry.label= text + "##" + key;
		entry.windowTitle= text + "###" + englishText;
		language.entries[key]= std::move(entry);
	}
}

std::string LocalizationManager::resolveStartupLanguage() const
{
	if (m_globalSettings != nullptr && !m_globalSettings->appLanguage.empty() &&
		isLanguageSupported(m_globalSettings->appLanguage))
	{
		return m_globalSettings->appLanguage;
	}

	// OS language, primary subtag only ("ja-JP" -> "ja")
	wchar_t localeName[LOCALE_NAME_MAX_LENGTH]= {};
	if (GetUserDefaultLocaleName(localeName, LOCALE_NAME_MAX_LENGTH) > 0)
	{
		char localeUtf8[LOCALE_NAME_MAX_LENGTH * 4]= {};
		if (StringUtils::convertWcsToMbs(localeName, localeUtf8, sizeof(localeUtf8)))
		{
			std::string primaryTag(localeUtf8);
			const size_t dashPos= primaryTag.find('-');
			if (dashPos != std::string::npos)
				primaryTag= primaryTag.substr(0, dashPos);
			if (isLanguageSupported(primaryTag))
				return primaryTag;
		}
	}

	return k_defaultLanguageCode;
}

std::vector<LocalizationManager::LanguageInfo> LocalizationManager::getSupportedLanguages() const
{
	// English first, the rest in code order (map order gives en first only by
	// luck of the alphabet, so be explicit)
	std::vector<LanguageInfo> languages;
	const auto englishIt= m_languages.find("en");
	if (englishIt != m_languages.end())
		languages.push_back(englishIt->second.info);
	for (const auto& [code, language] : m_languages)
	{
		if (code != "en")
			languages.push_back(language.info);
	}
	return languages;
}

bool LocalizationManager::isLanguageSupported(const std::string& langCode) const
{
	return m_languages.find(langCode) != m_languages.end();
}

bool LocalizationManager::setLanguage(const std::string& langCode)
{
	const auto it= m_languages.find(langCode);
	if (it == m_languages.end())
		return false;
	if (langCode == m_currentLanguageCode)
		return true;

	m_currentLanguage= &it->second;
	m_currentLanguageCode= langCode;

	if (m_globalSettings != nullptr)
	{
		m_globalSettings->appLanguage= langCode;
		m_globalSettings->save();
	}

	MIKAN_LOG_INFO("LocalizationManager::setLanguage") << "Language set to " << langCode;
	return true;
}

const char* LocalizationManager::fetchText(const char* key) const
{
	if (m_currentLanguage != nullptr)
	{
		const auto it= m_currentLanguage->entries.find(std::string_view(key));
		if (it != m_currentLanguage->entries.end())
			return it->second.text.c_str();
	}

	// Only reachable when the key is absent from en too (a code bug) or
	// nothing loaded: pass the key through so the UI shows something greppable
	if (m_warnedMissingKeys.insert(key).second)
		MIKAN_LOG_WARNING("LocalizationManager::fetchText") << "Unknown string key: " << key;
	return key;
}

const char* LocalizationManager::fetchLabel(const char* key) const
{
	if (m_currentLanguage != nullptr)
	{
		const auto it= m_currentLanguage->entries.find(std::string_view(key));
		if (it != m_currentLanguage->entries.end())
			return it->second.label.c_str();
	}

	if (m_warnedMissingKeys.insert(key).second)
		MIKAN_LOG_WARNING("LocalizationManager::fetchLabel") << "Unknown string key: " << key;
	return key;
}

const char* LocalizationManager::fetchWindowTitle(const char* key) const
{
	if (m_currentLanguage != nullptr)
	{
		const auto it= m_currentLanguage->entries.find(std::string_view(key));
		if (it != m_currentLanguage->entries.end())
			return it->second.windowTitle.c_str();
	}

	if (m_warnedMissingKeys.insert(key).second)
		MIKAN_LOG_WARNING("LocalizationManager::fetchWindowTitle") << "Unknown string key: " << key;
	return key;
}

const std::map<std::string, std::string>* LocalizationManager::getRawStrings(
	const std::string& langCode) const
{
	const auto it= m_languages.find(langCode);
	return it != m_languages.end() ? &it->second.rawStrings : nullptr;
}

void LocalizationManager::addLoadWarning(const std::string& warning)
{
	MIKAN_LOG_WARNING("LocalizationManager") << warning;
	m_loadWarnings.push_back(warning);
}

// -- LocText.h helpers ----
const char* locText(const char* key)
{
	LocalizationManager* manager= LocalizationManager::getInstance();
	return manager != nullptr ? manager->fetchText(key) : key;
}

const char* locLabel(const char* key)
{
	LocalizationManager* manager= LocalizationManager::getInstance();
	return manager != nullptr ? manager->fetchLabel(key) : key;
}

const char* locWindowTitle(const char* key)
{
	LocalizationManager* manager= LocalizationManager::getInstance();
	return manager != nullptr ? manager->fetchWindowTitle(key) : key;
}

std::string locFormat(const char* key, ...)
{
	const char* format= locText(key);

	va_list args;
	va_start(args, key);
	va_list argsCopy;
	va_copy(argsCopy, args);
	const int length= vsnprintf(nullptr, 0, format, argsCopy);
	va_end(argsCopy);

	std::string result;
	if (length > 0)
	{
		result.resize((size_t)length);
		vsnprintf(result.data(), (size_t)length + 1, format, args);
	}
	va_end(args);

	return result;
}
