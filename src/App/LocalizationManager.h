#pragma once

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

class GlobalSettings;

// UTF-8 UI string tables, one JSON file per language under
// resources/localization/ (en.json, ja.json, ...). API shape follows
// MikanXR's LocalizationManager; the storage and fallback behavior differ
// (see NOTICE.md).
//
// Keys are flat "section.key" strings. Every language is loaded once at
// startup and validated against English: missing keys, orphan keys, printf
// specifier mismatches, and embedded "##" all warn and fall back to the
// English text, so a bad translation degrades instead of crashing an ImGui
// format call or colliding a widget ID. Fetches return pointers into tables
// that live for the process (setLanguage only flips the active table), so
// per-frame ImGui use is safe.
//
// Main thread only, like the rest of the UI.
class LocalizationManager
{
public:
	struct LanguageInfo
	{
		std::string code;       // "en", "ja" (also the filename stem)
		std::string nativeName; // "English", "日本語"
	};

	// globalSettings may be null (headless tools): no persistence, language
	// resolves OS -> "en"
	bool startup(GlobalSettings* globalSettings);
	void shutdown();

	const std::string& getLanguage() const { return m_currentLanguageCode; }
	std::vector<LanguageInfo> getSupportedLanguages() const;
	bool isLanguageSupported(const std::string& langCode) const;
	// Live switch: flips the active table and persists appLanguage. ImGui
	// refetches every frame, so the UI changes immediately.
	bool setLanguage(const std::string& langCode);

	// "section.key" -> localized UTF-8. Unknown key returns the key pointer
	// itself (a passthrough, never a sentinel).
	const char* fetchText(const char* key) const;
	// Localized text + "##" + key: a widget label whose ImGui ID stays stable
	// and collision-free across languages
	const char* fetchLabel(const char* key) const;
	// Localized text + "###" + English text: a window/popup title whose ImGui
	// ID equals the English title, keeping imgui.ini layouts and by-name
	// window references working in every language
	const char* fetchWindowTitle(const char* key) const;

	// Load-time validation results, consumed by --loc-test
	const std::vector<std::string>& getLoadWarnings() const { return m_loadWarnings; }
	// Raw key -> text of one loaded language (pre-backfill texts, post-drop
	// keys), for test introspection. Returns nullptr for an unknown code.
	const std::map<std::string, std::string>* getRawStrings(const std::string& langCode) const;

	static LocalizationManager* getInstance() { return s_instance; }

private:
	struct StringEntry
	{
		std::string text;        // localized UTF-8
		std::string label;       // text + "##" + key
		std::string windowTitle; // text + "###" + English text
	};

	struct Language
	{
		LanguageInfo info;
		std::map<std::string, StringEntry, std::less<>> entries; // "section.key"
		std::map<std::string, std::string> rawStrings;           // pre-backfill, for tests
	};

	bool loadLanguageFile(const std::filesystem::path& path, Language& outLanguage);
	// Fills gaps and replaces invalid translations from English, then
	// precomputes the label/windowTitle variants
	void finalizeLanguage(Language& language);
	std::string resolveStartupLanguage() const;
	void addLoadWarning(const std::string& warning);

	std::map<std::string, Language> m_languages; // by code
	Language* m_currentLanguage= nullptr;
	Language* m_english= nullptr;
	std::string m_currentLanguageCode;
	GlobalSettings* m_globalSettings= nullptr;
	std::vector<std::string> m_loadWarnings;
	mutable std::set<std::string> m_warnedMissingKeys;

	static LocalizationManager* s_instance;
};
