#pragma once

#include <string>

// Call-site helpers over LocalizationManager, safe without one (headless
// tools): every helper degrades to returning the key itself.
//
// Usage per string kind:
//	locText("section.key")        Text/TextWrapped/tooltips and validated
//	                              printf format strings ("...Fmt" keys)
//	locLabel("section.key")       any interactive widget label (Button,
//	                              Checkbox, MenuItem, combo, slider): the ID
//	                              is the key, not the translation
//	locWindowTitle("windows.key") ImGui::Begin / popup titles and every
//	                              by-name window reference (DockBuilder,
//	                              SetWindowFocus, OpenPopup)
//	locFormat("section.keyFmt")   printf-expansion into a std::string, for
//	                              labels assembled at runtime
const char* locText(const char* key);
const char* locLabel(const char* key);
const char* locWindowTitle(const char* key);
std::string locFormat(const char* key, ...);
