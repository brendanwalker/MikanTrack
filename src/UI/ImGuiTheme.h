#pragma once

#include "imgui.h" // ImWchar

// The app's ImGui look: a dark pastel palette over StyleColorsDark, rounded
// metrics, and the Mochiy Pop One UI font (with Japanese glyph coverage).
// Both are applied once in App::startup, after ImGui::CreateContext and
// before the first frame.
namespace ImGuiTheme
{
void applyStyle();
void loadFonts();
// The exact glyph ranges the font atlas bakes ([start,end] pairs, zero
// terminated). Shared with the localization self-test so translated strings
// are validated against what actually renders.
const ImWchar* getJapaneseGlyphRanges();
} // namespace ImGuiTheme
