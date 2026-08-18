#pragma once

// The app's ImGui look: a dark pastel palette over StyleColorsDark, rounded
// metrics, and the Mochiy Pop One UI font (with Japanese glyph coverage).
// Both are applied once in App::startup, after ImGui::CreateContext and
// before the first frame.
namespace ImGuiTheme
{
void applyStyle();
void loadFonts();
} // namespace ImGuiTheme
