#pragma once

#include "CalibrationPatternFinder.h"

#include <filesystem>

// Renders a printable ChArUco board image and writes it to a PNG file.
// Adapted from MikanXR's MarkerObjectSystem::printMarker (minus the libharu PDF output).
// pixelsPerMM controls the output resolution (e.g. 11.811f ~= 300 DPI).
bool generateCharucoBoardPng(const std::filesystem::path& pngPath, int charucoCols, int charucoRows,
							 float charucoSquareLengthMM, float charucoMarkerLengthMM,
							 eCharucoDictionaryType charucoDictionaryType, float pixelsPerMM);

// Renders a printable ArUco marker image and writes it to a PNG file.
// Adapted from MikanXR's MarkerComponent::printMarker (minus the libharu PDF output).
bool generateArucoMarkerPng(const std::filesystem::path& pngPath, int arucoId,
							eCharucoDictionaryType arucoDictionaryType, int markerSizePx);
