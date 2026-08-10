#pragma once

#include <array>

#include "glm/ext/quaternion_float.hpp"
#include "glm/ext/vector_float2.hpp"
#include "glm/ext/vector_float3.hpp"
#include "nlohmann/json.hpp"

#include "HandFusion.h" // FusionDiagnostics
#include "TrackingTypes.h"

// Shared JSON (de)serialization for tracking types, used by the diagnostic
// dump, the tracking recorder/replay, and the headless replay CLI. The toJson
// side preserves the dump.json schema exactly; the fromJson side is tolerant
// of nulls (nlohmann emits null for non-finite floats) and reads them as 0.

namespace TrackingJson
{
nlohmann::json vec2ToJson(const glm::vec2& v);
nlohmann::json vec3ToJson(const glm::vec3& v);
nlohmann::json quatToJson(const glm::quat& q); // xyzw, matching the OSC schema
const char* sideName(int side);
nlohmann::json fingersToJson(const std::array<FingerAngles, FINGER_COUNT>& fingers);
nlohmann::json landmarksToJson(const std::array<glm::vec3, HAND_LANDMARK_COUNT>& points);
nlohmann::json imageQualityToJson(const HandImageQuality& quality);
nlohmann::json fusionDiagnosticsToJson(const FusionDiagnostics& diagnostics);

// null-tolerant scalar read: null (nlohmann's encoding of NaN/Inf) -> 0
float floatFromJson(const nlohmann::json& value);

glm::vec2 vec2FromJson(const nlohmann::json& j);
glm::vec3 vec3FromJson(const nlohmann::json& j);
glm::quat quatFromJson(const nlohmann::json& j); // xyzw
void fingersFromJson(const nlohmann::json& j, std::array<FingerAngles, FINGER_COUNT>& outFingers);
void landmarksFromJson(const nlohmann::json& j, std::array<glm::vec3, HAND_LANDMARK_COUNT>& outPoints);
} // namespace TrackingJson
