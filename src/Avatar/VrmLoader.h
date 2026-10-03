#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "AvatarTypes.h"

// Reads a VRM file (glTF 2.0 binary plus the VRM 0.x "VRM" or VRM 1.0
// "VRMC_vrm" extension) into an AvatarModel. Pure CPU work: no GL, no app
// state, so a self test can run it on an in-memory file and a headless tool on
// a path. The glTF core goes through cgltf; the VRM extension JSON is read
// directly for the subset the app uses (humanoid map, meta, the MToon
// material fields).
namespace VrmLoader
{
struct LoadResult
{
	std::shared_ptr<AvatarModel> model; // null on failure
	std::string error;                  // human-readable, empty on success
	std::vector<std::string> warnings;  // non-fatal oddities, also logged
};

// Loads from a file on disk. The path is remembered on the model (UTF-8).
LoadResult loadFile(const std::filesystem::path& path);

// Loads from the bytes of a .vrm/.glb file already in memory. External
// buffer or image URIs cannot be resolved here and fail the load; VRM files
// embed everything in the GLB, so this is the path the file loader takes too.
LoadResult loadMemory(const uint8_t* bytes, size_t byteCount);

// Human-readable name of a VRM generation ("VRM 0.x" / "VRM 1.0")
const char* versionName(eVrmVersion version);
} // namespace VrmLoader
