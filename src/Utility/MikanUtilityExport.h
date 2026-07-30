#pragma once

// MikanMediaPipe compiles everything statically — the DLL export macros from
// MikanXR's MikanUtility library collapse to nothing here.
#define MIKAN_UTILITY_CLASS
#define MIKAN_UTILITY_FUNC
