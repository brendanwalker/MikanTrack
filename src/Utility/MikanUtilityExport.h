#pragma once

// MikanTrack compiles everything statically, so the DLL export macros from
// MikanXR's MikanUtility library collapse to nothing here.
// MIKAN_UTILITY_FUNC is used function-style: MIKAN_UTILITY_FUNC(bool) foo();
#define MIKAN_UTILITY_CLASS
#define MIKAN_UTILITY_FUNC(rval) rval
