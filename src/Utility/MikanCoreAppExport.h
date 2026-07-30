#pragma once

// MikanMediaPipe compiles everything statically — the DLL export macros from
// MikanXR's MikanCoreApp library collapse to nothing here.
#define MIKAN_COREAPP_CLASS
#define MIKAN_COREAPP_FUNC
