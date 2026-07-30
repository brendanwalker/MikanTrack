#pragma once

// MikanMediaPipe compiles everything statically — the DLL export macros from
// MikanXR's MikanCoreApp library collapse to nothing here.
// MIKAN_COREAPP_FUNC is used function-style: MIKAN_COREAPP_FUNC(void) foo();
#define MIKAN_COREAPP_CLASS
#define MIKAN_COREAPP_FUNC(rval) rval
