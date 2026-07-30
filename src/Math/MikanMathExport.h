#pragma once

// MikanMediaPipe compiles everything statically — the DLL export macros from
// MikanXR's MikanMath library collapse to nothing here.
#define MIKAN_MATH_CLASS
#define MIKAN_MATH_FUNC(rval) rval
