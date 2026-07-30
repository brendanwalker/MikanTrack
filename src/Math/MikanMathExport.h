#pragma once

#ifndef MIKAN_MATH_CLASS
#if defined(MIKAN_MATH_EXPORTS)
#if defined _WIN32 || defined __CYGWIN__
#define MIKAN_MATH_FUNC(rval) __declspec(dllexport) rval
#define MIKAN_MATH_CLASS __declspec(dllexport)
#else // Not Windows
#if __GNUC__ >= 4
#define MIKAN_MATH_FUNC(rval) __attribute__((visibility("default"))) rval
#define MIKAN_MATH_CLASS __attribute__((visibility("default")))
#else
#define MIKAN_MATH_FUNC(rval) rval
#define MIKAN_MATH_CLASS
#endif
#endif // defined _WIN32 || defined __CYGWIN__
#else
#define MIKAN_MATH_FUNC(rval) rval
#define MIKAN_MATH_CLASS
#endif // MIKAN_MATH_EXPORTS
#endif //! defined(MIKAN_MATH_CLASS)