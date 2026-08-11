#include <string>

#include "App.h"
#include "TestRegistry.h"

#ifdef _WIN32
#include <windows.h>
#endif

// Entry point and nothing else. The command-line self-tests and headless
// diagnostic tools live one-per-file in src/Tests and register themselves with
// the TestRegistry, so this file does not grow when one is added.
static int runApp(int argc, char** argv)
{
	for (int argIndex= 1; argIndex < argc; ++argIndex)
	{
		if (std::string(argv[argIndex]) == "--list-tests")
		{
			TestRegistry::printAvailable();
			return 0;
		}

		int exitCode= 0;
		if (TestRegistry::tryRun(argc, argv, argIndex, exitCode))
			return exitCode;
	}

	App app;
	return app.exec(argc, argv);
}

#if defined(_WIN32) && !defined(_CONSOLE)
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
	return runApp(__argc, __argv);
}
#else
int main(int argc, char** argv)
{
	return runApp(argc, argv);
}
#endif
