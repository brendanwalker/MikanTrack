#pragma once

#include <string>
#include <vector>

// Registry for the command-line self-tests and headless diagnostic tools.
// Each one lives in its own file in this folder and registers itself with
// MIKAN_REGISTER_TEST, so main.cpp only asks the registry whether a flag
// belongs to a command and never grows when one is added.
//
// The runner owns the boilerplate every command used to repeat: it points the
// logger at "<flag without the dashes>.log" with console output on, runs the
// command, tears the logger down and returns the exit code (0 = pass).

// Whatever followed the command's own flag on the command line
using TestArgs= std::vector<std::string>;
using TestFunction= int (*)(const TestArgs& args);

enum class eTestCategory
{
	// Deterministic, no hardware, no input files - safe to run as a batch
	SelfTest,
	// Needs a camera or controller physically connected to mean anything
	Hardware,
	// Headless diagnostic that operates on a file the user names
	Tool,
};

struct TestRegistration
{
	const char* flag= nullptr;
	const char* description= nullptr;
	eTestCategory category= eTestCategory::SelfTest;
	TestFunction function= nullptr;
};

namespace TestRegistry
{
// Called from the static initializers the registration macro creates
void add(const TestRegistration& registration);
const std::vector<TestRegistration>& getAll();

// Runs the command named by argv[argIndex] with the arguments after it,
// including logger setup and teardown. Returns false when that flag is not a
// registered command, so the caller keeps parsing.
bool tryRun(int argc, char** argv, int argIndex, int& outExitCode);

// Prints every registered command, grouped by category (--list-tests)
void printAvailable();
} // namespace TestRegistry

// Registers one command. Put it directly below the command's function so the
// flag, the description and the code live together in one file.
#define MIKAN_REGISTER_TEST(flag, description, category, function)                     \
	namespace                                                                          \
	{                                                                                  \
	struct TestRegistrar_##function                                                    \
	{                                                                                  \
		TestRegistrar_##function() { TestRegistry::add({flag, description, category, function}); } \
	} g_testRegistrar_##function;                                                      \
	}
