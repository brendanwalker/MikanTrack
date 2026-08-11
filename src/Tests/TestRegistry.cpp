#include "TestRegistry.h"

#include <cstdio>

#include "Logger.h"

namespace
{
// Function-local storage, not a file-scope global: registrations arrive from
// static initializers in other translation units, and their order relative to
// a file-scope container's construction is not defined.
std::vector<TestRegistration>& getMutableRegistry()
{
	static std::vector<TestRegistration> registry;
	return registry;
}

// "--test-fusion" -> "test-fusion.log"
std::string makeLogFilename(const char* flag)
{
	std::string name= flag != nullptr ? flag : "test";
	while (!name.empty() && name[0] == '-')
		name.erase(name.begin());
	return name + ".log";
}

const char* categoryName(eTestCategory category)
{
	switch (category)
	{
	case eTestCategory::Hardware: return "hardware required";
	case eTestCategory::Tool: return "tools (take a file argument)";
	default: return "self-tests";
	}
}
} // namespace

void TestRegistry::add(const TestRegistration& registration)
{
	getMutableRegistry().push_back(registration);
}

const std::vector<TestRegistration>& TestRegistry::getAll()
{
	return getMutableRegistry();
}

bool TestRegistry::tryRun(int argc, char** argv, int argIndex, int& outExitCode)
{
	if (argIndex < 0 || argIndex >= argc)
		return false;

	const std::string flag= argv[argIndex];
	for (const TestRegistration& registration : getAll())
	{
		if (registration.function == nullptr || flag != registration.flag)
			continue;

		TestArgs args;
		for (int nextIndex= argIndex + 1; nextIndex < argc; ++nextIndex)
			args.push_back(argv[nextIndex]);

		LoggerSettings loggerSettings= {};
		loggerSettings.min_log_level= LogSeverityLevel::info;
		loggerSettings.log_filename= makeLogFilename(registration.flag);
		loggerSettings.enable_console= true;
		log_init(loggerSettings);

		outExitCode= registration.function(args);

		log_dispose();
		return true;
	}

	return false;
}

void TestRegistry::printAvailable()
{
	const eTestCategory categories[]= {eTestCategory::SelfTest, eTestCategory::Hardware,
									   eTestCategory::Tool};
	for (eTestCategory category : categories)
	{
		printf("\n%s:\n", categoryName(category));
		for (const TestRegistration& registration : getAll())
		{
			if (registration.category != category)
				continue;
			printf("  %-22s %s\n", registration.flag,
				   registration.description != nullptr ? registration.description : "");
		}
	}
	printf("\n");
}
