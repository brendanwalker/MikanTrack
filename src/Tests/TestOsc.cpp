#include "TestCommon.h"

static int runOscTest(const TestArgs& args)
{
	const bool bSuccess= runOscWriterSelfTest();

	return bSuccess ? 0 : 1;
}

MIKAN_REGISTER_TEST("--selftest", "OSC writer packet round-trip", eTestCategory::SelfTest, runOscTest);
