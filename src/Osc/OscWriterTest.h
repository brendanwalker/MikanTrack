#pragma once

/// Golden-byte self test for the OSC 1.0 encoder (OscWriter.h).
/// Logs pass/fail per case via MIKAN_LOG_INFO/ERROR.
/// @returns true if all test cases passed
bool runOscWriterSelfTest();
