#pragma once

// Drains the GL error queue, logging each error via MIKAN_LOG_ERROR with the given label.
// Returns true if any GL error was found (i.e. "did an error occur?").
bool checkGlError(const char* label);
