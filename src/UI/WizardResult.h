#pragma once

// How a calibration wizard's last run ended. None while a run is active (or
// before the first run). Completed means the accept path ran and results were
// applied to the config; every other close is Cancelled. The setup flow
// (guided new-project chain) branches on this; manual wizard runs ignore it.
enum class eWizardResult
{
	None,
	Completed,
	Cancelled,
};
