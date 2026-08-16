#pragma once

// Which wire format the OSC streamer speaks. The two are mutually exclusive:
// they describe the same pose in incompatible terms (absolute world transforms
// plus confidences vs parent-relative avatar bones), so a client would have to
// pick one anyway, and sending both doubles the bundle for nobody's benefit.
//
// Lives in its own header so the config layer and the streamer can both name
// the mode without the config layer pulling in the socket and encoder.
enum class eOscOutputMode : int
{
	// The native schema: /mikan/* world-space poses, finger angles and
	// confidences. Documented in OscStreamer.h.
	Mikan= 0,
	// VMC protocol (https://protocol.vmc.info): /VMC/Ext/Bone/Pos humanoid
	// bones in Unity convention, for VMC receivers such as VMC4UE.
	Vmc= 1,
	Count= 2,
};

inline const char* oscOutputModeName(eOscOutputMode mode)
{
	switch (mode)
	{
	case eOscOutputMode::Mikan:
		return "Mikan";
	case eOscOutputMode::Vmc:
		return "VMC";
	default:
		return "Unknown";
	}
}
