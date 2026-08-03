#pragma once

#include <array>
#include <cstdint>

#include "glm/ext/quaternion_float.hpp"

#include "ImuService.h" // MountingCaptureResult

class AppConfig;
class VisionThread;
struct TrackingFrameResult;

// Wrist IMU mounting calibration wizard.
//
// The mounting rotation needs two things measured in two different ways, and
// they cannot be gathered at the same moment - which is why this is a wizard
// and not a button:
//
//  1. TWIST. Pronating/supinating the forearm rotates it about its long axis
//     and nothing else, so the dominant axis of the angular-velocity scatter
//     IS the forearm axis. This is the part the elbow rides on, and it is
//     measured over seconds of motion rather than one instant.
//  2. A HELD POSE. Motion cannot observe roll about that axis; a straight
//     wrist supplies it for free, because the forearm frame is DEFINED as the
//     palm frame at a neutral wrist.
//
// Doing (2) alone is what the earlier single-button flow tried, and it had to
// land all three degrees of freedom at one instant; in practice it got the
// arm axis wrong by ~60 deg and the error was invisible until the elbow
// started sweeping a cone.
class MountingWizard
{
public:
	enum class eState
	{
		VerifyDevices,
		CalibrateBias,
		TwistForearms,
		HoldStraight,
		Review,
	};

	MountingWizard(AppConfig* config, VisionThread* visionThread);

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }

	// Returns false once the wizard wants to close
	bool update(float deltaSeconds, const TrackingFrameResult& fusedResult);

private:
	// A side takes part only if its controller is actually usable; a user with
	// one controller should not be blocked by the other side never arriving
	bool isSideParticipating(int sideIndex) const;
	void beginTwistStage();

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifyDevices;
	bool m_bActive= false;
	bool m_bWantsClose= false;

	bool m_bParticipating[2]= {false, false};
	// Latched once a side's twist clears every bar, so that easing off the
	// motion (as everyone does when reading the next instruction) doesn't
	// un-earn it
	bool m_bTwistReady[2]= {false, false};

	// The motion reset is serviced on the vision thread, so the status read
	// right after requesting one still describes the PREVIOUS session. Latching
	// on that is what let the twist stage complete instantly. Wait for the
	// epoch to move before believing anything.
	uint32_t m_epochAtReset= 0;
	bool m_bWaitingForMotionReset= false;

	float m_holdCountdown= 0.f;
	bool m_bCaptureRequested= false;

	// Review-stage outcome
	bool m_bAccepted[2]= {false, false};
	MountingCaptureResult m_captured[2];
};
