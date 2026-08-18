#pragma once

#include <array>
#include <cstdint>

#include "glm/ext/quaternion_float.hpp"

#include "ImuService.h" // MountingCaptureResult
#include "WizardResult.h"

class AppConfig;
class VisionThread;
struct TrackingFrameResult;

// Wrist IMU mounting calibration wizard.
//
// Two motions, and the geometry comes entirely from the IMU:
//
//  1. TWIST (pronation/supination) turns the forearm about its long axis and
//     nothing else, so the dominant axis of that window IS the forearm axis,
//     measured in the sensor's own frame. This is the only degree of freedom
//     the elbow estimate consumes.
//  2. CURL (elbow flexion) turns it about the elbow hinge, near perpendicular
//     to the long axis. Two independent axes close the frame, which fixes the
//     roll that a twist alone cannot see.
//
// Vision decides exactly one bit - which side of the hinge axis the palm is
// on - and nothing else. That is a choice between two candidates 180 degrees
// apart, so it survives a palm estimate far too noisy to BE a mounting.
//
// There is deliberately NO held pose. Earlier versions took the roll, and the
// arm axis SIGN, from vision: first from a single instant, then from an
// average over the twist. Both failed the same way, because a wrist that does
// not hold still during the capture corrupts the average just as thoroughly
// as it corrupts one frame - measured at 54 degrees of spread on a left arm
// whose calibration was reliably garbage.
class MountingWizard
{
public:
	enum class eState
	{
		VerifyDevices,
		CalibrateBias,
		TwistForearms,
		CurlElbows,
		Review,
	};

	MountingWizard(AppConfig* config, VisionThread* visionThread);

	void enter();
	void exit();
	bool isActive() const { return m_bActive; }
	eWizardResult getResult() const { return m_result; }

	// Returns false once the wizard wants to close
	bool update(float deltaSeconds, const TrackingFrameResult& fusedResult);

private:
	// A side takes part only if its controller is actually usable; a user with
	// one controller should not be blocked by the other side never arriving
	bool isSideParticipating(int sideIndex) const;
	// Starts one motion stage: begins recording it, and waits for the reset to
	// land before believing anything the status says
	void beginMotionStage(eState state, eMountingMotion motion);
	// Shared by both motion stages - they watch the same three numbers, since
	// "enough back-and-forth rotation about one axis" is the bar either way
	bool drawMotionStage(const ImuSideStatus status[2], bool bReady[2], const char* elbowHint);

	AppConfig* m_config;
	VisionThread* m_visionThread;

	eState m_state= eState::VerifyDevices;
	bool m_bActive= false;
	bool m_bWantsClose= false;
	eWizardResult m_result= eWizardResult::None;

	bool m_bParticipating[2]= {false, false};
	// Latched once a side's motion clears every bar, so that easing off (as
	// everyone does when reading the next instruction) doesn't un-earn it
	bool m_bTwistReady[2]= {false, false};
	bool m_bCurlReady[2]= {false, false};

	// The motion reset is serviced on the vision thread, so the status read
	// right after requesting one still describes the PREVIOUS session. Latching
	// on that is what let the twist stage complete instantly. Wait for the
	// epoch to move before believing anything.
	uint32_t m_epochAtReset= 0;
	bool m_bWaitingForMotionReset= false;

	bool m_bCaptureRequested= false;

	// Review-stage outcome
	bool m_bAccepted[2]= {false, false};
	MountingCaptureResult m_captured[2];
};
