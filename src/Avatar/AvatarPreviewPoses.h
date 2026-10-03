#pragma once

#include "BodyPoseSolver.h" // BodyDimensions
#include "TrackingTypes.h"

class AvatarSkeleton;

// Canned tracked frames for posing an avatar without cameras: the Avatar
// panel's Rest and Demo previews and the headless render tool. Both take the
// user's lengths to be the avatar's own, so reach maps 1:1.
namespace AvatarPreviewPoses
{
// A frame that reproduces the avatar's rest: each palm on the avatar's rest
// palm frame, zero finger angles, the shoulders on the avatar's upper-arm
// joints, the head facing forward, no measured forearm. With no rig trims
// every bone the retarget places sits at identity.
void makeRestFrame(const AvatarSkeleton& skeleton, TrackingFrameResult& outFrame, BodyDimensions& outUser);

// Hands up in front of the chest, the right hand curled and the left spread,
// the head turned, no measured elbows (so the elbow hints show) and no
// shoulders (so the fixed root places the avatar)
void makeDemoFrame(const AvatarSkeleton& skeleton, TrackingFrameResult& outFrame, BodyDimensions& outUser);
} // namespace AvatarPreviewPoses
