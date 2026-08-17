# Wire protocol

The OSC output contract: transport, the native `/mikan/*` schema, the VMC mode, and the delivery rules both obey. This is the load-bearing seam with client applications, so schema changes ripple: the authoritative bundle layout lives in the `OscStreamer` class comment (`src/Osc/OscStreamer.h`), the human-facing tables in [README.md](../../README.md), and the byte-level self tests decode real encoded packets. See [conventions.md](./conventions.md) for the frames and units the messages are expressed in, [hand-tracking.md](./hand-tracking.md) and [body-pose.md](./body-pose.md) for where the values come from, and [debugging.md](./debugging.md) for diagnosing a receiver that sees nothing.

---

## Transport

- OSC 1.0 bundles over UDP unicast, default target `127.0.0.1:8000` (`OscStreamerConfig` in `src/Osc/OscStreamer.h`). One bundle per frame, rate-limited by `maxRateHz` (default 60, non-positive disables).
- The encoder is hand-rolled (`src/Osc/OscWriter.h`), the socket is Winsock2 (`src/Osc/UdpSocket.h`). `OscStreamer::encodeFrame` produces the exact datagram bytes without touching the socket, which is how `--selftest` and `--test-vmc` verify the wire format against a spec-written decoder rather than a reconstruction.
- Two mutually exclusive wire formats selected by `OscStreamerConfig::outputMode` (`src/Osc/OscOutputMode.h`): the native Mikan schema and the VMC protocol. Each mode keeps its own target port (VMC defaults to its conventional 39539), so switching formats cannot aim a stream at a listener that speaks the other.
- The Mikan format sends one bundle per frame, deliberately unchunked. VMC mode splits a frame into complete bundles of at most `OscStreamer::k_maxDatagramBytes` (1400) each: the full VMC frame is about 3.4 KB, which would both ride IP fragmentation past a 1500-byte MTU and overflow the 2048-byte default receive buffer of Rug.Osc-based receivers, where the packet silently never arrives.
- Windows grants a unicast UDP port to one process at a time. A generic OSC monitor left bound to the port starves the real receiver silently; close it before testing.

## The Mikan schema

Positions are meters in the marker-anchored world frame when extrinsics exist, otherwise OpenCV camera space; `/mikan/info` reports which. Finger angles are DEGREES on the wire (radians everywhere inside the app). Full field-by-field semantics live in the `OscStreamer` class comment; the shape:

- `/mikan/frame` `,iifi` frameId, timestampMs, fps, sendSequence. `sendSequence` increments by exactly one per bundle sent and is the client's loss counter; `frameId` identifies the capture and repeats, skips, and steps backwards with several cameras, so it cannot measure loss.
- `/mikan/hand/{left,right}/tracked` `,iff` tracked flag, presence, confidence.
- `/mikan/hand/{left,right}/elbow` `,ffff` position plus confidence.
- `/mikan/hand/{left,right}/shoulder` `,ffff` position plus confidence.
- `/mikan/hand/{left,right}/palm` `,fffffff` position plus orientation quaternion xyzw.
- `/mikan/hand/{left,right}/forearm` `,ifffffff` valid flag, position, orientation. The origin is the WRIST JOINT, not the palm center: the wrist is the one chain point a consumer cannot recover from the palm transform alone, and +X runs toward the hand so the elbow is one forearm length back along -X.
- `/mikan/hand/{left,right}/fingers` `,ffffffffffffffffffff` per finger thumb to pinky: lateral, proximalBend, intermediateBend, distalBend.
- `/mikan/hand/{left,right}/skeleton` 45 floats at 1 Hz: per finger, base position in the palm frame, phalanx lengths, and the neutral zero-angle direction in the palm frame.
- `/mikan/body/head` `,ffffffff` position, orientation, confidence.
- `/mikan/info` `,ss` conventions string and app version, at most once per second.

Delivery rules the schema is built on:

- **Always-send with confidence-carried validity.** Elbow, shoulder, and head are sent every frame, tracked or not, because a consumer holds the last value for an address that goes silent; confidence 0 means do not use this value. `OscStreamer::resolveElbowOutput`, `resolveShoulderOutput`, and `resolveHeadOutput` are static precisely so the self test exercises the contract without a socket.
- **Withhold below threshold.** A hand whose fused confidence falls below `minConfidence` is reported `tracked=0` and its palm/forearm/fingers/skeleton messages are withheld, so the client blends to a rest pose instead of following a jittering estimate.
- **Dropout hold.** After a dropout, the last good pose keeps streaming for `holdOnDropoutMs` (default 250) with confidence decaying linearly to zero, bridging short dropouts before the client sees `tracked=0` (`OscStreamer::resolveOutputPose`).

## VMC mode

`src/Osc/VmcRetarget.h` documents the receiver model this was written against: a VMC bone transform is the bone's local transform in Unity convention, identity rotation means the avatar's rest pose (VRM-style T-pose rig), and a receiver REPLACES both the rotation and the translation of every bone it is sent. The consequences drive the design:

- Every streamed bone carries a real offset (a zero translation collapses the bone onto its parent), so the avatar takes the measured proportions: shoulder width, upper arm, and forearm from the body config, finger offsets from the calibrated hand skeleton, and the one unmeasurable length, neck to head, as the `vmcHeadOffsetMeters` setting.
- The avatar frame is the world frame unchanged; `/VMC/Ext/Root/Pos` is deliberately identity because this is a desk-anchored upper-body tracker, not a room-scale root.
- Bones streamed, by Unity `HumanBodyBones` name: `Head`, both `Shoulder`/`UpperArm`/`LowerArm`/`Hand`, and all 30 finger bones (`eVmcBone`). Torso, neck, legs, eyes, and jaw are never streamed and stay at the avatar's rest pose, which is also the reference the streamed rotations are measured against.
- The arm chain is streamed whenever the hand is: with no measured elbow the upper arm aims straight at the wrist and the forearm takes the hand's own orientation (a neutral wrist), because an unstreamed arm snaps to the avatar's T-pose while the hand keeps its world orientation, and the whole arm error then surfaces as a spin at the wrist joint (`VmcRetarget.cpp`).
- VMC carries no confidence and no tracked flag, so loss is expressed as stillness: past the dropout hold the last streamed bones freeze (`vmcFreezeOnLoss`, default on) rather than going silent.

## Consumers and change discipline

Known consumers: the MikanTrack Unreal Engine plugin, a separate project that shares this app's name and consumes every `/mikan/*` address (UE-side conversion is the axis swap documented in [README.md](../../README.md)), and VMC receivers (developed against VMC4UE). Changing an address, a type tag, or a field meaning is a breaking protocol change: update the `OscStreamer.h` bundle-layout comment, the README tables, the self tests that decode the encoded bytes (`--selftest` OSC sections, `--test-vmc`), and coordinate the consumer update.
