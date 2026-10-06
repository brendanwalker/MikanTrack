# Wire protocol

The OSC output contract: transport, the VMC protocol MikanTrack speaks, and the delivery rules it obeys. This is the load-bearing seam with client applications, so changes ripple: the authoritative message layout lives in the `OscStreamer` class comment (`src/Osc/OscStreamer.h`), the human-facing table in [README.md](../../README.md), and the byte-level self tests decode real encoded packets. See [conventions.md](./conventions.md) for the frames and units the bones are derived from, [hand-tracking.md](./hand-tracking.md) and [body-pose.md](./body-pose.md) for where the values come from, and [debugging.md](./debugging.md) for diagnosing a receiver that sees nothing.

---

## Transport

- OSC 1.0 bundles over UDP unicast, default target `127.0.0.1:39539`, VMC's conventional port (`OscStreamerConfig` in `src/Osc/OscStreamer.h`). Rate-limited by `maxRateHz` (default 60, non-positive disables). The project file keeps the port under the `vmcPort` key.
- The encoder is hand-rolled (`src/Osc/OscWriter.h`), the socket is Winsock2 (`src/Osc/UdpSocket.h`). `OscStreamer::encodeFrame` produces the exact datagram bytes without touching the socket, which is how `--selftest` and `--test-vmc` verify the wire format against a spec-written decoder rather than a reconstruction.
- A frame is split into complete bundles of at most `OscStreamer::k_maxDatagramBytes` (1400) each: the full frame is about 3.4 KB, which would both ride IP fragmentation past a 1500-byte MTU and overflow the 2048-byte default receive buffer of Rug.Osc-based receivers, where the packet silently never arrives.
- Windows grants a unicast UDP port to one process at a time. A generic OSC monitor left bound to the port starves the real receiver silently; close it before testing.

## The VMC stream

`src/Osc/VmcRetarget.h` documents the receiver model this was written against: a VMC bone transform is the bone's local transform in Unity convention, identity rotation means the avatar's rest pose (VRM-style T-pose rig), and a receiver REPLACES both the rotation and the translation of every bone it is sent. The consequences drive the design:

- Every streamed bone carries a real offset (a zero translation collapses the bone onto its parent), so the avatar takes the measured proportions: shoulder width, upper arm, and forearm from the body config, finger offsets from the calibrated hand skeleton, and the one unmeasurable length, neck to head, as the `vmcHeadOffsetMeters` setting.
- The avatar frame is the world frame unchanged; `/VMC/Ext/Root/Pos` is deliberately identity because this is a desk-anchored upper-body tracker, not a room-scale root.
- Bones streamed, by Unity `HumanBodyBones` name: `Head`, both `Shoulder`/`UpperArm`/`LowerArm`/`Hand`, and all 30 finger bones (`eVmcBone`). Torso, neck, legs, eyes, and jaw are never streamed and stay at the avatar's rest pose, which is also the reference the streamed rotations are measured against.
- The arm chain is streamed whenever the hand is: with no measured elbow the upper arm aims straight at the wrist and the forearm takes the hand's own orientation (a neutral wrist), because an unstreamed arm snaps to the avatar's T-pose while the hand keeps its world orientation, and the whole arm error then surfaces as a spin at the wrist joint (`VmcRetarget.cpp`). Without a shoulder the clavicle and upper arm are omitted.
- With a VRM avatar loaded (see [avatar.md](./avatar.md)) the bones come from retargeting the resolved poses onto that avatar instead: the same bone set, with each local rotation measured against the nearest streamed ancestor and each local position the avatar's own rest offset, so a receiver running the same file keeps the character's proportions. The root stays identity either way.
- With an avatar loaded, `/VMC/Ext/VRM ,sss` announces the file at most once a second: the local path (UTF-8), the avatar's title, and the lowercase hex SHA-256 of the file bytes. The path only means something on the same machine, which is the case it exists for: a local receiver loads the very file the bones were retargeted onto.
- While the phone face stream is live (see [face.md](./face.md)), `/VMC/Ext/Blend/Val ,sf` carries each mapped blendshape (the names are the avatar's own, see the face map in [avatar.md](./avatar.md)), followed by one `/VMC/Ext/Blend/Apply` with no arguments. The frame the stream stops sends every value once more at zero, because a receiver holds the last value of a blendshape that stops arriving. The phone's head rotation also replaces the camera head on the `Head` bone while the stream is live, and a face sample no camera frame carried goes out as a frame of its own.

## Delivery rules

VMC carries no confidence and no tracked flag, so validity is expressed through which bones arrive and whether they move:

- **Withhold below threshold.** A hand whose fused confidence falls below `minConfidence` is treated as lost, so the avatar does not follow a jittering estimate.
- **Dropout hold.** After a dropout, the last good pose keeps streaming for `holdOnDropoutMs` (default 250), bridging short losses before the loss rule applies (`OscStreamer::resolveOutputPose`). The held pose's confidences decay linearly to zero over the window, which the avatar retarget reads when it blends the measured elbow against its hint.
- **Loss is stillness.** Past the hold, the last streamed bones of that hand keep streaming frozen (`vmcFreezeOnLoss`, default on, `OscStreamer::resolveVmcOutputPose`) rather than going silent, since a silent bone drops that arm back to the avatar's T-pose. With the setting off the bones stop. A hand needs a world-anchored palm to be streamed at all, so before extrinsics calibration nothing hand-side is sent.

## Consumers and change discipline

Known consumers are VMC receivers: VRM4U's VMC node (`AnimNode_VrmVMC`) in Unreal Engine, and VMC4UE, which the stream was developed against. Changing an address, a type tag, a bone name, or a field meaning is a breaking protocol change: update the `OscStreamer.h` class comment, the README table, and the self tests that decode the encoded bytes (`--test-vmc`), and check the receivers still bind.
