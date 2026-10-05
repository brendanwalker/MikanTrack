# Face stream (iFacialMocap)

MikanTrack receives ARKit face data from the iFacialMocap iPhone app. The app streams 52 blendshape weights plus head and eye rotation as UDP text. `FaceService` (`src/Face/FaceService.h/.cpp`) owns the socket, the handshake, and the parser. It publishes the newest sample as a `FaceSample` (`src/Face/FaceTypes.h`).

## Protocol

The app streams to a machine that asks it to. The ask is a datagram carrying a fixed ASCII text starting with `iFacialMocap_` and no terminator, sent to the phone address on the stream port. The service sends it as a burst of five copies when its socket binds or the phone address changes, and again every 3 seconds while no stream datagram has arrived for 3 seconds. The default phone address is the broadcast address `255.255.255.255`, which reaches a phone anywhere on the LAN.

The socket is bound to the stream port with `SO_BROADCAST`. A broadcast handshake loops back to the same port on this machine, so any datagram starting with `iFacialMocap_` is dropped as our own.

A stream datagram is one frame of text, well under the 4096-byte receive capacity:

- Blendshape section before the first `=`: `|`-separated entries `name-value` (v1) or `name&value` (v2, allows negatives). Values are percent and are stored as value/100.
- Bone section after the `=`: `|`-separated entries `name#numbers`. `head#rx,ry,rz,px,py,pz` needs all six values. `rightEye#rx,ry,rz` and `leftEye#rx,ry,rz` need three.

Two status fields share the blendshape section. `trackingStatus` is the phone's face-tracked flag (1 tracked, 0 lost) and becomes `FaceSample::faceTracked`. `hapihapi` is an app field with nothing a face needs and is skipped. While the phone reports the face lost its values are stale, so the face is not applied: the blendshapes relax once and the camera head stands. Blendshape names come from the ARKit table in `FaceTypes.cpp`. The phone's `_L` and `_R` suffixes fold to `Left` and `Right`. Unknown names are counted and skipped. A datagram where no entry parsed counts as a parse failure and changes nothing. Values a datagram omits keep their previous state.

Samples stay in the phone's convention: rotations in degrees, positions in the app's units. The head rotation enters MikanTrack's head frame at one conversion point, `faceHeadDeltaFromPhoneEuler` (`src/Face/FaceHead.h`, see [conventions.md](./conventions.md)).

## Service and threading

`App` owns the service like `ImuService`: created in `App::startup`, `startup()` on project activation, `shutdown()` wherever the IMU service shuts down. `App::applyFaceConfig()` maps `AppConfig::face` to `FaceServiceConfig` and calls `setConfig`, which binds or releases the socket on a port or enable change. `App::getFaceService()` gives the UI its handle.

The service has no thread of its own. The vision thread calls `update(steadyNowMs())` once per iteration, in its own `Face` hitch-watchdog phase right after the capture stage. `update` drains every waiting datagram and the last one wins, then runs the handshake schedule. One mutex guards all state, so `setConfig`, `getStatus`, and `getLatestSample` are safe from any thread. A sample counts as streaming while its datagram is younger than 0.5 seconds. `timestampMs` is the `update` time on the shared `steadyNowMs()` clock, the same clock camera frames use.

## Output

`VisionThread::applyFaceToOutput` puts the live sample on every output frame just before the OSC send:

- The 52 blendshapes go on `TrackingFrameResult::face`. In VMC mode the streamer sends them as `/VMC/Ext/Blend/Val` under the avatar's names (the face map in [avatar.md](./avatar.md)), then `/VMC/Ext/Blend/Apply`. The Mikan format does not carry them. The published result carries them too, so the 3D scene's avatar shows the face through its morph targets in either format.

- The phone's head rotation replaces the camera head's orientation while the stream is live, with the head marked valid at full confidence. The phone measures the head far better than the body-pose stage. The position stays the camera head's, since the phone's is relative to the phone.

The phone runs faster than the cameras and keeps streaming when they stop, so a sample no camera frame carried goes out on its own (`runFaceOnlyOutput`). That frame repeats the last output with the new face and the sample's own timestamp. Once the last camera frame is older than the dropout window, the repeated hands go out untracked, so the streamer's hold and freeze rules see a loss rather than a pose that never ages.

The phone reports its head rotation away from looking straight at the phone. The anchor is the world orientation of that straight ahead. Unaligned it is identity, so looking at the phone faces world +X, the avatar's rest facing. "Align head to camera" captures the anchor at the next camera frame where both the body-pose head and the phone head are measured (`faceHeadAnchorFromWorld`), so a phone mounted off to one side still turns the head about the right axes.

## Config and UI

`FaceConfig` on `AppConfig::face`, persisted under `face` in the project JSON:

- `bool enabled` (default false)
- `int port` (default 49983)
- `std::string phoneAddress` (default `255.255.255.255`)
- `glm::quat headAnchor` with `headAnchorPresent`, written as `headAnchor` `[x, y, z, w]` only once captured

The Tracking panel's Face section, after Wrist IMU, holds the enable checkbox, the port, the phone address, and a status readout: bound or not, streaming or quiet, datagrams per second, unknown-name and parse-failure counts, the last unknown name, and the last head rotation. Below it sit the head alignment button, its reset, and whether the head is aligned.

## Test

`--test-face` covers the name table, both blendshape formats, bone parsing and rejection, the handshake filter, garbage input, a loopback round trip through the service, and the socket primitives. `--test-vmc` covers the head conversion, the anchor round trip, the face map rules, and the blendshape and avatar messages on the wire.
