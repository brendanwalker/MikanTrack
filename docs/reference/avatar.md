# Avatar

The VRM avatar system: what the loader reads from a VRM file, the humanoid bone table and its version differences, the rest skeleton derived from it, the MToon subset the renderer implements, the retarget that poses the avatar from the tracked frame, the per-avatar rig settings that correct the mapping and the retarget's assumptions, the avatar-driven VMC output, and the panel that drives it. See [conventions.md](./conventions.md) for the avatar space and its single conversion point, [architecture.md](./architecture.md) for where the modules sit, and [testing.md](./testing.md) for the tests.

External references, both public: the VRM specification (https://github.com/vrm-c/vrm-specification), including `VRMC_vrm-1.0`, the `VRM` 0.x extension, and `VRMC_materials_mtoon-1.0`, and the pixiv three-vrm renderer (https://github.com/pixiv/three-vrm), whose VRM 0.x material migration the loader follows.

---

## Files and the two VRM generations

A VRM is a glTF 2.0 binary (`.vrm` is a `.glb`) plus one top-level extension that names which glTF node plays which humanoid bone and carries the metadata. Two generations exist and both load into one `AvatarModel`:

- VRM 0.x: extension `VRM`, `specVersion` "0.0", exported by UniVRM 0.x and VRoid Studio. The model faces glTF -Z with its left at -X. Humanoid bones are an array of `{bone, node}` in Unity `HumanBodyBones` spelling. Materials are a Unity property bag per material (`materialProperties[]`).
- VRM 1.0: extension `VRMC_vrm`, `specVersion` "1.0". The model faces glTF +Z with its left at +X. Humanoid bones are an object keyed by bone name, with the thumb renamed to metacarpal/proximal/distal. MToon lives in a per-material `VRMC_materials_mtoon` extension.

When a file carries both extensions, `VRMC_vrm` wins. Everything version-specific is resolved in `VrmLoader` and `AvatarSkeleton`; nothing downstream branches on the generation.

The CC0 VRoid Studio samples `models/avatars/fem_vroid.vrm` and `masc_vroid.vrm` (fetched by `InitialSetup_x64.bat`, provenance in `NOTICE.md`) are VRM 0.x. A VRM 1.0 file is covered by the synthetic test only until a redistributable 1.0 sample is added.

## The loader (`src/Avatar/VrmLoader.h`)

`VrmLoader::loadFile` and `loadMemory` produce an `AvatarModel` (`src/Avatar/AvatarTypes.h`) or an error string, plus non-fatal warnings. Pure CPU work with no GL and no app state, so the self test runs it on an in-memory file and `--vrm-info` on any path. The glTF core goes through `cgltf` (`thirdparty/cgltf`, one `CGLTF_IMPLEMENTATION` in `VrmLoader.cpp`): parse, load the embedded buffers, validate, then read through `cgltf_accessor_read_float` / `read_uint` / `read_index`, which resolve normalized, strided, and sparse accessors. The VRM extension JSON is read directly with nlohmann for the subset the app uses. What lands in the model:

- nodes: name, parent, children, the parent-relative local transform as authored (TRS composed, or the matrix as given), mesh and skin indices
- skins: joint node indices and inverse bind matrices (identity when absent)
- meshes: triangle primitives only, with positions, normals (empty when absent), `TEXCOORD_0`, `JOINTS_0` as `uvec4`, `WEIGHTS_0` renormalized to sum one, and `uint32` indices (generated when the file has none)
- materials: base color factor and texture, alpha mode, alpha cutoff (default 0.5), double sided, and the MToon subset (below)
- textures and images: wrap and filter flags per texture, and every image decoded through `cv::imdecode` to RGBA8 with rows kept top-down (glTF's UV origin is the top-left texel, which is also the first row GL receives, so nothing flips anywhere); an undecodable image becomes a 1x1 white placeholder with a warning
- the humanoid map: one node index per `eHumanoidBone`, -1 when absent, kept twice: `fileHumanoidNodes` as the file names it and `humanoidNodes` as in effect after the rig settings' overrides
- meta: name, version, author, license (1.0 `licenseUrl`, 0.x `licenseName`)
- morph target names per mesh (glTF `extras.targetNames`, the index as a string when absent), and per primitive each target's position and normal offsets kept sparse, only the vertices it moves
- expressions: each `AvatarExpression` has the name as the file spells it, a VRM 1.0 preset name (0.x `presetName` values are mapped onto that vocabulary, an unknown preset is empty like a custom expression), `isBinary`, and morph binds of `(mesh, morphIndex, weight)` with weight 0..1. 0.x reads `blendShapeMaster.blendShapeGroups` (mesh index in the bind, weight divided by 100), 1.0 reads `expressions.preset` and `expressions.custom` (node index in the bind, resolved to the node's mesh). Binds out of range warn and are dropped. Material color and texture transform binds are ignored. `findExpressionByPreset` and `findExpressionByName` look them up
- `sha256Hex`: SHA-256 of the exact file bytes (Windows CNG), computed for memory loads too

A load fails when the glTF is invalid, when neither VRM extension is present, or when a required humanoid bone is missing (hips, spine, head, both arms with hands, both legs with feet). Unknown bone names warn and are ignored.

## The humanoid table (`eHumanoidBone`)

The 55 VRM humanoid bones in Unity `HumanBodyBones` vocabulary, the spelling VMC receivers and the 0.x extension both use, so `humanoidBoneName` matches `VmcRetarget::boneName`. Fingers are laid out as 3 bones x 5 fingers x 2 hands in app finger order (thumb, index, middle, ring, little), proximal first, so a `(side, finger, phalanx)` triple indexes as `firstHumanoidFingerBone(side) + finger * 3 + phalanx`.

The VRM 1.0 thumb rename is folded back into this table: 1.0 `thumbMetacarpal` maps to `ThumbProximal` and 1.0 `thumbProximal` to `ThumbIntermediate`. The app's thumb phalanx [0] is the metacarpal (`FINGER_JOINTS` thumb `{CMC, MCP, IP, TIP}`), so app thumb proximal = 1.0 metacarpal = 0.x proximal, and the three thumb slots line up with the three app phalanges like every other finger.

`humanoidBoneParent` is the spec hierarchy. Optional bones (upper chest, shoulders, toes, eyes, jaw, all fingers) may be absent, so `AvatarSkeleton` resolves each bone's nearest PRESENT ancestor instead (an avatar without an upper chest hangs its neck and shoulders off the chest).

## The rest skeleton (`src/Avatar/AvatarSkeleton.h`)

`AvatarSkeleton` is immutable rest-pose data derived from a model, in the app's world frame, built once on load and shared by pointer. Everything comes from composed rest WORLD transforms, never from node-local rotations: a VRM 1.0 rig is not required to be normalized (nodes may carry rest rotations), and VRoid rigs put helper nodes between humanoid bones. It holds:

- `worldFromAvatar`, the conversion from the file's glTF frame into the world frame (see [conventions.md](./conventions.md)), and the rest globals per node in avatar space, which is the pose rendered before any retarget
- per humanoid bone: presence, node, resolved parent, rest world position and rotation, and the rest offset from the parent (the offset a bone keeps whatever rotation it is given)
- derived lengths: upper arm and forearm per side, shoulder width (between the upper-arm joints), hand length (hand joint to middle finger base), and the height above the hips for framing
- per side a `HandRest`: the avatar palm frame in the app's palm convention and a `HandSkeleton` describing the avatar's fingers

The hand rest comes from the shipping hand code rather than a second palm-frame derivation. Fingers are whatever the rig has: a finger's physical bones are its present humanoid slots in order (a Blender-authored rig that maps two bones per finger into the intermediate and distal slots has physical bones 0 and 1 there), and `HandRest::fingerBoneCount` and `fingerSlots` record them. A 21-landmark hand is assembled from the rest joints (wrist = hand node, physical bone i as joint i, the joints the rig lacks down to the fingertip extrapolated along the last bone at 0.8 per step, and a finger with no bones filled with a flat-hand stand-in that keeps it off the output) and run through `HandPoseModel::computePalmFrame` and `computeSkeleton`, with the palmar memory seeded to world down because every VRM rests palms-down in a T-pose. A rig with no middle finger takes +X along the forearm. The chirality `buildFingerJoints` reads from `baseInPalm[Index].y` therefore comes out exactly as for a tracked hand: positive on a left hand, negative on a right, provided the rig's index really sits on the thumb side of its middle finger; a rig whose index and middle are mapped the other way round gets a skeleton warning (shown in the Avatar panel and the log) because its thumb will pronate the wrong way until the mapping is corrected in the rig settings. `neutralDirInPalm` is then overwritten with the avatar's own rest finger directions, an explicit exception to the flat-hand rule for tracked skeletons: zero angles must reproduce the avatar's rest hand, and this skeleton never goes on the wire.

## MToon subset and the renderer

`GlSkinnedMeshRenderer` (`src/Render/GlSkinnedMeshRenderer.h`) draws a model into whatever framebuffer is bound: one VAO per primitive, GPU skinning with four weights per vertex reading joint matrices (`global[joint] * inverseBind`, the mesh node's own transform ignored per the glTF spec) from a `GL_TEXTURE_BUFFER`, since VRoid rigs exceed the 64 matrices GL 3.3 guarantees for a uniform array. Textures upload as `GL_SRGB8_ALPHA8` with mipmaps and a sampler object per glTF texture.

The fragment shader is the MToon 1.0 ramp in linear space: `shading = linearstep(-1 + toony, 1 - toony, NdotL + shift)` and `rgb = mix(shade, base, shading)` with one directional light, a small constant ambient, and gamma encoding on output because the scene framebuffer is plain RGBA8. The material fields read for it:

- `shadeColorFactor`, `shadeMultiplyTexture`, `shadingShiftFactor`, `shadingToonyFactor`
- `renderQueueOffsetNumber` and `transparentWithZWrite`, which order the BLEND draws
- the glTF `alphaMode`, `alphaCutoff` and `doubleSided`

Draw order follows the spec: OPAQUE and MASK, then BLEND with z-write sorted by offset, then BLEND without z-write with depth writes off, sorted by offset. Double-sided materials disable culling and flip the normal on back faces. A material that is not MToon shades as a half-brightness copy of its own base color.

VRM 0.x materials arrive as the `materialProperties` bag and are migrated the way UniVRM and three-vrm do: `_Color` and `_ShadeColor` from gamma to linear, `_MainTex` and `_ShadeTexture` as texture indices, `_ShadeToony` and `_ShadeShift` through the toony and shift remap, `_CullMode` 0 as double-sided, `_ZWrite` into `transparentWithZWrite`, and the Unity `renderQueue` values of the transparent materials ranked into offsets (z-write off onto -9..0 ending at 0, z-write on onto 0..9 starting at 0).

Morph targets deform on the CPU. A primitive with targets keeps its unmorphed vertices and the vertex range its targets touch. When `setMorphWeights` changes any of its weights, that range is recomposed from the base and re-uploaded with `glBufferSubData`. The 3D scene feeds it the shown frame's face through `AvatarFaceMorphs` (the face map below), so the preview shows the face exactly as a receiver holding the same file applies it.

Deferred, tracked in `docs/plan.md`: outline, rim, matcap, UV animation, GPU morphing, spring bones.

## The retarget (`src/Avatar/AvatarRetarget.h`)

`AvatarRetarget::solve` poses the avatar from the measured frame. The avatar keeps its own proportions: every bone keeps its rest offset from its humanoid parent and only rotates, so a pose is a root placement plus one world rotation DELTA per bone (`AvatarPose`), measured against the bone's rooted rest rotation. A bone's posed world rotation is `delta * rootRotation * restRotationWorld`, its posed position hangs off its parent's posed position by the parent's delta applied to the rest offset, and an absent bone (`present == false`) sits at identity, following its parent. `computePosedBones` composes that chain and `computePosedGlobals` turns it into node globals for the renderer, where present humanoid bones override their node and every other node (helpers, hair, clothing bones) composes from its parent as authored. The solve is pure over the frame it is given plus one filter state for the root, advanced on frame timestamps, so a replayed recording poses identically.

Per frame, in the world frame:

- Root: with `followShoulders` on and both shoulders tracked, the shoulder line fixes the yaw and the midpoint the translation (the avatar's rest shoulder midpoint lands on the measured one) through a first-order follow with `rootFollowTimeConstantS` (0.3 s), snapping on the first measurement or a timestamp regression. A shoulder dropout holds the last followed placement. Never followed, or follow off, the configured fixed root applies.
- Head: the measured head frame is +X facing, +Y left, +Z up, which is the avatar's rest frame, so the measured orientation is the posed canonical head and the delta is what remains after the root yaw and the head trim: `head * inverse(rootRotation * trim)`.
- Clavicle: with both shoulders measured and the avatar carrying a shoulder bone, swung from rest onto the measured shoulder's direction from the shoulder midpoint, as the measured-length VMC retarget does.
- Wrist target: `avatarShoulder + (userWrist - userShoulder) * (avatarArm / userArm)`, the user's reach scaled onto the avatar's arm (user lengths from `BodyDimensions`, avatar lengths from the skeleton). Without a measured shoulder the avatar's own stands in, which keeps the hand where it was seen. Beyond the avatar's reach the wrist is clamped to it and the arm straightens.
- Elbow: a two-bone solve over the avatar's upper arm and forearm (the circle where the two bone spheres meet), at the point nearest a pole. The pole starts at the rig's elbow hint, `avatarShoulder + rootRotation * elbowHintOffset`. With a measured forearm it moves toward the measured elbow (`getElbowPositionWorld`, scaled the same way) by `clamp(forearmConfidence / elbowHintConfidence, 0, 1)`, so a confident forearm decides the bend alone and a fading one hands it back to the hint.
- Upper arm: direction only, a swing from rest. Forearm: direction from the solve, roll from the measured forearm orientation (IMU or body solver) or, with none, from the palm, which puts the wrist at zero bend rather than inventing a twist. The forearm roll trim adds to that roll.
- Hand: the full measured palm frame against the avatar's rest palm frame turned by the hand trim, `restPalm * trim`.
- Fingers: the measured angles scaled by the rig's curl gain (the three bends) and splay gain (lateral), played through `HandPoseModel::buildFingerJoints` on the AVATAR's hand skeleton and its untrimmed posed palm, then each physical bone swung from its rest direction relative to the bone before it, so the hand's roll carries down the chain and zero angles reproduce the avatar's rest hand. A finger with fewer bones than the measured three folds the bends past its last bone into that bone, so a fist still closes on a two-bone finger. Fingers the avatar lacks, and fingers the rig disables, stay absent. An explicit thumb pronation from the rig replaces the one the hand skeleton implies.

Two instances run: `OscStreamer` solves on the vision thread over the RESOLVED poses (after the confidence gate, the dropout hold and the freeze-on-loss rule, so a dropout shapes the avatar exactly as it shapes the measured-length stream), and `MainWindow` solves for display on whatever fused result the 3D scene shows, live or replayed (with the recording's body lengths), resetting its root follow when the feed switches. The vision thread receives the skeleton and the rig settings together through `VisionThread::setAvatarSkeleton`, a pending slot adopted between frames rather than a config refresh, which would end a recording. `makeAvatarRetargetConfig` in `AppConfig.cpp` is the one mapping from the project's avatar settings and the avatar's rig settings to the solver config, shared by both instances.

## Rig settings (`src/Avatar/AvatarRig.h`)

Real rigs break the retarget's rest assumptions (a T-pose, palms down, the head facing forward, a human thumb), and some name their bones wrongly. `AvatarRigSettings` holds the corrections. It persists in a sidecar beside the model, `<stem>.mikanrig.json` (`Brendan_VRM.vrm` gets `Brendan_VRM.mikanrig.json`), not in `project.json`, so the corrections travel with the file and every project that loads it shares them. A missing sidecar means defaults. An unreadable one warns and means defaults. The JSON carries a `version`, a `boneNodes` object of overrides, a `rotationTrimDegrees` object holding only the nonzero trims, and a `left` and `right` block of per-side fields.

- Mapping overrides: per humanoid bone, a node NAME (an empty name unmaps the bone), so the sidecar survives a re-export that reorders nodes. `applyRigToModel` rebuilds `humanoidNodes` from `fileHumanoidNodes` plus the overrides. An unknown name warns and keeps the file's mapping. So does unmapping a bone every VRM must have, since the skeleton and the retarget assume those exist. Two bones on one node warns but applies, because that is the normal state halfway through swapping two fingers.

- Rotation trims: Euler degrees (glm's XYZ) rotating the rest frame the retarget matches a measurement against, in that frame's own axes. They exist only where the retarget measures a full frame against an assumed rest: the head (the canonical +X facing, +Y left, +Z up frame), the hands (the palm frame), and the forearms (a single roll, stored in the x component). A trim re-orients what counts as rest. It cannot fix a misassigned bone, and it cannot change the hinge axis the thumb curls about.

- Elbow hint: per arm, a point in the rooted torso frame (+X facing, +Y the avatar's left, +Z up) relative to the avatar's upper-arm joint, default `(-0.15, 0, -0.30)` m, down and back. Living in the torso frame, it turns with the root yaw. `elbowHintConfidence` (default 0.5) is the forearm confidence at which the measured elbow takes over completely.

- Hand tweaks: per hand, the thumb pronation (automatic, or an explicit signed angle), curl and splay gains (default 1), and five finger enables. The thumb pronation is its own control rather than a trim because it sets the hinge the thumb's MCP and IP bends rotate about. The automatic sign comes from which side of the palm the index sits on (`HandPoseModel::getThumbPronationRad`), which a rig with swapped fingers gets wrong. The override only reaches the avatar retarget: tracking, `computeFingerAngles` and the wire never see it.

`App` owns the settings beside the model and the skeleton. `loadAvatar` reads the sidecar, applies the mapping and builds the skeleton. `setAvatarRig` applies an edit at once. A mapping change reapplies the overrides and rebuilds the skeleton. The model's map is rewritten in place on the main thread, which is safe because only main-thread code reads it and the vision thread holds only the skeleton. Any other edit keeps the skeleton, so the streamer's root follow keeps running while a control is dragged. Every edit goes to the vision thread, and the sidecar is written once edits settle for 3 seconds, on unload, and on shutdown.

## Avatar-driven VMC

With a skeleton set, `OscStreamer` streams the retargeted pose through `VmcRetarget::buildPoseFromAvatar` instead of the measured-length chain: each VMC bone's local rotation is its delta measured against the nearest streamed ancestor's delta (the torso is never streamed, so an arm hangs off the rest chest as before), and its local position is the avatar's own rest offset from its humanoid parent, so a receiver loading the same file keeps the character's proportions. Bones the pose did not place, or the avatar lacks, are left out and rest on the receiving side. The root stays identity: the measured-shoulder placement is for the 3D view only, and a whole-body yaw lands in the root rather than the arms, so the streamed arms stay relative to the receiver's own torso. `humanoidBoneForVmc` maps the two Unity-spelled enums by name. Without an avatar the measured-length `VmcRetarget::buildPose` path is unchanged.

The loaded file is announced as `/VMC/Ext/VRM` (path, title, SHA-256) once a second, so a receiver on the same machine can load the very file the bones were retargeted onto and confirm by the hash that it is unchanged. `App::loadAvatar` hands the identity and the face map to the vision thread through `VisionThread::setAvatarIdentity`, adopted between frames like the skeleton.

## The face map (`src/Avatar/AvatarFaceMap.h`)

`AvatarFaceMap` turns the face stream's 52 ARKit blendshapes (see [face.md](./face.md)) into the blendshape names a VMC receiver matches on, built once per loaded avatar. Each output is a weighted sum of ARKit columns, clamped to [0, 1]. Which names an avatar gets:

- An avatar with expressions named after ARKit blendshapes (perfect sync, compared without case) gets those expressions alone, in its own spelling.

- Otherwise each VRM preset the avatar carries gets the ARKit columns that shape it: the sided blinks from each eye, `aa` from the jaw, `ih` from the smile, `ee` from the stretch, `ou` from the pucker, `oh` from the funnel, and the four gaze directions from the eye looks. The two-eyed blink is driven only when the avatar lacks a sided pair, since both would close the eyes twice. The emotion presets have no ARKit counterpart and stay untouched.

- Also in that second case, the ARKit names the avatar carries as morph targets go out under their own names. A standard receiver ignores them, since it matches expressions only. A receiver that also matches morph targets gets the full face, which is how an avatar whose ARKit shapes are morphs without expressions keeps its detail.

- With no avatar loaded the ARKit names go out verbatim.

`AvatarFaceMorphs` is the receiving half on the same file, what the 3D scene's preview runs: an output naming an expression drives that expression's morph binds scaled by the bind weight, an output naming a morph target drives it directly, and a morph several outputs drive takes the largest weight rather than the sum, so a preset and the ARKit shape it was derived from never double up.

## Integration

`AvatarConfig` on `AppConfig` persists only the path and placement under the project's `avatar` JSON section: `modelPath` (absolute, or relative to the project folder, then to the exe folder, so `models/avatars/fem_vroid.vrm` names a shipped sample), `showInScene`, `followShoulders`, `rootPosition` and `rootYawDegrees`. Loading is never a side effect of reading the config, since replay reconstructs configs headlessly without a GL context. `App` owns the loaded `AvatarModel` and `AvatarSkeleton` as shared pointers with a generation counter; it loads the configured avatar at the end of `activateProject` on the main thread and clears it on the way back to the main menu. `Scene3dPanel` draws the avatar posed by the display retarget (or at rest when no pose is set) before the line pass, with the tracked skeleton lines drawn depth-disabled over it so they stay visible inside the character.

The Avatar panel (`src/UI/AvatarPanel.h`, View menu) loads and unloads a file above three tabs:

- Model: the meta and counts, the avatar's arm lengths against the user's measured ones (the ratio the hand retarget will scale reach by), the rig and skeleton warnings, the sidecar's name, and the placement controls.

- Mapping: a schematic figure (`src/UI/AvatarMapFigure.h`) with Body, Head, Left Hand and Right Hand views and a dot per bone, colored green when mapped, grey when an optional bone is unmapped, red when a required one is, and orange on a warning (one node on two bones, or the index and middle of a hand whose index sits on the far side of the middle from the thumb). Below it, the bones of the shown view, each with a node picker (filtered by the text box, plus an unmapped entry that required bones refuse), a reset to the file's mapping when overridden, and a trim on the head, hands and forearms. Reset all clears every override.

- Retarget: the preview pose (Live, or the canned Rest and Demo frames from `src/Avatar/AvatarPreviewPoses.h`, which pose the scene's avatar so the rig can be tuned without cameras; the VMC output always follows live tracking), the elbow hint and its confidence blend per arm, and the hand tweaks per hand.

Each arm's Show gizmo toggle draws its elbow hint in the 3D scene as a camera-facing ring joined to the shoulder. Pressing within 8 pixels of a ring drags it in the camera-facing plane through its starting point, suppressing the orbit drag for that gesture. The dragged world point goes back into the torso-frame offset against the current pose's root rotation and posed upper-arm joint.

## Tools

- `--test-vrm`: self test on synthetic in-memory files in both generations
- `--test-avatar-retarget`: self test of the retarget and the avatar VMC stream on the shared synthetic rig (`src/Tests/SyntheticAvatar.h`)
- `--test-avatar-rig`: self test of the rig settings: the sidecar round trip, the overrides, and each retarget adjustment
- `--test-vrm-samples`: loads the shipped VRoid samples, skipping when absent
- `--vrm-info <file.vrm>`: prints what the loader read and the derived skeleton for any file
- `--render-avatar <file.vrm> [out.png] [yaw] [pitch] [rest|demo]`: renders the rest pose, or the Demo preview frame put through the retarget (`demo`), through the scene renderer from a hidden window into a PNG, with the model's rig sidecar applied, the way to see what the renderer and the retarget produce without driving the UI
