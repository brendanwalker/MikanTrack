# Avatar

The VRM avatar system: what the loader reads from a VRM file, the humanoid bone table and its version differences, the rest skeleton derived from it, the MToon subset the renderer implements, and the panel that drives it. A loaded avatar is the one humanoid skeleton the app will retarget the tracked pose onto, so the retarget and the avatar-driven VMC output (planned, see `docs/plan.md`) build on the rest data described here. See [conventions.md](./conventions.md) for the avatar space and its single conversion point, [architecture.md](./architecture.md) for where the modules sit, and [testing.md](./testing.md) for the tests.

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
- the humanoid map: one node index per `eHumanoidBone`, -1 when absent
- meta: name, version, author, license (1.0 `licenseUrl`, 0.x `licenseName`)

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

The hand rest comes from the shipping hand code rather than a second palm-frame derivation. A 21-landmark hand is assembled from the rest joints (wrist = hand node, each finger's three bones as its base, middle and distal joints, the fingertip extrapolated at 0.8 of the distal bone since a VRM has no tip bones, and a missing optional finger filled with a stand-in along the palm) and run through `HandPoseModel::computePalmFrame` and `computeSkeleton`, with the palmar memory seeded to world down because every VRM rests palms-down in a T-pose. The chirality `buildFingerJoints` reads from `baseInPalm[Index].y` therefore comes out exactly as for a tracked hand: positive on a left hand, negative on a right. `neutralDirInPalm` is then overwritten with the avatar's own rest finger directions, an explicit exception to the flat-hand rule for tracked skeletons: zero angles must reproduce the avatar's rest hand, and this skeleton never goes on the wire. A hand with no index, middle or little finger has no valid rest and is skipped by consumers.

## MToon subset and the renderer

`GlSkinnedMeshRenderer` (`src/Render/GlSkinnedMeshRenderer.h`) draws a model into whatever framebuffer is bound: one VAO per primitive, GPU skinning with four weights per vertex reading joint matrices (`global[joint] * inverseBind`, the mesh node's own transform ignored per the glTF spec) from a `GL_TEXTURE_BUFFER`, since VRoid rigs exceed the 64 matrices GL 3.3 guarantees for a uniform array. Textures upload as `GL_SRGB8_ALPHA8` with mipmaps and a sampler object per glTF texture.

The fragment shader is the MToon 1.0 ramp in linear space: `shading = linearstep(-1 + toony, 1 - toony, NdotL + shift)` and `rgb = mix(shade, base, shading)` with one directional light, a small constant ambient, and gamma encoding on output because the scene framebuffer is plain RGBA8. The material fields read for it:

- `shadeColorFactor`, `shadeMultiplyTexture`, `shadingShiftFactor`, `shadingToonyFactor`
- `renderQueueOffsetNumber` and `transparentWithZWrite`, which order the BLEND draws
- the glTF `alphaMode`, `alphaCutoff` and `doubleSided`

Draw order follows the spec: OPAQUE and MASK, then BLEND with z-write sorted by offset, then BLEND without z-write with depth writes off, sorted by offset. Double-sided materials disable culling and flip the normal on back faces. A material that is not MToon shades as a half-brightness copy of its own base color.

VRM 0.x materials arrive as the `materialProperties` bag and are migrated the way UniVRM and three-vrm do: `_Color` and `_ShadeColor` from gamma to linear, `_MainTex` and `_ShadeTexture` as texture indices, `_ShadeToony` and `_ShadeShift` through the toony and shift remap, `_CullMode` 0 as double-sided, `_ZWrite` into `transparentWithZWrite`, and the Unity `renderQueue` values of the transparent materials ranked into offsets (z-write off onto -9..0 ending at 0, z-write on onto 0..9 starting at 0).

Deferred, tracked in `docs/plan.md`: outline, rim, matcap, UV animation, morph targets (expressions), spring bones.

## Integration

`AvatarConfig` on `AppConfig` persists only the path and placement under the project's `avatar` JSON section: `modelPath` (absolute, or relative to the project folder, then to the exe folder, so `models/avatars/fem_vroid.vrm` names a shipped sample), `showInScene`, `followShoulders`, `rootPosition` and `rootYawDegrees`. Loading is never a side effect of reading the config, since replay reconstructs configs headlessly without a GL context. `App` owns the loaded `AvatarModel` and `AvatarSkeleton` as shared pointers with a generation counter; it loads the configured avatar at the end of `activateProject` on the main thread and clears it on the way back to the main menu. `Scene3dPanel` draws the avatar at rest before the line pass, placed by the fixed root transform, with the tracked skeleton lines drawn depth-disabled over it so they stay visible inside the character.

The Avatar panel (`src/UI/AvatarPanel.h`, View menu) loads and unloads a file, shows the meta and counts, the avatar's arm lengths against the user's measured ones (the ratio the hand retarget will scale reach by), the placement controls, and the read-only humanoid bone table (bone to node name, or missing). The table is the automatic mapping; per-bone adjustment is a later item.

## Tools

- `--test-vrm`: self test on synthetic in-memory files in both generations
- `--test-vrm-samples`: loads the shipped VRoid samples, skipping when absent
- `--vrm-info <file.vrm>`: prints what the loader read and the derived skeleton for any file
- `--render-avatar <file.vrm> [out.png] [yaw] [pitch]`: renders the rest pose through the scene renderer from a hidden window into a PNG, the way to see what the renderer produces without driving the UI
