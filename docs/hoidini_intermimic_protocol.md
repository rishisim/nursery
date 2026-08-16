# HOIDiNi + InterMimic protocol: Stages 1–2

This is the canonical Stage 1–2 protocol for the contact-precision candidate.
It ends at an exact, validated binding to the shared EmbodiedGen scene. It does
not claim HOIDiNi motion generation, InterMimic execution, object settling,
humanoid contact, or video rendering.

## Stage 1: activity contract

The checked-in contract is
`configs/hoidini_intermimic_pilot.json`. Its activity is:

> Pick up the red mug from the table, bring it to your mouth, and place it back
> on the table.

The contract freezes the information that later systems must not reinterpret:

- 10 seconds, 30 Hz, 300 half-open frames;
- all seven shared seeds;
- the canonical red-mug target and table support IDs;
- `goal` as an explicit alias of the same table instance as `support`;
- an adult, neutral, zero-shape SMPL-X actor using the right hand;
- target-relative initial placement `[-1, 0, 0]` with identity orientation;
- six authoritative phase intervals in both frames and seconds;
- intended right-hand and table-contact windows;
- start and intended final `target ON table` relations;
- the same seven acceptance metrics used by the InteractMove protocol.

The resolved initial humanoid pose is translation
`[-0.8851, 0.1831, 0.8242]` metres and XYZW rotation `[0, 0, 0, 1]`.
It comes from the target-relative Stage 1 rule, not the native Franka pose.
No proactive accessibility/reachability gate is added at this stage; failure
would be attributed only if later motion generation or physics execution
actually demonstrates one.

Validate the contract with:

```bash
python3 scripts/prepare_hoidini_intermimic_scene.py validate-task \
  --task-contract configs/hoidini_intermimic_pilot.json
```

## Stage 2: exact shared-scene binding

### Authoritative input chain

The HOIDiNi pipeline does not independently generate a second scene. It binds
the accepted durable package at:

```text
/work/dal503972/interactmove_intermimic/scene_packages/red-mug-mouth-return
```

The fail-closed binder consumes and cross-checks:

1. the Stage 1 task contract;
2. the shared-scene handoff;
3. the canonical ActivitySpec;
4. the EmbodiedGen generation receipt;
5. `layout.json`;
6. every file listed in the 171-file inventory;
7. the validated SceneBundle and its semantic hash.

It rejects an alternate root, the superseded package, a handoff/path/hash
mismatch, any missing or additional scene file, a changed byte count or hash,
an unsafe path or symlink, a different activity/timing/seed block, a different
asset backend, canonical-ID drift, robot insertion, coordinate drift, or a
SceneBundle that lacks the required complete-scene capabilities.

### What is preserved

The binding retains:

- target `red mug` as `egv2_028a54da21c1691ae18ea225`;
- support and goal `table` as the same
  `egv2_0d4fc4a78d3706edccafb665` instance;
- object URDF, visual mesh, collision mesh, pose, mass, friction, and all hashes;
- the finite metric room reference mesh;
- the explicit `z=0` floor and four wall collision definitions;
- right-handed `+Z`-up metres/kilograms/radians/seconds, XYZW quaternions, and
  `T_parent_child` transforms;
- the exact shared world-placement metadata for the declared neutral adult
  SMPL-X actor;
- native Franka metadata only as ignored provenance, with zero robot actors.

The accepted object assets are SAM3D outputs, pinned to source commit
`01417d16fb5cc762a60f370c1bf7f59d603ddfaf` and checkpoint revision
`2e73555018d2741ccd486e56c24fac41155a1dc6`. The repaired room uses the
panorama/Pano2Mesh path. Neither SAM3D nor TRELLIS is required at binding time:
the binder consumes the already-generated URDF/OBJ/PLY assets and does not
regenerate them.

The accepted generation receipt records `gpt-5.6-luna` through the Responses
API with OpenAI SDK 3.1.0 and reasoning effort `none`. The binder itself makes
no OpenAI request.

The world placement is not yet a complete SMPL-X tensor state. The later
embodiment adapter must resolve the SMPL-X root/pelvis convention and construct
explicit `transl`, global orientation, body, jaw/eye, and articulated-hand pose
tensors before HOIDiNi or InterMimic can consume it.

### Output

The durable compact receipt is:

```text
/work/dal503972/hoidini_intermimic/compact_records/shared_scene_binding_red-mug-mouth-return.json
```

Its governed checked-in copy is
`docs/hoidini_intermimic_shared_scene_binding.json`. The receipt points to the
canonical external assets; no scene asset is copied into Git or the HOIDiNi
durable scene namespace.

Run the canonical Juno binding with:

```bash
bash scripts/juno_bind_hoidini_intermimic_shared_scene.sh
```

`inspect-layout` remains a low-level layout/URDF inspection command for fixtures
and diagnostics. A naked `layout.json` resolution is not sufficient to complete
Stage 2 because it cannot prove inventory, generation, SceneBundle, or canonical
instance identity.

## Physics boundary

The SceneBundle is complete enough for later consumer preparation: visual room,
metric reference mesh, object assets, explicit floor/walls, and InteractMove
scene input are present. It is not yet physically complete:

- the SceneBundle blocker list reports missing restitution for the plate, mug,
  and spoon, while the table URDF also lacks restitution;
- some upstream inertia values are placeholders;
- `physics_material_complete=false`;
- `dynamic_settle_ready=false`;
- settling remains `not_run`.

The 10-second scene-only SAPIEN canary is intentionally narrow. It used no
humanoid or controller, recorded no intended-contact trace, did not apply URDF
mass, and hard-coded restitution. It therefore cannot clear the InterMimic,
mass/material, grasp, contact, balance, or settling gates.

## Stage completion

Stage 1 is complete because the exact shared ActivitySpec is represented and
validated without ambiguity. Stage 2 is complete because the full canonical
package and provenance chain passed on Juno, canonical IDs and transforms were
retained, room collision definitions were validated and bound from the
SceneBundle, and zero scene assets were copied or resampled.

The next stage must investigate HOIDiNi's actual released inference boundary
before claiming it can consume this arbitrary object and scene. No current
record claims HOIDiNi motion, explicit contact-pair output, InterMimic tracking,
or a final RGB clip.
