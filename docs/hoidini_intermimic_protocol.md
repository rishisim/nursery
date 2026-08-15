# HOIDiNi + InterMimic protocol: Stages 1–2

This is the canonical protocol for the contact-precision candidate. These two
stages stop at a validated, robot-free EmbodiedGenV2 scene manifest. They do
not run HOIDiNi, InterMimic, a simulator policy, or rendering.

## Stage 1: explicit activity contract

The checked-in pilot is
`configs/hoidini_intermimic_pilot.json`. It turns the prompt “Pick up the mug
from the table, carry it to the tray, and put it down.” into explicit data:

- target (`mug`), initial support (`table`), and goal receiver (`tray`);
- a ten-second action timeline and intended contact windows;
- a neutral adult SMPL-X initial state placed by a deterministic world-space
  offset from the table;
- initial object poses supplied by the generated layout;
- scene, layout, motion, physics, and rendering seeds;
- measurable success criteria and forbidden failures.

The contract deliberately does not hide prompt interpretation behind an NLP
adapter. New activities must be authored and reviewed in the same schema before
scene generation. The adult body is only the public-system pilot embodiment;
no infant embodiment claim is made.

Validate it with:

```bash
python scripts/prepare_hoidini_intermimic_scene.py validate-task \
  --task-contract configs/hoidini_intermimic_pilot.json
```

The contract uses a right-handed, Z-up world in meters and radians. World
quaternions are stored as `[x, y, z, w]`. This upstream interpretation is
frozen against EmbodiedGenV2 `v2.0.1` commit
`9b333554254af196bace88c1a171a3bf047fa09c`; the manifest records that
assumption because `layout.json` itself has no format-version field.

## Stage 2: EmbodiedGenV2 scene boundary

### Input

The resolver accepts an EmbodiedGenV2 task directory containing `layout.json`.
The required upstream maps are `tree`, `relation`, `objs_desc`,
`objs_mapping`, `assets`, `quality`, and `position`.

For every non-background object, `assets[name]` must reference either its URDF
or the directory containing `<name_with_underscores>.urdf`. The URDF is the
source of truth for:

- visual and collision mesh paths;
- mesh scale and mesh-local `xyz`/`rpy` transforms;
- mass and center of mass when present;
- inertia when present and not the known EmbodiedGen template placeholder;
- static/dynamic friction (`mu1`/`mu2`);
- restitution when explicitly present.

The resolver preserves raw mesh-local transforms. It does not infer a
collision mesh from its filename and does not substitute the visual mesh when
the declared collision mesh is missing.

### Robot policy

Current EmbodiedGen layouts may retain semantic robot names, tree edges, or a
robot pose even when robot insertion was not requested. Those values are
accepted only as upstream metadata and stripped. No robot asset or robot
instance is included in the resolved manifest. Do not use EmbodiedGen's
current convenience SAPIEN entry point for this robot-free boundary because
the public implementation may still construct a robot.

No proactive humanoid-reachability gate is performed. The manifest records
that fact, and an actual placement failure should be reported only if one is
later observed.

### Output

The output manifest contains:

- deterministic instance IDs and source names;
- target, support, and goal bindings;
- normalized world poses;
- URDF and mesh paths plus SHA-256 hashes;
- body mode (`dynamic` or `static`);
- available mass, inertia, friction, and restitution with provenance;
- the original VLM-estimated mass range, real height, friction provenance, and
  upstream quality marker when present;
- verified target-on-support and goal-on-support scene-graph relationships;
- background visual references;
- an explicit absence of background collision unless separately designated;
- the resolved initial adult-human transform;
- robot-removal evidence and current limitations.

Resolve a scene into an ignored working directory:

```bash
python scripts/prepare_hoidini_intermimic_scene.py resolve-scene \
  --task-contract configs/hoidini_intermimic_pilot.json \
  --layout /path/to/embodiedgen/task_0000/layout.json \
  --output tmp/hoidini_intermimic/scene_manifest.json
```

`tmp/` is disposable and ignored. Real generated scene packages, downloads,
media, and complete runs must not be committed to this repository.

## Done when

Stage 1 is complete when the pilot contract validates and every phase covers
exactly the ten-second interval. Stage 2 is complete when the resolver can
bind all three scientific roles, parse every declared object asset without a
path escape, publish a robot-free manifest, and distinguish known physical
properties from missing or placeholder values.

The checked-in test fixture is representative text data only. It proves this
file boundary and does **not** claim that a real EmbodiedGenV2 scene was
generated or loaded in a simulator.

## Open limitations

- The first task is one rigid manipulated object; articulated, deformable, and
  multi-target activities are out of scope.
- EmbodiedGen's generated identity-like inertia tensor is treated as a template
  placeholder. A later simulator stage must calculate or validate inertia.
- EmbodiedGen does not currently provide object restitution, so it remains
  unavailable instead of inheriting a simulator hard-code.
- Background `mesh_model.ply` is visual only unless a separate validated
  collision asset is supplied. The table URDF supplies the pilot's support
  collision.
- A real EmbodiedGen generation run and physical scene validation remain future
  work; no checkpoints, assets, or full scenes were downloaded here.
