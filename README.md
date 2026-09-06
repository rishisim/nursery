# Nursery Embodied Scene

This branch contains the lightweight Nursery adapters for generating reusable
EmbodiedGen rooms and preparing the human-motion integration boundary.
EmbodiedGen remains the native implementation for room generation, planning,
asset retrieval and generation, URDF export, and collision-aware placement.

The active tracked implementation is organized as:

```text
configs/embodiedgen_scene.json
nursery/__init__.py
nursery/embodiedgen_scene.py
nursery/humanoid/
├── embodiedgen_hoidini/
│   ├── __init__.py
│   ├── prepare_scene.py
│   ├── run_motion.py
│   └── validate_motion.py
└── hoidini_intermimic/
```

`nursery/humanoid/embodiedgen_hoidini/` deterministically prepares a complete
EmbodiedGen layout or Nursery composed room and one selected object/support
interaction condition. Its `run_motion` entry point aligns an explicitly supplied
pre-contact GRAB prefix and runs one HOIDiNi interaction. The
`hoidini_intermimic` entry point converts the saved interaction into an
InterMimic SMPL-X motion reference for an explicitly supplied humanoid XML.

Historical Nursery research that is not part of this workflow is retained
under `archive/legacy_research/`. It is inactive on this branch and remains
unchanged on the synthetic research branches.

Generated rooms, assets, scenes, and previews belong under the ignored
`outputs/embodiedgen_scene/` root. The external EmbodiedGen checkout belongs
under the ignored `.external/EmbodiedGen/` path. Local upstream humanoid
checkouts likewise belong under `.external/HOIDiNi/`,
`.external/InterActMove/`, `.external/InterAct/`, and `.external/InterMimic/`; they are dependencies,
not Nursery-owned source.

## Commands

Prepare the configured reusable room bank:

```bash
python -m nursery.embodiedgen_scene prepare-bank \
  --config configs/embodiedgen_scene.json
```

Build one static activity scene:

```bash
python -m nursery.embodiedgen_scene build \
  --activity "A person folds laundry beside a bed" \
  --seed 7 \
  --config configs/embodiedgen_scene.json
```

Render a fast preview from a room ID, scene directory, `scene.blend`, or
composed `scene_updated.urdf` path:

```bash
python -m nursery.embodiedgen_scene preview \
  --scene LivingRoom_seed11 \
  --config configs/embodiedgen_scene.json
```

Prepare a built room for one object interaction using the existing Python API:

```python
from pathlib import Path
from nursery.embodiedgen_scene import build_scene
from nursery.humanoid.embodiedgen_hoidini import prepare_scene

scene = build_scene("Pick up the red mug from the table", seed=7)
manifest_path = Path(scene["scene_urdf"]).parent.parent / "scene_manifest.json"
bundle = prepare_scene(manifest_path, target="red mug", seed=7)
```

Use an exact requested object name from `scene["object_instances"]` as `target`.
The builder records the actual inserted or reused instance and its placement
relation in this existing manifest. Preparation resolves native room names to
URDF links, preserves all links and mesh references, and uses the composed
URDF's joint transforms, mesh origins, and scale for the selected object and
support. The bundle names these nodes by their URDF link names. Relative
`scene_urdf` references resolve beside the manifest.

Older room manifests without `object_instances` must be rebuilt with the current
builder. Native `layout.json` inputs continue to work. Room preparation requires
an explicit target on one horizontal rectangular support, fixed URDF joints,
and OBJ target/support geometry; ambiguous or missing instance bindings fail
explicitly. The prepared bundle still reports `ready_for_hoidini_inference=False`
because the human prefix and inference are outside this step.

Focused interface validation (no upstream generation or inference):

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_embodiedgen_hoidini.py
```

The room tests call the real composer with a substituted native placement
command and check composed-world geometry, reused objects, deterministic
sampling, and rejection of invalid connections. Native GPU execution is not
covered by these tests.

## One scene-conditioned motion

Inside the existing HOIDiNi CUDA environment, with the upstream checkout on
`PYTHONPATH`, supply a built scene and one explicitly chosen GRAB sequence:

```bash
python -m nursery.humanoid.embodiedgen_hoidini.run_motion \
  --scene /path/to/scene_manifest.json --target mug \
  --prompt "The person lifts a mug." \
  --prefix /path/to/GRAB_RETARGETED_compressed/s10/mug_lift.npz \
  --model /path/to/runtime_model/model000120000.pt \
  --config /path/to/HOIDiNi/hoidini/configs/0_base_config.yaml \
  --output outputs/hoidini_scene/run --yaw -90 --seed 90323

python -m nursery.humanoid.embodiedgen_hoidini.validate_motion \
  outputs/hoidini_scene/run --render
```

The output directory must be new and ignored. Keep upstream `TMP_DIR`,
`HF_HOME`, and `XDG_CACHE_HOME` under ignored dependency/cache roots, and set
`SMPL_MODELS_DATA` to the installed models. No dependencies are installed by
these commands. The checkpoint's normalization files and `args.json` must be
beside the checkpoint, as in the existing Juno runtime-model directory.

`--start-frame` indexes the source after upstream resampling to 20 fps. Sixteen
source frames provide the velocity information for the 15-frame prefix. This
bounded adapter rejects source windows with hand contact instead of transferring
a grasp to an unrelated mesh. `--yaw` explicitly controls the human's heading;
alignment translates the human in XY and grounds the reconstructed prefix body
on the room's Z=0 floor. Both hand orientations receive the heading rotation. The mug
uses the prepared scene pose and geometry. `--prepare-only` saves the input and
prefix without sampling.

The runner reuses the upstream encoder, normalizer, model, and two-phase
`SamplingFlow`. Its GRAB-specific phase-one geometry lookup is temporarily bound
to the scene's object and support; upstream source is not modified. Outputs are
`input.json`, `prefix.npz`, `sampling.yaml`, `motion.npz`, and optimization plots.
The validator reloads the artifacts and records explicit mug-lift checks in
`validation.json`; optional rendering saves a Blender scene and selected frames.

This demonstrates kinematic object interaction. The optimization sees the mug
and desk, not every room obstacle, and does not execute physics or generate
validated forces. Prefix search, grasp retargeting, and InterMimic physics
execution remain deferred.

### Current validation result

The scene-conditioned inference runs, but the mug-lift example has **not passed
the motion-quality gate**. Juno job `377826` used the existing checkpoint
`model000120000.pt`, upstream HOIDiNi commit
`0fc3a78da4d0a6372efd4e52f89c54fd590ce32b`, and the upstream 350-step settings
for both optimization phases. Inputs were `s10/mug_lift.npz`, start frame 0,
yaw -90 degrees, seed 90323, and a real generated LivingRoom scene with one
retrieved mug on its desk. The explicit prefix was grounded by 0.0213614 m.

| Saved-artifact check | Measured result | Gate |
|---|---:|---|
| Motion length | 100 frames at 20 fps | Pass |
| Prefix position error | 0.00000024 m | Pass: below 0.001 m |
| Initial mug position error | 0.00000026 m | Pass: below 0.001 m |
| Maximum mug lift | 0.2289 m | Pass: 62 lifted frames with a hand within 0.03 m |
| Mug penetration within the desk footprint | 0.01554 m | Pass: at most 0.02 m |
| Lowest body vertex | -0.11228 m | **Fail:** no lower than -0.05 m |

The prefix's lowest body vertex is approximately zero; penetration develops in
the generated continuation. A corrected 200-step diagnostic also failed this
check (-0.10471 m). The existing upstream foot-skating loss penalizes horizontal
motion, not penetration below the room floor. These are permissive kinematic
smoke checks, not physical validation or a demonstrated success rate.

Run artifacts remain ignored on Juno at
`/work/dal503972/nursery/outputs/hoidini_scene/run/`. Its `input.json` records the
generated scene manifest at
`/work/dal503972/nursery/outputs/embodiedgen_scene/scenes/20260905T022111338639Z_81fa0b9e43/scene_manifest.json`.
The 24 focused local tests pass. The user accepted the failed floor check as a
documented limitation for proceeding with motion conversion. It remains failed;
no floor penalty or broader collision correction is implemented here.
The Blender animation and four preview frames saved successfully. The command
exited with status 1 for the failed floor check; rendering itself completed.

### HOIDiNi to InterMimic conversion

Convert the saved run using the existing Torch environment:

```bash
python -m nursery.humanoid.hoidini_intermimic \
  --run outputs/hoidini_scene/run \
  --humanoid-xml .external/InterMimic/isaacgym/src/intermimic/data/assets/smplx/omomo.xml \
  --object-name mug \
  --output outputs/hoidini_intermimic/run
```

The output directory must be new and ignored. Outputs are the native
`nursery_mug_000.pt` tensor, the supplied humanoid XML, the unchanged canonical
object mesh, 1,024 sampled object points, and a provenance manifest. The source
motion validation is copied into the manifest unchanged, including failures.

The converter maps HOIDiNi's 52 body/finger joints to the XML's body order,
applies the upstream body-local axis convention, and reconstructs body positions
and orientations using that skeleton. It preserves metre, +Z world coordinates,
pelvis motion, and object motion. It resamples translations linearly and rotations
with SLERP to 30 fps; contacts use the nearest source frame. The final source
sample is held through its remaining frame interval. A five-second, 100-frame
input at 20 fps produces 150 frames with 591 fields: root pose, local exponential-map
DOFs, world body poses, object pose, and contact intent. Quaternions use XYZW.

The 60 source hand-anchor scores are thresholded at 0.4 and aggregated per hand
at its wrist. Positive labels mean desired contact; zero means unconstrained.
Object contact means any predicted hand contact, not contact with the desk.
Finger rotations are retained, but per-finger and full-body contact labels are
not inferred. Transferring joint angles to the supplied body proportions does
not preserve hand-object distances or establish physical feasibility.

Juno job `379121` completed conversion and the isolated upstream check:

```bash
python tests/check_hoidini_intermimic_upstream.py \
  --output outputs/hoidini_intermimic/run \
  --intermimic .external/InterMimic \
  --interact .external/InterAct
```

The check executes the actual InterMimic loader body and InterAct PoseLib forward
kinematics without constructing a simulator. Only SDK tensor creation and angle
wrapping are bound to equivalent Torch primitives. It verified finite loaded
data of shape `[1, 150, 1211]`, references of shape `[1, 1, 150, 332]`, unchanged
motion/contact fields, and maximum FK position error of `7.16e-7` m. The tested
upstreams were [InterMimic](https://github.com/Sirui-Xu/InterMimic) commit
`60d6d6e0895a308ff8dc8f4c53af211b739cd5e7` and
[InterAct](https://github.com/wzyabcas/InterAct) commit
`96180a34f7b516e7f3520b853c19ea8679b8204f`. All 44 focused local tests pass.

The target body's minimum joint height is -0.10534 m. Its positions differ from
HOIDiNi's separate predicted joint-position channels by 0.07754 m on average
and 0.44745 m at maximum; this comparison includes both body proportions and
inconsistency between the source's pose and position channels. It is not a
contact-preservation check. The original floor check remains **failed**.

Converted artifacts and `upstream_validation.json` remain ignored on Juno at
`/work/dal503972/nursery/outputs/hoidini_intermimic/run/`. This establishes reference
format and skeleton compatibility. Registering the room/object physics assets,
loading a controller, executing the interaction under physics, and recording
synchronized video and simulator signals are still required for the combined
workflow; none is claimed by this conversion check.

On Juno, run generation and rendering commands inside an appropriate Slurm
allocation. See the upstream [EmbodiedGen documentation](https://horizonrobotics.github.io/EmbodiedGen/docs/index.html)
for native installation and pipeline details.
