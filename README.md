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
│   └── prepare_scene.py
└── hoidini_intermimic/
```

`nursery/humanoid/embodiedgen_hoidini/` deterministically prepares a complete
EmbodiedGen layout or Nursery composed room and one selected object/support
interaction condition. It
does not generate the required human prefix or run HOIDiNi. The later HOIDiNi
to InterMimic modules remain purpose-only scaffolding.

Historical Nursery research that is not part of this workflow is retained
under `archive/legacy_research/`. It is inactive on this branch and remains
unchanged on the synthetic research branches.

Generated rooms, assets, scenes, and previews belong under the ignored
`outputs/embodiedgen_scene/` root. The external EmbodiedGen checkout belongs
under the ignored `.external/EmbodiedGen/` path. Local upstream humanoid
checkouts likewise belong under `.external/HOIDiNi/`,
`.external/InterActMove/`, and `.external/InterMimic/`; they are dependencies,
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

On Juno, run generation and rendering commands inside an appropriate Slurm
allocation. See the upstream [EmbodiedGen documentation](https://horizonrobotics.github.io/EmbodiedGen/docs/index.html)
for native installation and pipeline details.
