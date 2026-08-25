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
├── embodiedgen_hoidini.py
└── hoidini_intermimic/
```

The modules under `nursery/humanoid/` are purpose-only scaffolding for two
future bridges: EmbodiedGen to HOIDiNi, followed by HOIDiNi to InterMimic. They
do not yet implement conversion or simulation behavior.

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

On Juno, run generation and rendering commands inside an appropriate Slurm
allocation. See the upstream [EmbodiedGen documentation](https://horizonrobotics.github.io/EmbodiedGen/docs/index.html)
for native installation and pipeline details.
