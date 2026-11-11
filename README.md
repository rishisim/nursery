# Nursery

Nursery asks whether video plus bodily experience supports better language
learning than video alone, as independent training experience increases.
The simulator supplies egocentric video and body signals; caregiver speech
provides intentionally uncertain language supervision.

Start with [PLAN.md](PLAN.md). The current sequence is
[16×16 vision](docs/step_1_visual_baseline.md),
[richer body input](docs/step_3_body_representation.md),
[a controlled representation comparison](docs/step_4_body_representation_comparison.md),
[the body training recipe](docs/step_5_body_training_recipe.md), and
[language acquisition curves](docs/step_6_language_acquisition_curves.md).
[Past experiments](archive/language_learning/past_experiments.md) explains what
we tried earlier; the completed cross-situational and acquisition studies are
indexed in the [archive](archive/README.md#completed-study-worktrees).
Steps 2–5 remain design references and do not imply their code is implemented.
The matched Step 1 comparison is complete: at 630 clips, video accuracy rose
from the recorded 60.12% with 4×4 tokens to 61.93% with 16×16 tokens across
three paired seeds. See [the results and limits](docs/step_1_visual_baseline.md#completed-results--7-november-2026).

## File map

```text
PLAN.md                         Current direction and study links
nursery/
  learner/
    video_only.py               Video → word matching (visual baseline)
    additive.py                 Video score + learned body → word score
    body.py                     Contact, fingers, arms, load/skin encoding
    windows.py                  Clip-bounded body windows and episode context
    losses.py                   Shared language and diagnostic loss functions
    train.py                    Shared data splits, training and checkpoints
    evaluate.py                 Shared picture-only and body-present scoring
    embed.py                    Frozen DINOv2 feature extraction
    config.json                 Upstream pin, training defaults, evaluation paths
    hand_patches.json           Contact-patch definitions
    __main__.py                 One CLI: embed / train
    __init__.py                 Pinned upstream loading
  speech/                       Caregiver utterances, timing, vocabulary and cues
  simulator/                    GRAB replay, body signals, assets and rendering
  embodiedgen_scene.py          Room builder; also used by the renderer
  common.py                     Paths, configuration and ignored-output guard
configs/                        Room-generation configuration
scripts/                        Render/pipeline/seed launchers and run summaries
tests/                          Current learner, speech, physical and simulator tests
docs/
  step_1_visual_baseline.md             Step 1: 16×16 vision and recorded 4×4 reference
  step_2_child_gaze.md                  Step 2: simulated gaze drives naming and feeds the learner
  step_3_body_representation.md         Step 3: recoverable sensory detail and missing signals
  step_4_body_representation_comparison.md  Step 4: current versus richer body, additive
  step_5_body_training_recipe.md        Step 5: decide wordless pretraining after representation
  step_6_language_acquisition_curves.md Step 6: main video versus video + body curves
  learner.md                    Current implementation and scoring reference
  speech_coupling.md             Speech-generation evidence and assumptions
archive/
  code/learner/                 Completed heavy/light, pretraining and old joint pilot
  code/motion/                  Inactive HOIDiNi/InterMimic code and its tests
  language_learning/            Past experiments, protocols and curated results
  legacy_research/              Earlier simulation source/config/test archive
  README.md                     Historical evidence and artifact locations
```

**Video-only and additive share the same visual base, training and evaluation.**
The additive learner extends `video_only.py`; it does not carry a second copy
of the visual implementation. Both use the pinned
[vlm-headcam](https://github.com/mcfrank/vlm-headcam) upstream implementation.
The preserved acquisition runner, joint interaction and 16×16 support are
integrated into this checkout; see [source and worktree locations](archive/README.md#completed-study-worktrees).
Step 1 selects only video-only training at the full-data point.
The archived width-128 joint pilot is a different implementation; `--arch joint`
is retained only for compatibility. New representation comparisons start with
additive fusion.

## Existing entry points

Run these from the repository root with the existing `.external/cue-env/` and
upstream checkout pinned in `nursery/learner/config.json`.
These examples explain the existing runner; study-specific data amounts and
budgets still come from the study plans.

```bash
# Shared preparation, using an existing episode dataset.
.external/cue-env/bin/python -m nursery.speech transcripts --episodes outputs/renders/clean_856 --output outputs/runs/study --seed 0
.external/cue-env/bin/python -m nursery.learner embed --episodes outputs/renders/clean_856 --output outputs/runs/study

# Video-only; direct language objective with soft pooling.
.external/cue-env/bin/python -m nursery.learner train --run outputs/runs/study --condition none --seed 0 --pool soft --body none --w-pred 0 --name video_only

# Additive; same visual/speech data, with body inputs and the same objective.
.external/cue-env/bin/python -m nursery.learner train --run outputs/runs/study --condition none --seed 0 --pool soft --body real --w-pred 0 --name additive
```

`--physical <directory>` adds saved load/skin streams to the additive learner;
the examples above use the base contact/finger/arm streams. Historical defaults
and diagnostic flags are documented in [the learner reference](docs/learner.md).
No run is launched by this documentation.

Embedding defaults to 16×16; `--grid 4` retains the historical option. Use a
fresh output root when changing grids. The exact Step 1 commands and frozen
comparison are in [the visual baseline](docs/step_1_visual_baseline.md#execution-protocol).

`scripts/render.sh <output>` renders sessions; `scripts/pipeline.sh` chains
speech, embedding and training. `scripts/train_seeds.sh` runs several seeds;
`scripts/summarize.py` reads saved metrics; `scripts/merge_render.py` combines
per-session renders. The simulator needs licensed GRAB/SMPL-X assets, Blender,
`.external/egocentric-env/`, and the room bank specified by its configuration.
Use `python -m nursery.simulator --help` or
`python -m nursery.embodiedgen_scene --help` for those entry points.

## Local data and verification

`outputs/` is the destination for new generated files and runs; it also holds
retained datasets, features and checkpoints. Databrary agreement/access PDFs
are in `outputs/exports/databrary/`. Use system temporary storage for scratch work.
`.external/` holds environments and upstream checkouts; `data/` and `downloads/`
retain local assets. These roots are ignored, but their contents are not
automatically disposable. Reusable inputs and mixed run/input folders retain
their existing paths. Completed standalone runs and retired reviews/motion
artifacts are under `outputs/legacy/`; see the
[relocation map](archive/README.md#local-artifact-relocation).
Archived source snapshots are excluded from default test discovery.

```text
outputs/
  assets/                 Object assets and evaluation images
  embodiedgen_scene/      Room bank and previews used by the simulator
  renders/clean_856/       Retained source dataset
  speech_data/            Source speech research data
  runs/                   Reusable features plus existing baseline/run records
  exports/                Documents and conversation exports
  legacy/
    motion/               Retired motion trials and previews
    reviews/              Completed audits, literature and handoff material
    runs/
      heavy_light/        Completed property-word diagnostic
      sensory_pretraining/ Completed sensory-pretraining comparison
    …                       Earlier archives remain in place
```

`outputs/runs/clean_body/`, `outputs/runs/clean_video_only/`,
`outputs/runs/clean_perfect_naming/` and `outputs/runs/step1_matched/`
contain both saved inputs and experiment records; their
presence does not queue new runs. For new studies, use descriptive roots such
as `outputs/runs/body_representation_comparison/`, with explicit condition/seed
names. This is a naming example, not a created run. Completed study worktrees,
including their original run roots, live under `outputs/legacy/worktrees/` with
compatibility links at their previous locations.

```bash
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m pytest -q -p no:cacheprovider tests/test_learner.py tests/test_speech.py tests/test_physical.py
PYTHONDONTWRITEBYTECODE=1 .external/egocentric-env/bin/python -m pytest -q -p no:cacheprovider tests/test_simulator.py
```
