# Nursery

Nursery asks whether video plus bodily experience supports better language
learning than video alone, as independent training experience increases.
The simulator supplies egocentric video and body signals; caregiver speech
provides intentionally uncertain language supervision.

Start with [PLAN.md](PLAN.md). The two proposed studies are
[acquisition curves](docs/acquisition_curves.md) and the smaller
[cross-situational comparison](docs/cross_situational_learning.md).
[Past experiments](archive/language_learning/past_experiments.md) explains what
we already tried. The small study is implemented; its current execution and
joint-interaction follow-up are recorded in its document.

## File map

```text
PLAN.md                         Current direction and study links
nursery/
  learner/
    video_only.py               Video → word matching (visual baseline)
    additive.py                 Video score + learned body → word score
    interaction.py              Direct visual/body scores + learned interaction
    cross_situational.py        Frozen small-study preparation and reporting
    heavy_light.py              Canonical load-only property-word diagnostic
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
  acquisition_curves.md          Main proposed study
  cross_situational_learning.md  Small-study protocol, results and joint follow-up
  learner.md                    Current implementation and scoring reference
  speech_coupling.md             Speech-generation evidence and assumptions
archive/
  code/learner/                 Completed pretraining and old joint pilot
  code/motion/                  Inactive HOIDiNi/InterMimic code and its tests
  language_learning/            Past experiments, protocols and curated results
  legacy_research/              Earlier simulation source/config/test archive
  README.md                     Historical evidence and artifact locations
```

**Video-only and additive share the same visual base, training and evaluation.**
The additive learner extends `video_only.py`; it does not carry a second copy
of the visual implementation. Both use the pinned
[vlm-headcam](https://github.com/mcfrank/vlm-headcam) upstream implementation.
The revised joint learner in `interaction.py` preserves both direct pathways
and adds a zero-initialized learned interaction. It is reused from the
acquisition-curves implementation at commit `08ebb82c`, selected through
`--arch interaction`. The archived width-128 joint pilot is a different,
completed implementation; `--arch joint` is retained only for compatibility.

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
automatically disposable. Existing assets and saved runs stay at their recorded
paths; see the [archive index](archive/README.md).
Archived source snapshots are excluded from default test discovery.

```bash
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m pytest -q -p no:cacheprovider tests/test_learner.py tests/test_speech.py tests/test_physical.py
PYTHONDONTWRITEBYTECODE=1 .external/egocentric-env/bin/python -m pytest -q -p no:cacheprovider tests/test_simulator.py
```
