# Nursery

Nursery tests whether extra modalities (contact, finger pose, arm position) raise
the language-acquisition ceiling beyond what video alone reaches. A simulator
replays GRAB grasps as egocentric head-camera sessions, a speech generator adds
caregiver utterances tied to what the infant does, and a learner built on the
pinned [vlm-headcam](https://github.com/mcfrank/vlm-headcam) compares video-only
with video + body. Start with the [current plan](PLAN.md) and
[learner reference](docs/learner.md). The active objective is direct acquisition
curves across independent training-data amounts. Read [Past experiments](archive/language_learning/past_experiments.md)
for the prior results and their limits; detailed records are in the
[research archive](archive/README.md).

The EmbodiedGen room builder and HOIDiNi/InterMimic motion adapters
(`nursery/embodiedgen_scene.py`, `nursery/humanoid/`) supply rooms to the
simulator and retain the earlier physics work. The motion experiments are
inactive; their commands and measured failures are preserved in the archive.
The simulator still imports a shared output helper from the motion package, so
that source remains in place. Retained diagnostic learner modules likewise
support reproduction and shared helpers; their presence is not a run queue.

Generated rooms, assets, scenes, and previews belong under the ignored
`outputs/embodiedgen_scene/` root. External checkouts (EmbodiedGen, HOIDiNi,
InterAct, InterActMove, InterMimic, vlm-headcam) live under the ignored
`.external/`; they are dependencies, not Nursery-owned source.

## Layout

```text
nursery/
  simulator/    GRAB replay, Objaverse assets, mesh swaps, Blender render
  speech/       Caregiver speech: transcripts, utterance templates, cue weights, speech config
  learner/      Embedding and training on the pinned vlm-headcam, body input, Konkle eval, learner config
  common.py     Repo root, merged config loading, ignored-output guard
  humanoid/, embodiedgen_scene.py   EmbodiedGen rooms and HOIDiNi/InterMimic motion
tests/          test_simulator.py (egocentric-env), test_speech.py and test_learner.py (cue-env)
docs/           Current learner implementation reference and speech evidence
archive/        Completed experiments, historical commands and curated aggregates
scripts/        render.sh, pipeline.sh, train_seeds.sh, summarize.py, merge_render.py
outputs/        Ignored data and runs
  assets/       objaverse, konkle
  renders/      clean_856 (retained source dataset)
  runs/         Retained speech, features, trained conditions and historical run evidence
  speech_data/  osf_254yq (research use only, never commit)
  legacy/       Earlier experiments, working directories and superseded runs (patched/)
.external/      Ignored envs (cue-env, egocentric-env) and checkouts (vlm-headcam, HOIDiNi, ...)
```

Configs: `nursery/simulator/config.json` (render), `nursery/speech/config.json`
(generator, cues, vocabulary map), `nursery/learner/config.json` (training,
Konkle root, upstream pin). `nursery.common.load_config()` merges the speech and
learner files into the single dict stored in each run's `generator.json`.

## How to run

Render, then speech, embed, train, summarize. Paths are relative to the repo root.
The ignored venv `.external/cue-env/` was created with `uv venv --python 3.11 .external/cue-env`;
use it with the upstream checkout at the commit in `nursery/learner/config.json`.
Embedding and training pick CUDA, then Apple MPS, then CPU.

```bash
scripts/render.sh outputs/renders/new                      # resumable; episodes land in outputs/renders/new/episodes
.external/cue-env/bin/python -m nursery.speech transcripts --episodes outputs/renders/new/episodes --output outputs/runs/new --seed 0
.external/cue-env/bin/python -m nursery.learner embed --episodes outputs/renders/new/episodes --output outputs/runs/new
.external/cue-env/bin/python -m nursery.learner train --run outputs/runs/new --condition none --seed 0
python scripts/summarize.py 'outputs/runs/new/none_s*/metrics.json'
```

Body conditions (design in `docs/learner.md`; the body flags need `--pool soft`):

```bash
.external/cue-env/bin/python -m nursery.learner train --run outputs/runs/new --condition none --seed 0 --epochs 60 --pool soft --name video_only
.external/cue-env/bin/python -m nursery.learner train --run outputs/runs/new --condition none --seed 0 --epochs 60 --pool soft --body real --name body_real
```

`scripts/pipeline.sh <episodes> <run_dir> [seed]` chains speech, embed and train.
`scripts/train_seeds.sh <run_dir> none "--condition none" 0 1 2 3 4` trains several seeds, two at a time.

## Room generation

The shared room builder remains available:

```bash
python -m nursery.embodiedgen_scene prepare-bank --config configs/embodiedgen_scene.json
python -m nursery.embodiedgen_scene build --activity "A person folds laundry beside a bed" --seed 7 --config configs/embodiedgen_scene.json
```

The previous scene-motion, conversion and controller commands and validation
records are in [the motion archive](archive/motion_pipeline.md).

## Egocentric GRAB replay

`nursery/simulator/` renders 120-second, 1 Hz head-camera sessions
from HOIDiNi's retargeted GRAB with neutral SMPL-X and recorded full hand poses.
Inputs are licensed GRAB/SMPL-X at the paths in package-local `config.json`,
the existing `.external/egocentric-env/`, Blender, and a real exported room bank.
Session and dev/test probe rooms are disjoint; outputs must be Git-ignored.
Each clip draws a target of 3–5 static distractors from a shuffled list of other
GRAB categories. Bounded placement attempts preserve tabletop and object/hand
clearance; the realized count may be zero. The session render receipt records
per-clip target and realized counts, and the build manifest reports their distribution.
Frame `objects` records all realized categories, including zero-pixel visibility.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
.external/egocentric-env/bin/python -m nursery.simulator build \
  --rooms /Users/rishisim/Documents/research/nursery/outputs/embodiedgen_scene/room_bank \
  --subjects s1 s9 s10 --sessions-per-subject 2 --seed 7 \
  --output outputs/renders/run
.external/egocentric-env/bin/python -m nursery.simulator preview \
  --episodes outputs/renders/run --output outputs/renders/run/contact_sheet.png
PYTHONDONTWRITEBYTECODE=1 .external/egocentric-env/bin/python -m pytest -q -p no:cacheprovider tests/test_simulator.py
```

Omit `--subjects` for all subjects; Phase 2 uses `--subjects s1 s2 s3 s4 s5 s6 s7 s8`.
Probes still render. `--workers N` packs jobs; `--session-index N` renders one
scheduled session without probes. Gates: labelled contact median <1 cm,
held object visible in ≥70% of contact frames, mean visible categories ≥2.
The fraction of frames with ≥2 visible categories is a statistic only.
Full-body SMPL-X is batched over the rendered 1-Hz frames. Hand-only upstream
skinning retains exact full-rate hand clearance bounds, and object bounds use
all source frames. There is no sampled-bound displacement inflation. The unchanged
1-cm contact gate uses 1-Hz frames plus up to 200 evenly spaced labelled-contact
frames per clip, with exact triangle-surface distances. Whole-hand diagnostics
continue to use the rendered frames.

The [archived replay acceptance record](archive/language_learning/past_experiments.md#replay-acceptance-record)
preserves prior pilot measurements. Adult replay is an infant proxy; head turns
can hide objects, placement rejection biases visible categories, and kinematic
replay does not provide a physical contact-force solve. Data and measurements
remain in ignored run roots.
