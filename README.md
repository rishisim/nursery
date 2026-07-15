# BabyWorld-Lite

A tiny working prototype for **synthetic developmental multimodal data**.

The point is not photorealism. The point is to show that a synthetic infant-like world can be deeply instrumented in ways real baby datasets usually cannot be: synchronized egocentric frames, child-directed language, action/proprioception traces, touch/contact signals, object state, causal event labels, and counterfactuals.

## What this demonstrates

Each generated episode contains:

- **RGB frames / GIF** from a toy egocentric scene.
- **Caregiver language** before and after the action.
- **Action command**: `tap`, `push`, `poke`, or `grasp`.
- **Proprioception**: hand position and velocity per frame.
- **Touch/contact**: contact binary, force, normal, slip/vibration proxies.
- **Object states**: position, velocity, angle per frame.
- **Causal event label**: e.g. `rolls_far`, `topples`, `slides`, `grasp_success`.
- **Causal graph and counterfactuals**: e.g. action/contact/material/impulse -> event.

It also includes a mini-benchmark that trains the same classifier under different input conditions:

1. vision + language proxy
2. plus action
3. plus sensorimotor/touch
4. oracle physical state

This gives a quick **data-usability** story: richer synthetic modalities can be evaluated by whether they improve action-effect prediction under fixed model/training conditions.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_dataset.py --n 1200 --out data/sample --render-first 12
python train_benchmark.py --data data/sample/manifest.csv --out data/sample
python make_demo_html.py --data data/sample --out demo.html
open demo.html  # or double-click it
```

## Leakage-aware forward-prediction eval

The honest eval reads `episodes.jsonl` directly and recomputes all features from
frames `[0, k]`, where `k` is the contact/prediction frame. It does not use the
legacy manifest summaries (`max_force`, `contact_count`, `slip_estimate`,
`vibration_energy`) or `hardness_proxy` as inputs. The training objective is
continuous rollout prediction: post-contact displacement, x/y delta, final x/y
position, and topple angle. Event labels are used only for coverage/diversity
reporting, not as model targets.

```bash
python3 generate_dataset.py --n 6000 --out data/honest --render-first 12
python3 run_honest_eval.py --episodes data/honest/episodes.jsonl --out data/honest/eval
```

By default the full run evaluates mechanism splits: held-out material,
held-out high impulse x high mass region, and held-out object/action
composition. A random split is still available explicitly as a contrast.

Outputs:

- `results.json`
- `forward_prediction_report.txt`
- `coverage_report.txt`
- `coverage_report.json`
- `{split}_displacement_mae.png`
- `k_sweep_error_vs_k.png`

For quick smoke tests on the bundled 800-episode sample, lower the exploratory
support threshold or run a subset of splits:

```bash
python3 run_honest_eval.py --episodes data/sample/episodes.jsonl --out data/sample/eval --splits random held_out_material --min-test-cell-size 1 --bootstrap 100 --skip-k-sweep
python3 -m pytest
```

## Pre-access grounded-language pilot

The grounding pipeline is deliberately labeled
`provisional_not_babyview_matched`. It creates the experimental infrastructure
needed before restricted BabyView access without claiming that its provisional
rates are empirical estimates.

It generates the same base episodes under three language-alignment conditions
(`strong`, `weak`, and episode-shuffled) and four whole-trajectory motor
conditions (`null`, synchronized, split-local episode-shuffled, and
time-shifted). Model-visible records contain only raw frame paths, timestamped
utterances, and low-level hand trajectories. Oracle state and causal labels are
written to a physically separate file. The persistent audit checks allowlisted
inputs, no-self and no-cross-split donors, composition/hash-disjoint splits,
equal inventories across alignment arms, and a caption/contact-marker-free
renderer.

The same config also drives provisional utterance rate/count/length/timing,
activity-window duration, visibility/occlusion, camera motion, and distractor
density. The audit records both configured and realized distributions,
including target-word frequencies and silent-frame fraction, so measured
BabyView aggregates can later replace the provisional values through the same
schema.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m babyworld_lite.grounding \
  --config configs/grounding_provisional.yaml \
  --out data/grounding_provisional

python train_grounding_pilot.py \
  --episodes data/grounding_provisional/examples.jsonl \
  --out data/grounding_pilot \
  --seeds 11 22 33 \
  --alignment weak \
  --device auto
```

The learner uses actual rendered RGB frames, utterance text, and continuous
hand `x/y/vx/vy` sequences. All arms share the same video, text, and motor
encoders, initialization, data order, and optimizer schedule. The locked
primary evaluation is a balanced held-out-composition action contrast; it
calls only the video and text encoders, with motor omitted at test. Results are
written to `pilot_results.json` with paired lift intervals, manipulation
checks, shortcut diagnostics, and an explicit claim gate. The small default
run is an infrastructure validation, not evidence that synchronized motor cues
improve language grounding.

### Official Machine-DevBench integration

The learner can now be saved as a standalone checkpoint and evaluated through
the public EgoBabyVLM Machine-DevBench protocol. The local runner imports the
official dataset discovery, task metadata, frequency-balanced aggregator, and
feature-extractor protocol from a pinned EgoBabyVLM checkout. It changes only
the Slurm/CUDA launcher so the same trial equations can run on Apple MPS.

```bash
git clone https://github.com/facebookresearch/egobabyvlm .external/egobabyvlm
PYTHONPATH=.external/egobabyvlm python -m scripts.eval_data.download_machine_devbench \
  --cache-dir .external/cache/machine_devbench

python scripts/train_nursery_checkpoint.py \
  --episodes data/grounding_provisional/examples.jsonl \
  --out output/machine_devbench/nursery_sync_seed11.pt \
  --arm synchronized --seed 11 --alignment weak --epochs 8 --device auto

python scripts/run_machine_devbench.py \
  --model nursery \
  --checkpoint output/machine_devbench/nursery_sync_seed11.pt \
  --out output/machine_devbench/nursery_full.json \
  --device auto

python scripts/audit_machine_devbench_coverage.py \
  --checkpoint output/machine_devbench/nursery_sync_seed11.pt
```

The public archive has 20 manifests and 3,721 trials across realistic and
cartoon styles. The provisional learner completes all of them, but only 13 of
1,414 unique target words are in its training vocabulary. Its current score is
therefore an integration diagnostic, not evidence about developmental language
learning. The post-access corpus must expand and calibrate the vocabulary before
Machine-DevBench becomes a substantive secondary outcome.

As an external correctness check, the full OpenAI CLIP ViT-L/14 run yields
86.27 lexical, 70.29 grammatical, and 78.28 overall, versus the published
CLIP-L row of 87.3 / 70.4 / 78.8. The 0.52-point overall difference provides a
close macOS/MPS reproduction of the public reference.

The optional two-page pre-access research attachment can be rebuilt with:

```bash
python scripts/build_frank_attachment.py
pdftoppm -png output/pdf/frank_preaccess_experimental_plan.pdf tmp/pdfs/frank-plan
```

## Aria Everyday Activities real-data phase

The real-data adapter uses Meta's Aria Everyday Activities (AEA) release as an
**adult, partly scripted sensor-format analogue**. It is not developmental
evidence and is not represented as BabyView-matched. The locked question is
whether synchronized six-axis head accelerometer+gyroscope input during
training improves motor-withheld video/ASR action grounding relative to a
split-local, whole-sequence-shuffled IMU control.

The initial plan selects 40 recordings: eight from each location and eight
from each script, using only predeclared ASR action-anchor counts and safe
manifest metadata before VRS inspection. The selected raw VRS budget is about
106.3 GiB. Annotations and raw VRS are acquired initially; all MPS outputs are
excluded.

```bash
# The user must personally accept the AEA license and obtain the expiring JSON.
python scripts/plan_aea_subset.py \
  --links "$HOME/Downloads/AriaEverydayActivities_download_urls.json" \
  --annotations-root data/aea_raw \
  --out configs/aea_subset_40.yaml

# This flag acknowledges an acceptance already completed by the user; the
# script never accepts a license. Signed URLs are never printed or copied.
python scripts/download_aea_subset.py \
  --links "$HOME/Downloads/AriaEverydayActivities_download_urls.json" \
  --plan configs/aea_subset_40.yaml \
  --out data/aea_raw \
  --components annotations main_vrs \
  --confirm-license-accepted-by-user

# VRS preprocessing uses a separate Project Aria environment. The existing
# project venv is Python 3.12, so use its interpreter to create it.
.venv/bin/python -m venv .venv-aria
.venv-aria/bin/python -m pip install 'projectaria-tools[all]'
.venv-aria/bin/python scripts/preprocess_aea.py \
  --config configs/aea_real.yaml --out data/aea_processed

# Training remains in the existing project environment.
.venv/bin/python scripts/run_aea_experiment.py \
  --examples data/aea_processed/examples.jsonl \
  --config configs/aea_real.yaml --out output/aea_real

# Build the canonical, source-backed technical-report payload. The Codex
# portable report builder can package this as a self-contained HTML report.
.venv/bin/python scripts/build_aea_report_artifact.py \
  --results output/aea_real/aea_results.json \
  --preprocess data/aea_processed/preprocess_summary.json \
  --out output/aea_real/artifact.json
```

The pipeline resamples the three accelerometer and three gyroscope axes together
at 50 Hz in SI units, keeps every window from a sequence together, groups or
purges concurrent recordings, uses paired seeds and a hierarchical bootstrap,
and omits IMU from the primary test code path. Held-out location,
wearer-session-proxy, and action-object-composition evaluations are reported
separately. The wearer field is explicitly a release-visible
location+script+recording proxy, not persistent person identity.

A restricted-data-free end-to-end smoke is available:

```bash
.venv/bin/python scripts/run_aea_smoke.py
```

Its outputs are labeled `infrastructure_smoke_test_not_a_real_data_finding`.
The report artifact records source provenance, metric definitions, the chart
contract, and the technical-report section mapping alongside the results.
See [the AEA phase protocol](docs/aea_real_data_phase.md) for acquisition,
audit, positive-control, inference, and reporting details.

## Why this is deliberately small

The full idea is huge. This pilot isolates the core claim:

> Synthetic developmental data can be cheap, scalable, perfectly instrumented, and benchmarked for usefulness even before it is photorealistic.

Next extensions:

- replace 2D physics with PyBullet, TDW, Habitat, or OmniGibson;
- add depth, segmentation masks, richer audio, and object IDs;
- train a small VLM/world-model instead of a classifier;
- test transfer to real egocentric data or held-out simulated worlds;
- add a cost-normalized leaderboard: performance per generated hour, per dollar, or per causal event.
