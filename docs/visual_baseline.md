# Step 1: retain visual detail and keep the historical comparison

Step 1 of [PLAN.md](../PLAN.md), authorized on 6 November and completed on
7 November 2026. The preserved 16×16 implementation is integrated. All three
matched scientific runs completed, and the independent final audit passed.

## The idea

Compare body against a visual learner that retains the spatial detail already
extracted by DINO. The old encoder averaged a 16×16 patch grid down to 4×4.
Keep the original crop, frozen DINOv2 model and five-frame visual window; retain
the 256 patch tokens plus CLS instead of 16 averaged tokens plus CLS.

The question is how much this helps the broader word task. The four-noun
diagnostic found a substantial improvement, but it did not measure the
acquisition curves or their eventual ceiling.

## What we already have

The saved four-noun diagnosis reported video-only accuracy of 76.88% with 4×4
tokens and 92.19% with 16×16 tokens over ten paired seeds: +15.31 percentage
points. It used matched crops, examples and test panels. These are exploratory
results on a reused small panel, not an expected improvement for every task.
Its local source report is identified in the
[archive index](../archive/README.md#body-diagnosis-evidence).

The original acquisition study used 4×4 features and three paired seeds:

| Independent training clips | Recorded 4×4 video accuracy, mean ± sample SD |
|---:|---:|
| 63 | 30.0 ± 1.6% |
| 158 | 39.5 ± 1.7% |
| 315 | 49.6 ± 1.2% |
| 473 | 56.4 ± 0.8% |
| 630 | 60.1 ± 0.6% |

These are measurements from the
[completed acquisition record](../archive/language_learning/acquisition_curves.md#completed-results--27-october-2026).
Keep its per-seed results and trial records; do not substitute the four-noun
92.19% into this table.

## How we use the baseline

All new live conditions in the [representation comparison](body_representation_comparison.md)
use the same 16×16 features. Include the recorded 4×4 video row beside them,
with its original data amount, task and protocol labeled. Reuse the historical
measurement rather than automatically adding another training arm.

Before attributing a difference to resolution, check matching source clips,
speech, crop/window, vocabulary fitting, test candidates, seeds, training
budget, scoring and checkpoint selection. If these differ, retain the old row
as historical context and label the difference as non-matched. A new evaluation
panel does not make an old-versus-new difference a resolution-only effect.

## Implementation and retained inputs

Commit [`edb6b1a4`](https://github.com/rishisim/nursery/commit/edb6b1a47474bc08cd72ebc795e782b7eed0675a)
on `research/acquisition-curves` adds 16×16 embedding support, region-count-aware
training/evaluation and guards against mixing cache dimensions. Its 83 passing
synthetic software checks establish code behavior, not learning performance.
The full branch/worktree is preserved in the
[worktree archive](../archive/README.md#completed-study-worktrees).

That implementation and its original acquisition runner are reused in the
working learner. New train and picture-evaluation caches live under
`outputs/features/visual_grid16/`; old caches and provenance remain unchanged.
The token count rises from 17 to 257 per frame. The complete new arrays need
about 38.6 GiB; upstream encoding is sharded and merged with a disk-backed
array to fit the 32-GiB machine without changing crops or model arithmetic.

## The output we want

| Main-task comparison at 630 clips | Held-out accuracy, mean ± sample SD | Paired difference from recorded 4×4 |
|---|---:|---:|
| Recorded 4×4 video reference | 60.12 ± 0.56% | — |
| 16×16 video | 61.93 ± 0.93% | +1.81 ± 1.47 percentage points |

This comparison is matched as specified below. The four-noun result above
remains separate. This baseline feeds Steps 3 and 5.

## Execution protocol

Use the 100% point from the original acquisition study: 630 independent clips,
18,613 utterances and 63,153 replay seconds. Train only video-only models, with
paired seeds 0, 1 and 2, each fresh for 60 epochs. Match batch 256, AdamW learning
rate 0.0003, weight decay 0.1, soft pooling temperature 0.1 and the five-frame
window. Use the same speech, clip split, vocabulary, ordered examples, candidates,
unknown-word rule and earliest development-maximum checkpoint selection.

Reuse the original local DINO snapshot
`f9e44c814b77203eaa57a6bdbbd535f21ede1415` offline, with its resize and centre crop.
Verify CLS agreement and that pooling new patch tokens back to 4×4 reproduces
the saved cache within floating-point/float16 rounding. Verify prepared subsets,
panels and cache row mappings against the archived run before training. The
intervention retains more tokens in the same readout; the relative contribution
of CLS and patch tokens to soft pooling changes with token count.

Use local MPS and the original timing gate: one development-only epoch, discard
its weights, and project all three 60-epoch models with 25% headroom. The
24-hour cap begins at the timing gate; feature preparation is recorded
separately. Preserve partial work if the cap is exceeded. No paid compute,
body training, pretraining or other data amounts are included.

The new ignored scientific root is `outputs/runs/visual_baseline/`; preparation
and cache validation records live in `outputs/validation/visual_baseline/`.
The feature root retains exact copies of the original speech/provenance inputs.
Commands after feature creation are:

```bash
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m nursery.learner acquisition prepare \
  --root outputs/runs/visual_baseline --inputs outputs/features/visual_grid16 \
  --episodes outputs/renders/clean_856 --fractions 1.0 --conditions video_only
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m nursery.learner acquisition timing \
  --root outputs/runs/visual_baseline
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m nursery.learner acquisition run \
  --root outputs/runs/visual_baseline
```

Report all seed scores, means and sample standard deviations, and paired
16×16-minus-recorded-4×4 differences. New-clip accuracy is primary; s8 and
picture/Konkle transfer are separate supporting results. This reused-panel,
fixed-data comparison does not establish an acquisition ceiling or replace
the later curves.

## Completed results — 7 November 2026

The three 16×16 video-only models completed 60 epochs at execution revision
[`6c868e23`](https://github.com/rishisim/nursery/commit/6c868e232915ee3fd6e0c4b6f87d10d34ca68939).
The reference measurements are the saved 4×4 models from the original
acquisition study; they were not retrained. Both used 630 independent clips,
18,613 utterances, the same three seeds and the same development selection.

Primary new-source-clip word matching:

| Seed | Recorded 4×4 | 16×16 | Paired change |
|---:|---:|---:|---:|
| 0 | 60.35% | 61.23% | +0.88 pp |
| 1 | 59.47% | 62.98% | +3.51 pp |
| 2 | 60.53% | 61.58% | +1.05 pp |
| Mean ± sample SD | 60.12 ± 0.56% | 61.93 ± 0.93% | +1.81 ± 1.47 pp |

Separate supporting panels, also mean ± sample SD across the three seeds:

| Panel | Recorded 4×4 | 16×16 | Paired change |
|---|---:|---:|---:|
| Held-out person, s8 | 55.57 ± 0.26% | 58.15 ± 0.78% | +2.58 ± 0.51 pp |
| Rendered picture-only transfer | 69.10 ± 0.15% | 70.24 ± 1.35% | +1.14 ± 1.20 pp |
| Konkle transfer | 63.80 ± 2.58% | 64.83 ± 2.24% | +1.03 ± 2.58 pp |

New-clip, s8 and rendered-picture scores improved in all three paired seeds.
Konkle declined by 1.90 points in seed 2. These standard deviations describe
training-seed variation on fixed, reused panels; they are not confidence
intervals across independent new test panels.

The larger token grid gives a modest improvement on the broader weak-speech
task. Its +1.81-point primary gain is much smaller than the four-noun
diagnostic's +15.31 points. That diagnostic's roughly 92% accuracy does not
describe saturation of this benchmark. One data amount cannot establish a
plateau or determine the gain at smaller experience amounts.

For the next build step, keep 16×16 as the visual reference and proceed with
the [body representation design](body_representation.md). The result supports
retaining the available visual detail, while useful body representations and
their acquisition benefit remain unresolved. No body comparison or new curve
point was launched by this result.

## Validation and retained evidence

All 104,988 images were encoded using the original offline DINO snapshot.
The arrays occupy 41,444,223,360 bytes (38.6 GiB); encoding took 33.1 minutes.
Every feature value is finite. All frame tables and cache key-to-row mappings
match the old caches. Across 64 evenly spaced sampled images in each of the
three caches, CLS tokens matched exactly and pooling back to 4×4 had relative
RMSE about 0.00021, consistent with float16 rounding.

Preparation reproduced the original speech ordering, splits, vocabulary,
normalization, fit mask and every frozen panel. Software validation passed
102 tests. The development-only timing gate projected 10.3 hours including
25% headroom; actual summed scientific model time was 6.59 hours. Timing
weights were removed, and every scientific model started fresh.

The recorded 4×4 video runs took 23.9 minutes in total, so the new runs used
about 16.5 times their recorded wall time. The comparison matches examples
and optimization steps; it does not match compute. These are observed timings
from different dates, not a dedicated hardware-speed benchmark.

The independent final audit checked 68 source hashes, 16 prepared-file hashes
and 1,726 external input hashes; reconstructed trial credits and scores;
verified earliest development maxima; and strict-loaded all six finite
selected checkpoints. Source/input freezing passed before this result update.

Retained local, ignored evidence:

- `outputs/features/visual_grid16/`: complete new caches, original encoder
  snapshot and copied speech/provenance inputs.
- `outputs/runs/visual_baseline/`: frozen protocol and inputs, timing record,
  all three model histories, per-trial scores and selected checkpoints,
  `summary.json`, `summary.csv` and `report.md`.
- `outputs/validation/visual_baseline/`: preparation and encoder checks,
  `comparison_preflight.json`, `final_audit.json` with exact historical/new
  seed values and paired differences, audit scripts and execution/test logs.

Generated arrays, runs and reports remain local and untracked. The code,
protocol description and compact result tables are versioned here. The
historical worktrees, caches and models remain preserved.
