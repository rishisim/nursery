# Step 1: retain visual detail and keep the historical comparison

Step 1 of [PLAN.md](../PLAN.md), authorized for execution on 6 November 2026.
The preserved 16×16 implementation is integrated. Feature preparation is in
progress; the three new scientific runs have not yet completed.

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

| Main-task comparison at the selected data amount | Held-out accuracy | Difference from recorded 4×4 |
|---|---:|---:|
| Recorded 4×4 video reference | Existing matched-size result | — |
| 16×16 video | Pending | Pending, or labeled non-matched |

Identify each row's source and whether the difference is matched. Keep the
four-noun result separate. This baseline feeds Steps 3 and 5.

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
