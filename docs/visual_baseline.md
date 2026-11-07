# Step 1: retain visual detail and keep the historical comparison

Design reference for [PLAN.md](../PLAN.md). Full 16×16 token support has been
implemented and software-tested on the preserved acquisition branch. No new
acquisition experiment is launched by this document.

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

When implementation resumes, reuse that change in the one working learner.
Prepare new train and picture-evaluation caches under a fresh ignored output
root; preserve old caches and provenance. The token count rises from 17 to 257
per frame, so measure storage, memory and timing before choosing a budget.
Do not silently shrink data or context to fit.

## The output we want

| Main-task comparison at the selected data amount | Held-out accuracy | Difference from recorded 4×4 |
|---|---:|---:|
| Recorded 4×4 video reference | Existing matched-size result | — |
| 16×16 video | Pending | Pending, or labeled non-matched |

Identify each row's source and whether the difference is matched. Keep the
four-noun result separate. This baseline feeds Steps 3 and 5.

## Before execution

Confirm the archived change is integrated, compatible caches cover the chosen
data and panels, and fixed data/seed/budget rules are recorded. New main-task
16×16 scores and runtime remain unmeasured. This document does not queue a
rendering, training or cache-generation job.
