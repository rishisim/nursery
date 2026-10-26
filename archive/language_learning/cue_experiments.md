# Contact-cue experiments and superseded hand gates

These experiments are complete. They are not the active learner protocol.
See [the archive index](../README.md) and [current plan](../../PLAN.md).

## What the cue studies asked

Could contact timing improve picture-to-word learning by giving more training
weight to some image/speech pairs? The learner was not given the rich bodily
stream now used by Nursery. No-cue, real-contact, shuffled, shifted, noise and
oracle conditions separated cue timing from indiscriminate weighting.

The weakness of the speech relationship was deliberate: if all speech already
identified the visible object perfectly, there would be less ambiguity for a
cue to resolve. This is different from deliberately creating inaccessible
cross-clip evidence; see [the design-history clarification](design_intent.md).

## Initial grid: 90 runs

Record: [cue_experiment.json](results/cue_experiment.json), committed in
`e4d9dffb`. Five seeds per cell, 18 cells, 90 completed runs. The held-share
settings were 0.3, 0.6 and 0.9. These are the old protocol's parameters, not
the current generator's calibrated held-channel share.

At the primary held-share 0.6 setting, contact minus no cue was **+2.27
percentage points**, 95% paired-seed interval **[−9.34, +13.89]**. The
preregistered verdict was **inconclusive**. Some high-held-share contrasts were
positive; they did not replace the primary outcome.

A staging/render defect invalidated early half-A artifacts. The recorded final
grid used corrected caches and rendered data. Keep invalidated evidence apart;
do not combine it with corrected results. The detailed local execution report
remains at `outputs/legacy/cue_experiment/report.md` (ignored, not shipped in Git).

## Independent confirmation: 100 runs

Protocol: `77cc0901`; result: `139a445c`, preserved unchanged in
[cue_confirmation.json](results/cue_confirmation.json). Twenty-five fresh seeds
(5–29), four conditions, 100 runs; no pooling with the original five seeds.

| Contrast | Mean gain, percentage points | 95% paired-seed interval |
|---|---:|---:|
| Contact − no cue | +0.94 | [−0.95, +2.83] |
| Contact − shuffled | +3.78 | [+1.63, +5.93] |
| Contact − shifted | +4.33 | [+2.19, +6.48] |

The frozen verdict was **timing_only**. Correctly timed weights beat corrupted
weights, but a gain over ordinary no-cue training was not established. This
does not demonstrate a higher language-learning ceiling, nor does it reject
feeding richer sensory information into a learner.

## Hand-location gate attempts

The subsequent implementations (`7ec9bd17`, `b2ed6599`) used body information
to influence visual-region selection. The learned trust in that cue collapsed
toward zero; these approaches were replaced by the information-bearing body
pathway in `8b70cca2`.

The early design also treated some legitimate category information in contact
and hand shape as leakage, coarsening/noising it and demanding weak body-only
word prediction. This was an overly broad restriction, not a standing scientific
requirement. Excluding hidden identity labels is different from removing what
touch or proprioception legitimately reveals.

The local design is preserved at `outputs/legacy/step3/body_input_design.md`;
Git preserves the original implementations. The current implementation reference
is [docs/learner.md](../../docs/learner.md). The later direct-label diagnostic
showed visual headroom, so the old “perfect naming adds only about two points”
observation must not be promoted into a general ceiling claim.

## What this means for the build

Do not repeat the 190-run cue program as if no timing controls existed. It tested
weighting and corrupted timing under its own data/protocol. The remaining
question concerns richer bodily inputs and learning transferable meanings from
weak language, with matched exposure and valid tests. Results from steps
4.1–4.13 are preserved separately in [experiments.md](experiments.md).
