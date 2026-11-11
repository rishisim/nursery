# Current Nursery plan

## Active objective

Measure direct language-acquisition curves for video-only (V) versus the same
video plus bodily inputs (V+B), across amounts of independent training
experience. The long-term hypothesis is a higher language-learning ceiling;
the existing experiments neither establish that benefit nor rule it out.

## Current direction and six steps

First add simulated child gaze to the world: it drives what the parent names
and is a new learner input. An error breakdown of Step 1 showed most errors
come from naming small background objects the child isn't attending to.
Then improve the bodily information available to the learner before choosing how to
pretrain it or introducing another fusion design. Preserve useful detail already
present in the recordings and explicitly record sensations that require new
physical modeling or measurements. Active exploration remains deferred.

The documents below follow the earlier study format: the idea, the experience
or comparison, how it works, and the output we want. Step 1 now includes its
frozen protocol and completed results. Step 2's world generation is complete,
and all 82,393 production close-ups and frozen-DINO rows are independently
validated. All twelve frozen additive runs and the independent Phase D audit
are complete at 630 clips. Video only scored 63.61%; video + gaze scored
63.25% with gaze and 60.13% with gaze removed. Picture-only transfer also
declined. Steering, the alternative gaze wiring, was then run under the same
frozen settings: 75.72% with gaze and 66.31% with gaze removed, beating video
only in every seed, with picture transfer up 6.5–10.6 points. Steering video +
gaze is the candidate to carry to the acquisition curves.
Step 3's richer body input is now built and frozen. Step 4 is complete:
neither current nor rich body adds to steering gaze, because dev selection
switches the body score off; rich body is carried to Step 5. Steps 5–6 remain
design references, with settings still to be settled.

| Step | Document | Question and current status |
|---|---|---|
| 1 | [Visual baseline](docs/step_1_visual_baseline.md) | Complete and audited: at 630 clips, recorded 4×4 video scored 60.12%, 16×16 scored 61.93%; paired gain +1.81 points over three seeds. |
| 2 | [Child gaze](docs/step_2_child_gaze.md#completed-results--9-november-2026) | Complete and independently audited: 12/12 frozen runs, four conditions × seeds 0–2 × 60 epochs, 630 clips. New-world video 63.61%; video + gaze 63.25% with gaze (−0.36 pp), 60.13% removed (−3.48 pp). Shuffled 50.54%/59.47%; point-only 62.59%/62.35%. Picture transfer declined. One US 8×5090 attempt cost $4.416; verified results back in 66m51.818s, rental destroyed, zero instances and key removal confirmed. Current additive gaze is not selected for curves. [Steering](docs/step_2_child_gaze.md#steering-gaze--9-november-2026) (6 runs, audited): 75.72% with gaze (+12.11 pp), 66.31% removed (+2.70 pp), positive in every seed; s8 +13.45/+2.90 pp; pictures 82.09%/84.58% vs 75.62%/73.98%. Shuffled steering at or below video. Steering is the candidate for curves. |
| 3 | [Body representation](docs/step_3_body_representation.md#frozen-candidate--10-november-2026) | Built and frozen (input only, no language result yet). Rich B keeps the ±5 s window but summarizes 30 Hz source frames per step, adds contact events and wrist-frame contact geometry, finger speed and 30 Hz arm motion: 193 dims vs 153. 22 contact patches kept by a fixed train→dev rule. All checks passed. On dev, held-category probe 23.7% → 31.7%; brief contacts visible 32% → 100%. Pressure, friction, texture, deformation and spatial heat stay deferred. |
| 4 | [Representation comparison](docs/step_4_body_representation_comparison.md) | Does richer body help under the same additive setup, without wordless pretraining? [Complete](docs/step_4_body_representation_comparison.md#completed-results--11-november-2026), 12/12 verified runs at 630 clips. New-clip test: video 63.67%, + gaze 75.60%, + gaze + current body 76.26%, + gaze + rich body 76.14%. Body − gaze +0.66 / +0.54 pp; on s8 −0.67 / −0.34 pp; rich − current −0.12 pp. Dev chose body weight ≈ 0 in all six body runs; at the as-trained weight body costs 6.5–8.2 pp. Rich body is the Step 5 candidate. Spend $10.48. |
| 5 | [Body training recipe](docs/step_5_body_training_recipe.md) | After the representation is understood, does wordless pretraining help? Additive remains the default; joint interaction is deferred. |
| 6 | [Language acquisition curves](docs/step_6_language_acquisition_curves.md) | Does the selected video-plus-body setup help across amounts of experience? This is the main outcome. |

Step 2 changes the world the later steps run in: speech becomes gaze-driven, so
Steps 3–6 use it and re-establish their video baselines there. Steps 1 and 3
meet in Step 4. Step 5 settles the learning recipe before Step 6.
A weak intermediate result is a design signal for the build, not a standalone
paper framing or an automatic instruction to launch more conditions.

Step 1 supports retaining 16×16 as the visual reference. Its modest gain on the
broader task is separate from the small four-noun diagnostic's roughly 92%
accuracy. Step 2 retains the gaze-driven world and its new video baseline, but
the current additive close-up/point input supplied no overall gain and reduced
video-only performance after removal in every seed. Steering gaze, where gaze
guides which patches a word is matched to, raised accuracy with gaze removed and
on gaze-free pictures in every seed, so gaze improved what the video path
learned. That is one data amount; whether gaze raises the ceiling needs the
acquisition curves. No new gaze acquisition curves have been run. Step 3's rich body input is
frozen; whether it helps word learning is Step 4's question.

## Decisions to keep consistent

- Count gaze as embodied experience. G denotes steering gaze location plus its
  attended sharp close-up; B denotes the other body channels. All conditions
  share gaze-driven speech and main video; V receives neither G nor B.
- At fixed data compare V, V+G, V+G+current B and V+G+richer B, all at 16×16.
  Measure gaze gain, additional body gain, representation gain and total gain.
  Body effects here are conditional on gaze; no automatic body-without-gaze arm.
  Preserve 4×4 results as explicitly historical references.
- Plan three acquisition curves: V, V+G and V+G+selected B. Keep current B in
  the fixed-data comparison. An unsupported body candidate does not prevent
  a V versus V+G study or require a positive outcome.
- Gaze steers visual matching; B adds a calibrated word score. Keep steering
  settings matched and visual weights trainable. Primary evaluation uses each
  learner's assigned inputs; report all-extra-inputs-removed and picture transfer
  separately. Shuffle B with real gaze retained to test its alignment.
- Use additive fusion and no wordless pretraining for the initial representation
  comparison. Apply the same development-only body-weight selection rule to
  both body conditions; test outcomes do not tune the design.
- Preserve natural weak speech, source-clip partitions and accessible time
  spans. Finer temporal sampling does not grant a longer body window. Record
  necessary encoder-capacity and compute changes.
- If wordless pretraining is selected for the curves, use only each point's own
  selected training clips, including normalization. Extra optimizer passes are
  compute, not new independent experience. Both arms share DINO's pretrained
  visual prior; this is not perception learned from scratch.
- Count independent source clips on the experience axis. Separate uncertainty
  across training seeds from uncertainty across test clips. Reused panels
  remain exploratory; repeated seeds are not new test situations.
- Keep new physical-property tasks, substantial new sensory simulation, active
  exploration and joint-architecture searches deferred.

## Preserved starting points

The completed [cross-situational study](archive/language_learning/cross_situational_learning.md)
and [original acquisition curves](archive/language_learning/acquisition_curves.md)
are archived with their protocols, results and source revisions. The
[archive index](archive/README.md#completed-study-worktrees) identifies their
full local worktrees, runs and compatibility paths.

The 16×16 implementation is commit
[`edb6b1a4`](https://github.com/rishisim/nursery/commit/edb6b1a47474bc08cd72ebc795e782b7eed0675a)
on `research/acquisition-curves`. That implementation and its acquisition
runner are now integrated into this checkout for Step 1. Its original 83
software checks were not a scientific experiment; the new comparison has its
own protocol and retained output root.

- Keep `outputs/renders/clean_856/`, saved speech/features and past run roots in
  place. They are retained inputs/evidence, not an active experiment queue.
- Read [Past experiments](archive/language_learning/past_experiments.md) before
  designing a new comparison; [experiments.md](archive/language_learning/experiments.md)
  remains the detailed historical protocol/result record.
- Use [the learner reference](docs/learner.md), [speech evidence](docs/speech_coupling.md)
  and [design intent](archive/language_learning/design_intent.md) for existing
  implementation and weak-alignment constraints. The [archive](archive/README.md)
  also holds earlier simulator work and the historical literature review.
