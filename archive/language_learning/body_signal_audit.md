> Historical step 4.5 diagnostic, archived 26 October 2026. The recommendation
> below was followed by the body-only loss and later beyond-vision tests; it is
> not the current work queue. See [subsequent experiments](experiments.md).
> Inline source and output paths remain relative to the repository root.

# Body-signal audit (no training of the language learner)

Data: `outputs/renders/clean_856` (856 sessions, 102,720 frames, 59,681 held frames, 50 held categories, 51 GRAB
source objects, 1,063 held clips). Held frame = a hand touches the object. Features come from `nursery/learner/body.py`.
Scripts: `outputs/audits/body_signal/` (ignored). The probe is a diagnostic. It never feeds the learner and is not
language evidence.

## Result

- Grasp sharing is **not** the main cap. Only 4 of 50 words (pen, spoon, lock, cheese; 2.6% of held frames) share a
  grasp source. Grasp-only predictability is 98.5% on s8 (96% balanced).
- Contact is healthy and recomputed per mesh. Finger pose is replayed unchanged, as designed.
- The real body features do carry word information across subjects: 18% balanced accuracy over 50 words
  (chance 2%, shuffled labels 2-3%). The learner's body-only score on s8 is at chance (25.5 on a 4AFC, 25 = chance).
- **Recommendation: prioritise the learner's body representation and word alignment, not the simulator.**

## 1. Grasp ambiguity

Source = GRAB object in the clip name (`s1/toothbrush_pass_1.npz`). Which word and instance a clip got is **not
recorded per frame**. The frame records `clip` and `held_category` only. I rebuilt it with `swaps.assignment()` on the
clip name and `swaps_v2.json`. The rebuilt word equals `held_category` on all 35,454 held frames that have a swap. The
other 24,227 frames use the GRAB mesh itself or an extra word from `vocabulary.json` (no instance).

- Sources that give more than one word: `toothbrush` (pen 322 frames, spoon 118; 6 pen clips, 3 spoon clips) and
  `cubesmall` (lock 844, cheese 290; 12 and 9 clips). Each clip gets one word and one instance (hash of the clip name),
  and the same clip has the same word in every session.
- Words that come from several sources (the reverse case): tube (cylinder small, medium, large), pyramid (small, large).
- All other 45 words map to one source each.

"Grasp-only predictability" (predict the majority word of the clip's source; not a ceiling on the body stream):

| | frequency-weighted | category-balanced |
|---|---|---|
| all held frames | 99.3% | 96.0% |
| s8 test, majority from s1-s7 | 98.5% | 96.0% |

Only cheese and spoon (the minority word of their source) score 0.

## 2. Contact quality

Mean distance between two frames (L1 over the 22 patch fractions; Jaccard over the 1,556 contacted-vertex bits). Pairs
are random frame pairs. "Between" = different word. "Same instance" = same word and instance, different clip.
"Same word, other instance" = same word, different instance. "Same clip" = same clip, other frame.

| group | between L1 / Jac | same instance L1 / Jac | same word, other instance L1 / Jac | same clip L1 / Jac |
|---|---|---|---|---|
| toothbrush (pen/spoon) | 1.14 / 0.129 | 1.25 / 0.121 | none (1 instance per word) | 0.79 / 0.352 |
| cubesmall (cheese/lock) | 1.94 / 0.142 | 1.89 / 0.135 | 2.22 / 0.166 | 2.02 / 0.217 |
| cube sizes (lock, cheese, juice, sandwich) | 3.17 / 0.083 | 3.55 / 0.088 | 3.45 / 0.086 | 2.65 / 0.205 |
| spheres (rock, ball, balloon) | 3.01 / 0.157 | 3.08 / 0.211 | 3.27 / 0.234 | 2.15 / 0.308 |
| torus (cookie, bagel, tape) | 1.79 / 0.113 | 1.90 / 0.126 | 1.96 / 0.117 | 1.43 / 0.265 |
| all held frames | 2.76 / 0.130 | 2.60 / 0.182 | 2.85 / 0.148 | 2.15 / 0.290 |

Between-object contact is no farther than the variation between clips of the same object. Contact is clip-specific,
not object-specific. Clip-mean patch L1 agrees (between vs same instance: toothbrush 0.97 vs 1.08, cubesmall 1.54 vs
1.32, spheres 2.57 vs 2.16, torus 1.49 vs 1.56).

Health, all held frames:

- Zero contact: 0% (held is defined by contact). 42% of all frames have no held object (approach and release).
- Saturated: 0.5% of frames have at least 50% of one hand's vertices on (swapped 0.8%, unswapped 0.02%); none at 80%.
  Median 12% of the contacting hand's vertices, 6 of 22 patches on.
- Distinct patterns per clip (mean 56 frames): 50 distinct vertex sets (90%) and 16 distinct binary patch patterns
  (median 11). Contact changes frame to frame.

Recomputed per mesh? Yes. I replayed the build's own `_swapped_contact` on 9 toothbrush and 21 cubesmall clips (up to
40 frames each) against all 12 pool meshes (pen and spoon; cheese and lock) of the same source:

| | contacted-vertex Jaccard | patch L1 | frames identical |
|---|---|---|---|
| toothbrush: same clip, pen mesh vs spoon mesh | 0.49 | 0.51 | 0% |
| toothbrush: same clip, two instances of one word | 0.66 | 0.24 | 0.5% |
| cubesmall: other word | 0.32 | 1.01 | 0% |
| cubesmall: same word | 0.38 | 0.88 | 0% |

So contact depends on the mesh. Pen vs spoon differs more than pen vs pen for the same clip, so a word-level cue exists
there. For cubesmall the cue between words is barely larger than between instances. In the rendered data a clip has one
mesh only, so clip variation hides this cue.

Finger pose: identical across objects by construction. `build.py` writes `fullpose[clip, frame]` from the GRAB clip;
the swap changes only contact and the rendered mesh. In the data, the distance between clip means (mean absolute pose
difference) is 0.183 between pen and spoon clips vs 0.171 between clips of one word, and 0.150 vs 0.148 for cubesmall.
That is clip-to-clip variation, with no object effect.

## 3. Supervised diagnostic (labels for diagnosis only)

MLP, one hidden layer of 128, 20 epochs, Adam 1e-3, batch 256, no sweep, no early stopping. 3 seeds (mean). Inputs from
`load_store`: contact = 22 patch fractions; pose/arm = 90 finger rotations + 27 arm features. "+win" adds the mean of
the same features over +-5 s inside the clip (the encoder's window). Train s1-s7 (52,484 held frames), test s8 (7,197
frames, all 50 categories). Whole clips stay together because clip names include the subject.

Split A (clip-disjoint, subject s8 held out):

| input | balanced | frequency-weighted | pen/spoon pair acc | lock/cheese pair acc |
|---|---|---|---|---|
| contact | 10.1% | 12.1% | 30.1% | 58.3% |
| pose/arm | 15.8% | 15.0% | 38.4% | 58.3% |
| both | 18.3% | 18.6% | 38.4% | 63.5% |
| contact +win | 14.2% | 15.3% | 22.4% | 64.2% |
| pose/arm +win | 11.9% | 12.6% | 38.4% | 55.6% |
| both +win | 18.0% | 17.6% | 38.4% | 55.6% |
| both, shuffled training labels | 2.4% | 2.5% | 39.7% | 46.9% |

Baselines on the same test frames: majority category 2.0% balanced, 6.0% weighted; grasp-only 96.0% and 98.5%; pair
majority 38.4% (pen/spoon) and 54.8% (lock/cheese).

- The features beat the majority and shuffled controls by about 8x, but are far below grasp-only. Predicted-word source
  is right only about as often as the word (19%), so the main failure is recognising the grasp across subjects. Each
  subject has different clips and motion.
- Within shared grasps, the probe is at the pair majority (pen/spoon 38.4%; balanced pair accuracy 50%). Lock/cheese is
  slightly above it (63.5% vs 54.8%, 61% balanced; 1 test subject, 135 frames, so weak).
- The window adds little. Contact alone gains (10 to 14%); pose/arm and both do not.

Split B (instance-disjoint) is **not scored**. Dropping every instance that s8 uses leaves training data for only
21 of the 50 categories (18 have no instance concept, 11 swapped ones lose all instances, including all four
shared-source words). Those 21 words have a mean of 2.5 training clips (min 1, max 12) against 18 in split A. Scores are
at chance (1-3%) for all inputs, including pose/arm, which cannot depend on the instance. That reflects clip scarcity,
not instance generalisation. On the same 21 test categories the clip-disjoint split gives 8-14% (13% for both).
No instance-disjoint conclusion is possible with this render.

## Limits

- One test subject (s8), 3 seeds of one fixed MLP; no sweep.
- Instance is rebuilt from the clip name, not recorded. Pen and spoon each have a single instance in the rendered data.
- Word-level contact cue in shared-grasp clips was measured on replayed meshes, not on rendered frames.

## Recommendation

Prioritise the learner's body representation and word alignment. The simulator signals are not defective (contact is
per mesh, pose is intact, grasp ambiguity is 4 words), and the features carry about 8x-chance word information across
subjects that the learner does not use (body-only 25.5 vs 25 chance). One change to try: add a body-only word-matching
loss on the same named-object pairs (body latent against the word, without the picture score), so the body path cannot be
bypassed by the picture path. Pilot: s8 body-present 4AFC, body-only and combined, against the existing control and a
shuffled-body run. Body-only recognition is not required for a multimodal gain; this only checks that the path learns.
This also corrects the PLAN 4.4 note that shared grasps cap what the body can know: they affect 4 words.
