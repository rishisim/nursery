# Step 7: property and action words

Design reference in [PLAN.md](../PLAN.md), following the
[noun acquisition curves](step_6_language_acquisition_curves.md). No code,
speech or run is launched by this document.

## The idea

Test words whose meaning is partly bodily, in the same four-choice format as
nouns. Given a word, the learner picks the matching moment from four. If extra
modalities raise the ceiling, the body gain should be larger here than on
nouns.

## Why this step

- On nouns, the body adds nothing over gaze at any data amount (Step 6). The
  learner keeps the body weight near zero.
- The test nouns are mostly visual. An earlier diagnosis found about 80% of
  what the body knows about them is already in vision, which caps any body
  gain on nouns at roughly 10 points.
- Children's early vocabulary includes property words (heavy, hot, wet) and
  action words (lift, push, give), so this is part of ordinary acquisition,
  not a constructed task.

## Word sets

| Rung | Candidate words | Where the meaning comes from |
|---|---|---|
| Nouns (control) | Existing noun vocabulary | Object category; vision should do well |
| Actions | GRAB intents: lift, pass, inspect, offhand, and object uses such as drink | Real recorded hand and arm motion |
| Properties | Heavy/light first; others only if honestly supplied | Load from mass and motion, plus grip and motion differences |

Properties must be physically consistent: mass is a persistent property of each
object, matched to its size and material, so vision gets partial cues (a metal
object looks heavier than a plastic one). The earlier
[heavy/light control](../archive/language_learning/past_experiments.md#heavylight-positive-control-415)
drew mass at random per clip, invisible to vision; it is not repeated. Recorded
motion does not react to assigned mass, so load is the only weight cue the
body gets. Softness, friction, texture and temperature words stay deferred
until they have believable physics.

## Speech

Extend the gaze-driven caregiver generator to name actions and properties
during the matching events, keeping the same weak alignment as nouns: delays,
names for things the child is not handling, and names with no visible
referent. Utterances may combine words ("lift the heavy cup"); the learner
still scores each word separately, so no new text encoder is needed at this
step. Word frequencies should follow the speech evidence where it exists and be
recorded where they are assumed.

## Learner change: temporal video

The video path currently pools patches from a five-frame window without using
their order, so it cannot represent motion. The body path reads about 11
seconds through a GRU. On action words that would favor the body by design.

All arms get the same small temporal encoder over the video frames, covering
the same accessible time span as the body window. Gaze steering applies to it
as before. Because this changes the shared learner, the noun rung is
re-measured with it so all rungs use one learner.

## The comparison

| Arm | Inputs |
|---|---|
| V | 16×16 video with temporal encoder |
| V+G | V plus steering gaze |
| V+G+B | V+G plus body, additive, no wordless pretraining |

630 clips, matched seeds, the Step 4 body-weight rule. Body representation
(current or rich) is fixed in the protocol before launch. Controls: shuffled
body with real gaze retained, and all-extra-inputs-removed. Each rung gets its
own test panel, sized before launch to detect about a 3-point difference, with
foils drawn from the same rung (an action foil is a different action).

## The output we want

One table per rung: V, V+G and V+G+B accuracy, with gaze gain, body gain and
total gain and their intervals, plus the selected body weights. The key
reading is whether body gain is larger on property and action words than on
nouns.

A null body gain here, with the body weight still switched off, points to how
the body is trained rather than what it receives; that is the
[Step 8](step_8_body_training_recipe.md) question. A body gain at 630 clips
moves to the curves in Step 9.

## Draft protocol — 17 November 2026

Status: draft for review. Nothing is built or launched. Counts come from
read-only scouting in `outputs/step7_scouting/` (`data_facts.md`,
`learner_facts.md`, `round2_facts.md`, `round3_facts.md`). No input reads
Databrary or BabyView recordings or derivatives; all data is GRAB-derived
synthetic scenes, speech and gaze.

### 1. Vocabulary

**Actions: lift, pass, inspect, offhand, drink.** Each GRAB clip has exactly
one intent, so the clip's intent is the word's meaning. Dropped: `use` (one
word covering six unrelated objects) and every other intent (at most 14 train
clips, nearly all tied to a single object).

| Word | Train clips / objects | Dev | Test | s8 | Projected train utterances |
|---|---:|---:|---:|---:|---:|
| pass | 189 / 50 | 45 | 43 | 48 | ≈537 |
| lift | 131 / 46 | 26 | 25 | 37 | ≈372 |
| inspect | 71 / 19 | 14 | 21 | 17 | ≈202 |
| offhand | 37 / 29 | 11 | 8 | 11 | ≈105 |
| drink | 33 / 5 | 11 | 9 | 8 | ≈94 |

**Properties: heavy, light.** Labels come from the instance mass below: top
third of the 126 rendered instances is heavy, bottom third light, middle third
unused (never named, never on the panel). Only clips where the object is lifted
off the surface while in contact count.

| Word | Instances / nouns | Mass | Lifted train | Dev | Test | s8 | Projected train utterances |
|---|---:|---:|---:|---:|---:|---:|---:|
| heavy | 42 / 25 | 250–858 g | 278 | 52 | 55 | 71 | ≈789 |
| light | 42 / 22 | 4.7–110 g | 140 | 45 | 31 | 35 | ≈398 |

Projected utterances use the current generator's own-object naming rate
(2.84 per train clip; own-object names are 8.6% of utterances). This rate is
assumed: the speech evidence gives no verb or adjective frequencies.

Known confounds, reported rather than removed:
- `drink` occurs only with five containers, so vision can partly answer it
  from the object. Object-matched foils (below) test this directly.
- Heavy/light is mostly predictable from the rendered noun. Five nouns (bell,
  bottle, doll, rock, train) have instances on both sides; their 27 held-out
  lifts form the subset where the noun alone cannot answer.

### 2. Mass

Use the simulator's existing per-instance mass: a deterministic log-uniform
draw inside the rendered noun's authored range, keyed by the rendered instance
(`physical.py`, `key_by="instance"`). Every clip showing that instance gets the
same mass, and the range comes from what is rendered (a cup rendered as a bell
gets bell mass), so it fits the object's look. The draw spreads instances of
one noun across the range, so the body has information beyond the noun.

Step 6 read the per-clip store (`physical_by_clip/`, a new random object every
clip). The instance store (`outputs/renders/clean_856/physical/`) exists
but predates the `contact_area` field the current-B loader reads, so loading
it fails. Step 7 rebuilds it with the current `build_physical(key_by="instance")`
(CPU only, no rendering) into a new store, and switches every arm to it. Alternatives rejected: GRAB mesh volume × density (mismatches the 58%
of train clips that render a replacement asset), and the range midpoint
(makes heavy/light a pure function of the noun).

Body representation: current B (153 dims), as in Step 6. Its load channels
carry the mass during lifted contact. Recorded motion does not react to mass.

### 3. Temporal video encoder

Same in every arm:
- Frames: the cached 1 Hz main frames, 11 steps from −5 to +5 s around the
  same jittered alignment frame the body window uses, with the same
  same-clip mask. About 82% of windows are cut by a clip boundary (about 8 of
  11 frames available on average); body and video lose the same steps.
- Per frame: the existing gaze-steered patch readout (Gaussian reweighting and
  sharp-view token in V+G; plain in V) gives one 768-d vector, projected to
  128-d.
- Over time: a one-layer bidirectional GRU, 64 hidden per direction, giving a
  128-d latent and a word score.
- The temporal score is added to the existing order-free video score, with a
  trainable weight. About 173k new parameters (projection 98k, GRU 74k),
  matched to the body encoder's 180k.

Known gap: each body step summarizes 30 Hz motion within its second; 1 Hz video
sees only positions once per second. Pre-launch check on dev, no language
involved: a probe trained on train clips predicts the action from frozen
temporal-video features and from body features. If video trails body by more
than 10 points, I come back with a 4 Hz rendering option (about 27k new frames)
before launch.

### 4. Test panels

All panels are four-choice; foils come from the same rung. Each held-out source
clip appears in about 10.4 (test) or 7.4 (s8) replay sessions with different
gaze and scene context. Each appearance gives one trial, at a moment inside the
relevant event. Intervals are bootstrapped by source clip (heavy/light: by
rendered instance).

| Panel | Source | Clips / clusters | Trials (≈) | MDE at 5 seeds |
|---|---|---:|---:|---:|
| Actions | test + s8 | 227 / 227 clips | 1,980 | 2.9 pp |
| Heavy/light | test + s8 lifted | 192 / 70 instances | 1,650 | 3.6 pp |
| Nouns | Step 6 test panel, reused | 115 clips | 556 | ≈5 pp (exploratory) |

The minimum detectable differences (MDE) assume 80% power, two-sided α = .05.
They use the noun panel's gaze − video variance as a conservative stand-in.
The action panel meets the 3-point target. Heavy/light does not: there are
only about 70 independent objects, so a 3-point property effect may be missed.
The noun panel is reused and remains exploratory.

Foils:
- Actions: the target is a moment of contact within the action. Foils are
  contact moments from three clips with other actions. Clips of the same object
  are preferred; 80 of 227 have three such foils. That object-matched subset is
  reported separately.
- Heavy/light: the target is a lifted-contact moment. Foils are three
  lifted-contact moments of the opposite label. The same rendered noun is
  preferred, then the same action. The straddling-noun subset is reported
  separately.

Test and s8 halves are also reported separately. Development panels are built
the same way from dev clips.

### 5. Speech and rendering

No new rendering. Speech is generated from saved gaze, contact and visibility
metadata. The generator is extended to emit action and property words in the
same utterances as names ("lift the heavy cup"), at the own-object naming rate.
They follow the same weak alignment as nouns: 1 s jitter, 35% during the
matching event, 41% while something else is visible, 24% with no matching
referent. Action words are timed to the action (lift onset from object height;
other actions by contact interval). Property words are timed to lifted contact.
Noun naming is unchanged, so the noun rung is re-measured on the new speech
and learner together.

### Runs

| Arm | Inputs |
|---|---|
| V | temporal video |
| V+G | V + steering gaze |
| V+G+B | V+G + current B, instance mass |
| V+G+B shuffled | B shuffled across clips, real gaze kept |

- Training: 630 clips, seeds 0–4, Step 6 epochs and optimizer settings. That
  is 20 models.
- The all-extra-inputs-removed result is scored from the trained models.
- One body-weight rule: Step 4's development rule, applied to the mean dev
  accuracy over the noun, action and property dev panels, weighted equally.
  Selected weights are reported.
- Cost: about 25–35 min per model on one RTX 4090; about $18 at Step 4's
  $0.87 per model including overhead. Proposed cap: $25.

Output: one table per rung (V, V+G, V+G+B; gaze, body and total gain with
intervals), selected body weights, and the object-matched and straddling
subsets. The key reading is whether the body gain on actions and heavy/light
exceeds the body gain on nouns.

### Build order after approval

1. Speech generator extension and regenerated transcripts. Check the counts
   against the table above.
2. Panels for dev, test and s8, with counts and foil-matching checked.
3. Temporal video encoder. Run the dev probe check (section 3).
4. Learner switched to the instance mass store. Run a short smoke run per arm.
5. Freeze the protocol, then launch the 20 models.

## Completed results — 19 November 2026

All 20 models (V, V+G, V+G+B and V+G+B with shuffled body; seeds 0–4; 630
clips) ran on Juno (Slurm array 521876, A30/H100/H200), were fetched and
verified against the frozen protocol. Analysis:
`outputs/runs/step7_word_types/analysis/` (`analyze.py`, `results.md`).

Changes from the approved protocol, made before launch:
- 13,878 sharp close-ups that the 11-frame window and the new panels needed
  were rendered and embedded locally with the existing path (min cosine to
  cached views 0.995) into a new cache, `outputs/features/child_gaze/step7_compact/`.
- The frozen runtime cap was raised from 4 to 5.5 hours (operational only).
- Final panels: actions 223 clips / 1,925 trials (4 clips had no cached frame
  inside the event); heavy/light 192 clips / 70 instances / 1,628 trials.
- Rental attempts failed on slow uploads to most Vast.ai hosts; spend $7.29,
  no runs. The runs used Juno instead.

Held-out accuracy (%, mean of 5 seeds) and gains (pp) with joint 95% intervals
over seeds and source clips (heavy/light: rendered instances):

| Rung | V | V+G | V+G+B | Shuffled B | Gaze gain | Body gain | Total gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nouns (test) | 50.65 | 65.65 | 66.55 | 65.65 | +15.00 [11.11, 19.05] | +0.90 [−2.11, 3.85] | +15.90 [11.73, 20.44] |
| Actions (test + s8) | 65.64 | 66.45 | 77.87 | 65.79 | +0.81 [−2.34, 3.83] | +11.42 [8.57, 14.62] | +12.23 [8.96, 15.61] |
| Heavy/light (test + s8) | 40.01 | 52.38 | 98.43 | 52.17 | +12.37 [7.29, 17.16] | +46.04 [39.26, 53.29] | +58.42 [53.63, 63.42] |

- Shuffled body adds nothing on any rung (−0.66 to 0.00 pp), so the gain needs
  body input aligned with the moment.
- Object-matched actions (637 trials): body gain +13.69 [8.42, 19.15].
  Straddling-noun heavy/light (235 trials): +60.43 [43.66, 73.23].
- Halves: actions test +7.80, s8 +15.75; heavy/light test +46.38, s8 +45.67.
- By action word, body gain: lift +17.09, offhand +17.48, pass +12.47,
  inspect +3.37, drink +0.47.
- Selected body weights: real body 1.10, 0.70, 0.95, 1.40, 1.35; shuffled
  0.05, 0.10, 0.20, 0.10, 0.05. The dev rule turns the body on without
  wordless pretraining.
- All extra inputs removed (V / V+G / V+G+B / shuffled): nouns 50.65 / 55.50 /
  56.44 / 55.58; actions 65.64 / 67.82 / 66.78 / 67.57; heavy/light 40.01 /
  40.11 / 37.48 / 40.47.

Reading: the body gain grows from nouns (+0.9, null) to actions (+11.4) to
heavy/light (+46.0), as predicted.

What it means for the next step:
- **Epoch-curve audit (completed 20 November 2026, zero GPU cost):**
  Inspection of all 20 saved `history.json` files rules out two hypotheses:
  1. *The 3-rung dev stopping rule did not cause the noun drop.* Across all
     four arms, selecting the best noun-only epoch instead of the best
     3-rung mean epoch recovers only +0.90 to +1.02 pp on dev (`V` 53.18% →
     54.20%, `V+G` 69.57% → 70.47%, `V+G+B` 68.51% → 69.53%).
  2. *`V` on actions is not under-trained.* `V` peaks on dev actions at
     epoch 8 (72.68%; avg single-seed peak epoch 11.2) and overfits steadily
     down to 59.18% by epoch 60. All arms peak between epochs 8 and 22 and
     overfit after epoch 25–30. This confirms the 96% supervised probe vs.
     65.6% test `V` gap reflects rapid overfitting under weak caregiver
     speech rather than too few training epochs, and means diagnostic runs
     only need ~30 epochs.
- **Diagnose the noun drop before Step 9 (priority #1).** With the 3-rung
  stopping rule ruled out, the 10–13 pp noun drop (`V` 63.67 → 50.65, `V+G`
  75.60 → 65.65) comes from the **11-frame temporal video encoder**, the
  **new multi-word action/property speech**, or their interaction. Next: run
  a small 2×2 ablation on `V+G` (Step 6 vs. Step 7 speech × static vs.
  temporal encoder, 1–2 seeds, 30 epochs) to isolate and fix the regression
  so all rungs share one clean learner.
- **Treat heavy/light as a completed proof of concept unless hardened for
  curves.** At 98.4% (and +60.4 pp on straddling nouns), heavy/light proves
  weak speech can ground a physical property in wrist load, but its clean
  load channel and dropped middle third would saturate too early on Step 9
  curves. Either leave heavy/light out of Step 9 (running curves on nouns +
  actions) or restore the middle third of masses and add realistic load noise
  if a property curve is wanted.
- **Reframe [Step 8](step_8_body_training_recipe.md).** Because dev selection
  already turns the body on in every seed (weights 0.70–1.40) without
  pretraining, Step 8 is no longer needed to "wake the body up." It is now
  either skipped or narrowed to testing whether wordless pretraining speeds
  up low-data action learning (e.g., 63–158 clips).

## Noun-drop fix: per-word zero-init temporal weight — 21 November 2026

The temporal video encoder caused the noun drop: its single scalar weight
starts at 1.0, so the temporal score enters every word, nouns included, from
the first step. Diagnostic on Step 7 speech and panels (630 clips, seeds 0–1,
30-epoch cap, best-dev checkpoint, Juno job 525910) compared two starting
values against static `cell_B` and the original Step 7 learner. Mean of the
two seeds, accuracy in %:

| Learner / arm | Nouns dev / test | Actions dev / test / s8 / pooled | Heavy / light | Temporal weight |
|---|---|---|---|---|
| Static `cell_B`, V+G | 78.6 / 73.6 | 70.1 / 72.3 / 56.8 / 65.2 | 60.5 / 56.1 | — |
| Scalar 1.0 (Step 7), V | 52.3 / 51.2 | 70.7 / 72.0 / 58.3 / 65.7 | 44.6 / 39.8 | 1.07 |
| Scalar 1.0 (Step 7), V+G | 69.3 / 66.2 | 70.9 / 71.9 / 63.4 / 68.0 | 55.0 / 53.2 | 0.98 |
| Scalar 0.0, V | 57.7 / 55.0 | 71.0 / 72.5 / 56.7 / 65.3 | 43.2 / 42.8 | −0.43 |
| Scalar 0.0, V+G | 73.7 / 70.0 | 71.3 / 72.9 / 58.5 / 66.3 | 57.3 / 55.8 | +0.32 / −0.32 by seed |
| Per-word 0.0, V | 58.3 / 57.6 | 76.4 / 76.2 / 63.3 / 70.3 | 48.7 / 44.2 | see below |
| Per-word 0.0, V+G | 79.0 / 73.3 | 74.1 / 75.7 / 60.1 / 68.6 | 57.4 / 58.5 | see below |

The Step 7 scalar 1.0 rows used a 60-epoch cap; `cell_B` has no V arm at 30
epochs.

Per-word zero-init gives each vocabulary word its own temporal weight,
starting at 0. It restores nouns to the static level (V+G test 73.3 vs 73.6)
and gives the best action dev and test scores in both arms.

**Forced-zero readout (CPU, no GPU).** Re-scoring the four per-word
checkpoints reproduced all 16,436 saved test choices. Setting every temporal
weight to 0 at inference then lowers pooled actions by 2.5–6.0 pp in every
checkpoint, back to about the static level (64–67%), while noun test moves
by −0.7 to +1.6 pp. The temporal path therefore supplies the action gain and
leaves nouns alone. Action-word weights are ±0.2–0.3 with mixed signs
(their mean, near zero, hides this); `lift` is negative in all four
checkpoints and `pass` positive in three. `pass` gains 6–11 pp test accuracy
from the path in every checkpoint; the other words vary by seed.

**Decision.** Per-word zero-init (`temporal_weight_init="per_word"`) is the
default temporal video learner from here on; the Step 7 protocol builder now
writes it. The recorded Step 7 results above used scalar 1.0, and saved
protocols without the key still load as scalar 1.0. Runs are under
`outputs/runs/step7_temporal_init/` (`results.md`, `zero_readout.md`).

