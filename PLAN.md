# Current Nursery plan

## Active objective

Show that embodied inputs raise language acquisition beyond what video alone
reaches, across amounts of independent training experience. Gaze counts as
embodied experience alongside the other body channels.

Language acquisition is measured as a ladder of tests, all in the same
four-choice format so their curves stay comparable:

1. **Nouns**: which scene matches the word. Done for one test panel; kept as
   the control, where vision should already do well.
2. **Property and action words**: words whose meaning is partly bodily, such as
   heavy, lift or pass.
3. **Simple sentences**: the wrong choice has the roles or relation swapped,
   for example "cup in bowl" against "bowl in cup".

The prediction is that the embodied gain grows from nouns to property and
action words to sentences. Each rung is reported separately; scores are never
mixed across rungs.

## Where we are

Steps 1–4 and 6 are complete. Steering gaze, where gaze guides which video
patches a word is matched to, lifts the noun curve at every data amount:
+9.95 points at 63 clips, about +18 at 158–315, +11.93 at 630 (63.67% → 75.60%).
The other body channels add nothing over gaze at any amount; development
selection keeps their weight at 0–0.15. Video and gaze are both still rising at
630 clips, so the ceiling question is open.

Nouns are the one test where vision has the advantage: an earlier diagnosis
found about 80% of what the body knows about the test nouns is already in
vision. Step 7 therefore built the action and property tests where the body
has something to add, and confirmed the ladder prediction at 630 clips: body
gain grows from +0.90 pp on nouns (null) to +11.42 pp on actions and +46.04 pp
on heavy/light, with shuffled body near zero and dev selection turning the body
weight on (0.70–1.40) without wordless pretraining.

Two diagnostics came before Step 9 curves and are done: the saved Step 7
training curves show every arm peaks by epoch 22, and the 10–13 pp noun drop
came from the temporal weight starting at 1.0; a per-word weight starting at 0
fixes it and is now the shared learner for every rung. Because Step 7 already showed the learner uses the
body without pretraining, Step 8 is now optional—either skipped or narrowed to
a low-data sample-efficiency test (63–158 clips) on actions. Sentences follow
(Step 10), and comparison with children's learning comes last (Step 11).

| Step | Document | Question and current status |
|---|---|---|
| 1 | [Visual baseline](docs/step_1_visual_baseline.md) | Complete and audited: at 630 clips, recorded 4×4 video scored 60.12%, 16×16 scored 61.93%; paired gain +1.81 points over three seeds. |
| 2 | [Child gaze](docs/step_2_child_gaze.md#completed-results--9-november-2026) | Complete and independently audited: 12/12 frozen runs, four conditions × seeds 0–2 × 60 epochs, 630 clips. New-world video 63.61%; video + gaze 63.25% with gaze (−0.36 pp), 60.13% removed (−3.48 pp). Shuffled 50.54%/59.47%; point-only 62.59%/62.35%. Picture transfer declined. One US 8×5090 attempt cost $4.416; verified results back in 66m51.818s, rental destroyed, zero instances and key removal confirmed. Current additive gaze is not selected for curves. [Steering](docs/step_2_child_gaze.md#steering-gaze--9-november-2026) (6 runs, audited): 75.72% with gaze (+12.11 pp), 66.31% removed (+2.70 pp), positive in every seed; s8 +13.45/+2.90 pp; pictures 82.09%/84.58% vs 75.62%/73.98%. Shuffled steering at or below video. Steering is the candidate for curves. |
| 3 | [Body representation](docs/step_3_body_representation.md#frozen-candidate--10-november-2026) | Built and frozen (input only, no language result yet). Rich B keeps the ±5 s window but summarizes 30 Hz source frames per step, adds contact events and wrist-frame contact geometry, finger speed and 30 Hz arm motion: 193 dims vs 153. 22 contact patches kept by a fixed train→dev rule. All checks passed. On dev, held-category probe 23.7% → 31.7%; brief contacts visible 32% → 100%. Pressure, friction, texture, deformation and spatial heat stay deferred. |
| 4 | [Representation comparison](docs/step_4_body_representation_comparison.md) | Does richer body help under the same additive setup, without wordless pretraining? [Complete](docs/step_4_body_representation_comparison.md#completed-results--11-november-2026), 12/12 verified runs at 630 clips. New-clip test: video 63.67%, + gaze 75.60%, + gaze + current body 76.26%, + gaze + rich body 76.14%. Body − gaze +0.66 / +0.54 pp; on s8 −0.67 / −0.34 pp; rich − current −0.12 pp. Dev chose body weight ≈ 0 in all six body runs; at the as-trained weight body costs 6.5–8.2 pp. Spend $10.48. |
| 5 | — | Skipped: Step 6 used additive fusion without wordless pretraining. The training-recipe question moves to Step 8, after the new tests exist. |
| 6 | [Noun acquisition curves](docs/step_6_language_acquisition_curves.md#completed-results--17-november-2026) | Complete, 45/45 models, 63–630 clips, seeds 0–2. Gaze above video at every amount (+9.95 to +18.41 pp). Current body − gaze between −0.66 and +0.72 pp, every interval touching zero; body weight kept at 0–0.15. No plateau yet. Spend $29.96. |
| 7 | [Property and action words](docs/step_7_word_types.md#completed-results--19-november-2026) | Complete, 20/20 models at 630 clips, seeds 0–4, temporal video in every arm. Body gain over gaze: nouns +0.90 pp [−2.11, 3.85], actions +11.42 [8.57, 14.62], heavy/light +46.04 [39.26, 53.29]; shuffled body ≈ 0. Dev selected body weights 0.70–1.40. Nouns fell about 10–13 pp under the new shared learner; fixed by per-word zero-init of the temporal weight (21 Nov). Rental spend $7.29 (no runs); runs on Juno. |
| 8 | [Body training recipe](docs/step_8_body_training_recipe.md) | Reframed / optional post-Step 7: Step 7 already showed the learner turns the body on without pretraining (weights 0.70–1.40). Either skip to Step 9 or test whether wordless pretraining speeds up low-data action learning (63–158 clips) now that the noun drop is fixed. |
| 9 | [Curves by word type](docs/step_9_acquisition_curves.md#completed-results--22-november-2026) | Complete, 85/85 models, 63–630 clips, seeds 0–4, nouns + actions, fixed learner. Body gain on actions +6.4 to +9.5 pp at every amount, nouns ≈ 0; gaze gain on nouns +9.5 to +20.0, actions ≈ 0. Shuffled body ≈ 0. V+G+B at 315 clips ≈ V at 630 on actions. V still rising at 630: no plateau yet. Juno only, $0. |
| 10 | Sentence comprehension | Structured caregiver speech from GRAB events, a text encoder that reads word order (for example a small transformer trained from scratch), and a scene representation that can express relations and roles. Not yet designed. |
| 11 | Comparison with children | Order in which words are learned against Wordbank age of acquisition, and DevBench-style word and sentence tasks. Not yet designed. |

### Immediate next steps (post-Step 7)

1. ~~**Inspect saved Step 7 training curves (done 20 Nov 2026):**~~ Confirmed `V` on actions peaks early (epoch 8, 72.7% dev) and overfits to 59.2% by epoch 60 (not under-trained), and the 3-rung dev stopping rule only costs ~0.9–1.0 pp on nouns across all arms. All arms peak between epochs 8 and 22, so diagnostics only need ~30 epochs.
2. ~~**Diagnose the 10–13 pp noun drop (done 21 Nov 2026):**~~ The temporal encoder's scalar weight starting at 1.0 caused it. A per-word temporal weight starting at 0 restores nouns to the static level (V+G test 73.3% vs 73.6%) and raises actions (V+G pooled 68.6% vs 65.2% static); forcing the weights to 0 at inference removes 2.5–6.0 pp of actions and leaves nouns unchanged. Per-word zero-init is now the default temporal video learner ([details](docs/step_7_word_types.md#noun-drop-fix-per-word-zero-init-temporal-weight--21-november-2026)).
3. ~~**Two design calls before Step 9 (decided 21 Nov 2026):**~~ Step 8 skipped (revisit only if `V+G+B` lags at low data); heavy/light kept as a completed Step 7 proof of concept, so Step 9 curves cover nouns + actions.
4. **Review the [Step 9 protocol](docs/step_9_acquisition_curves.md), then build and launch on Juno.**

A weak intermediate result is a design signal for the build, not a standalone
paper framing or an automatic instruction to launch more conditions.

## Decisions to keep consistent

- Count gaze as embodied experience. G denotes steering gaze location plus its
  attended sharp close-up; B denotes the other body channels. All conditions
  share gaze-driven speech and main video; V receives neither G nor B.
- Every rung compares V, V+G and V+G+B on the same trials. Report gaze gain
  (V+G − V), body gain (V+G+B − V+G) and total gain (V+G+B − V) per rung.
  Body effects are conditional on gaze; no automatic body-without-gaze arm.
  Preserve 4×4 results as explicitly historical references.
- Video must have a fair chance on each rung. When a rung depends on motion,
  every arm gets the same temporal video encoder over the same accessible time
  span as the body window. Re-measure the noun rung whenever the shared
  learner changes.
- Gaze steers visual matching; B adds a calibrated word score. Keep steering
  settings matched and visual weights trainable. Primary evaluation uses each
  learner's assigned inputs; report all-extra-inputs-removed and picture
  transfer separately. Shuffle B with real gaze retained to test its alignment.
- Apply one development-only body-weight rule to every body condition; test
  outcomes do not tune the design. Report the selected weights.
- Size each new test panel before launch to detect about a 3-point difference.
  Reused panels remain exploratory; repeated seeds are not new test situations.
- Preserve natural weak speech, source-clip partitions and accessible time
  spans. New property, action and sentence speech keeps the same weak
  alignment (delays, nonheld or absent referents). Record necessary
  encoder-capacity and compute changes.
- Properties must be physically consistent with each object and persistent
  across its clips, so vision gets partial cues. Use what the simulator already
  supplies (load from mass and motion, contact, hand pose and motion). Softness,
  friction, texture and spatial temperature stay deferred until they have
  believable physical models or measurements.
- If wordless pretraining is selected, use only each point's own selected
  training clips, including normalization. Extra optimizer passes are compute,
  not new independent experience. Both arms share DINO's pretrained visual
  prior; this is not perception learned from scratch.
- Count independent source clips on the experience axis. Separate uncertainty
  across training seeds from uncertainty across test clips.
- Active exploration and joint-architecture searches stay deferred. Extending
  the video curve past 630 clips, to test for a plateau, is a later separate
  study.

## Preserved starting points

The completed [cross-situational study](archive/language_learning/cross_situational_learning.md)
and [original acquisition curves](archive/language_learning/acquisition_curves.md)
are archived with their protocols, results and source revisions. The
[archive index](archive/README.md#completed-study-worktrees) identifies their
full local worktrees, runs and compatibility paths. The earlier
[heavy/light control](archive/language_learning/past_experiments.md#heavylight-positive-control-415)
showed the additive route can learn a property word from load, but its mass was
random per clip and invisible, so it is not a fair property test.

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
