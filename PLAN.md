# Plan: do extra modalities raise the language-acquisition ceiling?

Question: does adding bodily input during learning improve language acquisition
over video alone? The long-term hypothesis is that it raises the plateau.
Measure two outcomes separately:

- **Understanding with the body available:** real vision + body versus a matched
  video-only learner on the same held-out situations, words and candidates.
- **Learning that carries over:** picture-only performance after learning with
  real bodily input versus after learning with video alone.

A reliable gain in either outcome advances the goal. A body-present gain does
not establish transfer to vision, and absent transfer does not negate a
body-present gain. Oracle identity inputs are privileged diagnostics, not
evidence of benefit from real bodily experience. Learning curves and data
scaling distinguish immediate multimodal performance, learning efficiency and
a higher eventual plateau.

## Order

1. **Evaluation suite and vocabulary.** Done 2026-10-14.
   - 1a. Pick evals. Done 2026-10-14. See "Evaluation suite".
   - 1b. Widen vocabulary to cover all 60 Konkle test categories. Done. See
     "Vocabulary widening". Needed because 41 GRAB words share only ~16
     usable words with Konkle test-60, and 41 words may saturate.
2. **Speech that depends on the body.** Caregiver speech tied to what the
   baby is doing, strength set from developmental studies (not chosen by us).
   Done 2026-10-15. See "Speech coupling".
3. **Smallest learner that tests the question.** Design: `docs/learner.md`;
   sources: `docs/literature/body_input_review.md`.
   - Body-input pathway: contact, finger pose and arm position encoded
     separately, then combined with vision during language training.
   - Random body dropout in training; evaluate both body-present understanding
     and picture-only transfer as distinct outcomes.
   - Vision-to-body prediction as an auxiliary loss on the shared visual
     projection. Supplements the pathway.
   - Contact-episode grouping optional (flag, off by default).
   - Visual architecture identical across conditions (linear projection,
     soft pooling). No object identity in body inputs.
   - Finish with cheap checks: language loss depends on the body
     representation, gradients reach its encoder, dropout and body-free eval
     work. These show the pathway works, not that it helps.
   - Earlier attempts (hand-location prior, v1/v2) gave no gain; in Git
     history.
   - Open: gaze. Decide later.
   - Done 2026-10-15 (commit 8b70cca2). Checks pass.
4. **Staged comparison at max data.**
   1. Matched video-only vs video + body, 1 seed, same data, speech and dev
      rule. Report both outcomes; compare body-present scores against the
      video-only learner on the same trials, not just the multimodal learner
      with its body removed. Reuse a baseline only if its settings match.
      Predefine dev selection for each outcome; never select on test scores.
   2. Check learning curves at a set checkpoint (epoch 30); extend only if
      still rising.
   3. If a promising real-body gap in either outcome: shuffled body and more
      seeds; 5 seeds to confirm that outcome.
   4. If no gap in either outcome: pick the next diagnostic from the failure
      (body probe or oracle). If only transfer is absent, investigate transfer
      without dismissing the body-present result. No automatic ablation sweep.
   - 4.1 result (seed 0, s8 held out, 60 epochs; selected epochs 45 and 47,
     broadly saturated by ~30): video-only 76.4 rendered / 69.1 Konkle;
     video + body 75.4 / 69.7. No clear picture-only gain for this
     configuration.
     - Body-present test (s8): picture 44.8, body score 25.5 (32 on
       held-referent trials), combined 44.0. Body weight 0.74, not
       collapsed. The pathway is active but does not help this test or
       transfer to picture-only. Open causes: weak speech alignment, poor
       generalization, score calibration, or the fusion form.
     - Fusion is an additive body-word score: it cannot model body-vision
       interactions. Minimal by design; revisit after the oracle check.
     - Body probe: next-second contact 90% vs 74% persistence, but the
       encoder sees +-5 s both ways, so the future is in its window. Not
       evidence of prediction. Held word 16.5% vs 6.0% majority, same as raw
       features (17.0%). The grasp-family split is degenerate (100%
       majority). No evidence yet of category information across grasp
       families.
   - 4.2 oracle pathway check (true held-object identity as body, vision-to-
     body loss off, seed 0): picture-only 75.5 rendered / 68.8 Konkle (no
     change). Body-present test, all / held-referent trials:
     picture 43.7 / 61.7, body 50.5 / 94.3, combined 49.8 / 76.6.
     - The pathway can use a strongly informative body when present.
     - Combined (76.6) is far below body alone (94.3) on held trials: the
       additive score is poorly calibrated (body weight 0.61).
     - Nothing reaches picture-only; picture held-referent accuracy dips
       (65.1 video-only -> 61.7), as if the body explains away held cases.
     - So two separate gaps: transfer to the visual path, and the real body
       carrying far less word information than the oracle (32 vs 94 on
       held-referent trials).
   - Next: retain the bounded oracle-to-picture transfer diagnostic. It tests
     whether explicit teaching can improve picture-only learning with
     privileged input. Success must subsequently be tested with real body
     inputs; failure does not rule out benefits with the body available.
   - 4.3 oracle transfer (diagnostic; commit 33dd9863): frozen oracle teacher,
     picture-only student with distillation (T=2, weight 1), matched control.
     - With training-matched (soft) scoring the teacher's combined score is
       52.8 / 84.6 (all / held-referent) vs picture 46.0 / 61.7.
     - Student = control on picture-only: 76.3 vs 76.4 rendered, 68.5 vs
       69.1 Konkle (selected epoch); same at epoch 30.
     - Student agrees with the teacher more at epoch 30 (58 vs 48% top-1 on
       train), then stalls and ends below the control (59 vs 67%).
     - Reading: the student cannot fit the teacher on train. Most likely the
       frozen features + linear projection, not the transfer loss. Unchecked:
       the max picture-only accuracy these frozen features allow with true
       labels (supervised region probe). Run that before more transfer work.
   - 4.4 next (in order, then stop and report):
     1. Direct-label check (bounded diagnostic). Same frozen features and
        visual architecture, 1 seed, no sweep, max 30 epochs, checkpoint on
        dev. Correct object labels from training data only, and only where
        the object is visible in the frame window (not every frame labelled
        with the held object). Report train fit and held-out accuracy. A gain
        = visual headroom exists. A weak result is inconclusive; it does not
        justify unfreezing the encoder or more probes.
     2. Re-score the existing real-body model and the video-only control with
        soft pooling (no retraining). Same held-out trials: video-only
        control, body model with body removed, body only, combined. All /
        held / non-held referent. Main comparison: vision + body vs the
        separately trained video-only control.
     3. If no promising real-body gain: retrain the real-body learner with the
        vision-to-body loss off (pathway, dropout, data, speech unchanged).
        1 seed, inspect at epoch 30, extend only by a dev rule set in
        advance. Report body-present and picture-only separately; a reliable
        body-present gain counts as progress.
     4. Stop and report. Gain -> shuffled body and matched seeds. No gain ->
        name the remaining uncertainty and propose one change.
     Not now: more oracle distillation, unfreezing the image encoder, broad
     ablations. Priority is whether real body signals improve word learning.
5. **Scaling.** Track both outcomes across smaller data sizes from the same
   render, fit Frank's
   free-asymptote logistic. Separate faster learning / better at limited data
   from a higher final plateau.

Held vs seen-only words are a diagnostic breakdown. Better shared visual
features could help both, so a seen-only gain is not a failure.

## Evaluation suite

| Benchmark | Use |
|---|---|
| Body-present word understanding, held-out situations | Real vision + body versus matched video-only on identical trials. Existing s8 4AFC is the initial measure. Report all, held-referent and non-held-referent trials. |
| Rendered probes, held-out rooms | Picture-only transfer. All vocabulary words. Dev split picks the epoch; test split reported. Report held and seen-only words separately. |
| Konkle test-60, Frank's exact protocol | Real-photo transfer, comparable to Frank. Epoch picked on rendered dev (we won't cover Konkle dev-117). |
| LEVANTE vocab (159 items) | Not usable yet (0 overlap). |
| EgoBabyVLM Machine-DevBench | Later. Lexical noun 2AFC from own vocab; harness not found yet. Grammar tasks need a non-bag-of-words text model. |


## Vocabulary widening (step 1b)

Checked Objaverse LVIS labels (46k labeled of ~800k objects) for Konkle test-60:

- 57/60 have Objaverse assets; ~50 have 20+ instances.
- Missing: leaves, rock, tent. Thin (<20): crib 6, cheese 11, tape 13,
  button 15. Fill from full Objaverse tag search or EmbodiedGen text-to-3D
  (already wired in `nursery/embodiedgen_scene.py`).
- Weak label matches to check by eye: bill (dollar), tree (Christmas tree
  only), trumpet (cornet), rug (doormat/runner), pipe, fan.
- Objaverse quality varies: filter meshes and normalize scale.

Konkle words by how they enter the simulator:

- **Held, native GRAB grasp (11):** airplane, apple, ball, bottle, bowl,
  camera, clock, knife, phone, train, watch.
- **Held, mesh swap on a similar-shape GRAB grasp (13):** bagel←torus,
  tape←torus, rock←small sphere, balloon←large sphere, pen←toothbrush,
  spoon←toothbrush, bird←duck, doll←bunny, cheese←small cube, lock←small cube,
  cookie←small cylinder, juice←cube, sandwich←cube.
- **Held, loose fit (8):** bell, key, turtle, trumpet, shoe, cake, pipe,
  umbrella. Grasp shape is off; fix later with HOIDiNi grasps for new meshes.
- **Seen only (28):** basket, bed, bike, bill, boot, bucket, butterfly,
  button, cat, chair, crib, dog, fan, guitar, hat, jacket, leaves, meat,
  necklace, pants, pizza, ring, rug, socks, stool, tent, tree, tv.
- **Extra:** GRAB words outside Konkle (cup, hammer, scissors, ...) stay.

Speech names seen-only objects too, so the learner hears all words.

Done 2026-10-14 (branch `cue/vocabulary`, commits 68a6dea7, 16425a6a, 164bd04d):

- Assets: 1,210 screened instances, ≥10 per Konkle word (most 20+), split
  train/dev/test by instance. Fills: procedural meshes for balloon, bill,
  button, tape, bagel (not EmbodiedGen). 35 instances are CC-NC; licenses in
  `assets.json`. cookie comes from small torus, not small cylinder.
- Swaps: every held word (native too) uses Objaverse instances on GRAB grasps,
  contact recomputed. Loose fits (<30% of contact within 5 mm): watch, bottle,
  rock, balloon, bird, doll, juice, sandwich, key, turtle, umbrella.
  v2 alignment (`--align v2`, not yet used in the render) brings 22/32
  word-source pairs to ≥50% within 5 mm. Still loose (shape mismatch with the
  GRAB source): bird, turtle, doll, sandwich, key, juice, watch, umbrella.
  Untested better sources: gamecontroller/mouse (sandwich, doll, bird),
  alarmclock (turtle), wristwatch (watch), stapler/waterbottle (juice).
  watch and umbrella penetrate more under v2. Use v2 in the clean render.
- Placement: tent shrunk to 1.0 m; hat, pizza, cake, trumpet, airplane go on
  the floor (table too small); big floor objects push the actor to the edge
  of the open square. 104 extra sessions boost the rare words; all 78 words
  are now in the learner's vocabulary (bed barely: 5 mentions).
- Render: 856 sessions (752 + 104 extras), 102,720 frames, body signals saved
  per frame (joints, hand pose, object pose, per-vertex contact).
- Video-only smoke run (1 seed, 20 epochs): rendered test 45.3% (held 41.0,
  seen-only 39.8, extras 61.5); Konkle test-60 45.2% (chance 25%). Not
  saturated, still rising at epoch 20.

Before the final experiment (step 4):
- One clean full render with all fixes (not the patched set; step 5 slices
  data by size). New placement code shows ~30% fewer floor words per
  session; check before that render.
- Train long enough that video-only plateaus.
- Step 2 speech must keep every word ≥5 mentions (seen-only words are rarely
  named in real data, ~0.04/min).

## Speech coupling (step 2)

From literature and the Karmazyn-Raz & Smith OSF data (12 mo, 16 dyads);
details in `outputs/step2/speech_coupling_lit.md` (worktree):
- Named object held by infant: 23% (12 mo) to ~42% (12–26 mo).
- Handled objects named ~6× more per minute than never-handled ones.
- ~30% of utterances name an object; ~3.6–6 names/min.
- Name falls −10 to +10 s around grasp onset; at 12 mo before ≈ after.
- No parent hands in the simulator: parent-held naming (~29%) needs a
  design choice.

Done 2026-10-15 (branch `cue/vocabulary`, commit cea4af3d):

- 25% of utterances name an object (~5/min). Each name picks a channel:
  infant-held within ±5 s (35%), visible not held (41%, also stands in for
  parent-held), in the room but out of view (24%). Uniform within channel.
  Held share is calibrated after redraws (achieved 0.341).
- Implied lift, held vs never-held objects per object-minute: ~2.8x, below
  the ~6x in the data (holds are frequent here: 55% of frames). Anchored on
  the share; lift is a sensitivity knob for step 4.
- Every word ≥20 named mentions (median 97; lowest tent 20, tree 22).
- Video-only smoke (1 seed, 20 epochs, same render): rendered test 69.5%
  (held 69.3, seen-only 56.4, extras 89.9); Konkle 67.7%. Up from 45% mostly
  because names are ~3x more frequent. Still rising at epoch 20, so the
  plateau and remaining headroom are unknown; check before step 4.
