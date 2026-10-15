# Plan: do extra modalities raise the language-acquisition ceiling?

Claim: video-only word learning plateaus; adding embodied modalities as learner
inputs lifts that plateau. Measure gain in word learning, video-only vs
video + body.

## Order

1. **Evaluation suite and vocabulary.** Done 2026-10-14.
   - 1a. Pick evals. Done 2026-10-14. See "Evaluation suite".
   - 1b. Widen vocabulary to cover all 60 Konkle test categories. Done. See
     "Vocabulary widening". Needed because 41 GRAB words share only ~16
     usable words with Konkle test-60, and 41 words may saturate.
2. **Speech that depends on the body.** Caregiver speech tied to what the
   baby is doing, strength set from developmental studies (not chosen by us).
   Done 2026-10-15. See "Speech coupling".
3. **Learner with a body input.** Third input next to frozen image encoder and
   word tower. Trained with the body input randomly switched off; tested
   vision-only (picture + word, like children).
   - One body input; contact, hand pose, joints (and any other simulator
     signal) each encoded separately inside it, so each can be tested alone.
   - Inventory every signal the simulator can give; use as many as possible.
     GRAB replay has full-body pose, hand pose, object pose, per-vertex contact;
     replay currently saves only left/right contact flags.
   - No object identity in the body signal (leakage).
   - Open: gaze. Could weight the 17 image regions near the gaze point. Needs a
     gaze source: rules (risk: we decide the answer) or a model trained on real
     infant eye-tracking. Head-cam gaze sits near center, so may add little.
     Decide here.
4. **Single point at max data.** Conditions: video-only, video + body,
   video + shuffled body (capacity control), oracle (upper bound). 5 seeds,
   epoch picked on dev every epoch, test at that epoch.
5. **If there's a gap:** add two smaller data sizes (subsets of the same
   render), fit Frank's free-asymptote logistic, compare asymptotes.

## Evaluation suite

| Benchmark | Use |
|---|---|
| Rendered probes, held-out rooms | Primary. All vocabulary words. Dev split picks the epoch; test split reported. Report held and seen-only words separately. |
| Konkle test-60, Frank's exact protocol | Real-photo transfer, comparable to Frank. Epoch picked on rendered dev (we won't cover Konkle dev-117). |
| LEVANTE vocab (159 items) | Not usable yet (0 overlap). |
| EgoBabyVLM Machine-DevBench | Later. Lexical noun 2AFC from own vocab; harness not found yet. Grammar tasks need a non-bag-of-words text model. |

Held vs seen-only words is the built-in check: touch should help held words
and not seen-only ones.

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
