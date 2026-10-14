# Plan: do extra modalities raise the language-acquisition ceiling?

Claim: video-only word learning plateaus; adding embodied modalities as learner
inputs lifts that plateau. Measure gain in word learning, video-only vs
video + body.

## Order

1. **Evaluation suite and vocabulary.** In progress.
   - 1a. Pick evals. Done 2026-10-14. See "Evaluation suite".
   - 1b. Widen vocabulary to cover all 60 Konkle test categories. See
     "Vocabulary widening". Needed because 41 GRAB words share only ~16
     usable words with Konkle test-60, and 41 words may saturate.
2. **Speech that depends on the body.** Caregiver speech tied to what the
   baby is doing, strength set from developmental studies (not chosen by us).
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

To do:
- Map GRAB `airplane` to `airplane`, not `plane`.
- GRAB torus is not a Konkle ring; rename it (bagel or tape after swap).
  Konkle `ring` comes from Objaverse ring meshes.
- Swaps: scale new mesh to the original's size, recompute contact.
