# Plan: do extra modalities raise the language-acquisition ceiling?

Claim: video-only word learning plateaus; adding embodied modalities as learner
inputs lifts that plateau. Measure gain in word learning, video-only vs
video + body.

## Order

1. **Evaluation suite.** Check vocabulary overlap with benchmarks, pick evals.
   - Done 2026-10-14. See "Evaluation suite" below.
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

Our vocabulary: 41 words from GRAB objects.

| Benchmark | Overlap | Use |
|---|---|---|
| Rendered probes, held-out rooms | 41/41 | Primary. Dev split for epoch selection, test split reported. |
| Konkle real photos (test-60 + dev-117) | 12 + 7 = 19 | Transfer to real photos: 4AFC over the 19 shared categories, Frank's trial rules. Also report standard test-60 (OOV = chance) for comparison. |
| LEVANTE vocab (159 items) | 0 | Not usable. |
| EgoBabyVLM Machine-DevBench | built from own vocab | Later. Lexical noun 2AFC with generated images; harness not found yet. Grammar tasks need a non-bag-of-words text model. |

- To do: map GRAB `airplane` to the word `airplane`, not `plane`, to match
  Konkle (already counted in the 12).
- Small shared vocabulary limits real-photo tests; more object categories in
  the simulator would widen it.
