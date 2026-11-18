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

## Open decisions

- Exact property and action vocabulary, and how mass is assigned per object.
- Form of the temporal video encoder (GRU or small transformer) and its size.
- Panel sizes, seeds, epochs and the rental budget.
- Whether new action speech needs new rendering, or reuses existing clips.
