# Step 2: simulated child gaze drives naming and feeds the learner

Design reference in [PLAN.md](../PLAN.md). It follows the
[visual baseline](visual_baseline.md) and comes before the body steps. No gaze
model, close-up render, speech change or run exists yet.

## The idea

Add one gaze process to the simulated world. It does two jobs:

1. **It drives naming.** When the parent names something the child isn't
   holding, it's usually what the child is looking at. The link has real-life
   strength, not 100%.
2. **It feeds the learner.** The video + gaze learner gets where the child is
   looking, plus a sharp close-up of that spot.

Then test video against video + gaze. Gaze is a body-derived signal, so this
directly tests whether an extra modality lifts language learning.

## Why this comes first

An error breakdown of the Step 1 16×16 learner (630 clips, three seeds)
traced most errors to the simulated world, not the learner. Results are in
`outputs/analysis/visual_baseline_errors/` and `outputs/analysis/not_held_errors/`
(ignored).

| Finding | Number |
|---|---|
| Errors on targets nobody holds | 84% |
| Held vs not-held accuracy, same noun (26 nouns) | 77.2% vs 56.2% |
| Not-held accuracy when the hand holds something else | 48.1% |
| Not-held accuracy, tiny target (≤16 px) vs large (>1,024 px) | 19.0% vs 66.2% |
| Questions wrong in all three seeds | 158 of 570 |
| Correlation of noun accuracy with naming frequency | r = 0.06 |

The cause is in the speech generator. It names held objects (35%), unseen
objects (24%), and otherwise any object with at least one visible pixel, picked
uniformly ([transcripts.py](../nursery/speech/transcripts.py)). Nothing ties
that pick to what the child is looking at. Frames are 320×240, so a background
key is often one or two DINO patches. The learner doesn't confidently attach
"key" to the wrong object; it never pins the word down.

Real naming of non-held objects is tied to attention. In our
[speech evidence](speech_coupling.md), naming is about 1.9× more likely when
the infant looks at an object (row 1c). The referent is present and attended
for about 50–60% of object words (row 5b). Our world has no gaze, so that link
is missing.

## How it works

**Gaze in the world.** Generate gaze first, from the scene alone. Speech reads
it afterward. If gaze followed speech, gaze would encode the answer.

- The gaze target is one object in view at a time. Looks last a second or two.
  Most go to the held object; the rest go to visible objects, weighted by size
  and motion.
- Take every number from head-mounted eye-tracking studies of toddler object
  play, with sources recorded like [speech_coupling.md](speech_coupling.md).
  Candidate sources: Yu & Smith 2013 (PLOS ONE, eye–hand coordination and
  joint attention); Franchak et al. 2011 (Child Development, head-mounted eye
  tracking). Verify before using.
- First version: eyes only. The head keeps following GRAB, so the video is the
  same for every condition. Head turns toward objects come later.

**Gaze drives naming.** Keep the held (35%) and unseen (24%) shares. Split the
remaining visible-object share. When a non-held object is named, it's the gazed
object about 50–60% of the time; otherwise it's another visible object. Sweep
this strength later. Never set it to 100%.

**Gaze feeds the learner.** Two inputs:

- **Gaze point:** where in the frame the child is looking.
- **Close-up:** a second, narrow camera at the eye, aimed at the gaze point,
  rendered sharp and encoded by the same frozen DINO. Cropping the 320×240 frame
  is not enough: it enlarges pixels without adding detail.

## The comparison

All conditions use the same new world: same gaze, same gaze-driven speech, same
16×16 video, clips, splits and training budget.

| Condition | Learner input | What it answers |
|---|---|---|
| Video only | 16×16 video | Baseline in the new world |
| **Video + gaze** | Video + gaze point + close-up | **Does gaze help?** |
| Video + shuffled gaze | Gaze from another moment | Is it gaze information, or just a bigger model? |
| Video + gaze point only | Video + gaze point, no close-up | Is it *where* to look, or *seeing it sharply*? |

**Two test modes for every gaze learner:**

| Test mode | What it shows |
|---|---|
| With gaze | Gaze helps answer questions |
| **Gaze removed, video only** | **Gaze helped it learn words:** it now recognizes objects from video alone better than the video-only learner |

The second mode is the stronger claim. Test questions come from naming moments
([acquisition_data.py](../nursery/learner/acquisition_data.py), named and
visible rows). Once naming follows gaze, the child will often be looking at the
target. The with-gaze score can rise partly from that. The gaze-removed score
and the picture-only transfer panels, which have no gaze, avoid it.

Report results split by: target held or not, target gazed or not, and target
size. The old-world 16×16 result (61.93%) stays as a labeled reference for how
much the world change alone moves video. It is not a matched baseline: the
speech differs.

## The output we want

At one data amount, first 630 clips with seeds 0–2 to match Step 1: accuracy
for each condition in both test modes, with the splits above. If video + gaze
helps, especially with gaze removed, carry it to the
[acquisition curves](language_acquisition_curves.md). A higher ceiling needs
the curves; one data amount can't show it.

## Before the comparison

- **Gaze checks:** looks land only on objects in view. The share on held vs
  other objects and the look durations match the sourced values.
- **Speech checks:** achieved held, unseen, gazed and other-visible shares hit
  their targets. The gaze model must put enough looks on non-held objects to
  reach 50–60%.
- **Close-up checks:** time a sample render first and project the full cost.
  Check the close-up shows the gazed object, and that the main video frames
  are unchanged.
- **Leakage checks:** gaze is generated without reading speech. No labels
  enter the learner.
- **Cache checks:** existing 16×16 frame features cover the new utterance
  windows, or the gap is recorded.

## Open choices

- **How gaze enters the learner.** (a) A separate gaze branch added to the video
  score, the same additive setup used for body. (b) Gaze steering which video
  patches a word is matched to during training. With (a), the video path may
  learn less because gaze carries the easy cases; earlier body results showed
  this. With (b), the gaze-removed test is more natural. Decide on
  development data, not test scores.
- Gaze model parameters, once sourced.
- Close-up field of view and resolution.

## Deferred

Head turns toward objects, parent hands that hold up objects, gaze following
of the parent, and the body steps after this one.

## How code is run

Codex (CLI) writes and runs the code: gaze model, speech change, close-up
render, learner input, training and audits. Claude plans, writes briefs and
reviews. Outputs go under `outputs/`; no saved runs or renders are overwritten.
