# Step 2: simulated child gaze drives naming and feeds the learner

Design reference in [PLAN.md](../PLAN.md). It follows the
[visual baseline](visual_baseline.md) and comes before the body steps. Phase A
implements and validates the simulated world; learner selection and scientific
comparisons remain separate phases.

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

## Gaze evidence and parameter assumptions — Phase A

**[V]** means the primary full text was opened and the value checked;
**[A]** means a modeling assumption, with a sensitivity range, rather than a
measured infant parameter. A *look* here is a continuous object-level bout;
eye movements within one object need not end it. The simulator conditions on
looking at a visible object and does not model face looks or looks away.

| Quantity | Verified evidence | Sample and location in paper | Use and limitation |
|---|---|---|---|
| Object-look duration | **[V]** Mean 2.334 s, reported SD 0.982 s. All object/face ROI looks together: 2.068 s (SD 0.642). | Yu & Smith (2013), 17 dyads, 11–15 months (mean 13.5), three toys at a time; [Table 1, PDF p. 5](https://www.la.utexas.edu/users/dil/papers/2013_Yu_POne.pdf), [DOI](https://doi.org/10.1371/journal.pone.0079659). | Anchor the object-bout timescale near two seconds. The reported SD is not a fitted within-child dwell distribution. |
| Duration distribution and context | **[V]** About 75% of object looks were shorter than 3 s; observed durations extended from the imposed 0.5-s analysis floor to nearly 31 s. Mean duration was 1.26 s without overlapping parent attention and 3.03 s with it. | Suarez-Rivera, Smith & Yu (2019), 40 infants, 12.1–16.1 months; Results I–II and Fig. 2 in the [author manuscript](https://discovery.ucl.ac.uk/id/eprint/10123731/7/Suarez%20Rivera_nihms-993247.pdf), [DOI](https://doi.org/10.1037/dev0000628). | Supports short bouts with a right tail. Parent attention/talk effects are context, not inputs to this scene-only model. The 0.5-s floor is an analysis choice, not a biological minimum. |
| Own-held versus other-object looking | **[V]** Yu & Smith show gaze coupled to the child's and parent's hand actions; the own-hand association is stronger for infants than parents. **Exact eligible-choice probability not found.** | Yu & Smith (2013), Results and Fig. 3. | **[A]** Prefer the child's held object with probability 0.60 when held and other visible objects are eligible; sensitivity range 0.50–0.70. This is not the paper's 62.13% object-looking time, which includes all objects. Report achieved held/non-held frame shares separately. |
| What draws gaze: ongoing manual action | **[V]** Infants looked at a reaching target in the final second before contact 76.4% of the time; 62.2% of reaching encounters included looking at contact. | Franchak et al. (2011), six 14-month-olds in free play; [PDF pp. 10–11](https://faculty.ucr.edu/~franchak/publications/2011-FranchakKretchSoskaAdolph-ChiDev.pdf), [DOI](https://doi.org/10.1111/j.1467-8624.2011.01670.x). | Supports action-linked attention. These event-conditioned rates are not shares of all looks to held objects. |
| What draws gaze: image size and motion | **[V]** Franchak et al. discuss size/motion salience but explicitly caution that gaze also depends on actions; their study does not estimate selection coefficients. | Franchak et al. (2011), introduction, PDF pp. 2–3. | **[A]** Non-held weights are `visible_px**0.5 * (1 + 2 * normalized_image_motion)`. Size exponent range 0.25–1.0; motion coefficient range 0–2. These sensitivity ranges are not confidence intervals. |

The sticky categorical transition rule is also **[A]**: its dwell target has a
sensitivity range of 1.5–3.0 s around the 2.334-s object-look anchor. Its
transition probability accounts for the chance of resampling the same target;
this is a modeling construction, not a fitted infant transition matrix.
Visibility changes and source-clip boundaries may truncate bouts, and low
object diversity can prevent the target duration; validate realized durations
and record those limits. The fixed
head trajectory, visible-object-only gaze and omission of parent handling
make this a constrained synthetic model, not a fitted reproduction of an
infant eye tracker.

For speech, the specified 55% is the gazed share of the **visible, non-held
naming channel** (the remaining 41% after held 35% and unseen 24%), rather
than a share of all non-held names including unseen names. At the central
values it contributes 0.41 × 0.55 = 22.55% of all naming events. This
denominator follows the spec's instruction to split the remaining visible
share. The 50–60% attended-referent statistic in
[speech_coupling.md](speech_coupling.md) has a different age/context and
denominator; 55% here is a design target, not a directly measured conditional
probability from toddler object play. Channel assignment must respect
available non-held gaze at the preserved utterance times; independent
channel draws can make this target unattainable. Gaze itself must never
read speech or be changed to meet speech quotas.

The checked primary PDFs and text extractions are retained locally under
`outputs/validation/child_gaze/literature/`; they are ignored, not redistributed
in Git. Exact held-choice and salience coefficients were not found in the
checked papers; no figure was silently treated as an exact numeric estimate.

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

## Phase A implementation and validation protocol

The retained input is `outputs/renders/clean_856/`, with the seed-0 speech
schedule in `outputs/features/visual_grid16/manifest.parquet`. Its 102,720
frames contain visibility counts and coarse cells, but no exact object image
positions. The close-up exporter reconstructs the saved scenes and renders
instance masks to recover a visible pixel nearest each instance's visible
centroid. It checks every exported instance's pixel count against the saved
metadata. Instance IDs remain distinct even when their word categories match.
The existing main video is read-only.

Gaze uses those pixels, saved holding, image size and image motion. Its
independent per-session seed does not depend on speech. A sticky categorical
transition gives a stationary held preference of 0.60 when a held object and
alternatives are visible, and a 2.334-second mean contiguous bout in a fixed,
feasible scene. These are model settings; dynamic-scene durations and shares
are measured separately. Concentrated or single-object scenes can limit the
achievable switching rate. Frames without visible objects have null gaze;
they receive neither an invented target nor an invented close-up.

Speech first reproduces the original schedule, then changes referents using
a separate seeded stream. Channel probabilities respond to the already-fixed
gaze, preserving utterance times, named flags, templates and timing jitter.
The 35% held channel retains its original clip-local ±5-second holding window;
it is distinct from physical holding at the source or jittered frame. The
55% gazed-visible target applies at source time. Both source and aligned
gaze/holding shares are recorded. An infeasible availability calibration or
achieved named-gaze share outside 50–60% stops generation; held and unseen
channel shares must be within two percentage points of their targets.

The close-up camera remains at the original eye and rotates toward the gaze
ray, preserving the head trajectory. A 24-degree horizontal field of view and
224×224 output are engineering assumptions (open choices in this spec), with
depth of field disabled. It renders the scene anew, rather than enlarging
the main image. Encoding uses the exact offline Step 1 DINO snapshot
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`, retaining 256 patches plus CLS.
Only image paths and frame keys enter the encoder; instance/category fields
are for world generation and audits, not learner inputs.

Time scene reconstruction, geometry export, setup and close-up rendering
before production. The union of five-frame windows around all 34,140 original
utterances is a conservative 82,952-frame rendering envelope; exact timing
preservation keeps that bound valid without freezing Phase B panels. Missing
gaze can reduce the number of images. Reuse prepared replay jobs between the
geometry and close-up passes. Record a measured local-worker configuration
and headroom in the gate; do not infer parallel speedup from worker count.
Production generation follows a source commit. Source hashes distinguish
precommit validation samples from production artifacts.

New gaze/geometry and images live under `outputs/renders/child_gaze_closeup/`,
speech under `outputs/speech/child_gaze/`, features under
`outputs/features/child_gaze/`, and validation evidence under
`outputs/validation/child_gaze/`. `outputs/runs/child_gaze/STATUS.md` records
the current gate and next action. Phase A does not select a learner, freeze
new test panels, or launch a language-learning comparison.

### Local rendering gate — 8 November 2026

Two local workers exported 240 frames from two complete sessions in 27.025 s,
including reconstruction and scene setup. The same saved replay jobs rendered
239 valid close-ups in 42.158 s. Scaling both **complete session passes** to
856 sessions gives 3.21 hours for geometry and 5.01 hours for close-ups;
25% headroom gives **10.28 hours**, below the 12-hour cap. This deliberately
does not discount the close-up pass to its smaller 82,952-key envelope, so it
does not incorrectly scale away per-session setup. Encoding is recorded
separately. The gate record is
`outputs/validation/child_gaze/render_gate.json`.

The initial 11-frame sample spans 11 source clips. Every instance visibility
count matches the retained metadata, and every exported gaze point lies on
its own instance mask. Replayed main RGB is not claimed bit-identical (small
render differences remain); the original main images are preserved and used
unchanged. The actual close-ups were visually inspected. An 11-image offline
DINO smoke test produced finite `[11, 257, 768]` float16 features in 0.95 s;
the first image exactly matched direct encoding from the pinned snapshot.
These checks validate the implementation, not a language-learning benefit.
