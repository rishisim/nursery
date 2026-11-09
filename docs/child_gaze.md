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

- **How gaze enters the learner — decided: (a) additive.** (a) A separate gaze
  branch added to the video score, the same additive setup used for body.
  (b) Gaze steering which video patches a word is matched to during training.
  The user chose (a) by design on 8 November 2026, without a pilot; see
  [Readout decided by design](#readout-decided-by-design--8-november-2026).
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

### Phase A takeover — 8 November 2026

**At this earlier checkpoint, world generation was complete and production
close-ups/cache were blocked before rental launch. The later hybrid checkpoint
below supersedes that execution status; Phase A remains incomplete.** All 856 sessions
finished geometry export with two local workers in 6,314.50 seconds (105.24
minutes). Only after the complete 102,720-frame table and all session receipts
were saved was the old Phase A Codex process stopped, at 15:23:29 UTC. No
production local close-up render was started.

The geometry job started from implementation revision `0c6f9be5`. Its completion
manifest and subsequent gaze/speech generation record `84ce80f1`, the then-current
HEAD; the Phase A simulator, speech and encoder source is unchanged between
those commits. The later learner implementation remains separate from this
world validation. No learner was selected or trained here.

| Full-world measurement | Result |
|---|---:|
| Geometry sessions / frames | 856 / 102,720 |
| Valid object gaze / null gaze frames | 102,022 / 698 |
| Held share among valid gaze frames | 33.29% |
| Held gaze when a held object is available | 60.51% |
| Observed object-bout mean / median / 95th percentile | 1.888 / 1 / 5 seconds |
| Frames with diversity-limited switching | 4,377 |
| Preserved utterances / named utterances | 34,140 / 8,489 |
| Named held / unseen / visible non-held channels | 34.54% / 24.43% / 41.03% |
| Gazed share of named visible non-held channel, source time | 55.64% |
| Same share at the jittered aligned frame | 40.48% |
| Physically held referent among names, source / aligned time | 21.82% / 20.76% |
| Required close-up images / masked null-gaze keys | 82,393 / 559 |

The 60% gaze preference is conditional on an available held object, not a
prediction that most frames show held-object gaze. Realized bouts include
clip cuts and disappearance; the 1-Hz timeline also limits their resolution.
Speech meets its source-time 50–60% gaze gate and held/unseen tolerances while
preserving timing jitter. The lower aligned gaze agreement is retained weak
alignment, not repaired by moving utterances or gaze. Held-channel and
instantaneous physical holding remain different quantities.

Independent world audit passed: all 822,168 instance records agree with saved
visibility, 3,424 complete masks were independently checked, and all 102,022
valid gaze points land on their own instance mask. Gaze regenerated exactly
even with arbitrary speech columns added to the scene table. All 34,140 speech
rows preserve timing, named/referential flags, cue weights and template form;
referent flags and achieved shares were independently reconstructed. Seven
frozen original input hashes remain unchanged. The 82,952-key envelope contains
82,393 required valid-gaze close-ups and 559 null-gaze keys to mask, not render.
The focused implementation suite passed 34 tests; the independent auditor also
ran a 19-test subset. Output hashes are retained with the audit. Production
image/cache coverage and frozen-DINO validation remain pending.

The approved rental was one 8× RTX 4090 host, with a **$10 total and 90-minute
launch-to-return hard cap**. Preflight found a 21.03-Mbps upload-only rate.
The completed replay arrays, room trees, object library and pinned DINO alone
total at least 20,928,119,418 bytes: **132.68 minutes to upload**, excluding
jobs, metadata, source bundles, Blender, remote setup, rendering, encoding,
download and verification. This fails the time gate before launch. **No
instance was created, no assets were uploaded, and actual rental cost is $0.00.**
There is no rental to tear down. The earlier approximately 56-minute/$5 plan
assumed much faster transfer and is not a qualified estimate for this network.

That checkpoint required a faster connection passing a fresh complete
time/cost projection, or an explicit instruction to render locally. The later
hybrid authorization below superseded that gate.
The existing local two-worker estimate is about five hours for close-ups,
with encoding separate; no fallback job was launched. Preserve the completed
world, required-key envelope and full frozen-DINO feature design when resuming.
For the next build step, this supplies the intended weak gaze-driven naming
world. The missing sharp-view cache still prevents completing Phase A and
moving to the Phase B development comparison; these checks say nothing yet
about a language-acquisition gain.

Retained ignored evidence includes `outputs/renders/child_gaze_closeup/`,
`outputs/speech/child_gaze/`, and `outputs/validation/child_gaze/takeover.log`,
`takeover.json`, `takeover_audit/`, `network_upload.json` and `rental_gate.json`.
The rental plan and prepared scripts remain in `outputs/runs/child_gaze/`;
`STATUS.md` identifies the blocked next action. Original renders, speech,
features and runs remain preserved.


### Hybrid close-up checkpoint — 8 November 2026

**Phase A is partial, with a verified resumable checkpoint.** The user authorized
local rendering plus up to eight rented RTX 4090 GPUs, with a $10 cap and
three-hour wall cap. Execution began at 15:45:45 UTC. A four-RTX-4090 Vast host
was created at 15:48:16 UTC at $1.96444/hour including 120 GB disk. The source
renderer and encoder stayed unchanged at execution revision `4b7f1f38`.

Sessions were claimed atomically from one local SQLite queue. Local workers
preferred larger inputs; the uploader preferred the smallest missing asset
sets, reusing already transferred room/object assets. Rendering preserved the
complete candidate pools needed to reconstruct the original scenes. Public
Blender and model files were downloaded on the rental. New per-session images
were returned continuously, hash-checked, and encoded on the Mac with the same
pinned DINO and all 257 tokens, avoiding a roughly 32.5-GB feature download.

The measured two-minute local pilot improved from 14.05 images/s with four
workers to 15.07 with six (+7.2%); concurrent encoding reached 43.4 images/s
with 37% memory free. Six workers were retained. A remote session produced
93 images in 34.67 seconds including reconstruction; a three-worker sample
produced 280 images in approximately 43 seconds. These are execution samples,
not a controlled hardware comparison. Transfer/setup dominated remote startup.
The 1.358-GB first payload required resumed transfers; its complete original
SHA256 and all 358 input-file hashes passed before rendering. A transfer-boundary
corruption was detected and repaired before any affected bytes were used.

At 16:37 UTC the Mac reported 3% battery and roughly seven minutes remaining.
New claims and encoder batches were stopped, all in-flight work drained, and
all finished remote shards retrieved before the rental was destroyed at
16:39:42 UTC. The one remote claim that had never started was released only
after destruction. No shard was rendered twice. No rental remains running;
resumption requires AC power, not a new scientific design. The independent
partial audit finished about 55 minutes after execution began. The provider's
usage ledger records **$1.707 ($1.71 rounded)**: $1.662 GPU, $0.038 disk and
$0.007 inbound transfer, with 0.866 billed hours. Separately, Vast automatically
charged **$10.00 to add account credit** (`auto=true` in the invoice). That
funding is not an additional $10 of compute usage. Charges and funding are
retained separately in `rental_stream/billing_charges.json` and
`billing_invoices.json`.

| Verified checkpoint | Count |
|---|---:|
| Complete rendered sessions | 501 / 856 |
| Required close-ups saved | 48,137 / 82,393 |
| Local / rental images | 46,384 / 1,753 |
| Rental sessions returned | 18 |
| Complete full-token DINO feature rows | 46,863 |
| Rendered images awaiting encoding | 1,274 |
| Images still to render | 34,256 |
| Pending / claimed / failed sessions | 355 / 0 / 0 |
| Null-gaze envelope keys, still masked | 559 |

Independent validation passed 19 focused tests and a 101-session, 16-client
atomic-claim test with no duplicate claims. The partial completion audit checked
every saved image hash, dimensions, exact required-frame subset, original job
and source hashes, and agreement between receipts and queue. All 28 completed
feature shards passed full byte hashes, row mapping, finite float16
`[N, 257, 768]` values, pinned-model hashes and encoder provenance. The initial
direct MPS re-encoding sample matched exactly. Nine local and six remote views
were visually checked for sharp, centered gaze targets. All seven original
input hashes and twelve previously audited world/source hashes are unchanged.
These checks establish the saved partial artifacts, not full Phase A coverage.

The canonical session outputs are under
`outputs/renders/child_gaze_closeup/hybrid/sessions/`; complete feature batches
are under `outputs/features/child_gaze/hybrid/local_batches/`. The atomic queue,
stop markers, orchestration hashes and encoder state live in
`outputs/runs/child_gaze/hybrid/`. Independent evidence is
`outputs/validation/child_gaze/hybrid_audit/partial_completion.json` and
`partial_features.json`; rental transfer/teardown evidence is retained under
`outputs/runs/child_gaze/rental_stream/`. All are ignored. Original inputs and
prior outputs remain intact. No full-cache completion marker was written.

For the next build step, connect the Mac to power and resume the same queue and
encoder, preserving completed session receipts and feature batches. The
remaining local work is approximately 40 minutes including encoding/merge and
validation, based on observed throughput, with no additional rental cost.
Only after exact 82,393-key coverage and the final cache audit pass is Phase A
complete and the Phase B development comparison ready to proceed. No learner
was selected or trained during this execution.

### Local resumption attempt — 8 November 2026

The Mac was connected to AC power, but the new restricted execution session
could not run the saved local pipeline: PyTorch reported MPS built but
unavailable, and all six Blender scene-render attempts exited `-11` before
writing images. `blender --version` succeeded; the native scene startup
failed with an `Arch_ValidateAssumptions` warning. This is an execution
blocker; no scientific setting or renderer/encoder source was changed.

The six failed claims produced only empty task-created directories. After
independent inspection those directories were removed and the claims reset
to pending. The original checkpoint remains 501 complete sessions, 48,137
images and 46,863 encoded rows in 28 batches; 355 sessions / 34,256 images
remain pending, with zero claimed or failed sessions. Stop markers were
restored and all task processes exited. No rental was launched. Resume
logs and reset evidence are in `hybrid/resume_blocker.json` under the ignored
run root; preservation evidence is under `hybrid_audit/`.

Phase A remains incomplete. Its prior approximately 40-minute local ETA
applies only after restoring native Blender and MPS access. Phase B was
paused before pilot, panel preparation or protocol freezing; provisional
source edits were restored. The Phase B timing gate and qualified Phase C
rental plan are still pending, rather than passed or frozen.


### Completed Phase A local resume — 8 November 2026

**Phase A is complete.** Native Blender and MPS access worked in this session.
The same atomic done-manifest resumed only the 355 pending sessions, adding
34,256 close-ups without re-rendering the original 501 complete sessions.
The encoder reused the original 28 batches and added 35,530 missing rows with
frozen DINO snapshot `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
All 856 sessions now supply exactly 82,393 required images and feature rows;
559 null-gaze envelope keys remain masked. No rental or Juno was launched.
Despite the requested power connection, macOS continued to report battery
power; the local supervisor checked power and would drain and stop at 15%.

The 49 complete feature shards were losslessly merged into
`outputs/features/child_gaze/hybrid/full/closeup_cache/`, preserving full
float16 `[82,393, 257, 768]` features. Rendering finished in approximately
31 minutes after resumption; the merge took 86.92 seconds. Independent final
receipt and feature audits took 29.99 and 168.42 seconds respectively.

The independent audit verified every image hash/dimension, exact required-key
coverage, all shard and merged cache hashes, finite feature values, every
merged row against its source, seven original input hashes and twelve frozen
world/source hashes. Three fresh MPS encodings matched saved float16 values
bit for bit. The original 501 receipts and frame tables are unchanged, and
all original 28 cache hashes, ordered-image digests, row counts and recorded
provenance match both earlier partial audits. Three new actual close-ups
passed visual inspection. The focused source suite passed 74 tests, including
the separately prepared learner/protocol checks.

Retained evidence is `outputs/validation/child_gaze/hybrid_audit/`:
`final_receipts.json`, `final_features.json`, `final_preservation.json` and
`final_visual.json`; the merge marker is `hybrid/features_complete.json`
under the run root. Earlier blocked attempts and battery checkpoints remain
historical evidence. All outputs are ignored; original inputs and runs remain
preserved. The completed sharp-view cache now permits the Phase B development
comparison. It does not establish a language-acquisition benefit.

## Phase B execution protocol

`nursery.learner.child_gaze` prepares the new-speech input view, panels and
630-clip subset without changing the retained video caches. It checks every
accessible five-frame window against scene metadata, 16×16 video, gaze points
and valid-gaze close-ups. Null gaze stays masked. Close-up input is the exact
float32 mean of the 256 frozen patch features, excluding CLS; the full-token
Phase A cache remains preserved. Semantic gaze/held labels never enter the
learner. Point-only input does not load close-up features. The child-gaze
loader gathers bounded float16 batches and casts on the accelerator to the
same float32 model inputs; frame order, padding and shuffle RNG are unchanged.

The user chose the additive readout by design on 8 November 2026. This
supersedes the earlier development-pilot selection rule; no development pilot
or steering training is required. Gaze dropout stays 0.5 and the additive
initial weight stays 1.0. Steering remains implemented but unused.

Freeze additive for video-only, video plus gaze, shuffled gaze and point-only,
each at 630 clips with seeds 0–2 and 60 epochs. Keep Step 1 batch 256, AdamW
learning rate 0.0003, weight decay 0.1, soft pooling temperature 0.1, five-frame
window, vocabulary fitting, unknown-word rule and earliest development-maximum
checkpoint selection. Both gaze test modes use the same with-gaze-selected
checkpoint; picture transfer removes gaze and retains its separate
picture-development checkpoint selection.

The original `study/` is retained preparation and pilot evidence. The canonical
final protocol is `matrix/protocol.json`, sharing the unchanged prepared
panels and subsets; `matrix/frozen_inputs.json` binds it to committed source,
prepared inputs and source hashes. No new input preparation is performed.
Run `PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m
nursery.learner.child_gaze freeze --readout additive`, then `timing`. Source and
this protocol must be committed and pushed before freeze or timing. A later
reporting commit does not authorize rebinding or executing different source.

The timing gate trains two fresh epochs per condition from seed 0 using only
clips_dev and picture_dev. It records each epoch duration, discards only its
new timing weights and projects the twelve-run matrix serially with 25%
headroom. Development passes provide a conservative estimate of final
evaluation time; no test panels are scored. Record the Phase C transfer and
compute plan, then stop for rental approval. No rental is launched by these
commands. Outputs stay in the existing ignored child-gaze roots.


### Phase B measured pilot gate — 8 November 2026

Preparation and timing ran from committed revision `87e719b0`. The new world
has the same 630 selected training clips and 18,613 vocabulary-bearing
utterances. All 34,140 speech rows are covered; all 82,952 accessible window
keys have 16×16 video and gaze, with 82,393 valid close-ups and 559 masked null
keys. There are no cache gaps. New-world development, new-clip test, s8 and
picture-transfer panels are prepared, with the same split and sampling rules.

| Development-only timing candidate | Setup | Epoch including development selection | Conservative final evaluation estimate |
|---|---:|---:|---:|
| Additive gaze | 1.05 s | 155.39 s | 8.35 s |
| Gaze steering | 0.64 s | 175.63 s | 12.70 s |

**Blocked at the pilot timing gate.** Conservatively applying the brief's
no-cut-epochs rule to the pilot retained both 60-epoch candidates. They project
to 6.90 hours with 25% headroom, plus the 6.84-minute preflight already consumed
(about 7.02 hours total), exceeding the four-hour pilot cap. The unpadded full
training projection is already about 5.52 hours. The bounded float16 transfer
path was included in these timings; these are measurements, not a projected
speedup claim. No full pilot started, no readout was selected, and the final
four-condition matrix was not frozen or timed. Timing checkpoints were removed.

The independent timing audit verified that the rule predated both smokes,
only development panels were scored, held-out result maps are empty, full
pilot models do not exist and selection remains pending. Evidence:
`outputs/validation/child_gaze/phase_b_timing_audit.json`, and `pilot.json`,
`selection_rule.json`, `cache_coverage.json`, timing metrics and logs in the run
root. Both frozen pilot snapshots retain execution revision `87e719b0`; later
reporting commits do not change that binding.

The brief fixes pilot clips, seed and wall cap but does not explicitly state
pilot epochs. Clarifying that a fixed 20-epoch development pilot is permitted
would project about 2.41 hours including the retained preflight and headroom;
all twelve final runs would still retain 60 epochs. This change was presented
for clarification and was not made without approval. Alternatively, approve
GPU completion of the full candidates before selection. Real AC power is also
needed for long local training: macOS still reported discharging battery.

### Conditional Phase C compute plan

#### Twelve-run plan after the design decision — 8 November 2026

The current plan is
`outputs/validation/child_gaze/phase_b_design_rental_plan.json`. It replaces the
earlier pilot-inclusive plan, which remains retained as historical evidence.
Run only the four additive conditions × seeds 0–2 × 60 epochs. No pilot wave
or steering training is included. No new rental, Juno access or project-data
upload occurred while preparing this plan.

The stopped upload was already four parallel rsync streams over direct SSH,
with SSH compression enabled. Its final 84,364-byte/s aggregate rate was not
a single-stream or SSH-proxy limit. Local uplink instability, congestion or
loss on the route, and host ingress remain possible; the exact historical
cause cannot be separated without testing that destination again. Fresh
synthetic uploads to Cloudflare's neutral speed-test endpoint measured
3.49/3.42 Mbps with one stream and 4.32/1.79 Mbps with four. Larger timed-out
attempts are retained separately. These short tests show no reliable parallel
gain and do not establish sustained throughput to a rental.

| Transfer option | Measured basis and expected time |
|---|---|
| Four parallel streams | Already used on the old host. Keep resumable independent archives and hash checks; more streams alone have no demonstrated benefit. |
| Lossless compression | Count-only local zstd level-1 encoding of all eight archives reduced 1,927,782,400 bytes to 1,623,638,757 bytes in 0.90 s (15.8% smaller); originals unchanged. About 10.3 min at 21.03 Mbps, 50–121 min at the fresh neutral parallel rates, or 321 min at the historical stop rate. |
| Different host/region | Prefer a nearer US host over the old Bulgarian host. At a measured 21 Mbps, compressed transfer would take about 10.3 min; reserve 15 min with verification. Advertised host bandwidth does not establish local-to-host throughput. |
| Pre-stage before launch | Only with separate approval to upload project data: private Cloudflare R2 Standard storage, verified objects and short-lived download URLs. Local staging would take about 50–121 min on the current connection before the rental clock; GPU retrieval would take 1.1–10.8 min at an assumed 200–20 Mbps. Reserve 15 min and measure it. Nothing was staged or uploaded. |
| Faster local network | Join UTD CometNet or a known faster wired connection and remeasure. At actual 100–21 Mbps upload, compressed transfer would take 2.2–10.3 min; reserve 5–15 min. Campus uplink speed is unmeasured. |

The current connection is **not qualified** for the proposed 15-minute transfer
reserve. The compressed base plus final-source allowance requires at least
14.90 Mbps sustained before protocol and verification overhead; target at
least 20 Mbps. Prefer the faster-network
option, then verify the actual destination after approval. R2 pre-staging is
an alternative requiring explicit data-upload permission, even if its unused
free allowance covers the payload. [R2 pricing](https://developers.cloudflare.com/r2/pricing/)
lists free egress; account usage is unknown.
[CometNet access](https://atlas.utdallas.edu/TDClient/30/Portal/Requests/Service/170/CometNet-Wi-Fi)
does not promise a particular upload rate.

Prefer one **8× RTX 4090 24-GB** host, at least 384 GB host RAM and 160 GiB
storage. Run eight learners, then four, one per GPU; run counts are
2, 2, 2, 2, 1, 1, 1, 1. A read-only Florida offer at 21:54:57 UTC quoted
**$5.39259/hour including disk**, with about 516 GB RAM; it is not reserved.
Refresh availability and price before approval is executed. The old Bulgarian
offer remains about $4.31/hour but its transfer path is unqualified.

Keep the old payload unchanged; include the new committed source and frozen
matrix in a separately hashed local transfer inventory. The local manifest
`outputs/validation/child_gaze/phase_b_payload_delta.json` binds revision
`bfa470c6`, all 122 required source/matrix/transport-reference files and
42,129,377 content bytes. Allowing 50 MiB for the complete delta and headers
gives a conservative 1,676,067,557-byte transfer ceiling: 10.6 minutes at
21.03 Mbps. Target at least 20 Mbps and retain the 15-minute transfer reserve.
Transfer compressed
archives with resumable direct SSH streams, without redundant SSH compression,
or fetch pre-staged objects if separately approved. Re-encode the common
video/picture caches once using the pinned DINO snapshot and explicit original
PIL preprocessing. Keep the nine predetermined parity rows and thresholds;
validate full coverage, finite values and hashes, and bind the shared CUDA
cache derivation explicitly to the original inputs and frozen revision. Do
not silently replace local frozen hashes or execute the old payload checkout.

Before any full run, measure two fresh development-only CUDA epochs per
condition at intended eight-worker concurrency, then discard those weights.
Start every final run fresh. Use condition-specific conservative epoch times,
run setup and final-evaluation estimates to balance the two waves and project
their maximum durations, with 25% headroom. Synthetic step times and local
MPS timings do not qualify CUDA execution.

Reserve **90 minutes** beyond full training epochs: setup 15, transfer/hashes
15, encoding/parity/cache audit 30, CUDA timing 10, result return/verification
and teardown 10, contingency 5, and 5 minutes for padded run setup/final
evaluation across the two waves. With every condition bounded by E seconds
per epoch and that last allowance sufficient, 90 + 2.5E gives **165–215 minutes
at 30–50 s/epoch**. The exact gate is 85 minutes plus 1.25 times the sum of
the two wave maxima, each including 60 epochs, measured run setup and final
evaluation. Use actual elapsed time; tighten the qualifying epoch ceiling
if final evaluation/setup exceeds its allowance. The quote gives about
**$15.08–$19.57** over that range, including a $0.25
transfer reserve. The **$20 cap** implies an earlier stop at about **219.75
minutes**, rather than using the whole four hours; preserve time and cost for
return and teardown. At four hours this host would cost about $21.82, so it
does not qualify for a four-hour stay at this cap. Reproject before full
training and on material delays; stop if either cap fails, without cutting
epochs or seeds. The previous attempt's $4.90 is separate.

[Juno](https://hpc.utdallas.edu/systems-resources/juno/) remains an alternative
only with approval, an active allocation, campus/VPN access and prompt GPU
availability. A suitable four-GPU H100/H200 node would run three learners per
GPU in three waves. With the same 90-minute overhead, including queue time,
90 + 3.75E minutes requires E ≤40 seconds before additional run overhead;
measure real concurrent CUDA epochs first. Allocation, queue, memory,
throughput and any quota charges are unverified. No Juno session was opened.
The next build step is the approved twelve-run comparison, then Phase D;
these execution checks do not establish a language-acquisition gain.

### Approved GPU preflight and transfer stop — 8 November 2026

The user approved the rental plan, including both full **60-epoch** development
pilots (630 clips, seed 0) before selection, followed by the twelve matched
runs. The limits remained **$20** and **four hours from rental launch to results
back**. Neither epochs nor seeds were shortened. One Vast instance, `59765261`,
provided eight RTX 4090s and 160 GiB storage at $4.311111/hour. It launched at
19:17:33 UTC using execution checkout `87e719b0`; the original prepared study,
local timing pilots and sensory inputs were preserved.

The trimmed payload contained 106,741 regular files, including all 1,712 frozen
provenance signals: 1,736,507,993 content bytes, packaged into eight archives
totalling 1,927,782,400 bytes. Transfers resumed partial files, checked archive
hashes, extracted each completed archive and removed it remotely. An unstable
uplink and the host's initial 20-second SSH client timeout interrupted transfer;
the timeout was corrected and partial bytes retained. Only two archives were
fully verified/extracted before the budget stop.

The exact DINO snapshot and its three file hashes passed. Before full encoding,
nine predetermined original-cache rows—first, middle and last in each cache—
were tested against unchanged thresholds: maximum absolute error ≤0.05,
relative L2 error ≤0.001 and cosine ≥0.99999. The rental's automatic processor
initially selected Torchvision, whereas the original Mac environment selects
PIL because Torchvision is absent. That attempt failed and was retained.
Explicitly selecting **PIL** restored the original preprocessing; all nine
processed input tensors then matched the Mac byte-for-byte. This corrected
the execution implementation without changing the rule or tolerances.

| Preflight check | Verified result |
|---|---:|
| Corrected fixed-sample CUDA feature parity | 9 / 9 passed |
| Worst absolute feature error | 0.0078125 |
| Worst relative L2 error | 0.000059321 |
| Minimum cosine | 0.99999999824 |
| Concurrent full-batch synthetic kernel/VRAM checks | 8 / 8 passed |
| Synthetic measured step time / peak allocated VRAM | 0.09–0.19 s / about 5.92 GiB |
| Full development pilots / matched Phase C runs completed | 0 / 0 |

Synthetic checks used the full batch, five-frame window, 257 tokens and both
readouts, but saved no weights. They exclude real cache gathering and
development evaluation, so they are **not measured scientific epoch times**
and cannot authorize runs or select a readout. No full CUDA cache was encoded;
its complete coverage/finite-value/merge audit and the real epoch gate remain
pending. The rental environment records PyTorch 2.7.1/CUDA 12.6, Transformers
5.19.0 and Pillow 12.3.0, alongside the full package/device snapshot.

**Stopped at the wall-time projection gate.** At the stop-trigger observation,
the latest 189.54-second transfer window achieved 84,364 bytes/s. The remaining
904,041,776 archive bytes projected another 178.60 minutes. Added to 63.72
minutes already elapsed, **upload alone projected 242.32 minutes**, exceeding
the 240-minute limit before encoding, either pilot, the twelve runs or result
return. Independent review confirmed the calculation. A subsequent partial
rate recovery did not establish a qualified complete-job projection.

All **42 preflight result files** returned and passed their recorded hashes
at 20:25:14 UTC, **67.68 minutes after launch**. The rental was destroyed at
20:25:37 UTC, **68.06 minutes after launch**; a live provider query confirmed
zero remaining instances. The provider usage ledger records **$4.90** for this
attempt ($4.848 GPU, $0.052 disk, bandwidth rounded to $0.00). Prior rental charges
and earlier credit funding are separate. The task-created private SSH key was
removed after teardown.

The independent execution audit passed: 42 returned hashes, all nine corrected
parity samples and their original failure record, eight synthetic workers,
19 original prepared files and 71 source hashes, the time gate and teardown
order. The unchanged-source focused suite had 49 passing tests. This is a
**preflight/stop audit, not Phase D**. No scientific pilot, learner held-out
scoring, readout selection, final protocol freeze or matrix run occurred. The
required `runs_complete.json` records `complete: false` and zero of twelve.

Evidence remains ignored under `outputs/runs/child_gaze/phase_c/`, including
the payload manifest, processor/parity records, rejected attempt, synthetic
checks, `budget_gate.json`, returned-file manifest, billing and destruction
receipts. The independent record is
`outputs/validation/child_gaze/phase_c_execution_audit.json`; `STATUS.md` gives
the current blocked state. Original renders, speech, features, prepared study
and local timing pilots remain preserved.

For the next build step, use a faster uplink and a fresh qualified rental
window to complete the original full development-only pilots, then select,
freeze and commit before the twelve matched runs. Recheck actual transfer and
epoch throughput; the earlier 21.03-Mbps network test did not describe this
attempt's sustained transfer. Phase D waits for all twelve results. These
checks establish execution compatibility, not a language-acquisition gain.

## Readout decided by design — 8 November 2026

The user chose the additive gaze branch, option (a), by design. No
development pilot is run and steering is not trained. Reasons:

- It is the test this step exists for: gaze added as one more input,
  score = video + β·gaze, the same way body is added in later steps, so gaze
  and body results stay comparable.
- The video path is the same as the video-only learner, so any gain comes from
  the added input. Steering changes how the video path reads the frame.
- Gaze dropout 0.5 already guards against the video path learning less.
- The gaze-removed test is a secondary question (does gaze improve video-only
  learning?).

This replaces the pilot selection rule above; its settings (dropout 0.5,
additive initial weight 1.0) stay. Steering code stays in place, unused; it is
the natural next variant if additive shows no lift. The next step is to freeze
the four-condition additive matrix (seeds 0–2, 630 clips, 60 epochs, Step 1
settings), time it, and plan Phase C compute without a pilot wave.

### Phase B additive freeze and measured timing — 8 November 2026

Source, focused tests and the execution protocol were committed and pushed
as `bfa470c64e5ca2c531bd0ee520e378a1bd467d67` before freezing or training.
The canonical final protocol is
`outputs/runs/child_gaze/matrix/protocol.json`; its frozen-input record binds
that revision, 71 source files, 16 prepared files and 1,730 external inputs.
The original `study/`, pilot timing records and stopped-rental evidence remain
preserved. The final matrix shares their unchanged prepared inputs; only the
selection status and design-decision record differ from the prepared protocol.

The freeze retains all four conditions, 630 clips, seeds 0–2, 60 epochs,
Step 1 optimizer/batch/selection, gaze dropout 0.5 and initial additive weight
1.0. Both gaze test modes use the with-gaze-selected checkpoint, and picture
transfer removes gaze. No development pilot or steering training was run.

Two fresh MPS timing epochs per condition used seed 0 and development panels
only. Each timing learner started fresh; the two epochs within that learner
were consecutive. All new timing weights were discarded after measurement.

| Condition | Epoch 1 | Epoch 2 | Mean seconds/epoch |
|---|---:|---:|---:|
| Video only | 145.64 s | 149.62 s | 147.63 |
| Video + gaze | 154.88 s | 161.31 s | 158.10 |
| Video + shuffled gaze | 156.05 s | 156.31 s | 156.18 |
| Video + gaze point only | 154.89 s | 152.06 s | 153.47 |

The serial twelve-run projection is **138,658.91 seconds / 38.52 hours**:
1.25 × 3 seeds × the sum over conditions of setup + 60 mean epochs + the
conservative final-evaluation estimate. That estimate is four times a final
development pass, not test scoring. Timing took 20.79 minutes after the initial
full provenance verification. Evidence is `timing_gate.json`,
`matrix/timing/*/{history,metrics,status}.json` and `timing_additive.log` in the
run root. These are execution measurements, not acquisition results.

The independent audit checks the complete frozen settings, committed source
binding, two actual epochs per condition, empty held-out result maps, removed
timing weights, unchanged earlier evidence, and projection arithmetic. The
focused source validation passed **19 tests**. The independent audit **passed**:
`outputs/validation/child_gaze/phase_b_additive_audit.json`. It verified all
frozen hashes and all 78 earlier evidence files plus eight payload archives
unchanged. The run-root `STATUS.md` records the
handoff. A later reporting commit does not change the frozen execution revision.

**Phase B is complete; Phase C is blocked pending rental approval and a
qualified transfer path.** The current twelve-run rental plan above includes
no pilot wave. No new rental, Juno access, project-data upload or test-panel
scoring occurred. Phase C remains 0/12 and Phase D has not started. The next
build step is to test whether the frozen additive gaze input improves language
acquisition under matched training, after the transfer and real CUDA gates pass.

### Approved additive rental stopped at the cost gate — 8 November 2026

The user approved one US 8× RTX 4090 host for the twelve frozen additive
runs, with **$20 total** and **four hours from launch to verified results
back**. No relaunch, extra machine, staging service or protocol change was
authorized. AC power and caffeinate were confirmed before launch. The
operator, monitor and independent auditor had separate roles.

One Florida host, instance `59831415`, launched at **03:58:51.857 UTC on
9 November** (8 November locally). It supplied eight standard 24-GB RTX
4090s, 515,738 MB host RAM and 172 GB disk, enough for the 160-GiB minimum.
The recorded GPU/disk quote was **$5.381111/hour**. Including the $0.25
transfer reserve, the cost deadline was 220.215 minutes after launch,
earlier than the four-hour limit.

The new neutral upload measurements were **106.26 and 106.95 Mbps**, each
using 100 MB. All eight retained archives and the separately hashed
122-file source/matrix delta were verified and losslessly compressed. The
**1,629,582,976-byte** payload reached this host in **145.115 seconds at
89.837 Mbps**. Compressed hashes, original tar hashes, extracted input hashes
and the pinned DINO snapshot passed. Original local files stayed unchanged.

The retained PIL encoder was reused with its output-root adjustment. All
nine predetermined early parity rows passed; complete encoding then passed
coverage, finite-value, exact key/order and lossless-merge checks for all
three shared caches. The final nine-row parity had worst absolute error
**0.015625**, worst relative L2 **0.00006469393** and minimum cosine
**0.999999997908**, within the unchanged thresholds. The environment records
PyTorch 2.14.1/CUDA 13.2, Transformers 5.19.0, Pillow 12.3.0 and Python
3.12.15; the new runtime was qualified by these checks.

An explicit child binding preserves the canonical `bfa470c6` source,
protocol, prepared files and non-derived inputs. Its only execution changes
are CUDA device selection and the recorded shared derived-cache paths and
hashes. The original frozen matrix was not rewritten. The superseded
pilot-selection orchestration was adapted to the additive design decision;
no pilot selection or steering training ran.

Eight fresh workers each completed **two development-only CUDA epochs**,
two workers per condition. The gate uses the largest actual epoch across
both workers, plus the largest measured setup and four-times-development
final-evaluation estimate:

| Condition | Conservative seconds/epoch | Setup + 60 epochs + final evaluation |
|---|---:|---:|
| Video only | 53.7473 | 3,247.269 s |
| Video + gaze | 78.3727 | 4,738.766 s |
| Video + shuffled gaze | 71.9435 | 4,352.045 s |
| Video + gaze point only | 67.6175 | 4,091.736 s |

**The mandatory cost gate failed.** At 18.502 minutes elapsed, the recorded
eight-then-four schedule put a gaze run in each wave. Its projection was
1,110.135 + 1.25 × (4,738.766 + 4,738.766) + 900 seconds for return,
verification and teardown + 300 seconds contingency = **14,157.051 seconds /
235.951 minutes**, costing **$21.4113** including the transfer reserve.
The wall-time check passed, but the $20 check failed. All timing weights
were discarded, and **zero of twelve scientific runs started**.

The monitor also enumerated all 495 possible eight/four partitions as a
stop-decision check. The best grouping, with the same measured bounds,
headroom and reserves, still projects **222.471 minutes / $20.2024**. Thus
$21.4113 describes the recorded schedule, rather than a minimum possible
cost; regrouping alone would still fail this attempt's cap. No schedule,
reserve, epoch, seed or scientific setting was changed after the failed gate.

All **219 returned files** passed their hashes at **04:19:01.918 UTC**,
**20 minutes 10.061 seconds after launch**. All six final derived-cache
hashes were rechecked remotely before teardown. Destruction and a live
zero-instance confirmation completed at **04:19:03.125 UTC**, 20 minutes
11.268 seconds after launch. The task-created provider SSH key and local
private/public keys were removed. Provider usage for this attempt is
**$1.499**: GPU $1.477, disk $0.016 and bandwidth $0.006. The initial
$1.441 ledger snapshot was unsettled; the later operator and independent
auditor queries agree on $1.499. The prior $4.90 attempt remains separate.

The incomplete `runs_complete.json` records **0/12**. The independent
execution/stop audit **passed** (`audit/execution_stop_audit.json`): all 219
returned hashes, six final cache hashes, 71 source files, 16 prepared files,
1,730 original external inputs and 78 earlier evidence files were checked.
Timing weights were absent and teardown/key cleanup was verified.
**Phase D has not started**. There are no matched
video-versus-gaze accuracies, gaze-removed results, control results or
per-seed acquisition statistics from this attempt. The Step 1 old-world
16×16 reference remains unchanged and cannot fill these missing results.

Evidence is retained, ignored, under
`outputs/runs/child_gaze/phase_c/approved_rental/`: transport and runtime
inventories, `preflight/`, `returned/`, cache binding, measured gate,
`audit/`, `monitor/`, billing and teardown receipts. Original renders,
features, speech, prepared studies, runs and earlier evidence were preserved.

For the next build step, transfer and cache execution are now qualified;
the matched language-acquisition comparison still needs a new authorized
compute window. At the recorded speeds and reserves, the existing schedule
would need a quote at or below **$5.0222/hour**, or a higher cost cap; the
best eight/four grouping would need at or below **$5.3265/hour**. A future
host's actual transfer and concurrent epochs must qualify again. No further
rental or scientific change was made. This stop supplies execution timing,
not evidence for or against a language-acquisition benefit from gaze.
