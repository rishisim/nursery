# Small experiment: learning words across ambiguous encounters

Protocol and completed result for a small fixed-data comparison alongside the
[main study](acquisition_curves.md). The six planned runs completed on
27 October 2026; see the results below and the Step 4 execution record.

## The idea

Heavy/light showed that the learner can connect informative body signals to
words. Here we introduce a different challenge: each encounter contains several
possible referents, so the learner has to accumulate evidence across encounters.

For example, suppose “dax” names a category of object:

| Encounter | Heard | Seen |
|---|---|---|
| 1 | “Look, dax.” | An object from that category and a spoon |
| 2 | “Look, dax.” | Another view or example from that category and a ball |
| 3 | “Look, dax.” | Another view or example from that category and a book |

The learner receives observations, not the category labels in this table.
Changing distractors make the repeated relationship informative. Body signals
may supply additional evidence about the objects and interactions.

## Keep it small and controlled

Start with four object categories, each assigned a fixed unfamiliar word form.
Choose them for independent clip coverage, visible ambiguity and available
sensors—not learner performance. Reuse existing renders and features after
confirming they supply suitable distractor variation.

Each training utterance names a visible category, but multiple objects remain
plausible within that encounter. This is intentionally more scaffolded than the
main weak-speech task. Include both held and nonheld named objects at a fixed,
recorded balance: “held” must not be a perfect answer rule. Keep distractor,
subject and background associations from deterministically revealing the word.

Use all five body streams, the same time windows and the same language loss as
the main study. Labels may construct and score this controlled dataset, but
category IDs, referent labels and hidden physical parameters are never learner
inputs. The vocabulary and language are deliberately simplified.

## How we run it

Use two conditions: video only and additive video + body. There is no joint
learner in this small study. Their architecture is defined in the
[main design](acquisition_curves.md#the-three-learners), not implemented separately here.

Use one fixed small training set, with three paired seeds. Balance word exposure
and report independent clips and utterances per word. Keep the word/category
mapping and test panel fixed. There is no training-data sweep.

Split by source clip so replays cannot cross partitions. Train fresh models for
each seed, select checkpoints on development data, and test on unfamiliar clips
and varied views. New-instance testing requires genuinely held-out instances.

Testing uses four candidate situations and one learned word, with the same
candidate choices for all learners. Exactly one candidate contains the target
category within the permitted context. Body-enabled learners receive each
candidate's own sensory history. Chance is 25%.

As a supporting check, evaluate the same trained additive models with complete
body histories shuffled across situations. This is evaluation only, not another
training condition. Keep channels together within each history and freeze the
donor and missing-data rules before evaluation.

## The output we want

**A compact results table**, like the heavy/light presentation, reporting mean
held-out accuracy and uncertainty across seeds. The desired result is stronger
word understanding with body input after the same ambiguous encounters. This
is the hypothesis, not a promised result. No acquisition curve is produced here.

| Condition | Held-out accuracy, mean ± sample SD |
|---|---|
| Video only | 74.48% ± 5.92 percentage points |
| Additive video + body | 71.88% ± 4.69 percentage points |

Shuffling complete body histories at evaluation gave the same additive models
**68.23% ± 5.92 percentage points** across three seeds.

Chance is 25%; each run uses the same 64 held-out target trials. Standard
deviation describes variability across three training seeds, not a confidence
interval over all possible clips. Additive minus video was **−2.60 percentage
points** on average: the body-enabled learner did not outperform video-only in
this controlled study. Shuffling reduced additive accuracy by 3.65 points on
average, consistent with use of aligned body history without an overall gain.

Three encounters selected before training illustrate the word **blicket**
(bottle) with changing distractors. They span partitions and were not selected
for successful predictions. Actual encoder-center crops are retained locally
at `outputs/runs/cross_situational/report/examples.jpg`.

| Encounter | Partition | Named object held? | Example visible distractors |
|---|---|---|---|
| `s7_0038:2` — “Look, blicket.” | Training | Yes | Apple |
| `s6_0102:48` — “Look, blicket.” | Development | No | Knife |
| `s5_0020:118` — “Look, blicket.” | Test | Yes | Bowl, apple |

For reference, the earlier heavy/light
table reported video 37.88%, video + load 92.42%, and shuffled load 46.21%.
Those were a different, two-choice task with 50% chance; they are a presentation
example, not numerical targets for this experiment. See the
[saved heavy/light record](../archive/language_learning/experiments.md#step-2-heavy-light-capability-test).

Include a few example encounters beside the table. Acquisition curves and the
joint-architecture comparison remain in the main study.

## Running in parallel

After shared implementation checks, train separate models with separate
manifests and ignored run directories. Do not use this study's weights or
measurements to change an already-running main comparison.

Before launch, confirm category/clip feasibility, freeze sampling and training
settings, and set a short runtime cap from a timing check. Runtime is not yet
measured. New rendering and open-ended sweeps are outside this small study.

## Agreed execution plan

Implement in four steps, reporting completion of each step:

1. **Establish and freeze the dataset.** Audit existing renders, cached features
   and all five sensor streams. Check representative actual encoder crops for
   usable ambiguity, changing distractors and nuisance associations. Use the
   existing seed-0 source-clip split (70/15/15), excluding s8. The agreed targets
   are 64 training, 16 development and 16 test encounters per word, with 50%
   held naming in each word/partition and at most four encounters per source
   clip per word across replays. Choose four feasible categories by independent
   clip coverage, with alphabetical tie-breaking; map alphabetical category
   order to dax, blicket, wug and fep. Use the template “Look, dax.” Freeze
   manifests and provenance only after feasibility checks. If these targets
   fail, report the shortfall before changing them.
2. **Make the minimum runner changes.** Reuse the existing models, body encoder,
   language loss and situation evaluator. Add a study-specific preparation and
   reporting entry point and a narrow shared-runner option for its frozen
   protocol. Fit vocabulary and normalization on selected training experience;
   bypass unrelated ordinary-noun picture probes. Keep existing runner defaults
   unchanged outside this study. Use paired seeds 0, 1 and 2, 512-dimensional
   representations, soft pooling at temperature 0.1, AdamW (learning rate
   0.0003, weight decay 0.1), batch size 32, 20 epochs, additive body dropout
   0.5 and no auxiliary losses. Match common initialization, example order
   and update counts.
3. **Freeze evaluation and verify readiness.** Share fixed seed-0 development
   and test panels across all runs, preserving strict foil rules. Select each
   checkpoint on its own development situation accuracy, taking the earliest
   epoch on ties. Freeze test-history shuffle donors within subject, from a
   different source clip; move all channels, masks and temporal boundaries
   together. Validate splitting, normalization, pairing, scoring, gradients,
   masks, shuffling and checkpoint selection. Measure a short training/dev
   timing check, discard its weights, and agree the proposed runtime cap before
   launching the six fresh runs.
4. **Run and report.** Run exactly two conditions by three seeds. Evaluate
   selected checkpoints on the fixed test panel and then evaluate shuffled
   histories for the additive checkpoints. Fill the two-row table with mean
   accuracy and sample standard deviation across seeds, label the 25% chance
   level, and add one supporting shuffled-body sentence and three example
   encounters selected before training. Report independent clips and utterances
   per word and distinguish new clips from unverified new instances. A null
   result still completes the study; it does not authorize another experiment.

New artifacts belong in the worktree's ignored
`outputs/runs/cross_situational/`; retained inputs remain at their existing
locations. Commit source and compact curated records, not checkpoints,
predictions, logs, generated images or reproducible intermediates.

## Step 1: frozen dataset

The preparation and independent artifact checks passed with the agreed
64/16/16 encounters per word and 50% held naming in every partition. The
mapping is bell → dax, bottle → blicket, glass → wug, tube → fep. Turtle was
screened out because the four-per-source-clip cap left only 29 eligible held
training encounters; glass met the original counts. Selection used coverage
and dataset constraints, never learner performance.

| Named category | Training source clips | Development source clips | Test source clips |
|---|---:|---:|---:|
| bell | 39 | 13 | 10 |
| bottle | 39 | 10 | 12 |
| glass | 28 | 7 | 7 |
| tube | 34 | 7 | 10 |

These are unique clips within each word/partition, not additive dataset totals.
There are 256 training targets and 64 targets in each evaluation partition.
Development uses 11 additional candidate-only foil rows (six source clips);
test uses three (one source clip). These are situations offered as candidate
answers, not extra training exposures or scored target queries. Every trial
has four distinct frame situations. The four-per-source-clip/word cap holds
including foil rows. All replays stay in their source partition.

The 64 development and 64 test trials passed the canonical strict foil checks.
The test shuffle is a complete 67-history permutation within subject and across
different source clips. Every subject and room represented among targets is
associated with at least two words; no distractor category is present in every
encounter of a word. These checks exclude deterministic shortcuts, not all
possible statistical correlations.

Visibility screening requires at least 100 full-frame pixels and overlap with
the central four coarse grid cells for both target and a distractor. Sampled
five-frame encoder-crop panels were reviewed for held/nonheld examples of each
word and partition. Some center views are small or occluded; sampled review
does not establish recognizability for every encounter. No new-instance claim
is made. Three bottle/blicket encounters were selected before training for the
eventual example panel: `s7_0038:2`, `s6_0102:48`, `s5_0020:118`.

Frozen artifacts are `manifest.parquet`, `panels.json` and `protocol.json` in
`outputs/runs/cross_situational/`. Their SHA-256 values are respectively:

```text
e25b6bca7f5816adebc873a85e2fa7b770455ec264a714fb8807e533f0a265d2
9c75fc78131a345b61266158d9edd5bcae3b04715571f673f8d6110ae3f24bf2
fee74841d7062c5db512f055053ef2b96a02d182b1214d3254d25f16d0e21bd7
```

Inputs remain in the main checkout's `outputs/renders/clean_856/`, with physical
streams from `physical_by_clip/`. The selected `clean_body/train_cache` and
`step1_matched/train_cache` are identical hardlinked caches with matching DINOv2
metadata. The embedding metadata SHA-256 is
`6cf0a5df2a4b4b63151b353cbdf603df71621f1d05de8fb585853d60577c6c7c`;
the cache index SHA-256 is
`9da1f5246c81a39490eb9cbf464b278cc3c6d92d36a4d91b3ea45e224ef910c2`.
Five focused preparation tests passed. No experiment training had started at
the Step 1 checkpoint.

## Step 2: shared runner and reporting

The study is implemented as a narrow `--cross-situational <protocol.json>`
option on the existing learner `train` command. It uses the same models, loss
and minibatch loop, fits vocabulary/centering/body statistics from selected
training experience, and bypasses unrelated picture probes. Fixed development
panels select checkpoints; test and shuffled-test scoring run only after the
selected state is restored. Metrics bind the exact protocol, manifest and
panels with SHA-256 hashes. `--timing-only` runs one training/development epoch,
discards weights and skips test evaluation.

```bash
# Timing only; repeat with --body real --name additive for the second condition.
PYTHONDONTWRITEBYTECODE=1 .external/cue-env/bin/python -m nursery.learner train \
  --run outputs/runs/cross_situational --condition none --seed 0 \
  --body none --name video_only \
  --cross-situational outputs/runs/cross_situational/protocol.json --timing-only
```

After the runtime cap is agreed, the six fresh runs use the same command
without `--timing-only`, with paired seeds 0, 1 and 2. The additive condition
uses `--body real --name additive`; the protocol supplies the physical stream
path. No timing weights are reused.

The study-specific reporter is available through
`python -m nursery.learner.cross_situational --report-protocol <protocol.json>
--metrics <six metrics.json paths> --output <ignored report directory>`.
It requires all six paired runs and matching input hashes, checks target/trial
counts, and writes the two-row mean/sample-SD table, supporting shuffled score,
per-seed results, exposure counts and the preselected example identifiers.

Validation: all 53 existing learner tests and the three new runner tests passed;
all six preparation/reporting tests passed. The final nine study tests also
passed together. The existing tiny-probe standard-deviation warnings remain.
No source-data training or timing had run at the Step 2 checkpoint.

## Step 3: readiness and timing

The active chat now owns the worktree at
`/Users/rishisim/.codex/worktrees/cross-situational-study/nursery`, on
`research/cross-situational`. The previous worktree and its artifacts remain
preserved. Copied protocol/manifest/panels were byte-verified against the frozen
hashes; the existing environment and retained input paths resolve successfully.

One fresh training/development epoch per condition completed on local MPS:

| Condition | Startup | Training + development epoch | Total timing check |
|---|---:|---:|---:|
| Video only | 1.028 s | 0.521 s | 1.550 s |
| Additive video + body | 1.880 s | 0.499 s | 2.379 s |

Both checks used all 256 training encounters, skipped test/shuffled scoring and
discarded their weights. Only `timing/video_only/timing.json` and
`timing/additive/timing.json` were retained, including the frozen input hashes.
The six-run extrapolation is approximately 70 seconds before final test scoring
and process overhead. The user approved a **five-minute total wall-clock cap**
for the six fresh runs, with interruption and failure reporting rather than
silently extending it. Launch was paused at the user's request and resumed only
after the user explicitly asked to continue. No final study models or held-out
measurements were created before that resume.

## Step 4: completed execution and interpretation

All six fresh 20-epoch runs completed in **39.60 seconds** under the approved
300-second total cap, with zero nonzero exit codes, timeouts or retries. The
source commit used was `06332d05bfcff6a9f2f3f0c6851d7f8fc9283ae0`.
Exact commands, individual durations, return codes and input hashes are in
`outputs/runs/cross_situational/execution.json`; logs, selected checkpoints,
vocabularies and per-trial metrics remain in the corresponding ignored run
directories. The canonical generated summary is in
`outputs/runs/cross_situational/report/results.{json,md}`.

| Seed | Video accuracy | Additive accuracy | Shuffled additive accuracy | Video / additive selected epoch (zero-based) |
|---|---:|---:|---:|---|
| 0 | 81.250% | 76.562% | 75.000% | 16 / 18 |
| 1 | 71.875% | 71.875% | 64.062% | 11 / 13 |
| 2 | 70.312% | 67.188% | 65.625% | 15 / 19 |

Independent result verification confirmed all six frozen input hashes, 256
training pairs, 20 epochs, earliest own-development-max checkpoint selection,
identical fixed test candidates/correct slots, and reproduction of every
reported accuracy from the 64 saved trial flags. Shuffled evaluations retained
the same trials and the frozen complete-history donor permutation. The
reported means and sample standard deviations were independently recomputed.

This completes the requested fixed-data comparison. Both learners acquired
above-chance word–situation associations in the scaffolded task, but adding
the five body streams did not improve held-out accuracy. The result is a design
signal for the larger build: aligned sensory input is used here, yet that use
does not establish a language-learning advantage. It does not measure an
acquisition ceiling, establish new-instance generalization, or settle the
separately planned acquisition-curve comparison. At completion of this original
comparison, no follow-up experiment, architecture change, new rendering or
parameter sweep had been launched.

## Requested joint-interaction comparison

The user subsequently requested testing the new joint design on both this task
and the heavy/light capability task. This explicitly expands the original
two-learner scope. Reuse the committed acquisition-curves model at `08ebb82c`:
`interaction.py` preserves the direct visual/word and body/word scores, then adds
a learned visual/body interaction with zero-initialized output. Its prerequisite
`score_encoded` methods and tests are imported from the same commit. The old
128-dimensional `--arch joint` pilot is not this design; use `--arch interaction`.
Do not modify the other chat's active worktree or its models.

Run fresh **video, additive and interaction** models for each of seeds 0, 1 and
2 on CPU, with matched common initialization, data order, body dropout and
training opportunity. Fresh CPU references avoid confounding a CPU joint run
with the original MPS controls; the acquisition chat is using MPS independently.
Keep the original results and artifacts intact and report the new matched table
separately. There is no model-selection sweep or acquisition curve here.

- **Cross-situational:** reuse the exact frozen manifest, word mapping, panels
  and donor permutation above; retain all five body streams, 20 epochs, batch
  32, 50% body dropout, training-only centering/normalization and the same
  development checkpoint rule. Write new results to ignored
  `outputs/runs/cross_situational_joint/`, referring to the original protocol
  rather than resampling. Evaluate the full interaction score during both
  development and test; the previous hand-written additive evaluator alone
  would omit the interaction. Report aligned and shuffled results for both
  body-enabled conditions. Mask repeated padding only in interaction attention,
  identically in training and evaluation; preserve the inherited direct visual
  pathway's padding convention across all three models.
- **Heavy/light:** restore the one canonical runner to
  `nursery/learner/heavy_light.py` and reuse the retained original manifest,
  normalization and ten fixed donor permutations at the main checkout's
  `outputs/legacy/runs/heavy_light/`. Preserve the 88/18/22 category-disjoint
  split, **load-only** input, zero visual center, 100 epochs, batch 16, 50% body
  dropout, first-best-development checkpoint and two-word test with 50% chance.
  Write fresh results to ignored `outputs/runs/heavy_light_joint/`. Adding the
  other four streams would change the original diagnostic and is outside this
  comparison.

Before launch, validate zero-initialization equivalence, interaction gradients,
masking, full-score train/evaluation agreement, fixed input hashes and reporter
coverage of all nine runs. Check short CPU training/development timing runs,
including the joint architecture for both tasks, without retaining weights or
consulting test outcomes. Retain
the existing bounded limits: five minutes total for cross-situational training
and thirty minutes total for heavy/light training, with no automatic extensions
or retries. Interpret this as an exploratory architecture comparison on already
consulted test panels, not a fresh confirmatory test or evidence of a ceiling.

### Joint comparison: cross-situational results

All nine fresh CPU runs completed at source commit
`753a526f1781b524a0f83b3f2d1db304071689c1` in **57.48 seconds** under the
300-second cap, with no failed runs or retries. Three preliminary one-epoch
training/development timing checks discarded their weights and did not score
test. The original frozen manifest, panels and protocol were reused unchanged.
Exact commands, timing, hashes and exit codes are retained in
`outputs/runs/cross_situational_joint/execution.json`; the canonical generated
table is `outputs/runs/cross_situational_joint/report/results.md`.

| Condition | Test accuracy, mean ± sample SD | Shuffled body accuracy, mean ± sample SD |
|---|---:|---:|
| Video only | 75.00% ± 6.81 pp | — |
| Additive video + body | 71.88% ± 4.13 pp | 68.23% ± 2.39 pp |
| Joint interaction video + body | 73.44% ± 4.13 pp | 69.27% ± 3.93 pp |

| Seed | Video | Additive | Interaction | Selected epochs V/A/I, zero-based |
|---|---:|---:|---:|---|
| 0 | 82.8125% | 75.0000% | 78.1250% | 16 / 11 / 13 |
| 1 | 71.8750% | 73.4375% | 71.8750% | 5 / 14 / 19 |
| 2 | 70.3125% | 67.1875% | 70.3125% | 17 / 19 / 18 |

Independent audit reproduced all saved trial accuracies, frozen input hashes,
candidate arrays/correct slots, CPU settings, 256 training pairs, 20 epochs and
earliest own-development-max checkpoint selection. These controls are fresh CPU
models, so the earlier MPS table should not be substituted into this comparison.

The joint model is **1.56 pp above additive but 1.56 pp below video**. Its body
shuffle drop is 4.17 pp, consistent with use of aligned sensory histories, but
this does not establish an advantage over the video learner.

Saved same-checkpoint diagnostics sharpen the interpretation. Across the joint
models, picture-only scoring is 72.40%, direct picture-plus-body scoring is
74.48%, and full interaction scoring is 73.44%. The interaction term fixes one
and hurts three correct/incorrect decisions across the three seeds (the same
64 situations repeated, not 192 independent observations). Compared with the
same checkpoint's picture-only scoring, full scoring fixes 12 and hurts 10.
The small advantage over the independently trained additive control therefore
cannot be attributed to a beneficial interaction term at inference: learned
direct pathways and selected checkpoints also differ. This fixed-weight score
ablation does not isolate the interaction's effect during training.

### Joint comparison: heavy/light results

The nine fresh CPU models completed at the same source commit in **64.82
seconds** wall time (62.59 seconds inside the training/evaluation runner), under
the 1,800-second cap. The manifest and normalization table are byte-identical
to the retained original study, including 88/18/22 train/dev/test clips and
the same ten within-test-category donor permutations. Every model trained
for 100 epochs; checkpoint selection remained the first maximum development
accuracy. The prepared protocol SHA256 is
`2dc5201cc4e6bc46a7cdd48d2dd1d1ef26f535fad2cb7798f03b8d7a13739d9b`.

| Condition | Test accuracy, mean ± sample SD | Shuffled load accuracy, mean ± sample SD |
|---|---:|---:|
| Video only | 34.85% ± 9.46 pp | — |
| Additive video + load | 90.91% ± 4.55 pp | 46.06% ± 2.78 pp |
| Joint interaction video + load | 90.91% ± 4.55 pp | 46.06% ± 2.78 pp |

| Seed | Video | Additive | Interaction | Selected epochs V/A/I, zero-based |
|---|---:|---:|---:|---|
| 0 | 27.2727% | 90.9091% | 90.9091% | 33 / 1 / 1 |
| 1 | 31.8182% | 95.4545% | 95.4545% | 1 / 3 / 3 |
| 2 | 45.4545% | 86.3636% | 86.3636% | 47 / 1 / 1 |

Both body conditions pass all original capability gates: +56.06 pp over video,
positive gain in every seed, a paired source-clip 95% interval of
[+34.85, +74.24] pp, and a 44.85 pp shuffled-load drop. This interval is
conditional on the three fitted seeds and only two held-out categories.
Joint and additive make identical test decisions at every seed, although
their scores differ and the selected joint interaction weights are nonzero.
Thus the interaction branch is learned, but adds no measured accuracy benefit
over additive on this diagnostic. The body models select very early epochs
(zero-based 1/3/1) under the frozen development rule; training ran the full
100 epochs and no later test checkpoint was substituted.

Run artifacts, predictions, histories and the generated comparison are under
`outputs/runs/heavy_light_joint/`. Exact launch records are separately under
`outputs/runs/heavy_light_joint_execution/`. An initial supervisor setup placed
logs inside the prepared directory, so the overwrite guard rejected launch
before any model started. Those failure records were preserved outside the
prepared root, the three frozen files and their hashes were rechecked, and the
corrected launch trained each model once. No started model was retried. The
preliminary joint timing check discarded weights and did not score test.

Independent review reproduced every test credit, all ten shuffle draws, the
paired bootstrap and all five gates. It verified all 859 source-file hashes,
ten implementation hashes, the frozen tables, identical same-seed common
initialization/minibatch order, and every development-selected checkpoint.
Implementation validation passed 61 shared learner tests before the attention
mask refinement, then eight affected tests afterward, plus seven reporter tests
and 20 heavy/light tests. Generated artifacts remain ignored; only the curated
protocol/results record is committed.

Across both tasks, the new design preserves the clear benefit of aligned load
for explicitly supervised synthetic weight words, but does not yet improve
cross-situational noun learning over video. This is consistent with a remaining
signal-to-referent alignment problem; these runs do not identify that as the
sole cause. The earlier noun-task audit found body histories often describing
an object other than the named nonheld referent. Adding interaction capacity
does not itself repair that training evidence. The next design decision should
consider that mismatch alongside the separate acquisition-curve results, rather
than treating this small architecture comparison as a ceiling measurement.
These tests reuse already-consulted panels and are exploratory. The heavy/light
signal remains analytic synthetic load whose sampled mass does not affect RGB
or motion, with explicit aligned labels and only 22 test clips. No additional
training, data generation, parameter sweep or acquisition-curve run was launched
as part of this follow-up.
