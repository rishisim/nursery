# Nursery research archive

This is the entry point for completed or inactive research. Start active work
from [PLAN.md](../PLAN.md). An archived proposal, command or “next step” is not
an instruction to resume it.

Read [Past experiments](language_learning/past_experiments.md) for the
architecture, training, results and limits of prior language work. The
[detailed experiment record](language_learning/experiments.md) remains the
canonical historical protocol; the [dated literature/audit review](language_learning/body_input_review.md)
is historical evidence. Active work now centers on direct V versus V+B
acquisition curves; no archived conditional follow-up is queued.

Initially archived on 26 October 2026 from checkpoint `4a3dd16d`; Step 3 result
4.16 and the explanatory guide were added on 27 October 2026. Documentation and the two small cue-result
JSON files were reorganized; source code, configurations, datasets, checkpoints
and run directories were preserved. Git retains the exact implementations and
original locations.

## What we tried and what remains useful

| Work | What happened | Where to find it |
|---|---|---|
| AEA real-data grounding and early weak-alignment work | Adult video/transcript/IMU infrastructure worked, but the exact-window formulation did not establish transferable language learning. The subsequent research direction explicitly retained weak alignment and independent generalization tests. | [Design intent](language_learning/design_intent.md); [AEA implementation](https://github.com/rishisim/nursery/commit/e6acf51c); [early project history](NURSERY_REPORT.md#project-timeline). |
| Synthetic-video qualification and generator comparisons | The four-family comparison was a qualitative screen, not a language-learning experiment. The later engineering-health seal was explicitly not confirmatory scientific readiness. Real-data learnability failures gated off synthetic comparisons in the recorded pilots. | [Qualitative comparison](https://github.com/rishisim/nursery/blob/cd5d0301/results/synthetic_video_model_bakeoff.json); [Phase 4 record](https://github.com/rishisim/nursery/blob/9be20c4e/results/synthetic_video_phase4.json). These retired records remain in Git; no second runnable pipeline was restored. |
| Procedural embodied simulation, MIMo/MolmoSpaces and Unity/MuJoCo gates | Engineering, geometry, appearance and contact gates produced useful successes and failures. They did not establish a language-acquisition advantage. | [Phase reports](legacy_research/docs/embodied_simulation/phase_reports.md), [recovery status](legacy_research/docs/embodied_simulation/recovery_status.md), and the existing `legacy_research/` source/config/test archive. |
| EmbodiedGen → HOIDiNi → InterMimic | Kinematic mug lifting and format conversion worked; floor validity and physical hand contact failed. The modular architecture proposal was a proposal, not a completed solution. | [Commands, runs and measurements](motion_pipeline.md); [September report](NURSERY_REPORT.md). |
| Contact-cue weighting: 90-run grid, then 100-run confirmation | The grid was inconclusive. Confirmation beat shuffled/shifted cues but did not establish an improvement over the no-cue learner. This tested weighting examples, not rich bodily inputs. | [Plain-language record and exact aggregates](language_learning/cue_experiments.md). |
| Hand-location gates, then actual body-input learners | Steps 4.1–4.14 found no confirmed noun gain, including the corrected matched comparison. Step 4.15 showed a scaffolded heavy/light capability; Step 4.16 sensory-target pretraining did not pass its picture-only weak-noun transfer gate. The broader acquisition question remains open. | [Past experiments](language_learning/past_experiments.md), [complete protocol/result record](language_learning/experiments.md), [step 4.5 signal audit](language_learning/body_signal_audit.md), [historical reassessment](language_learning/body_input_review.md). |

The September report and older phase documents contain historical schedules,
recommendations and environment details. They are preserved evidence, not the
current architecture decision or proof those environments still run today.

## Run and evidence locations

Paths below are repository-relative unless explicitly remote. They are pointers
to existing evidence, not new copies or a claim of fresh experiment validation.

- `outputs/legacy/cue_experiment/`: full early cue-study report and artifacts.
  The small curated aggregates are versioned [here](language_learning/results/).
- `outputs/legacy/step2/`, `outputs/legacy/step3/`: older speech and gate work.
- `outputs/legacy/`: historical artifacts, including superseded patched renders
  and completed valid experiments. Archiving does not imply invalidation;
  never pool invalidated or patched artifacts with the current clean render.
- `outputs/renders/clean_856/`, `outputs/runs/clean_body/`: retained saved data,
  speech, embeddings, and completed learner comparisons. Kept in place because
  run metadata and analysis refer to these paths.
- `outputs/runs/step1_matched/`: corrected matched speech/exposure, 4.14 models
  and audit records; also reused as inputs by 4.16.
- `outputs/legacy/runs/heavy_light/`: completed load-only property-word
  diagnostic, six models and independent verification records.
- `outputs/legacy/reviews/research/body_debate/`, `outputs/legacy/reviews/episodes/`: October 26
  read-only research/audit evidence, not new training results.
- `outputs/legacy/runs/sensory_pretraining/`: ignored artifacts for completed Step 3 / 4.16;
  nine verified MPS runs and preparation, comparison, and independent-audit
  records. The complete protocol and results are in
  [the 4.16 archive entry](language_learning/experiments.md#step-3-transfer-focused-sensory-pretraining).
- `/work/dal503972/nursery/outputs/` on Juno: earlier motion/controller artifacts;
  exact subpaths and job evidence are in the motion record. Availability was
  not rechecked during archiving.
- `.external/`: upstream repositories/environments. `runs/`, `outputs/` and
  other generated roots follow `.gitignore`; models, media and datasets are not
  added to this documentation archive.

The source cleanup on 27 October 2026 preserved artifact paths, separated
the current video-only and additive models, and moved inactive code out of
`nursery/`. The simulator now uses the common
output guard directly, so it no longer depends on the archived motion package.

## Local artifact relocation

The subsequent output cleanup on 27 October 2026 moved the following retired
artifacts within the ignored `outputs/` root. All 185 relocated files were
verified with SHA-256 before and after moving (175,831,415 bytes). File contents,
including frozen manifests and historical path strings, were not rewritten.
Use this map when following an older record; archived scripts with hard-coded
old paths may require explicit path updates before reuse. No compatibility
symlinks or new experiments were created. These local moves are not uploaded
with the documentation commit.

All locations in this table are relative to `outputs/`:

| Previous location | Current location |
|---|---|
| `activity_pipeline/` | `legacy/motion/activity_pipeline/` |
| `hoidini_intermimic/` | `legacy/motion/hoidini_intermimic/` |
| `hoidini_official_example/` | `legacy/motion/hoidini_official_example/` |
| `hoidini_scene/` | `legacy/motion/hoidini_scene/` |
| `archive_review/` | `legacy/reviews/chat_history/` |
| `audits/` | `legacy/reviews/audits/` |
| `episode_review/` | `legacy/reviews/episodes/` |
| `handoff/` | `legacy/reviews/handoffs/` |
| `literature/` | `legacy/reviews/literature/` |
| `research/` | `legacy/reviews/research/` |
| `step1_verification/` | `legacy/reviews/connection_and_scoring/` |
| `runs/step2_heavy_light/` | `legacy/runs/heavy_light/` |
| `runs/step3_pretrain/` | `legacy/runs/sensory_pretraining/` |
| `runs/step2_heavy_light.log` | `legacy/runs/heavy_light/training.log` |
| `runs/step3_pretrain.prepare.log` | `legacy/runs/sensory_pretraining/preparation.log` |
| `runs/step3_pretrain.train.log` | `legacy/runs/sensory_pretraining/training.log` |

Reusable inputs and mixed folders stay in place: `assets/`, `embodiedgen_scene/`,
`renders/clean_856/`, `speech_data/`, `runs/clean_body/`, `runs/clean_video_only/`,
`runs/clean_perfect_naming/` and `runs/step1_matched/`. Exports and earlier
`legacy/` contents are retained. The active source/configuration and these
mixed run folders had no references to the retired locations in the inspected
code, documentation and JSON/YAML metadata.

## Archived source

| Location | Contents and status |
|---|---|
| [code/learner/heavy_light.py](code/learner/heavy_light.py) | Completed property-word diagnostic snapshot. |
| [code/learner/sensory_pretraining.py](code/learner/sensory_pretraining.py) | Completed sensory-pretraining/transfer snapshot. |
| [code/learner/tests/](code/learner/tests/) | Tests accompanying those historical runners. |
| [code/learner/joint_pilot.py](code/learner/joint_pilot.py) | Old width-128 Transformer. Still imported explicitly by `--arch joint` and covered by current compatibility tests; not the proposed joint design. |
| [code/motion/](code/motion/) | Inactive motion adapters and their tests. |

Except for the explicitly supported joint pilot, these are historical reading
copies, excluded from current test discovery. They retain their original import
and path assumptions. For reproduction, use the complete package layout at
[pre-cleanup commit 20f1cc03](https://github.com/rishisim/nursery/tree/20f1cc03),
rather than running isolated relocated files. The original monolithic learner
and longer implementation reference also remain at that commit. Current source
has one visual baseline and an additive subclass; no second active copy exists.

## How to use the archive

Recover an old implementation from its Git commit if needed. Keep one current
implementation in `nursery/`; do not create competing copies from these notes.
Revisit a historical experiment only when it answers a specific unresolved
question in the current plan. Preserve the difference between an unsuccessful
implementation, an inconclusive diagnostic, and a rejected scientific claim.
