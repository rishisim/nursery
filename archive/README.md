# Nursery research archive

This is the entry point for completed or inactive research. Start active work
from [PLAN.md](../PLAN.md). An archived proposal, command or “next step” is not
an instruction to resume it.

Initially archived on 26 October 2026 from checkpoint `4a3dd16d`; Step 3 result
4.16 was added on 27 October 2026. Documentation and the two small cue-result
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
| Hand-location gates, then actual body-input learners | Steps 4.1–4.13 found no confirmed gain. Step 4.15 showed a scaffolded heavy/light capability; Step 4.16 sensory pretraining learned aligned sensory prediction but did not pass the weak-noun transfer gate. The broader representation hypothesis remains open. | [Complete protocol/result record](language_learning/experiments.md#step-3-transfer-focused-sensory-pretraining), [step 4.5 signal audit](language_learning/body_signal_audit.md), [current reassessment](../docs/literature/body_input_review.md). |

The September report and older phase documents contain historical schedules,
recommendations and environment details. They are preserved evidence, not the
current architecture decision or proof those environments still run today.

## Run and evidence locations

Paths below are repository-relative unless explicitly remote. They are pointers
to existing evidence, not new copies or a claim of fresh experiment validation.

- `outputs/legacy/cue_experiment/`: full early cue-study report and artifacts.
  The small curated aggregates are versioned [here](language_learning/results/).
- `outputs/legacy/step2/`, `outputs/legacy/step3/`: older speech and gate work.
- `outputs/legacy/`: superseded patched renders/runs from the earlier cleanup;
  never pool invalidated or patched artifacts with the current clean render.
- `outputs/renders/clean_856/`, `outputs/runs/clean_body/`: current saved data,
  speech, embeddings, and completed learner comparisons. Kept in place because
  run metadata and analysis refer to these paths.
- `outputs/research/body_debate/`, `outputs/episode_review/`: October 26
  read-only research/audit evidence, not new training results.
- `outputs/runs/step3_pretrain/`: ignored artifacts for completed Step 3 / 4.16;
  nine verified MPS runs and preparation, comparison, and independent-audit
  records. The complete protocol and results are in
  [the 4.16 archive entry](language_learning/experiments.md#step-3-transfer-focused-sensory-pretraining).
- `/work/dal503972/nursery/outputs/` on Juno: earlier motion/controller artifacts;
  exact subpaths and job evidence are in the motion record. Availability was
  not rechecked during archiving.
- `.external/`: upstream repositories/environments. `runs/`, `outputs/` and
  other generated roots follow `.gitignore`; models, media and datasets are not
  added to this documentation archive.

## How to use the archive

Recover an old implementation from its Git commit if needed. Keep one current
implementation in `nursery/`; do not create competing copies from these notes.
Revisit a historical experiment only when it answers a specific unresolved
question in the current plan. Preserve the difference between an unsuccessful
implementation, an inconclusive diagnostic, and a rejected scientific claim.
