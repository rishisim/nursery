# Juno two-video CLIP+ engineering pilot

## Purpose and non-claim

This pilot rehearses the complete BabyView CLIP+ path using only the two
authorized videos already stored on governed Juno. Its question is whether the
pipeline is correct, resumable, privacy-safe, and operationally ready to scale.
It does not ask whether a two-video model reproduces the published result.

Every pilot result must carry all three labels:

- `engineering_only`
- `non_comparable`
- `not_a_reproduction_result`

The unresolved difference among the published full-corpus duration totals is
out of scope for this pilot. It remains unresolved in the scientific spec and
must not be silently treated as solved.

## Fixed decisions

- Run end to end: audit, preprocessing, evaluator calibration, DINO, BERT,
  CLIP+, one pilot evaluation, and a promotion-readiness report.
- Use exactly the two governed videos already present. Do not start another
  video download as part of this pilot.
- Train a pilot-only 2,048-token WordPiece vocabulary.
- Use different recordings for training and validation when both yield usable
  examples. If only one yields usable examples, use a clearly marked leaky
  `engineering_only` split solely to execute validation code.
- Use real DINOv2 ViT-B/14 and BERT-base shapes, with random initialization and
  very short runs. Do not substitute smaller architectures.
- Calibrate Machine-DevBench once with off-the-shelf CLIP-L in an isolated
  calibration namespace, then evaluate one frozen pilot checkpoint once.
- If an implementation-valid minimum microbatch cannot fit GPU memory, stop
  after collecting the failure evidence and decide separately what to change.

## Storage and implementation boundaries

There is one canonical implementation for pilot and full execution. Scale,
input selection, and short-run budgets belong in a pilot configuration overlay;
scripts must not be copied into this directory.

All paths must be obtained through `storage.py`. Restricted inputs, transcripts,
frames, detailed manifests, model weights, predictions, and logs remain in the
owner-only Juno roots. Git may retain only this plan, frozen non-sensitive
configuration, checksums, aggregate measurements, and a concise decision report.
If Juno or its ownership/mode check fails, stop without a laptop fallback.

The full scientific run may reuse shared code and preprocessing outputs only
when source, configuration, tool, and model-revision hashes match. It must not
reuse the pilot tokenizer, DINO weights, BERT weights, CLIP+ weights, split, or
benchmark result.

## Pilot P0 — Freeze the pilot contract

**Question:** Are the goal, boundaries, inputs, and stop conditions unambiguous?

Work:

1. Verify the canonical scientific spec and Phase 1 aggregate record.
2. Resolve a pilot configuration from the pinned upstream code revision.
3. Record one opaque pilot run ID and one fixed engineering seed.
4. Freeze the two-input rule, 2,048-token vocabulary, architecture choices,
   step budgets, split fallback, and output locations.
5. Mark all pilot outputs as engineering-only at creation time.

Gate:

- `../../tests/verify_phase0.py` passes.
- `decision_record.json` is valid and agrees with the resolved pilot config.
- No scientific target or unresolved variable is changed.

Retain in Git: frozen pilot config and decision record only.

P0 uses `../../configs/pilot.json` as the single canonical machine-readable
configuration. Its two input slots are resolved at execution only through
opaque keys in the governed ledger; no record key, filename, or governed path
is frozen in Git. The fixed seed is a deterministic engineering seed for this
single rehearsal, not one of the scientific reproduction seeds. P0 completion
does not start P1, access either input, or change the blocked scientific status.

## Pilot P1 — Verify Juno and audit the two inputs

**Question:** Can both governed videos safely support the intended data paths?

Work:

1. Run `storage.py check-juno` before accessing restricted material.
2. Locate the two inputs through governed ledger records, never hard-coded
   filenames or worktree paths.
3. Recheck byte counts and SHA-256 values against the transfer records.
4. Run FFprobe and record privacy-safe aggregates for duration, codec,
   decodability, frame rate, dimensions, and audio-stream availability.
5. Assign stable opaque pilot record keys.
6. Mark the inventory `incomplete_inventory`; it is not the full release ledger.

Gate:

- Both files are checksum-consistent and video-decodable.
- At least one file has usable audio. If neither does, stop: the fixed two-video
  pilot cannot exercise BERT or CLIP+ and no additional download is authorized.
- Detailed ledger rows exist only in governed durable storage.

Retain in Git: aggregate counts and an opaque governed-record checksum only.

P1 completed on 2026-08-11 for the fixed two-video engineering sample. Both
governed transfers remained byte- and checksum-consistent, both videos were
decodable, and the sample contains usable audio. The inventory remains
`incomplete_inventory`; this result does not represent the full release ledger
or change the blocked scientific reproduction status. Detailed rows remain in
owner-only durable Juno storage. P2 subsequently verified the environment and
learned-weight boundary without accessing either input.

## Pilot P2 — Pin the executable environment and weight boundary

**Question:** Can the real model shapes be constructed without contaminating
pilot training with public pretrained weights?

Work:

1. Recreate the pinned Linux/Python/CUDA/PyTorch environment on Juno.
2. Pin every operational preprocessing revision selected for the pilot,
   including FFmpeg, VTC, WhisperX, and their permitted preprocessing weights.
3. Instantiate DINOv2 ViT-B/14 and BERT-base from configuration and random
   weights only.
4. Add a launch-time assertion that rejects Hugging Face, torch.hub, or other
   external initialization paths for learned pilot components.
5. Record GPU type, count, software versions, and the resolved configuration.

Gate:

- Both real architectures construct successfully from random initialization.
- No training component reads an external pretrained checkpoint.
- The permitted preprocessing-model boundary is explicit and checksum-recorded.

Retain in Git: non-sensitive resolved configuration and dependency digests.

P2 completed on 2026-08-11 in a one-GPU Juno allocation. The immutable upstream
Pixi lock resolved without modification, and the real DINOv2 ViT-B/14 and
BERT-base 12x768x12 shapes constructed from explicit configuration and random
initialization. Runtime guards rejected Hugging Face model-name/from-pretrained,
torch.hub, checkpoint-path, and network fallback for learned initialization.
No checkpoint or training step was created. FFmpeg, WhisperX large-v2, its VAD
and English alignment artifacts, and VTC 2.2 were revision/checksum-pinned in a
separate preprocessing-only namespace and were not executed.

The upstream lock has a reproducible internal CUDA detail worth preserving: its
system contract says CUDA 12.6 and torchvision resolves to cu126, while its
locked torch wheel resolves CUDA 12.8 runtime packages. P2 records that exact
upstream result and does not create a replacement environment protocol.
`../../configs/pilot_p2.json` is the canonical safe environment/boundary config,
`../../scripts/pilot_p2_preflight.py` is the construction preflight, and
`../../scripts/pilot_p2_juno_job.sh` is the canonical GPU job entry point. The tracked
aggregate is `p2_aggregate.json`; the detailed record and log remain owner-only
in durable governed storage. P3 or later has not started.

## Pilot P3 — Preprocess one video, then both

**Question:** Is preprocessing deterministic, resumable, and capable of
producing visual, text, and paired examples from the fixed inputs?

Work:

1. On the first input, extract mono 16 kHz audio and validate it.
2. Run the pinned VTC and WhisperX path, remove KCHI-overlapping speech, apply
   WhisperX `large-v2`, keep words at confidence `>= 0.50` for this pilot, and
   preserve raw versus normalized text stages. The threshold boundary remains a
   full-reproduction assumption rather than an author-confirmed fact.
3. Extract 1 Hz frames, retain timestamps and source hashes, and keep at most 32
   aligned frame candidates per utterance.
4. Build visual-only, text-only, and frame–utterance paired manifests using
   opaque record keys.
5. Spot-check timestamp alignment and KCHI filtering inside governed storage.
6. Force one controlled interruption and confirm completed work is not repeated.
7. After the first input passes, process the second with the same command.
8. Produce aggregate throughput, retained-utterance, pair-count, failure, RAM,
   GPU-memory, and storage measurements.

Gate:

- At least one input produces retained adult utterances and aligned pairs.
- Rerunning is idempotent and a partial run resumes correctly.
- Counts reconcile across inputs, frames, utterances, and pairs.
- No sensitive filename, transcript, frame, or identifier enters Git or public
  logs.

If both inputs decode but neither produces retained paired examples, stop. The
pilot cannot honestly proceed to paired CLIP+ training with the fixed inputs.

Retain in Git: privacy-safe aggregate preprocessing report and config hashes.

P3 completed on 2026-08-12 for exactly the two governed inputs. The first
input was intentionally stopped after its atomically completed audio stage;
resume validated that completion and skipped the stage before executing the
remaining path. Both inputs then completed the identical pinned VTC,
WhisperX, one-Hz frame, normalization, pairing, and QA contract. A final
all-two rerun produced twelve valid stage cache hits and no repeated stage
execution. The tracked `p3_aggregate.json` contains privacy-safe totals only;
raw and normalized text, frames, detailed manifests, checksum inventory, and
important logs remain owner-only in governed storage. This engineering result
does not resolve the full-corpus inventory discrepancy.

## Pilot P4 — Calibrate Machine-DevBench in isolation

**Question:** Is the evaluator correct before it sees any pilot checkpoint?

Work:

1. Download the pinned official evaluation archive directly to governed Juno
   scratch and verify its recorded size and SHA-256.
2. Run the released evaluator with the pinned runnable off-the-shelf CLIP-B model and
   compare with the approximate 78.8 sanity result.
3. Record per-task and aggregate scoring behavior, tie handling, failure
   handling, and lexical/grammatical aggregation.
4. Store CLIP-B weights and cache paths under an evaluator-calibration-only
   namespace that no training configuration can reference.
5. Freeze the evaluator after calibration. Do not tune the pilot using benchmark
   feedback.

Gate:

- Archive checksum and task inventory pass.
- The calibration result is acceptably consistent with the released sanity
  result, or the discrepancy is resolved before training continues.
- An automated configuration check proves that no calibration model path is referenced by
  DINO, BERT, or CLIP+ training.

Retain in Git: calibration provenance, aggregate result, and the explicit label
`external_CLIP-B_evaluator_calibration_only`.

Status: **incomplete after the single frozen calibration execution**. The
predeclared acceptance window was 78.3–79.3 around the released 78.8 result.
The observed pooled result was lexical 89.658468, grammatical 70.504965, and
overall 80.081716. Deterministic recomputation from the identical governed
predictions without the current low-frequency-bin merge yielded 79.890328,
which still misses the frozen window; the discrepancy is therefore not a
reporting-only correction. No second inference run or benchmark-driven tuning
is authorized, and P5 has not started. Full predictions, detailed results,
checksum inventory, diagnosis, and important logs remain owner-only on Juno.
The calibration weights are not the BabyView/pilot model, cannot initialize
training, and full-reproduction evaluator recalibration remains required.

### P4 CLIP-L provenance diagnostic

After preserving the failed frozen CLIP-B calibration, one separately frozen
offline diagnostic tested the model identity repeatedly named in paper Tables 3
and 5: OpenAI `ViT-L-14` with OpenAI weights. It scored lexical 87.196202,
grammatical 70.749617, and overall 78.972910, compared with the published 87.3,
70.4, and 78.8. Within lexical, nouns were 94.793720 versus 94.6 and adjectives
were 79.598683 versus 80.0. The ten-task mean absolute error improved from
3.306422 for CLIP-B to 2.847925 for CLIP-L. This strongly supports CLIP-L as
the published reference identity and falls inside the original overall window,
but individual grammatical discrepancies remain as large as 6.227273 points.
The diagnostic does not erase the original P4 record or retroactively change
its predeclared execution; full predictions remain governed and P5 has not
started.

## Pilot P5 — Short real-shape DINO rehearsal

**Question:** Can the real visual tower train, checkpoint, and resume on Juno?

Work:

1. Use DINOv2 ViT-B/14 with DINO, iBOT, KoLeo, and teacher/student behavior,
   initialized from random weights.
2. Use the smallest implementation-valid microbatch and mixed precision without
   changing architecture or objective.
3. Run a short health segment, save a checkpoint, terminate cleanly, resume, and
   complete a bounded target of 100 optimizer steps.
4. Confirm finite losses, teacher updates, checkpoint integrity, and deterministic
   step accounting.
5. Measure peak GPU memory, throughput, checkpoint size, and resume time.

Gate:

- The full ViT-B/14 shape completes 100 steps with a verified resume.
- If the minimum valid configuration still runs out of memory, stop after one
  reproducible failure record. Do not substitute a smaller vision model or
  continue into BERT/CLIP+ while the architecture decision is unresolved.

Retain outside Git: pilot checkpoint and full logs. Retain in Git: aggregate
health and resource measurements only.

## Pilot P6 — Pilot tokenizer and short real-shape BERT rehearsal

**Question:** Can the text path train and produce a loadable CLIP+ text tower?

Work:

1. Train a 2,048-token WordPiece tokenizer only on retained pilot BabyView text.
2. Train BERT-base (12 layers, hidden size 768, 12 heads) from random weights
   using masked-language modeling.
3. Run 50 steps, checkpoint, resume, and complete 100 optimizer steps.
4. Execute validation using the recording-level split when possible; otherwise
   use the explicitly leaky engineering split.
5. Verify finite loss, tokenizer/model compatibility, checkpoint reload, and
   peak memory.

Gate:

- The tokenizer, model, MLM head, optimizer, and scheduler reload together.
- Validation executes, but its value is labeled non-generalizing.
- The tokenizer and checkpoint are marked disposable and ineligible for the
  full reproduction.

Retain in Git: vocabulary size, corpus aggregates, config hashes, and resource
measurements—not tokenizer files or model weights.

## Pilot P7 — One complete CLIP+ cycle

**Question:** Does the full multimodal control flow work with the pilot towers?

Work:

1. Load the pilot DINO teacher backbone and pilot BERT checkpoint.
2. Construct the real 768-to-512 projections, normalized embeddings, learnable
   temperature, symmetric contrastive loss, MLM head, and DINO teacher/student.
3. Select the smallest batch that still provides valid contrastive negatives.
4. Run exactly one complete cycle: 100 contrastive steps, 20 MLM steps, and 10
   DINO/iBOT steps.
5. Assert the `100:20:10` counts and ensure auxiliary steps do not replace
   contrastive examples.
6. Checkpoint within the cycle, restart, and verify that mode position,
   optimizers, schedulers, RNG state, and DINO teacher state resume correctly.
7. Verify that the DINO teacher backbone is copied into the CLIP vision tower at
   the intended boundary.

Gate:

- All 130 steps complete with exact automated mode counts.
- Losses are finite, gradients reach the intended components, and frozen versus
  trainable parameters match the resolved config.
- Checkpoint/resume does not repeat or skip a mode transition.

Retain in Git: aggregate loss/step checks, configuration hash, memory, throughput,
checkpoint size, and resume evidence only.

## Pilot P8 — Evaluate one frozen pilot checkpoint once

**Question:** Can a pilot checkpoint pass through the frozen evaluator without
using the benchmark for development?

Work:

1. Select the pilot checkpoint using only engineering validation loss and the
   predeclared rule.
2. Run the frozen Machine-DevBench evaluator exactly once on that checkpoint.
3. Produce lexical, grammatical, overall, and per-task aggregates.
4. Attach the engineering-only/non-comparable labels to every output.
5. Do not change preprocessing, architecture, schedule, checkpoint choice, or
   seed in response to the benchmark score.

Gate:

- The adapter loads the checkpoint without manual conversion.
- Aggregation is identical to the calibrated evaluator.
- The score is never presented as evidence for or against reproduction of 53.6.

Retain in Git: one compact aggregate pilot result with its non-claim labels.
Full predictions remain governed and untracked.

## Pilot P9 — Promotion-readiness decision

**Question:** Is the canonical pipeline ready to scale when full-data access is
available?

Work:

1. Reconcile every pilot gate and unresolved failure.
2. Summarize per-video-hour preprocessing cost, training throughput, peak RAM
   and GPU memory, storage growth, checkpoint sizes, and retry behavior.
3. Classify each artifact as reusable code, conditionally reusable preprocessing,
   or disposable pilot training output.
4. Verify important governed records and checkpoints were checksum-promoted to
   durable storage before any scratch cleanup.
5. Produce a concise go/no-go decision and a full-scale compute/storage estimate.

Gate for `engineering_status = complete`:

- P0 through P8 passed, or every stopped gate has an explicit user-approved
  resolution and was rerun successfully.
- No sensitive or generated artifact is present in Git or a disposable worktree.
- Every task-created output path is ignored and every retained governed artifact
  has a checksum record.

The scientific reproduction status remains unchanged regardless of pilot
success.

## Mandatory reset before the full reproduction

Before full preprocessing or training:

1. Reverify the complete governed input manifest and resolve the scientific
   Phase 1 blockers separately.
2. Rerun CLIP-L evaluator calibration from the pinned archive in its isolated
   namespace and recheck that training configs cannot reference its weights.
3. Build new full-corpus splits, statistics, and a 30,522-token tokenizer.
4. Initialize full DINO, BERT, and CLIP+ training from scratch.
5. Never continue from pilot weights or use the pilot benchmark score for model
   selection, hyperparameter choice, seed choice, or protocol revision.
6. Reuse a pilot preprocessing output only when its source, config, tool, model,
   and completeness hashes all match the frozen full-run contract.

Passing this reset is required even if the pilot completed without errors.
