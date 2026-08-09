# ChildLens v1.3 local-inference firewall

Status: implemented and synthetic-tested. This document defines generic
infrastructure only; no restricted ChildLens content or model output was read
while implementing or testing it.

## Purpose and boundary

`scripts/childlens_local_inference_firewall_v1_3.py` is an import-only,
fail-closed boundary between separately acquired local measurement instruments
and restricted inference. It does not discover a quarantine, download a model,
or expose an operational command-line path. A quarantine-side coordinator must
provide an explicitly authorized root, a sealed instrument set, opaque
content-digest work references, and fixed adapter invocations.

The lifecycle is irreversible:

1. Acquire code and model artifacts before restricted inputs are mounted.
2. Independently establish the code and weight licenses, authoritative source,
   exact versions, and hashes. No automated click-through is allowed.
3. Construct strict instrument receipts and call `seal_instruments`. Extra or
   free-form receipt fields fail closed.
4. Enter restricted execution. Every adapter invocation runs under an
   OS-enforced network-denial profile after an active IPv4/IPv6 socket-denial
   sentinel succeeds.
5. Keep pseudo-label payloads and checkpoints only in the owner-private
   quarantine. Export only the fixed aggregate receipt.

The receipt asserts that local instruments are measurement aids. Instrument
weights, tokenizers, vocabularies, embeddings, features, scores, confidences,
and unaudited pseudo-labels are barred from learner ancestry. Pseudo-labels are
not primary evaluation truth and cannot replace simulator oracle labels.

## Fixed adapter contract

All adapters implement `childlens-offline-fd-adapter-v1.3.0`. Corpus paths,
source filenames, model-artifact paths, and pseudo-label paths do not appear in
the adapter command. The sole filesystem argument is the generic,
receipt-hashed local adapter executable itself. The parent opens an input media
descriptor, model-artifact descriptor, and owner-private output descriptor and
passes only descriptor numbers. Standard input is closed;
standard output and standard error are discarded. The subprocess environment
is built from an allowlist and forces common Hugging Face and experiment
telemetry paths offline.

The frozen profiles are:

| Profile | Task | Resource class | Required behavior |
| --- | --- | --- | --- |
| `silero_vad` | speech segmentation proposals, revision `6.2.1` | CPU | offline |
| `whisper_cpp_large_v3_turbo` | multilingual ASR and word-timing proposals using unquantized `ggml-large-v3-turbo.bin`, revision `5359861c739e955e79d9a303bcbc70fb988958b1` | MPS-heavy | offline, word timestamps |
| `conservative_role_aid` | deterministic local acoustic role rules/config `childlens-role-rules-v1.3.0` | CPU | offline, `UNCERTAIN` fallback, no identity inference |
| `qwen2_vl_2b_instruct` | referential candidate proposals | MPS-heavy | offline, bounded frame batches, frozen revision `895c3a49bc3fa70a340399125c650a463535e71c` |

These names freeze an interface, not permission to acquire or use an artifact.
Each actual instrument still requires a separately reviewed code license,
weight license, evidence digest, authoritative-source receipt, and artifact
digest. A gated or click-through artifact remains blocked until the user has
separately resolved it.

The required role aid may emit only `NON_CHILD`, `CHILD`, `OVERLAP`, or
`UNCERTAIN`; its required conservative fallback is `UNCERTAIN`. It must not
infer identity. Optional anonymous speaker clustering such as ECAPA is outside
the required completion profile because it cannot establish semantic role; if
separately enabled for engineering support, it remains quarantine-only and
cannot change the required rules profile or its conservative fallback.

## Network and resource controls

On macOS, the runner uses the system sandbox with `deny network*`. On Linux, it
requires Bubblewrap with a new network namespace. The same wrapper denies
subprocess filesystem writes outside the explicitly validated quarantine root;
the parent still validates every intended output path before opening it.
Unsupported hosts fail with `E_NETWORK_ISOLATION_UNAVAILABLE`. A sentinel
attempts local IPv4 and IPv6 socket creation/bind/listen inside the exact
isolation wrapper. If either family succeeds, restricted inference is refused.
The sentinel and write confinement were exercised successfully on the current
macOS host using synthetic code and no corpus data.

CPU worker budgets are restricted to 2–4. MPS-heavy adapters are serialized by
an owner-private nonblocking file lock, so no more than one can run at once.
Adapter stdout/stderr cannot enter logs. Child environments inherit no home
directory, credentials, cookies, authorization headers, proxy configuration,
or cloud API keys.

## Checkpoint, deduplication, and export

The checkpoint is an owner-private SQLite database inside quarantine. Work is
keyed by input content digest, instrument-receipt digest, and task. Registration
verifies the media digest and rejects duplicate digests or inodes within a
batch. A restarted process converts an interrupted `RUNNING` entry back to
`PENDING`. Each pseudo-label is written to a private temporary file, synced,
size-bounded, atomically renamed inside quarantine, and recorded by digest.
The public aggregate builder receives state counts only; it never reads a
pseudo-label payload. It reports `COMPLETE` only when the sealed set contains
all four fixed profiles and every task has exactly 15/15 completed selected
media items with no pending, running, or failed work. A partial instrument set,
synthetic smoke test, or incomplete checkpoint remains `INCOMPLETE` and cannot
satisfy the terminal readiness validator.

The allowlisted aggregate receipt includes the exact security fields required
by the v1.3 terminal synthesizer: offline-only execution, proven subprocess
network blocking, no hosted/cloud content path, no external API or upload, no
telemetry, no restricted-data egress, quarantine-only outputs, and explicit
false ancestry/training/outcome fields. It contains no item/task cells, paths,
identifiers, filenames, exact timestamps, transcript or lexical content,
frames, or model-label payloads.

## Validation

`tests/test_childlens_local_inference_firewall_v1_3.py` uses synthetic private
files and a synthetic FD adapter. It proves strict receipt validation,
pre-download sealing, Qwen revision pinning, offline environment scrubbing,
active network-sentinel failure when sockets remain available, forged backend
rejection, OS-enforced write confinement, quarantine confinement, 2–4 CPU
workers, one MPS-heavy process, checkpoint/resume, media deduplication,
quarantine-only pseudo-labels, and the fixed aggregate security/ancestry
schema.

No test downloads a model, accesses a runtime root, reads corpus content, or
performs hosted inference.
