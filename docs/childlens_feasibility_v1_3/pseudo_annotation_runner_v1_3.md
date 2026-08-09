# ChildLens v1.3 local pseudo-annotation runner

## Scope

`run_childlens_model_assisted_pseudo_annotation_v1_3.py` is the zero-argument,
fail-closed bridge from the frozen v1.2 pilot to v1.3 quarantine-only machine
hypotheses. It does not initialize, train, or evaluate a learner. Machine
outputs are pseudo-labels, never ground truth, and cannot replace simulator
oracle labels for the later primary evaluation.

## Two-phase boundary

The public phase opens no ChildLens input. It independently verifies the exact
Silero 6.2.1 ONNX cache, pinned whisper.cpp v1.9.1 build bundle and unquantized
large-v3-turbo weight, the complete 40-entry Qwen2-VL cache manifest, package
versions, executable hashes, licenses, and adapter code. The four strict
firewall receipts are sealed before runtime discovery.

The launcher then re-executes its restricted phase under the platform OS
network-denial backend. An active socket-denial sentinel must pass before the
runtime is discovered. Adapter subprocesses inherit the denial, receive media
and job data through file descriptors, discard stdout/stderr, use an
allowlisted credential-free offline environment, and write only inside the
owner-private quarantine. Hosted models, APIs, uploads, telemetry, URL loaders,
and remote model identifiers have no execution path.

The restricted phase validates all 15 content hashes and all 912 official
speech windows (135.25 minutes) against the frozen selection. It rejects
duplicate media and any prediction-, confidence-, lexical-, salience-, or
apparent-success-dependent selection field. A crash-safe SQLite checkpoint
supports restart without duplicate completed work. CPU work is limited to two
workers and an exclusive file lock permits only one MPS-heavy process.

Every visual candidate uses the same five fixed offsets and the same
content-independent resize to a maximum 448-pixel edge. The resize was frozen
after the first visual engineering attempt failed before producing any usable
pseudo-label. It bounds MPS memory without changing selection, frame times,
thresholds, or adapting to confidence, salience, lexical richness, or apparent
success.

The inherited storage controls are enforced before discovery-dependent work,
before every adapter invocation, while each child process is alive, and after
scratch cleanup: the total private namespace may not exceed 73 GiB and the
observed/projected free-space floor may not fall below 50 GiB.

## Fixed instruments

- Silero VAD 6.2.1 runs through a minimal direct ONNX Runtime 1.23.2 wrapper.
  This avoids the upstream Python helper's undeclared `torchaudio` import while
  retaining the audited model and reference state/context and hysteresis
  rules. Its boundaries remain hypotheses.
- whisper.cpp v1.9.1 uses the unquantized multilingual large-v3-turbo model,
  automatic language detection, deterministic transcription, and experimental
  word timing. Text, language, and timing remain hypotheses.
- The conservative source-role record uses `UNCERTAIN` when there is no
  defensible semantic speaker anchor. This is deliberately zero decision
  coverage, not fabricated child/adult classification. Qwen may separately
  propose a contextual source role, which also remains a pseudo-label. The
  author audit must therefore determine whether the role gate is feasible.
- Qwen2-VL-2B-Instruct at the frozen revision processes exactly five fixed
  frames at offsets -5, -2.5, 0, +2.5, and +5 seconds around each official
  window center. No prediction-, confidence-, lexical-, visual-, or
  success-dependent resampling is allowed. Object, action, reference, and role
  strings remain quarantine-only.

All instrument weights, tokenizers, vocabularies, prompts, scores, features,
embeddings, and pseudo-label payloads are barred from learner ancestry.

## Export and failure behavior

The only repository export is
`output/childlens_feasibility_v1_3/pseudo_annotation_receipt.json`. It contains
fixed aggregate completion, resource, storage, privacy, and scientific-boundary
assertions. It contains no item cells, identifiers, filenames, paths, exact
times, transcript or lexical content, frames, or model payloads. A bounded
instrument failure exports a truthful `INCOMPLETE` aggregate when checkpoint
counts are available so the terminal synthesizer can return REVISE. Integrity,
privacy, release-binding, network, or storage failures remain hard failures.

## Commands

Public synthetic audio and VLM adapter smoke, with no ChildLens input:

```bash
python3 scripts/smoke_childlens_local_pseudo_adapters_v1_3.py
```

Synthetic tests:

```bash
pytest -q tests/test_run_childlens_model_assisted_pseudo_annotation_v1_3.py \
  tests/test_childlens_local_inference_firewall_v1_3.py
```

Restricted run (zero arguments):

```bash
python3 scripts/run_childlens_model_assisted_pseudo_annotation_v1_3.py
```

The runner never creates or attaches an author prediction binding. Comparison
remains unavailable until the blinded author record is irreversibly locked.
