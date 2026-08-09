# ChildLens pseudo-calibration execution disposition

Date: 2026-07-22  
Scope: frozen prediction-independent 15-speech-minute sample  
Disposition: **CALIBRATION_REVISE**

## What ran

The execution reused the historical local Whisper/Qwen2 hypotheses read-only and ran the pinned Qwen3-ASR 1.7B plus forced aligner and Qwen3-VL 8B 4-bit challengers locally. The sample remained fixed at 15 participant-distinct items and 900 speech seconds; 137 pre-existing candidate windows crossed the half-open center-in-window rule. Memory-heavy paths ran serially. Restricted media, frames, audio, hypotheses, text, identifiers, and exact intervals stayed in the owner-private quarantine. Network access was denied during every restricted model process.

The public receipt contains only K=5-complement-suppressed, outward-rounded aggregates. It reports model disagreement, not truth or human reliability. No learner, corpus tokenizer, checkpoint, causal arm, acquisition effect, or simulator endpoint ran.

## Frozen gate result

- Qwen3-ASR nonempty item coverage: outward-rounded interval 1.0–1.0.
- Cross-model German language-ID agreement: 0.9–1.0, diagnostic only.
- Qwen3-VL schema validity: 1.0–1.0.
- Historical Qwen2 schema validity: 0.0–0.0.
- The single predeclared Qwen2 schema-only retry used the same checkpoint, Whisper hypotheses, windows, and five frame offsets and again produced 0.0–0.0 schema validity.

The failed Qwen2 path prevents a two-model visual envelope. Candidate multiplicity, null/irrelevant, and visibility envelopes are suppressed or incomplete. The speech paths also disagree substantially on boundary segmentation and source text; those diagnostics are explicitly non-human and cannot name either model as reference.

## Scientific disposition

The narrower existing v1 symbolic action-alignment mechanism proof remains reusable. The ChildLens-conditioned sensitivity outcome is not launch-ready and remains unauthorized. A one-model fallback would violate the frozen triangulation gate, so it was not used.

The next task requires a substantive user choice:

1. Freeze one replacement visual-instrument calibration revision using a distinct, unambiguously licensed local model, keeping the same sample, schema, thresholds, and one-shot rule; or
2. drop the ChildLens-calibrated layer and freeze an explicitly empirical-free synthetic sensitivity protocol.

For a Michael Frank-facing directional prototype, option 1 is recommended because it preserves the honest naturalistic-envelope story without representing model agreement as ground truth.
