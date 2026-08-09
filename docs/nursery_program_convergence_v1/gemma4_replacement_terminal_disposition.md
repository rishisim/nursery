# Gemma 4 E4B replacement disposition

Decision: `CALIBRATION_STOP_MODEL_TRIANGULATION`

## Outcome

The one-instrument Qwen2-to-Gemma replacement stopped at the mandatory provenance gate, before any Gemma inference on restricted ChildLens material. The completed Qwen3-ASR and Qwen3-VL results were not rerun or rewritten. No new ChildLens aggregate was computed, the one parser correction remains unused, no third visual model was tried, and no causal learner outcome ran.

This is a stop for the frozen Gemma triangulation route under the present artifact and storage state. It does not change the historical ChildLens lexical STOP, the earlier provisional-calibration REVISE receipt, or any v1–v1.3.1 evidence.

## Evidence-backed gate disposition

The official `google/gemma-4-E4B-it` snapshot is public, ungated, and Apache-2.0 at immutable revision `ee0ef6023621cff504d758262d4e04895a5af4a2`. Its single 15,992,595,884-byte weight file has SHA-256 `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503`.

The locally available 4-bit artifact is exactly manifested and operationally fits the M5/32-GiB host, but it cannot pass the frozen provenance gate. Its card declares `license: gemma`, while its stated upstream now declares Apache-2.0; it contains no LICENSE or NOTICE; and it says only that a local `mlx_vlm.convert` checkout produced it, without naming the converter version or revision. Its newly added source-revision statement and the byte identity of that source with the current official weights improve traceability but do not provide the exact conversion receipt the frozen amendment requires. A successful historical public synthetic run is technical evidence, not permission evidence.

The clean alternative was frozen: download the exact official snapshot and locally quantize it with MLX-VLM 0.6.6 at pinned commit `c9e27b08311f9c1f38690c46034b04c7598d73ee`, affine 4-bit, group size 64, under network denial. That path requires at least 22 GB decimal for source plus output before temporary-write headroom. Only 5.516 GiB remained above the required 50-GiB post-peak free-space floor. It was therefore not admitted and no upstream weights were downloaded.

The protocol's single correction is explicitly limited to removing one exact outer JSON fence. Spending it on a license, provenance, or disk failure would silently weaken the frozen design. Trying another visual model is also forbidden. Consequently the exact-common-schema canary, restricted 137-window run, Gemma-versus-Qwen3 aggregation, envelope formation, and outcome binding were not reached.

## What is frozen and ready

The additive replacement amendment freezes the unchanged 15-item, 900-second, 137-window sample; five fixed frame offsets; joint local audio/frame input; exact five-field categorical schema; deterministic decoding; K=5 suppression; outward rounding; unchanged coverage and abstention gates; one parser-only correction; and all privacy and no-ground-truth boundaries. It remains `FROZEN_NOT_ACTIVE_PENDING_PROVENANCE`.

A minimal symbolic runner now implements the five matched arms, paired bundles, cue-free evaluation, dual noun/object and verb/action corrective updates, side-only leakage checks, ancestry checks, and confidence-only-sharpening falsification on synthetic fixtures. Its scientific `run` entrypoint remains deliberately fail-closed because the calibration gate is not satisfied. This code is readiness work, not an acquisition-effect result.

## Privacy and scientific boundary

No restricted Gemma output was opened or produced. No audio, frames, transcript or translation text, identifiers, filenames, exact timestamps, per-window predictions, or small cells entered repository artifacts or commentary. The historical Qwen3 outputs remained quarantined and read-only. No hosted inference was used. No AEA or BabyView empirical artifact entered the work. ChildLens pseudo-labels remain provisional measurement hypotheses and simulator-oracle labels remain the only permitted future evaluation truth.

## Future reopening condition

There is no defensible one-shot outcome command from this terminal state. A fresh task may reopen this exact Gemma route only after enough additional local workspace exists to stage the pinned official source and locally converted output while preserving the 50-GiB floor (at least 22 GB decimal plus temporary-write headroom). It would then need a new immutable activation receipt, complete hashes, and the exact frozen joint audio/five-frame referential-schema canary under network denial before any restricted inference. That is a new activation attempt, not the parser correction and not permission to alter scientific gates.

Primary public sources: [Google Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4), [Google Apache-2.0 license](https://ai.google.dev/gemma/apache_2), [pinned official model snapshot](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2), and [MLX-VLM 0.6.6](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.6).
