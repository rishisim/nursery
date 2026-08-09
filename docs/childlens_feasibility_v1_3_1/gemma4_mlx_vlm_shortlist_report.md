# ChildLens v1.3.1 Gemma 4 / MLX-VLM public synthetic preflight

Status: **HISTORICAL PREFLIGHT — SUPERSEDED BY THE RUNTIME BAKEOFF BELOW**  
Date: 2026-07-22  
Scope: public documentation and code-generated synthetic fixtures only

## Boundary and result

No ChildLens content, quarantine file, identifier, transcript, timestamp, filename, frame, audio, or model-label payload was discovered, opened, or processed. No hosted inference was used. No model weights were downloaded, no click-through or access condition was accepted, and no model server was started.

This section records the pre-download state. It remains useful as a provenance record, but its `NOT_EXECUTED` status was subsequently resolved by the network-denied public/synthetic bakeoff described next.

## Runtime bakeoff addendum

Pinned MLX-VLM 0.6.6 executed both public 4-bit conversions locally under macOS network denial using one self-generated German TTS sentence and five synthetic frames. Both models consumed audio and all five frames and emitted schema-valid JSON. E4B used 6.247 GiB peak MLX unified memory in 17.135 seconds and achieved zero WER/CER on the synthetic source sentence. The 12B Unified model used 8.017 GiB in 16.204 seconds but had WER 1.0 and CER 0.7576 on the same sentence. Both recovered the synthetic visual object and audio–visual relation.

This is a one-fixture engineering canary, not a quality evaluation. E4B is now the preferred *conditional* joint pseudo-annotation instrument because it used less memory and was the only candidate with correct synthetic transcription. The E4B audio frontend emitted numerical preprocessing warnings, so any future use still requires a broader public suite and license-metadata reconciliation. Neither model is authorized here for a ChildLens rerun.

Machine-readable measurements are in `output/childlens_feasibility_v1_3_1/gemma4_public_synthetic_runtime_bakeoff.json`.

## Evidence matrix

| Capability | Gemma 4 E4B IT | Gemma 4 12B Unified IT |
|---|---|---|
| Upstream image input | Documented | Documented |
| Upstream audio input | Documented | Documented |
| MLX-VLM image path | Documented | Documented |
| MLX-VLM audio path | Documented; early defects were fixed in v0.5.0, but the fix author reported no Apple-Silicon end-to-end validation | Documented in merged Unified support |
| Mixed image + audio through MLX-VLM | Framework path exists; no E4B-specific mixed reproduction found | Mixed-input reproduction exists in the merged 12B support PR |
| Strict JSON Schema | Framework/server feature; not target-model validated in the implementation PR | Framework/server feature; not target-model validated in the implementation PR |
| Exact mixed-input + strict-schema composition | **NOT_EXECUTED** | **NOT_EXECUTED** |
| Disposition | `CONDITIONAL_SMOKE_REQUIRED` | `CONDITIONAL_SMOKE_REQUIRED` |

MLX-VLM's strict schema mode is constrained decoding through the server API. That can enforce syntactic/schema validity; it does not establish semantic correctness, modality use, or ChildLens-domain validity. The feature is not documented for the direct `generate()` convenience path, does not support `json_object`, and is incompatible with speculative decoding. Thinking-aware grammar handling is also documented as a limitation, so a future smoke must disable thinking and speculative decoding.

## Local preflight

The host reports 32 GiB installed memory. The inspected Python environment had neither `mlx` nor `mlx_vlm`, and neither shortlisted checkpoint was present in the two explicit public Hugging Face cache locations. Therefore runtime memory, model load success, schema-valid model output, semantic use of both modalities, and inference wall time are all unmeasured.

Published conversion artifact sizes are approximately 5.15 GB for `mlx-community/gemma-4-e4b-it-4bit` and 6.74 GB for `mlx-community/gemma-4-12B-it-4bit`. These are remote artifact sizes, not measured working-set or peak-memory figures. Google's 12B developer guide says the architecture targets a 16 GB GPU or unified-memory device, but that claim is not a measurement of this MLX conversion on this host.

The synthetic harness generated a 256×256 red-square image and a one-second, 16 kHz mono tone. It built allowlisted loopback request blueprints containing image, text, audio, and `strict: true` JSON Schema fields. It accepted a valid synthetic response and rejected an extra property. This validates only the schema/request harness; `server_call_executed` remains false.

## License and acquisition preflight

Google's current E4B and 12B model cards identify Apache-2.0. MLX-VLM code is MIT. Public MLX conversion metadata is inconsistent: the E4B conversion still displays a legacy `gemma` license label, while current upstream cards say Apache-2.0, and the 12B IT conversion does not present an equally clear license label. This is metadata drift, not permission evidence.

Before any later download, the coordinator must pin the exact upstream and conversion revisions, retain the applicable Apache-2.0 license/NOTICE materials, affirm that the chosen conversion inherits a usable license, and recheck whether access conditions have changed. This track did not accept terms or download weights.

## Required public/synthetic runtime smoke

Use a pinned MLX-VLM release no older than the version containing the relevant fixes (current evidence points to v0.6.6), the exact intended `-it` conversion revision, and a nonrestricted image/audio fixture with distinct semantic signals. Execute only after package, license, disk, and memory preflight.

For each model, freeze and test:

1. image-only and audio-only requests;
2. one mixed image → instruction text → audio request;
3. `/v1/chat/completions` with `response_format.type = json_schema`, `strict: true`, thinking disabled, and speculative decoding disabled;
4. parse and Draft 2020-12 schema validation;
5. correct recovery of both synthetic modality signals, not schema validity alone;
6. peak process memory and wall time, clearly identified as measured only after execution;
7. a network-denied rerun from a pinned local snapshot.

E4B should not be privileged merely because it is smaller: its audio path had prior defects and lacks a target-specific mixed-modal reproduction in the inspected evidence. The 12B candidate has stronger MLX mixed-input implementation evidence, but its exact public 4-bit checkpoint plus strict-schema path remains unproven. Both therefore stay conditional rather than approved.

## Decision for the v1.3.1 shortlist

The evidence supports retaining both models for a bounded public/synthetic runtime comparison, with no current winner. It does not authorize restricted inference and does not alter the frozen human-evidence boundary. A successful future smoke would establish technical plumbing only; model outputs would remain quarantined pseudo-label hypotheses, never human evidence or primary evaluation truth.
