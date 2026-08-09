# ChildLens v1.3.1 — M5 model-shortlist preflight

Date: 2026-07-22  
Track: public/synthetic-only local-instrument engineering  
Host: Apple M5, 10 CPU cores, 32 GiB unified memory

> Historical pre-download preflight. The later pinned runtime receipts supersede its unmeasured candidate rankings: Qwen3-ASR-1.7B/aligner and Gemma 4 E4B/12B were subsequently executed on self-generated fixtures under network denial. See `model_upgrade_appendix_v1_3_1.md`.

## Scope and hard boundary

This preflight evaluates whether candidate *measurement instruments* can plausibly run on the 32-GiB M5 host. It did not access ChildLens media, annotations, transcripts, filenames, identifiers, timestamps, the restricted manifest, or the quarantine. No candidate weights were downloaded. The only executions used already-installed public models and a public synthetic/sample input.

Any selected model remains a fixed, separately licensed annotation instrument. Its outputs are pseudo-label hypotheses, never human ground truth. Its weights, tokenizer, vocabulary, embeddings, hidden features, confidence scores, or other artifacts must not enter learner ancestry. Before restricted inference, the selected stack still requires an authoritative-download receipt, exact hash, license receipt, isolated pinned environment, successful public-input canary, and network-denied execution test.

## Host and admission constraints

- Unified memory: 32 GiB. macOS and the application must retain headroom; a model that nominally approaches 32 GiB is not operationally safe.
- Free disk observed during preflight: approximately 89 GiB.
- Existing public instrument cache: approximately 15 GiB.
- Retained operational disk floor: 50 GiB free on the shared volume.
- Candidate caches must therefore be installed and evaluated one family at a time. The plausible E4B + Qwen3-VL-8B + Qwen3-ASR-1.7B/aligner + full Whisper set would add about 39.6 GiB and violate the floor if kept together.

## Direct public synthetic measurements

| Existing instrument | Public workload | Wall time | Peak RSS | Interpretation |
|---|---:|---:|---:|---|
| whisper.cpp large-v3-turbo | approximately 11 s public audio | 1.88 s | 1.84 GiB | Observed real-time factor about 0.17; this is **turbo**, not full large-v3. |
| Existing v1.3 Qwen2-VL/Whisper smoke | public synthetic audio plus one five-frame vision window | 36.19 s | 7.75 GiB | Includes startup/model load and mixed stages; it is a compatibility baseline, not a per-window throughput estimate. |

At the observed turbo real-time factor, 135.25 audio minutes would be about 23 minutes of model compute before pipeline overhead. OpenAI's relative-speed table describes turbo as roughly eight times the speed of large; multiplying the measured factor by eight gives a deliberately rough full-large-v3 estimate of roughly 3.1 hours. That is an inference, not a local measurement, and must not be used as a quality or feasibility result.

## Candidate inventory

Disk sizes below are the selected BF16 safetensor payloads from authoritative repository metadata, not total repository sizes containing duplicate formats. Gemma memory figures are Google's published approximate inference memory and include its stated 20% loading overhead; they are decimal GB as published. Other working-memory ranges are engineering estimates that must be replaced by measured peak RSS in a public-input canary.

| Candidate | BF16 payload | Memory/runtime evidence | Disk after single install | Preflight disposition |
|---|---:|---|---:|---|
| Qwen3-ASR-0.6B + ForcedAligner-0.6B | 3.17 GiB combined | Estimated under 8–10 GiB working set; Apple MPS support and runtime not established by official docs. ASR covers 52 languages/dialects; timestamps require the aligner, which supports 11 languages. | about 85.8 GiB free | **CONDITIONAL_CANARY** — best next ASR efficiency canary; alignment is language-conditional. |
| Qwen3-ASR-1.7B + ForcedAligner-0.6B | 5.51 GiB combined | Estimated 8–12 GiB working set; same unproven MPS/runtime condition. Official paper positions 1.7B for highest open-source accuracy. | about 83.5 GiB free | **CONDITIONAL_CANARY** — accuracy escalation if 0.6B fails frozen quality needs. |
| Whisper large-v3 | 2.88 GiB selected safetensors | OpenAI lists approximately 10 GB required VRAM for large. Full model is expected to be substantially slower than installed turbo; not measured here. | about 86.1 GiB free | **LOCAL_PREFLIGHT_PASS** — fits; audit only if turbo quality is insufficient. |
| Gemma 4 E2B-it | 9.54 GiB | Google: 11.4 GB BF16 including loading overhead; text, image, and audio inputs. | about 79.5 GiB free | **CONDITIONAL_CANARY** — safest new unified audio/vision canary and fallback. |
| Gemma 4 E4B-it | 14.89 GiB | Google: 17.9 GB BF16 including loading overhead; text, image, and audio inputs. | about 74.1 GiB free | **CONDITIONAL_CANARY** — preferred quality/fit vision-referential candidate. |
| Qwen3-VL-8B-Instruct | 16.33 GiB | No audio input. Estimated 22–27 GiB for conservative five-frame inference; official Apple MPS support not established. Requires a newer Transformers environment than existing v1.3. | about 72.7 GiB free | **CONDITIONAL_CANARY** — secondary vision candidate if Gemma is inadequate; tight enough to require measured RSS. |
| Gemma 4 12B | 22.28 GiB | Google: 26.7 GB BF16 including loading overhead, leaving too little macOS headroom. Google Q4_0 figure: 6.7 GB. | about 66.7 GiB free | **QUANTIZED_ONLY** — BF16 operationally unsafe; quantized multimodal runtime must be proven. |
| Gemma 4 26B-A4B-it | 48.07 GiB | Google: 57.7 GB BF16; Q4_0 14.4 GB. Only 4B parameters active per token, but all weights still require storage/residency. | about 40.9 GiB free | **QUANTIZED_ONLY / DEFER** — BF16 violates both memory and disk floor. |
| Gemma 4 31B | 58.25 GiB | Google: 69.9 GB BF16; Q4_0 17.5 GB. Dense compute makes it unattractive even if a quantized runtime fits. | about 30.8 GiB free | **QUANTIZED_ONLY / DEFER** — BF16 impossible and violates disk floor. |
| Qwen3.6-27B | 51.75 GiB | Dense 27B multimodal model; official long-context examples target multi-GPU deployment. Four-bit weight size would still need a separately proven runtime. | about 37.3 GiB free | **LOCAL_BF16_STOP** — do not download for this prototype. |
| Qwen3.6-35B-A3B | 66.97 GiB | 35B total/3B active MoE. Sparse activation reduces compute, not full weight storage. | about 22.0 GiB free | **LOCAL_BF16_STOP** — do not download for this prototype. |

## Environment and license gates

- Qwen repository metadata/model cards identify Apache-2.0. Gemma 4 official cards identify Apache-2.0. No license was accepted in this track. A local clause/receipt and any unavoidable gated-download action remain mandatory before installation.
- Qwen3-ASR HF-native configs currently require a substantially newer Transformers build than the pinned v1.3 stack; use an isolated, hashed environment rather than upgrading the working pipeline in place.
- Qwen3-VL-8B likewise requires a newer Transformers environment than the existing stack.
- Official Qwen documentation emphasizes CUDA/vLLM paths and does not establish Apple MPS compatibility for these exact releases. MPS/CPU fallback, peak RSS, deterministic resume, and runtime therefore remain unresolved until a public-input canary.
- The forced aligner supports Chinese, Cantonese, English, German, Spanish, French, Italian, Portuguese, Russian, Korean, and Japanese. If the locally established corpus language is outside this set, Qwen ASR may still transcribe it, but this aligner cannot supply official word timing; a different separately licensed local alignment route would be required.

## Feasibility ranking

### Speech and timing

1. Keep installed Whisper large-v3-turbo as the proven compatibility and throughput baseline.
2. Canary Qwen3-ASR-0.6B, then its forced aligner only when the established language is supported. It has the best next-test storage/accuracy-efficiency profile.
3. Escalate to Qwen3-ASR-1.7B only if the 0.6B public canary passes technically but later fails a frozen quality need.
4. Audit full Whisper large-v3 only if turbo quality is inadequate and the slower runtime is acceptable.

### Vision and coarse referential candidates

1. Gemma 4 E4B-it is the preferred quality/fit canary; its official 17.9-GB loaded-memory estimate leaves meaningful host headroom.
2. Gemma 4 E2B-it is the safer fallback and fastest likely path to a stable local unified-instrument test.
3. Qwen3-VL-8B is a secondary canary only if Gemma quality or integration fails. It lacks audio, consumes more disk, and has an unproven MPS path.
4. Gemma 4 12B is quantized-only on this host. The 26B-A4B and 31B variants are deferred; their BF16 forms fail the memory/disk admission policy.
5. Qwen3.6-27B and 35B-A3B are stopped for this 32-GiB prototype. Their weights were not downloaded and must not be downloaded under this track.

## Frozen next-test sequence

The safe next step is an engineering canary, not restricted annotation:

1. Install exactly one candidate family from its authoritative source after recording model-card license, commit, selected files, SHA-256, and dependency lock.
2. Run only public/synthetic inputs first, with telemetry disabled and outbound network denied during inference.
3. Measure startup, peak RSS, per-audio-minute or per-five-frame-window time, output schema, resume behavior, and deterministic failure handling.
4. Require at least 4 GiB system-memory headroom under stress and at least 50 GiB disk free. Abort if either floor is threatened.
5. Remove a rejected public candidate before installing the next one; do not accumulate caches.
6. Only after the public canary, license, hash, privacy, and no-network validators pass may a separate coordinator consider the instrument for restricted pseudo-annotation. This report does not authorize that step.

Recommended order: Qwen3-ASR-0.6B (ASR first; aligner conditionally) and Gemma 4 E4B-it, with Gemma 4 E2B-it as the low-risk fallback. Qwen3-VL-8B is held in reserve. No 27B/35B download is justified.

## Exact repository revisions inspected

- Qwen3.6-27B: `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`
- Qwen3.6-35B-A3B: `995ad96eacd98c81ed38be0c5b274b04031597b0`
- Gemma 4 E2B-it: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- Gemma 4 E4B-it: `ee0ef6023621cff504d758262d4e04895a5af4a2`
- Gemma 4 12B-it: `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- Gemma 4 26B-A4B-it: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`
- Gemma 4 31B: `842da3794eaa0b77d5f08bae87a17459d91ff475`
- Qwen3-ASR-0.6B-hf: `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`
- Qwen3-ASR-1.7B-hf: `bcd2b5b7f32b480ab5790554cfa8347f246a14f3`
- Qwen3-ForcedAligner-0.6B-hf: `c07281df297b9905d24a508279258cccf987a064`
- Qwen3-VL-8B-Instruct: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- Whisper large-v3: `06f233fe06e710322aca913c1bc4249a0d71fce1`

## Primary sources

- [Gemma model overview](https://ai.google.dev/gemma/docs/core)
- [Gemma memory requirements](https://ai.google.dev/gemma/docs/get_started)
- [Gemma 4 E2B-it](https://huggingface.co/google/gemma-4-E2B-it) and [E4B-it](https://huggingface.co/google/gemma-4-E4B-it)
- [Qwen3-ASR official repository](https://github.com/QwenLM/Qwen3-ASR), [0.6B card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf), [1.7B card](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), and [forced-aligner card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Qwen3-ASR paper](https://arxiv.org/abs/2601.21337)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [OpenAI Whisper repository](https://github.com/openai/whisper), [Whisper large-v3 card](https://huggingface.co/openai/whisper-large-v3), and [whisper.cpp model documentation](https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md)

## Preflight conclusion

The 32-GiB M5 can support a carefully serialized local-instrument shortlist, but not the large BF16 candidates or a simultaneous cache of all feasible candidates. The defensible path is a Qwen3-ASR-0.6B public canary plus a Gemma 4 E4B public canary, retaining Whisper turbo and Gemma E2B as low-risk fallbacks. Qwen3-VL-8B is conditional; Gemma 12B+ is quantized-only; Qwen3.6 27B/35B are stopped for this host and were not downloaded.
