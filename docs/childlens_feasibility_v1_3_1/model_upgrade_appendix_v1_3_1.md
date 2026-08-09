# ChildLens v1.3.1 local annotation-instrument upgrade appendix

Date: 2026-07-22  
Scope: official public documentation and public/self-generated synthetic fixtures only

## Bottom line

Yes, there are better *candidate instruments*, but none fixes the missing qualified German human evidence.

- For transcript and timing, the best challenger is **Qwen3-ASR-1.7B-hf plus Qwen3-ForcedAligner-0.6B-hf**, with the existing `whisper.cpp` large-v3-turbo retained as the fast baseline and disagreement instrument. Full Whisper large-v3 is a slower control if a later human audit makes quality comparison meaningful.
- For a joint audio–five-frame pseudo-annotator, **Gemma 4 E4B IT 4-bit MLX** is the preferred canary. On one self-generated German fixture it beat the 12B Unified conversion while using less memory. This is not ChildLens evidence.
- For an optional visual-only disagreement judge, **Qwen3-VL-8B-Instruct** is lighter and more role-appropriate than Qwen3.6. Neither has audio.
- Qwen3.6 27B and 35B-A3B are poor defaults on this 32-GiB host: they are visual/text models, not ASR, and the MoE still stores all 35B parameters even though about 3B are active per token.

No historical ChildLens pseudo-label was replaced and no upgrade was run on restricted material.

## Transcript and timing

The official Apache-2.0 [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) supports automatic language identification and German among 30 languages. The Apache-2.0 [forced aligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) supports German and returns word/character timestamps for inputs up to five minutes; streaming ASR does not return timestamps. Official examples are CUDA-oriented, so Apple support had to be tested rather than assumed.

A pinned MPS smoke under OS network denial used one self-generated German TTS sentence. Qwen3-ASR identified German and achieved WER/CER 0.0, using 6.14 GiB peak process RSS and 28.458 seconds including load and inference. The aligner returned seven monotonic word timestamps, used 1.072 GiB peak RSS, and took 3.403 seconds. The exact model revisions and artifact digests are in `qwen3_asr_public_synthetic_runtime_receipt.json`.

The existing Whisper turbo public measurement took 1.88 seconds and 1.84 GiB peak RSS on approximately 11 seconds of public audio. Thus Qwen3 is technically feasible but materially slower at cold start. One clean TTS sentence cannot establish superiority on natural child-worn German audio. The defensible future design is Qwen3 as the higher-capacity transcript/timing proposal, Whisper turbo as an independent fast proposal, and disagreement as an uncertainty flag. Only a locked German human transcript can decide accuracy. [OpenAI documents](https://github.com/openai/whisper) that turbo is a faster, slightly degraded large-v3 variant and that the original code/weights are MIT-licensed; exact distribution licenses must still be bound.

Full Whisper large-v3 fits locally (`whisper.cpp` documents roughly 2.9 GiB disk and 3.9 GB memory) but is expected to be far slower than turbo. It was not downloaded because there is no qualified German audit against which its extra cost could be evaluated. Treat it as a later quality control, not the default.

## Joint audio–frame instrument: Gemma 4

Google's official [Gemma 4 overview](https://ai.google.dev/gemma/docs/core) lists E2B, E4B, 12B, 26B-A4B, and 31B. E2B, E4B, and 12B accept audio, images, and text; the 26B and 31B variants do not accept audio. [Audio is limited to 30 seconds](https://ai.google.dev/gemma/docs/capabilities/audio). Google's approximate Q4 load figures are 2.9, 4.5, 6.7, 14.4, and 17.5 GB respectively.

Pinned MLX-VLM 0.6.6 and public 4-bit conversions were tested under OS network denial on five generated frames plus one German TTS sentence:

| Model | Strict schema | German WER/CER | Visual and joint relation | Peak MLX memory | Wall time |
|---|---:|---:|---:|---:|---:|
| Gemma 4 E4B IT 4-bit | Pass | 0.0 / 0.0 | Pass | 6.247 GiB | 17.135 s |
| Gemma 4 12B Unified IT 4-bit | Pass | 1.0 / 0.7576 | Pass | 8.017 GiB | 16.204 s |

The E4B run emitted audio-preprocessor numerical warnings, so a broader public suite is still required before restricted use. MLX-VLM 0.5.0 initially failed to load E4B's KV-shared weights; pinned 0.6.6 resolved that compatibility failure. Conversion repositories lack fully consistent license metadata even though the current upstream Google models are Apache-2.0. Therefore E4B is **preferred but conditional** on conversion-license reconciliation, a multi-fixture public German smoke, exact hashes, and the existing network-denied instrument boundary.

Strict JSON validity proves interface behavior, not semantic truth. Gemma outputs remain pseudo-label hypotheses.

## Visual-only alternatives

[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) is Apache-2.0, supports image/video input and temporal/spatial grounding, and has roughly 17.5 GB of official BF16 weight payload. A pinned 4-bit MLX conversion could fit, but it was not downloaded or executed. Any community conversion must bind the upstream revision, conversion code/version, quantization configuration, and every shard hash. It is a reasonable optional visual disagreement judge on a small uncertain subset, not a joint audio model.

[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) and [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) are official Apache-2.0 image+text/video models. The 27B is dense. The 35B MoE activates about 3B parameters but stores about 35B total weights. Their observed unquantized payloads were approximately 55.6 and 71.9 GB, respectively, so neither was downloaded. Qwen documents llama.cpp-compatible quantizations and the community MLX stack supports them, but quantized feasibility does not make them better than the lighter Qwen3-VL-8B for this narrow visual-judge role.

## Speaker role is a separate problem

Diarization answers “who spoke when” using anonymous clusters. It does not type `CHILD` versus `NON_CHILD`. The [pyannote Community-1 card](https://huggingface.co/pyannote/speaker-diarization-community-1) documents local offline diarization but requires accepting access conditions; no such conditions were accepted here, and the output would still need semantic voice typing.

A child-centered VTC could be a quarantined annotation instrument only after affirmative code/weight license evidence, exact provenance, acceptance of its external empirical training ancestry under the annotation-instrument amendment, and in-domain German human validation. Without those conditions, source role remains `UNCERTAIN`; it cannot be inferred from diarization alone.

## Scientific boundary

Model–model agreement is useful for disagreement flags, candidate unions, and sensitivity ranges. It is not inter-human reliability or source-language ground truth. English machine translation checked by a non-German speaker is circular. GPT-5.6 Luna, Codex, and all hosted models remain ineligible to inspect restricted ChildLens content or label payloads.

An upgrade is worth implementing only after either (a) a qualified, authorized German annotator is obtained, allowing model–human comparison, or (b) the project explicitly pivots to a pseudo-label-only descriptive calibration study with no lexical-validity claim. It cannot reverse `CHILDLENS_LEXICAL_FEASIBILITY_STOP_CURRENT_RESOURCES`.
