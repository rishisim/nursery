# ChildLens v1.3.1 public-only scientific-role model shortlist

Status: **FROZEN PUBLIC-MODEL ANALYSIS — NO CHILDLENS CONTENT INSPECTED**  
Scope: model roles for local pseudo-annotation and disagreement analysis only  
Date: 2026-07-22

> Historical broad shortlist. The user-requested Qwen3.6/Gemma 4/Qwen3-ASR reassessment and the subsequent public synthetic runtime measurements supersede its implementation ranking. The controlling recommendation is `model_upgrade_appendix_v1_3_1.md`; the human-evidence boundary below remains unchanged.

## Boundary and decision

This analysis used public model documentation only. It did not access ChildLens media, audio, transcripts, identifiers, exact timestamps, filenames, manifests, quarantine files, or item-level model outputs.

No model listed here supplies human evidence. The scientifically defensible branch is:

1. **Qualified-German-human branch:** use local models only to create blinded proposals; obtain source-German human corrections and judgments before computing model–human agreement.
2. **No-qualified-human branch:** freeze the work as **DESCRIPTIVE_PSEUDO_ONLY**. Report instrument-specific aggregate coverage and sensitivity, use pseudo-labels for candidate generation and nonidentifying simulator calibration only, and retain simulator-oracle labels as the primary later evaluation truth. Do not call a ChildLens lexical-grounding feasibility gate passed from model consensus.

The second branch is a bounded scientific revision, not a substitute for human validation.

## Role-based shortlist

| Scientific role | Primary candidate | Secondary candidate | v1.3.1 disposition |
|---|---|---|---|
| Source transcript | Existing local `whisper.cpp` + Whisper large-v3-turbo | Phi-4 multimodal audio output as a diagnostic disagreement signal only | Keep the existing generator. Never score source-German WER/CER without locked human source text. |
| Word timing | WhisperX alignment-only path using a pinned German alignment model | Existing Whisper timing | Conditional shortlist: separately audit the exact alignment model/weights/license and verify network-denied CPU execution. Alignment refines the timing of hypothesized words; it does not validate their identity. |
| Voice/source role | VTC 2.2 | Anonymous diarization plus a separately validated voice-type mapping | Technical best fit, but blocked from acquisition/run until its code and weight licenses are affirmatively established. ChildLens-domain and German audit remain mandatory. |
| Joint audio-frame pseudo-annotation | Microsoft Phi-4-multimodal-instruct | Qwen2.5-Omni-7B | Phi-4 is conditional on pinned-code security review, authoritative hashes, and a public/synthetic network-denied Apple-MPS smoke test. Qwen2.5-Omni-7B is deferred for host-memory risk; its 3B sibling is blocked by its non-Apache custom license unless the user separately resolves it. |
| Visual disagreement judge | Florence-2-base | Qwen2.5-VL-7B-Instruct | Prefer Florence-2 for architecture/task diversity and host feasibility. Qwen2.5-VL-7B is a conditional high-capacity judge after a public/synthetic MPS and memory smoke test, but its Qwen lineage makes its errors less independent from the existing Qwen instrument. |

All candidates are measurement instruments. Their embeddings, hidden features, weights, tokenizers, vocabularies, confidence scores, and model-specific encodings must remain outside learner ancestry and scientific evaluation truth.

## Transcript and timing role

The existing `whisper.cpp`/Whisper large-v3-turbo path remains the practical source-transcript generator. Its output is a hypothesis, not a transcription record.

[WhisperX](https://github.com/m-bain/whisperX) is the strongest narrowly scoped timing addition because its official repository documents word-level timestamps via forced alignment, a German language path, a macOS CPU path, and a BSD-2-Clause code license. Its diarization option is not part of this shortlist. The exact German alignment model and downloaded weights still need their own provenance and license receipt. Forced alignment can move boundaries around the words it is given; it cannot establish that the words are correct. Therefore:

- boundary agreement may be measured only against locked human boundaries under the already frozen tolerance;
- source-language WER/CER requires human-corrected German text;
- failures or omissions in the transcript must not disappear through alignment;
- translated text cannot be substituted for source text.

Translation is not human evidence. It changes lexical identity, morphology, omissions, and segmentation, and can mask source-ASR errors. Even agreement between an ASR transcript, a translation, and a back-translation cannot show which German nouns or verbs were actually spoken. Translation may be a quarantined navigation aid only after source transcription exists; it must not define lexical labels, WER/CER, learner input, or evaluation truth.

## Voice/source-role role

[VTC 2.2](https://github.com/LAAC-LSCP/VTC) is the most task-specific public candidate. Its official README describes child-centered long-form recordings from a child-worn recorder, classes for key child, other child, female adult, and male adult, and Apple MPS support. It reports held-out F1 values of 68.5 for key child, 58.1 for other child, 70.9 for male adult, and 75.6 for female adult. These figures are evidence of imperfection, not a ChildLens accuracy guarantee.

VTC may be considered only under all of these conditions:

1. Establish an affirmative license for both code and weights. The inspected repository did not expose a root license, so absence of a prohibition is not permission.
2. Pin repository revision, weights, hashes, dependencies, and class mapping before restricted inference.
3. Complete public/synthetic Apple-MPS smoke tests, then execute restricted inference with network and telemetry disabled.
4. Audit the mapping on blinded, human-labeled source-language material. The official [VTC paper](https://www.isca-archive.org/interspeech_2020/lavechin20_interspeech.pdf) covers multilingual child-centered recordings but does not establish German or ChildLens validity; it also explains that other-child classification is strongly affected by distance from the wearable microphone.
5. Map female/male adult to `NON_CHILD` and key/other child to `CHILD` only for the frozen coarse gate. Preserve simultaneous class activation as an overlap hypothesis when the exact version demonstrably supports it; otherwise emit `UNCERTAIN`, not an invented `OVERLAP` label.

The [2025 VTC challenges analysis](https://arxiv.org/abs/2506.11074) reports that data relevance and quantity mattered more than many architecture or representation changes. That reinforces the need for an in-domain human audit.

Generic diarization is not voice typing. [pyannote](https://github.com/pyannote) defines diarization as answering “who spoke when”; anonymous speaker clusters do not say whether a speaker is a child or an adult. At most, diarization can support within-item speaker consistency and overlap hypotheses. A human or validated voice-type classifier must assign semantic role.

The official [pyannote Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1) requires accepting access conditions and supplying a Hugging Face token. It also documents optional telemetry. It is therefore not an autonomously selectable component here: accepting conditions is a user/legal action, and any later permitted local use would require a predownloaded snapshot plus telemetry and network disabled during restricted processing.

## Joint audio-frame pseudo-annotation role

[Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) is the preferred public candidate for a bounded local experiment because its official card documents joint image-and-audio input, German text and audio support, and a lightweight 5.6B-class architecture. Its [license is MIT](https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/5edcd2bcfc6b286a904f42c0de633c7ba978f300/LICENSE). The same card lists vision-language support as English, so German audiovisual grounding must be treated as unestablished even if German audio transcription works.

It remains conditional rather than approved:

- the official loader uses `trust_remote_code=True`, so the exact custom code must be pinned and security-reviewed before execution;
- the official tested hardware is NVIDIA rather than Apple MPS;
- the published joint examples/benchmarks do not establish validity for natural German child-worn audiovisual material;
- it must pass a nonrestricted synthetic MPS/eager-attention smoke test under a fixed memory ceiling and then run with network disabled;
- its outputs can propose source-role, visible-candidate, null/undecidable, noun/object, and verb/action hypotheses, but cannot adjudicate them.

[Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) is technically attractive because the official model card documents mixed text/image/audio/video input and temporal alignment. Its [license is Apache-2.0](https://huggingface.co/Qwen/Qwen2.5-Omni-7B/blob/main/LICENSE). However, the card's own memory table gives a 31.11-GB theoretical BF16 requirement for only 15 seconds of video and warns that actual use is typically at least 1.2 times theoretical. It is therefore operationally deferred on a 32-GiB host. The 3B sibling is not a drop-in remedy: its official repository uses a custom `qwen-research` license, which this task may not accept on the user's behalf.

If Phi-4 fails the public/synthetic local smoke test, the defensible outcome is to omit joint-model annotation or revise the engineering plan—not to invoke a hosted endpoint.

## Visual disagreement-judge role

The judge should receive only fixed local frames and source-language candidate strings, use a frozen output schema, and return `AGREE`, `DISAGREE`, or `UNDECIDABLE` hypotheses. It must not overwrite the primary instrument or resolve truth by vote.

Florence-2-base remains the preferred judge because it is small, MIT-licensed, architecture-diverse, and exposes explicit detection/phrase-grounding tasks. Before restricted use, a public/synthetic test must establish that the frozen source-German candidate interface is parsed reliably. Failure to support German should yield `UNDECIDABLE` or removal, not translation-defined truth.

[Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) is a secondary high-capacity candidate. Its official card documents object localization, structured output, long-video understanding, and event localization under Apache-2.0. It still requires exact artifact provenance and a public/synthetic Apple-MPS memory smoke test. Because the existing primary visual instrument is also from the Qwen family, agreement may reflect shared architecture, training sources, tokenization, and prompt behavior rather than independent confirmation.

## Why model–model agreement cannot replace human evidence

Agreement measures consistency under the chosen instruments, not correctness. Candidate models can share training data, architecture families, tokenizers, prompts, ASR hypotheses, VAD boundaries, selected frames, and systematic blind spots. Those common causes make their errors correlated and invalidate the interpretation of a vote as independent reliability.

A second model is useful for:

- disagreement and uncertainty flags;
- candidate unions/intersections;
- sensitivity ranges with each instrument reported separately;
- prioritizing a blinded human audit without using model confidence to select the frozen primary sample.

It cannot provide:

- human source-language WER/CER;
- true child versus non-child source role;
- visible-reference or null truth;
- inter-human reliability;
- publication-grade ChildLens evaluation labels.

No pooled consensus label should be reported as ground truth. The raw hypotheses stay quarantined, and repository outputs may contain only permitted cell-suppressed aggregates.

## Hosted-model exclusion

GPT-5.6 Luna, Codex, and every hosted/cloud model are categorically ineligible to inspect or annotate restricted ChildLens frames, audio, transcript text, filenames, timestamps, identifiers, or model-label payloads. The signed processing boundary prohibits third-party/cloud processing. Capability does not override that boundary. Hosted models may write generic code or analyze public documentation only when they receive no restricted input or output; they cannot serve as content annotators, translators, validators, or disagreement judges.

## Frozen v1.3.1 recommendation

If a qualified German annotator is available and authorized, retain a blinded source-German audit: the human works from raw local audiovisual material, records transcript/timing/role/referential judgments without predictions, locks the record, and only then reveals model proposals for agreement measurement. The role-based local shortlist is VAD plus Whisper/WhisperX for speech timing, VTC conditionally for voice type, Phi-4 conditionally for joint proposals, and Florence-2 first/Qwen2.5-VL second for visual disagreement.

If no qualified German annotator is available, freeze **DESCRIPTIVE_PSEUDO_ONLY**. Keep every model's aggregates separate, disclose that no model–human or inter-human reliability exists, prohibit primary ChildLens lexical evaluation, and use only simulator-oracle labels for later causal evaluation. This branch supports candidate generation and bounded aggregate calibration; it does not establish the lexical-grounding feasibility gates that require human truth.
