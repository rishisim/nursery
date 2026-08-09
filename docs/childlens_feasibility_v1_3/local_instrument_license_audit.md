# ChildLens v1.3 local-instrument and license audit

Version: `childlens-local-instrument-license-audit-v1.3.1`  
Audit date: 2026-07-22  
Scope: public software/model metadata only; no restricted ChildLens access or execution

## Disposition

The following fixed stack is approved for **public acquisition, hash verification, nonrestricted synthetic smoke tests, and later offline use inside the restricted quarantine**:

| Function | Fixed component | License disposition | Device and role |
| --- | --- | --- | --- |
| speech/VAD segmentation | Silero VAD `6.2.1`, bundled ONNX model | code and bundled weights: MIT; public, ungated, no click-through | ONNX Runtime CPU, 1 worker; boundary hypotheses only |
| multilingual ASR and timing | `whisper.cpp` `v1.9.1` plus unquantized `ggml-large-v3-turbo.bin` | code: MIT; upstream Whisper code/weights and converted model repository: MIT; public, ungated, no click-through | one Metal process; language, transcript, segment, and experimental word-timing hypotheses |
| anonymous speaker consistency | SpeechBrain `v1.1.0` plus `spkrec-ecapa-voxceleb` | code and model repository: Apache-2.0; public, ungated, no click-through | CPU; anonymous within-item cluster aid only |
| conservative acoustic role aid | librosa `0.11.0` pYIN and fixed local signal rules | code: ISC; no weights | CPU; F0/voicing evidence, never a direct child/adult fact |
| primary visual/referential candidates | `Qwen/Qwen2-VL-2B-Instruct` at a pinned repository commit, Transformers `4.53.3`, and PyTorch `2.7.1` | model repository and Transformers: Apache-2.0; PyTorch: BSD-3-Clause-style license; public, ungated, no click-through | observed synthetic MPS smoke pass; batch size 1; multilingual, multi-frame object/action/reference proposals |
| localization fallback | `microsoft/Florence-2-base` at a pinned repository commit, loaded through integrated Transformers `4.57.2` | model repository: MIT; Transformers: Apache-2.0; public, ungated, no click-through | optional engineering fallback for explicit object/region/phrase-grounding tasks |

No component above requires a model-card acceptance click, access token, contact-information disclosure, or gated repository approval. Public acquisition can therefore proceed autonomously before restricted processing. This is a technical license inventory, not legal advice; the signed ChildLens agreement remains controlling for the corpus.

The required stack has now been reconciled against the public local cache. Exact Silero 6.2.1 ONNX, an exact-tag whisper.cpp 1.9.1 Metal build with the unquantized large-v3-turbo model, and the pinned Qwen checkpoint each passed a synthetic/public smoke test while an outbound-network deny was active. These checks used no ChildLens input and do not certify a restricted run. Optional SpeechBrain, librosa, and Florence components remain nonrequired unless activated under their content-independent rules.

## Fixed artifacts and provenance

### 1. Silero VAD 6.2.1

- Official source tag: [`v6.2.1`](https://github.com/snakers4/silero-vad/releases/tag/v6.2.1), commit `7e30209a3e901f9842f81b225f3e93d8199902b1`.
- License: [MIT](https://github.com/snakers4/silero-vad/blob/v6.2.1/LICENSE). The project states that the released VAD has no telemetry, keys, registration, expiration, or vendor lock.
- PyPI wheel: `silero_vad-6.2.1-py3-none-any.whl`, SHA-256 `09de93c4d874bb19c53e62a47dd38be5f163cedad2b5599583231f2a84ef79cb`.
- Selected weight: `src/silero_vad/data/silero_vad.onnx`, 2,327,524 bytes, SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` at the pinned commit.
- Runtime: ONNX Runtime CPU `1.23.2` under the isolated CPython `3.10.20` instrument environment. The selected macOS arm64 wheel is `onnxruntime-1.23.2-cp310-cp310-macosx_13_0_arm64.whl`, 17,195,934 bytes, SHA-256 `a7730122afe186a784660f6ec5807138bf9d792fa1df76556b27307ea9ebcbe3`. This is the newest compatible release exposed for the fixed CPython/architecture pair by the authoritative package index during reconciliation; the initially audited `1.27.0` build was not available for that pair. ONNX Runtime is [MIT licensed](https://github.com/microsoft/onnxruntime/blob/main/LICENSE).
- Fixed operating mode: locally decoded mono 16-kHz audio, ONNX CPU execution, one thread per process. Never call `torch.hub` or a remote model identifier.
- Scientific limit: VAD output is a boundary hypothesis. It neither establishes an utterance nor a speaker role and must be scored against the locked author audit.

The exact ONNX file and wheel are present and hash-verified. A zero-signal structural smoke under an outbound-network deny loaded the CPU execution provider, produced the expected two-output structure, and returned finite arrays. The retained two-file public cache manifest has aggregate SHA-256 `68cb8972c6439d6c77ca79053b4d2ec222fe1fbcac4a214b4d7a1c96ea4990c0`. The older cached whisper.cpp-specific Silero 6.2.0 GGML file is explicitly unselected.

Silero documents 8/16-kHz operation, multilingual training breadth, and efficient one-thread CPU inference in its [official repository](https://github.com/snakers4/silero-vad). Its model is appropriate for content-independent full-pilot speech candidate generation on this host.

### 2. whisper.cpp 1.9.1 and Whisper large-v3-turbo

- Official source tag: [`v1.9.1`](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.9.1), commit `f049fff95a089aa9969deb009cdd4892b3e74916`.
- Code license: [MIT](https://github.com/ggml-org/whisper.cpp/blob/v1.9.1/LICENSE).
- Fixed model repository: `ggerganov/whisper.cpp`, revision `5359861c739e955e79d9a303bcbc70fb988958b1`, public, ungated, model-card license MIT.
- Fixed model artifact: `ggml-large-v3-turbo.bin`, 1,624,555,275 bytes, SHA-256 `1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69`.
- Upstream basis: OpenAI's [Whisper repository](https://github.com/openai/whisper) says both code and model weights are MIT licensed. The turbo checkpoint is multilingual, has 809M parameters, and has an approximate 6-GB memory requirement, which fits the 32-GiB host.
- Fixed build scope: build only the CLI needed for local file inference, with Metal enabled; do not build or launch server examples. Hash the source tree, compiler/toolchain receipt, CMake configuration, and final executable.
- Fixed inference posture: local model path only; transcription rather than translation; deterministic decode configuration; at most one Metal-heavy process; all JSON/text output stays in quarantine.
- Scientific limit: `whisper.cpp` word-level timing is experimental. The model's official documentation warns about hallucination, repetition, and uneven performance across languages and accents. Language identification, text, and timing are proposals until the blinded author audit measures WER/CER and declared-tolerance boundary agreement.

The preserved newer source checkout was not reset or treated as the runnable instrument. A separate clean detached worktree was pinned to commit `f049fff95a089aa9969deb009cdd4892b3e74916` (Git tree `f49541eaed447bce9b5e3598cc7a487ce5e54678`) and built only for `whisper-cli` with CMake `4.4.0`, Apple clang `21.0.0`, release mode, Metal enabled, tests disabled, and server disabled. The resulting CLI is 845,064 bytes with SHA-256 `9613b31e5380c184ae29ccb1d4046953d7037e8eb55308c9f1a34f145143b892`; its 19-entry executable/dylib bundle manifest has aggregate SHA-256 `9b13133224ff0e86ca48b3ab1138864080d9034718650d564eb91e665c2ee252`. A network-denied run on whisper.cpp's public JFK sample produced parseable, nonempty JSON. No transcript content was exported. The cached `q5_0` model remains present but unselected.

The unquantized artifact is selected instead of `q5_0` to avoid an unnecessary accuracy concession; it remains small enough for this host. The [whisper.cpp model documentation](https://github.com/ggml-org/whisper.cpp/blob/v1.9.1/models/README.md) documents the preconverted model route and Metal-oriented local use.

### 3. SpeechBrain ECAPA and deterministic acoustic evidence

- SpeechBrain source tag: [`v1.1.0`](https://github.com/speechbrain/speechbrain/releases/tag/v1.1.0), commit `36c180c7bfad3bf5c48bd76a24799812952c4565`.
- Code license: [Apache-2.0](https://github.com/speechbrain/speechbrain/blob/v1.1.0/LICENSE).
- Model repository: [`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb), revision `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`, public, ungated, repository license Apache-2.0.
- Principal weight: `embedding_model.ckpt`, 83,316,686 bytes, SHA-256 `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2`.
- Accompanying classifier weight: `classifier.ckpt`, 5,534,328 bytes, SHA-256 `fd9e3634fe68bd0a427c95e354c0c677374f62b3f434e45b78599950d860d535`.
- Serialization control: these official checkpoint files are not safetensors. Load only through the pinned SpeechBrain path in the isolated instrument runtime after exact hash verification. Do not execute arbitrary or unpinned checkpoint files.
- Offline control: use a local snapshot path with SpeechBrain fetching configured `allow_network=False`; never pass a Hub identifier during restricted execution.
- Output scope: anonymous segment embeddings and within-item clusters may exist only in quarantine. They must be discarded from any learner-facing export.

The ECAPA model was trained for speaker recognition on VoxCeleb and does **not** determine `CHILD`, `NON_CHILD`, or `OVERLAP`. It may only stabilize anonymous speaker consistency. A deterministic pYIN/F0 and voicing summary may be computed with [`librosa.pyin`](https://librosa.org/doc/latest/generated/librosa.pyin.html) from librosa `0.11.0` (ISC; wheel SHA-256 `0b6415c4fd68bff4c29288abe67c6d80b587e0e1e2cfb0aad23e4559504a7fa1`). Any local mapping from cluster/acoustic evidence to a role hypothesis must be frozen before author labels and must prefer `UNCERTAIN` rather than overclaim a semantic role.

This conservative design avoids a pretrained gender/age classifier. In particular, `audeering/wav2vec2-large-robust-6-ft-age-gender` was not selected because it jointly infers unnecessary sensitive attributes and is CC-BY-NC-SA-4.0 rather than permissively licensed.

### 4. Qwen2-VL-2B-Instruct primary visual/referential instrument

- Official Qwen model repository: [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct), revision `895c3a49bc3fa70a340399125c650a463535e71c`, public and ungated.
- Model repository license: [Apache-2.0](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/895c3a49bc3fa70a340399125c650a463535e71c/LICENSE). Qwen's [official Qwen2-VL release post](https://qwenlm.github.io/blog/qwen2-vl/) also states that the 2B and 7B variants are Apache-2.0 and documents multilingual visual-text understanding.
- Fixed weights:
  - `model-00001-of-00002.safetensors`, 3,988,609,112 bytes, SHA-256 `994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9`;
  - `model-00002-of-00002.safetensors`, 429,441,656 bytes, SHA-256 `92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833`.
- Local revision receipts: all 13 retained authoritative download metadata receipts name the same pinned revision; no mixed-revision evidence was found.
- Full retained public cache manifest: 40 entries and 4,429,622,733 aggregate entry bytes, SHA-256 `56f8306e4799e9cd9e5222831d5e856f9616a508f7b9761cd2de2ddd05eceb05` under the declared path/size/file-hash manifest scheme.
- Runtime code: Hugging Face Transformers `4.53.3`, tag commit `a5923d4de7df2fbd1f373dfcfe983216b79b6937`, [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.53.3/LICENSE). The PyPI wheel `transformers-4.53.3-py3-none-any.whl` has SHA-256 `5aba81c92095806b6baf12df35d756cf23b66c356975fb2a7fa9e536138d7c75`.
- Tensor runtime: Python `3.10.20`; PyTorch `2.7.1`, tag commit `e2d141dbde55c2a4370fac5165b0561b6af4798b`, under the project's [BSD-3-Clause-style license](https://github.com/pytorch/pytorch/blob/v2.7.1/LICENSE). The installed `torch-2.7.1-cp310-none-macosx_11_0_arm64.whl` is 68,630,914 bytes with SHA-256 `d72acfdb86cee2a32c0ce0101606f3758f0d8bb5f8f31e7920dc2809e963aa7c`. The paired `torchvision-0.22.1-cp310-cp310-macosx_11_0_arm64.whl` is 1,947,825 bytes with SHA-256 `3b47d8369ee568c067795c0da0b4078f39a9dfea6f3bc1f3ac87530dfda1dd56`.
- Installed-but-not-required helper: `qwen-vl-utils` `0.0.14`, Apache-2.0; wheel `qwen_vl_utils-0.0.14-py3-none-any.whl`, SHA-256 `5e28657bfd031e56bd447c5901b58ddfc3835285ed100f4c56580e0ade054e96`. The restricted runner does not import it because its URL-capable loaders are unnecessary for already-decoded local frames.
- Loader posture: instantiate the integrated Qwen2-VL model and processor from a local snapshot with `local_files_only=True`, `trust_remote_code=False`, and safetensors only, passing local PIL/in-memory frames directly.
- Device posture: the exact local snapshot passed a nonrestricted synthetic in-memory MPS load-and-one-token-generation smoke test under an outbound-network deny. It used `local_files_only=True`, `trust_remote_code=False`, and safetensors. Restricted operation remains batch size 1 with at most one MPS-heavy process and must repeat the network-denial sentinel before opening data.
- Fixed prompt/output posture: locally constructed, versioned prompts request a strict schema for coarse visible-candidate status, object/noun candidates, and action/verb candidates. Generation is deterministic. Free text, boxes, coordinates, and candidate labels remain restricted pseudo-labels.
- Scientific limit: output is a referential proposal, not reference truth. Multi-frame sampling must be frozen and content-independent; no confidence-, lexical-, salience-, or success-based resampling is permitted. Model-prompt language must follow the locally established language, with a frozen English control prompt allowed only if predeclared before author labels.

Qwen2-VL-2B is preferred over Florence-2-base as the primary instrument because it adds multilingual and multi-frame/video comprehension while remaining ungated, Apache-2.0, and small enough for a 32-GiB host. It is also preferred over Qwen2.5-VL-3B, whose official repository uses the Qwen Research License.

### 5. Florence-2-base localization fallback

- Official Microsoft model repository: [`microsoft/Florence-2-base`](https://huggingface.co/microsoft/Florence-2-base), revision `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac`, public, ungated.
- Model repository license: [MIT](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/LICENSE).
- Selected weight: `model.safetensors`, 463,221,266 bytes, SHA-256 `03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060`.
- Runtime code: Hugging Face Transformers `4.57.2`, tag commit `2915fb36cf8a48cad730f444b6057d18a6176d59`, [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.2/LICENSE). This version has an integrated Florence-2 implementation.
- Loader posture: instantiate the integrated `Florence2ForConditionalGeneration`/processor classes from a local snapshot with `local_files_only=True`, `trust_remote_code=False`, and safetensors only. Exclude `pytorch_model.bin` and repository custom-code files from the acquisition allowlist.
- Activation rule: use this fallback only if Qwen2-VL fails a nonrestricted synthetic compatibility/resource test or a required explicit-localization interface is structurally unavailable. It must not be activated or selected using ChildLens content, model confidence, apparent quality, or author labels.
- Device posture: MPS if a nonrestricted synthetic smoke test passes, otherwise CPU; batch size 1; at most one MPS-heavy process. Record the exact PyTorch wheel and device report in the runtime receipt.
- Fixed task family: object detection, dense-region captioning, detailed captioning, and caption-to-phrase grounding, as documented in the [official Transformers Florence-2 documentation](https://huggingface.co/docs/transformers/v4.57.2/model_doc/florence2).
- Scientific limit: generated text, boxes, regions, and phrase links are referential candidate proposals. They are not visible-reference truth. Frame selection must be frozen and content-independent; no confidence-, lexical-, salience-, or apparent-success-based resampling is permitted.

Florence-2-base is retained as an engineering fallback because it is a 0.23B-parameter, 463-MB, ungated, permissively licensed instrument with explicit detection and phrase-grounding tasks. Its compactness makes a bounded frame pass plausible on Apple Silicon while retaining the required object/region interface.

## Code-versus-weights license matrix

| Instrument | Code license | Weight/repository license | Gated or click-through | Autonomous public acquisition |
| --- | --- | --- | --- | --- |
| Silero VAD 6.2.1 | MIT | bundled in MIT repository | no | yes |
| ONNX Runtime 1.23.2 | MIT | no model weights | no | yes |
| whisper.cpp 1.9.1 | MIT | n/a | no | yes |
| Whisper large-v3-turbo GGML | conversion/runtime MIT; upstream Whisper MIT | model repository MIT | no | yes |
| SpeechBrain 1.1.0 | Apache-2.0 | ECAPA repository Apache-2.0 | no | yes |
| librosa 0.11.0 | ISC | no model weights | no | yes |
| Transformers 4.53.3 | Apache-2.0 | n/a | no | yes |
| PyTorch 2.7.1 | BSD-3-Clause-style project license | no model weights | no | yes |
| qwen-vl-utils 0.0.14, installed but not imported | Apache-2.0 | no model weights | no | yes |
| Qwen2-VL-2B-Instruct | integrated runtime Apache-2.0 | Qwen repository Apache-2.0 | no | yes |
| Florence-2-base fallback | separate integrated Transformers 4.57.2 runtime, Apache-2.0 | Microsoft repository MIT | no | yes |

Every full snapshot must receive a deterministic file manifest and aggregate SHA-256 receipt after download. The selected Qwen snapshot and exact whisper build bundle now have such receipts. The individual hashes above do not substitute for checking the sealed runnable artifact again immediately before restricted inference.

## Fail-closed acquisition and inference boundary

Public acquisition and restricted inference are separate phases.

### Public acquisition phase

1. Fetch only the pinned revisions from the authoritative sources above.
2. Do not accept a license, click a model agreement, authenticate to a gated repository, or substitute a community mirror.
3. Verify the listed artifact hashes and hash every retained file.
4. Build `whisper-cli`, create the isolated Python environment, and run only synthetic/public smoke tests.
5. Record package-lock/wheel hashes, compiler flags, model snapshot digest, and executable hashes.
6. Remove download helpers, Hub tokens, credential files, and remote model identifiers from the runnable restricted bundle.

### Restricted inference phase

1. Enforce a supervisor/OS-level outbound-network deny and run a denied-socket sentinel before opening any restricted input.
2. Set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, and `DO_NOT_TRACK=1`; clear proxy variables; use only absolute local model paths.
3. Pass `local_files_only=True` and `trust_remote_code=False` wherever supported; configure SpeechBrain `allow_network=False`.
4. Reject URLs, Hub identifiers, socket creation, subprocesses outside an executable allowlist, and output paths outside quarantine.
5. Keep raw media, frames, audio, transcripts, timestamps, item-level predictions, confidence values, boxes, embeddings, and cluster assignments in quarantine. Repository receipts contain only permitted aggregate coverage/resource facts.
6. Stop fail-closed if any model or library attempts network access, lazy file acquisition, unpinned code loading, or writing outside quarantine.

Environment variables are defense in depth, not proof of isolation. The OS/supervisor network-denial sentinel is mandatory.

## Learner and evaluation firewall

The annotation instruments are not learner ancestors. None of the following may enter corpus-local tokenizer construction, learner inputs, learner initialization, checkpoints, causal arms, or primary evaluation truth:

- instrument weights, tokenizers, vocabularies, token IDs, prompts, logits, confidence values, embeddings, hidden states, acoustic features, or cluster vectors;
- raw ASR, VAD, speaker, role, caption, object, action, region, or phrase-grounding hypotheses; or
- unaudited model-generated text, timestamps, labels, bounding boxes, or referential decisions.

Only the locked author-audited subset is human-labeled ChildLens evidence. Unreviewed full-pilot pseudo-labels may support permitted, nonidentifying aggregate simulator calibration and candidate generation, but never primary evaluation truth. The later synthetic causal evaluation must use simulator oracle labels.

## Alternatives not selected

- GPT-5.6 Luna, Codex, hosted APIs, cloud inference, and remote browser/model services are categorically excluded from restricted-content annotation because they would send ChildLens material or label payloads to a third party.
- `pyannote/speaker-diarization-community-1` is not selected. Its official model card requires accepting user conditions and sharing contact information before access, despite supporting offline use after download.
- PaliGemma is not selected because its official repository requires review and acceptance of Google's usage license.
- Qwen2.5-VL-3B-Instruct is not selected because its official model repository uses the Qwen Research License rather than the Apache-2.0 license used by the selected Qwen2-VL-2B artifact.
- SmolVLM2 is not selected as primary because its official card is English-centered, while the actual ChildLens language must not be assumed; it remains technically permissive but offers no advantage over the selected multilingual Qwen2-VL-2B for this audit.
- The audEERING age/gender checkpoint is not selected for the privacy and license reasons stated above.
- Quantized Whisper and community-converted VLM weights are not selected because the full permissive artifacts fit the host and avoid unnecessary conversion/quantization provenance and accuracy uncertainty.

## Residual limitations

- Actual language remains a human-established fact. Whisper's multilingual detection is diagnostic, not authoritative.
- Word timing, diarization, source role, and reference all require the blinded author audit. Model--human agreement is not inter-human reliability.
- ECAPA clusters do not semantically identify children or adults. The role heuristic must expose uncertainty and may have inadequate coverage; that is an empirical audit result, not a reason to tune the rule after labels.
- Qwen2-VL's multilingual claim does not establish accuracy for the encountered language, noisy egocentric video, or child-directed speech. The locked author audit must measure usable coverage and agreement.
- Florence-2 may be weaker for non-English phrase grounding and temporal actions than for static object proposals. If its engineering fallback is activated, detailed-caption/action candidates must be evaluated separately, with null/irrelevant/undecidable retained.
- The required artifacts have been installed/built and exercised only with public or synthetic inputs under network denial. No artifact in this audit has been run on restricted content. Restricted runtime readiness still requires the final instrument seal, denied-network sentinel, resource control, quarantine confinement, and privacy receipts immediately before opening data.

## Machine-readable record

The public provenance record is [`output/childlens_feasibility_v1_3/local_instrument_provenance.json`](../../output/childlens_feasibility_v1_3/local_instrument_provenance.json), and the cache/build reconciliation receipt is [`output/childlens_feasibility_v1_3/local_instrument_cache_reconciliation.json`](../../output/childlens_feasibility_v1_3/local_instrument_cache_reconciliation.json). Both contain only public package/model facts and no ChildLens payload.
