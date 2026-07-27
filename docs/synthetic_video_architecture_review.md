# Synthetic-video architecture review and frozen public pilot

**Status:** frozen decision record

**Evidence cut-off:** 2026-07-26

**Scope:** public sources and public/synthetic-only pilot design; no ChildLens or
BabyView media were accessed.

## Frozen learner and endpoint decision

The fixed learner is **EgoBabyVLM CLIP+ in `triple` mode**: image–text
InfoNCE steps interleaved 4:1:1 with BERT masked-language-model (MLM) and
DINOv2 DINO/iBOT self-supervised steps. The primary standardized endpoint is
**Machine-DevBench Lexical**, reported separately for nouns and adjectives and
as one prespecified macro-average of those two task accuracies. Chance is 50%.
CVCL is a conceptual predecessor and may be used only as an optional
interpretability reference; it is neither the primary learner nor endpoint.

The primary estimand is whether synthetic augmentation reduces learner-stage
real-data requirements:

| Arm | Allocated learner data |
|---|---|
| `Real-full` | \(H\) real hours |
| `Synthetic-full` | \(H\) synthetic hours |
| `Real-small` | \(r\) real hours |
| `Mixed` | \(r\) real + \(H-r\) synthetic hours, \(r < H\) |

Success requires `Mixed` to have **equivalent lexical-grounding performance**
to `Real-full` within a frozen two-sided equivalence margin and to be superior
to `Real-small`; the margin, alpha, confidence-interval procedure, seeds, \(H\),
and \(r\) must be preregistered from public/dummy engineering and blinded
real-only planning, before synthetic scores are opened. `Synthetic-full`
equivalence to `Real-full` is a stronger secondary result. This does not
establish “the same linguistic acquisition.”

There is a mandatory readiness gate before any synthetic-arm result is opened:
the identically configured real-only CLIP+ learner must (a) be meaningfully
above 50% on the frozen lexical aggregate, with its prespecified confidence
interval excluding 50% by the declared practical margin, and (b) show a
positive, uncertainty-qualified real-data learning curve over at least three
nested real-hour budgets. Failure stops the synthetic comparison and triggers
learner/protocol diagnosis using real-only results; it must not trigger
benchmark, seed, or arm-specific retuning.

Machine-DevBench images are generated (officially, FLUX-generated realistic
and cartoon styles), not held-out ChildLens frames. It therefore cannot be
called a held-out-real endpoint. The standardized primary endpoint is paired
with a required, separately reported **held-out-real ChildLens temporal
frame–utterance retrieval safeguard** defined below. Both remain in the
downstream evaluation family.

Consequences for the generator architecture remain:

- learner transcripts require reliable timing, but the learner does not consume
  waveforms;
- realistic speech remains necessary for blinded audiovisual fidelity;
- the canonical generator is modular video plus separately controlled,
  licensed German TTS driven from one episode plan;
- native joint audio remains diagnostic-only;
- the later public technical pilot remains capped at Wan 2.2 TI2V-5B and
  LTX-2/LTX-2.x.

## EgoBabyVLM compatibility pin and boundary

This review pins public upstream
[`facebookresearch/egobabyvlm@224621caf0628270b6115845ac75a65b984234a3`](https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3)
(2026-06-23), associated with
[EgoBabyVLM arXiv:2605.19130v1](https://arxiv.org/abs/2605.19130v1).
The repository is **CC BY-NC 4.0**, including a NonCommercial restriction.
It may support this noncommercial research with attribution and change notices;
it is not cleared for a future commercial product. Any product use requires a
separate implementation/license review rather than assuming research clearance.
Model and dataset licenses remain separate boundaries and must also be checked.

Exact inspected upstream environment bounds are Linux `linux-64`, Python
3.12.x, PyTorch 2.8.0, torchvision 0.23.0, CUDA 12.6, transformers 4.57.6,
torchmetrics 1.7.4, xformers 0.0.32.post2, and torchcodec 0.7.0. Some other
dependencies are ranges or unpinned Git `main` revisions; the later
implementation must archive the resolved Pixi lock and Git SHAs, not merely
repeat the declared ranges.

### Source-to-local component map

| Upstream pinned component | Nursery responsibility | Compatibility decision |
|---|---|---|
| `apps/data_preprocessing/{frames,transcription,manifests}` | Governed frame extraction, frozen ASR, split ledger, learner manifest | Adapt manifest builder to episode IDs and externally frozen child/session splits; do not use upstream random video-level splitting |
| `apps/baselines/clip/data/captions.py` + `configs/data/ego4d.yaml` | Load one utterance with one randomly selected in-window frame for train and deterministic validation frame behavior | Reuse interface; add no second learner implementation |
| `apps/baselines/clip/training` + `configs/mode/triple.yaml` | Four-arm CLIP+ training and checkpoint ledger | Vendor pinned code or a minimal attributed patch; freeze 4:1:1 interleave, initialization, schedules, and selection before governed runs |
| `apps/baselines/clip/modeling/{text_encoder,dinov2_ssl}.py` | English BERT MLM plus DINO/iBOT auxiliary objectives | Use identical arm-local text and images; no cross-arm auxiliary corpus |
| `apps/benchmark_creation/pipeline` | Build one corpus-grounded lexical asset | Run once on a frozen development vocabulary, never per arm; retain hashes and generation/filter manifests |
| `evaluation/multimodal/devbench` | Noun, adjective, and lexical macro-average | Evaluate every checkpoint on the same frozen generated asset |
| New thin Nursery adapter, later | Held-out-real temporal frame–utterance retrieval | Required because upstream Machine-DevBench has no real-frame transfer endpoint |

Upstream `triple` is a single training process that round-robins four
contrastive, one MLM, and one DINO/iBOT step, copying the SSL teacher backbone
into the CLIP vision tower after each SSL block. Its documented reference
invocation uses four processes/GPUs. **At the pinned commit, that example is
not runnable as written:** `triple` enables vision sync, but
`dinov2.pretrained_dir` defaults to null; trainer validation requires a
pretrained DINO/iBOT checkpoint and also requires the contrastive vision
backbone to match the SSL teacher's architecture and image size exactly. The
default Hub DINOv2 encoder plus bundled 224-pixel SSL config does not satisfy
that contract. This is an upstream compatibility issue, not evidence that
`triple` ran locally.

Nursery therefore freezes this initialization sequence: pin one public
DINOv2 ViT-B/14 prior; in public/dummy CUDA preflight, implement or verify the
smallest attributed bridge that loads those same hashed backbone weights into
both the bundled 224-pixel `CustomDINOv2VisionEncoder` and the DINO/iBOT
teacher/student checkpoint format expected by `pretrained_dir`; verify strict
state-dict sync before the first step. All arms and seeds begin from that same
public vision prior and public BERT-base prior, with only projection/MLM heads
seeded per the shared seed schedule. If a byte-identical, architecture-matched
bridge cannot be demonstrated, the CLIP+ protocol is no-go rather than falling
back to random or arm-specific initialization. This is a minimal documented
departure from the upstream example command, not from its trainer's enforced
contract.

Checkpoints contain model, optional MLM and DINO state,
optimizer/scheduler state, resolved Hydra config, epoch, step, and best
validation loss; `latest.pt` supports resume and `best.pt` follows validation
loss. Nursery will retain those mechanics but freeze a task-independent step
for primary evaluation rather than select on Machine-DevBench or held-out-real
test scores.

### Proposed learner input contract

Both real and synthetic episodes compile to the same flat learner manifest:

```json
{
  "episode_id": "opaque_stable_id",
  "source_kind": "real|synthetic",
  "split": "train|validation|test",
  "child_id": "governed_real_id|null",
  "session_id": "governed_session_id|synthetic_plan_id",
  "utterance_id": "episode_id:u0001",
  "utterance_de": "private_qa_only_german_text",
  "utterance_en": "offline_frozen_translation_or_asr_translation",
  "frame_filenames": ["relative/episode/frame_000123.jpg"],
  "timestamps_s": [4.1],
  "utterance_start_s": 3.7,
  "utterance_end_s": 4.8,
  "duration_credit_s": 6.0,
  "asr_pipeline_id": "frozen_local_pipeline_hash",
  "translation_pipeline_id": "frozen_local_pipeline_hash",
  "media_sha256": ["..."],
  "lineage_id": "governed_ledger_reference"
}
```

The direct upstream view presented to `Ego4DCaptionsDataset` contains only
`utterance` (from `utterance_en`), `frame_filenames`, `timestamps`,
`utterance_num`, `video_filename`, `transcript_filename`, and `num_frames`.
The richer governed sidecar is authoritative for split, privacy, accounting,
and lineage. Paths must be relative, stay inside the governed root, and resolve
only after access authorization. Synthetic rows use null `child_id`, but the
same schema, sampling, ASR, translation, and inclusion rules.

## Frozen protocol choices

### Language and tokenizer

**Decision: use an identical, offline German-to-English path for all real and
synthetic learner transcripts and build/evaluate Machine-DevBench in English.**
German audio remains the generated and fidelity-evaluation language. The
translator is frozen before data processing, runs only inside the governed
boundary, preserves utterance/timing IDs, and emits no network telemetry.
Synthetic oracle German text is QA metadata only: rendered synthetic audio must
pass through the same frozen ASR and translation path as real audio.

This is preferable now to a German-native benchmark because upstream lexical
caption templates, adjective antonyms, noun categories, filters, BERT model,
and evaluations are English-oriented; German case/gender/inflection would
require morphological lemmatization and native-speaker validation that the
project currently lacks. Model-derived German judgments cannot substitute for
that validation. Selecting translation narrows the claim to
**English-mediated lexical grounding from translated German speech**, with
translation/ASR noise part of the measurement pipeline. A future German-native
protocol is deferred, not run in parallel; it would require an authorized
German-speaking annotator, validated inflection/lemma and distractor rules, and
a separately preregistered study.

**Tokenizer decision: use the fixed public upstream
`bert-base-uncased` tokenizer and matching pretrained BERT-base initialization
in every arm, pinned by immutable Hugging Face revision and file hashes.**
Do not fit a tokenizer on ChildLens, synthetic text, combined full-real text,
or per-arm text. A public/dummy-fitted tokenizer would add unnecessary
departure and weak comparability; per-arm tokenizers would change model
capacity and leak arm identity. The public English tokenizer consumes no
ChildLens text and therefore cannot leak `Real-full` vocabulary into
`Synthetic-full`. This follows upstream exactly, but the immutable model
revision/hash is an additional Nursery reproducibility requirement.

### One common Machine-DevBench Lexical asset

Create exactly one English noun/adjective benchmark before learner training
from a **frozen development vocabulary**: public lexical resources plus only
the authorized generator-development/calibration corpus described in the
accounting section. Never derive it from `Real-full`, any arm-specific
manifest, or learner outcomes, and never regenerate it per arm. Freeze source
vocabulary/lemma list, part-of-speech and morphology processing, noun
categories, adjective antonyms, frequency-bin boundaries, prompts, image model
and weights, generation seeds, filters, final manifests, and hashes. Build both
official styles once; the realistic-style noun and adjective macro-average is
primary, with cartoon style diagnostic unless preregistration instead freezes
their average before any learner result.

Each score report includes per-concept accuracy and, for every arm/budget,
training exposure count after ASR/translation and frame filtering. Primary
inference uses the full frozen benchmark and reports exposure-stratified
sensitivity results (`0`, `1–k`, `>k`) without dropping zero-exposure concepts
post hoc. Vocabulary coverage is a property to report, not a reason to build an
easier arm-specific test. Test concepts and benchmark images/prompts are
firewalled from generator prompting, episode selection, retries, QA thresholds,
and learner checkpoint selection. Accidental overlap with naturally chosen
episode concepts is measured, not optimized.

The report must state that all Machine-DevBench visual stimuli are generated,
that the generator differs from or may resemble the training generator, and
that this creates an evaluation-domain confound. Results are standardized
generated-image lexical transfer, not held-out-real ChildLens performance.

### Learner fairness contract

- Start every seed/arm from byte-identical public BERT-base and the
  architecture-matched DINOv2 ViT-B/14 bridge specified above;
  projection/MLM heads use the same seeded initialization procedure. This
  common public prior is declared, hashed, and excluded from real-hour counts;
  no ChildLens-derived initialization pretraining is allowed.
- Keep `triple`, 4:1:1 interleave, DINO/iBOT configuration, MLM mask rate,
  augmentations, image resolution, optimizer, learning-rate schedule, batch
  sizes, precision, gradient accumulation, stopping step, and checkpoint rule
  identical. Validate that the common DINOv2 backbone and bundled SSL student
  architectures match before launch.
- Auxiliary data are arm-local: MLM lines are only that arm's allocated
  translated utterances and DINO/iBOT images are only that arm's allocated
  frames. Repetition needed to reach the common step budget is logged. No
  `Real-small` or `Synthetic-full` auxiliary objective may see `Real-full`.
- Sample frames at the frozen rate and upstream midpoint convention; use the
  same utterance-window rule and one seeded in-window frame draw per
  contrastive access. Store frame-list hashes and sampler seeds.
- Match credited input duration: every arm totals \(H\) hours except
  `Real-small` at \(r\). `Mixed` totals \(H\). Also match total optimizer steps
  and the count of each objective step across arms. Report unique
  utterances/frames, repetitions, accepted speech duration, and a secondary
  equal-unique-pair sensitivity analysis.
- Use at least three prespecified learner seeds shared across arms. Select one
  fixed optimizer step established without benchmark/test feedback; validation
  loss may diagnose training but must not create arm-specific stopping.
- Real audio and rendered synthetic audio use the same frozen local ASR,
  confidence threshold, utterance normalization, and offline translation.
  Oracle synthetic transcripts and plan labels are QA/lineage metadata only
  and cannot enter contrastive or MLM training.
- Resume only from a same-arm, same-seed checkpoint whose resolved config and
  data hashes match. Record all latest/epoch checkpoints in ignored governed
  run storage; retain only preregistered aggregate results and manifests in Git.

### Real-data accounting and estimand boundary

**Decision: the first study makes a learner-stage reduction claim conditional
on one fixed generator-development/calibration corpus \(C\), not an end-to-end
real-data reduction claim.** Before looking at final outcomes, freeze and
report \(C\)'s unique children, sessions, hours, permitted derived statistics,
and every use: vocabulary proposal, generator calibration, prompt/QA tuning,
benchmark construction, and fidelity reference. No record in \(C\) may enter
learner training or either evaluation. Final training/evaluation children and
sessions must be disjoint from \(C\).

The learner's `Real-small` and real portion of `Mixed` are the same nested
\(r\)-hour subset of the eligible training pool. They may not inherit
vocabulary, prompts, QA thresholds, translations, embeddings, or summaries
computed from the remainder of `Real-full`. The fixed generator may use only
\(C\), public inputs, and its episode plans to produce synthetic data. Thus an
\(r\)-hour arm never indirectly consumes \(H\) learner-pool hours. Report
learner real hours and \(C\) hours separately. Any later end-to-end reduction
claim requires a new protocol that charges \(C\) against the real-data budget;
it is not implied by this study.

### Held-out-real transfer safeguard

After authorized access, freeze a ChildLens temporal frame–utterance retrieval
test before training. Assign whole children to train/development/test where
sample size permits; otherwise the study is **no-go for confirmatory real-domain
claims** unless an approved leave-one-child-out design was preregistered.
Within each held-out child, assign whole sessions to exactly one split. Remove
near-duplicate/overlapping clips across sessions using timestamps and governed
hash/embedding duplicate checks. No frame, utterance, temporal neighbor, child,
or session from test may enter \(C\), generator development, benchmark
construction, learner training, thresholds, or checkpoint selection.

For each test utterance, choose a prespecified visible frame from its temporal
window and contrast the correct utterance against matched within-session
temporal negatives outside an exclusion buffer; run the reciprocal
utterance-to-frame retrieval as a secondary direction. Freeze negative count,
time-distance strata, utterance-length strata, frame rule, and Recall@1 /
mean-reciprocal-rank aggregation. Use the same asset and candidates for every
arm. Because current transcripts and visibility/alignment may be ASR- and
model-derived, label this endpoint **model-derived temporal alignment
transfer**, not referent ground truth or lexical accuracy. It becomes a
governed real-frame lexical subset only after independent authorized human
validation of referents and labels under a separate frozen annotation protocol.

### Compute, privacy, and governance

This Apple Silicon host can perform source review, JSON/schema tests, manifest
hashing, split-logic unit tests, configuration composition only if dependencies
are already safely available, and tiny dependency-light loader tests. It cannot
validate the official environment: upstream Pixi supports only `linux-64` and
declares CUDA 12.6; the trainer chooses CUDA or CPU, never MPS; the documented
`triple` recipe uses four GPUs; Machine-DevBench evaluation currently selects
`cuda` directly; benchmark image generation and filtering are GPU/SLURM
oriented. CPU code paths do not establish practical training support.

Approved compute is split:

1. Public/dummy CI and a tiny public-data triple/evaluation smoke test may run
   on institutionally approved or ordinary hosted CUDA because it contains no
   ChildLens material. First reproduce environment resolution, config,
   checkpoint/resume, one step of each objective, and lexical evaluator wiring
   at the pinned commit. A single-GPU triple attempt is exploratory until it is
   shown numerically equivalent to the reference distributed semantics.
2. ChildLens frames, audio, transcripts, translations, embeddings, prompts
   derived from them, and restricted metadata may run only on an approved
   governed CUDA system with storage, access, logging, retention, and egress
   controls. Ordinary hosted GPU services and external APIs are prohibited.
   If no suitable governed CUDA path exists, the empirical study is no-go.

## Decision table

| Topic | Frozen now | Deferred until authorized ChildLens access | No-go trigger |
|---|---|---|---|
| Learner | CLIP+ `triple`, pinned upstream, common public BERT/DINOv2 prior and strict initialization bridge | Governed CUDA runtime reproduction | Cannot prove matched strict sync, all three objectives, and exact resume |
| Language/tokenizer | Offline German ASR → English translation; public pinned `bert-base-uncased` | Local model hashes and translation audit on authorized samples | Any network egress or arm-specific processing |
| Primary benchmark | One frozen English Machine-DevBench Lexical asset for all arms | Vocabulary freeze using only authorized \(C\); generation/filter audit | Per-arm asset, learner-test steering, or unfrozen test |
| Readiness | Real-only above-chance and positive learning curve before unblinding synthetic | Margins, \(H/r\), nested budgets, seeds from governed design | Gate fails; synthetic outcomes remain sealed |
| Real safeguard | Child/session-disjoint temporal frame–utterance retrieval | Feasible split counts, duplicate audit, candid model-derived label | No independent test children or irreparable clip leakage |
| Accounting | Conditional learner-stage claim; \(C\) separate and excluded | Freeze \(C\) ledger and eligible learner pool | `Real-small` indirectly uses the rest of `Real-full` |
| Compute/privacy | Public/dummy hosted CUDA allowed; restricted data only governed CUDA | Infrastructure approval and egress test | Restricted material would leave governed boundary |
| Licensing | CC BY-NC research boundary and attribution | Institutional review of all model/data licenses | Commercial use without separate permission/reimplementation |

## Phased implementation sequence and stop rules

1. **Public compatibility package.** Vendor the pinned upstream revision or
   record an immutable dependency, freeze lockfile SHAs and licenses, define
   the richer manifest plus upstream view, and add dependency-light schema,
   split, accounting, and config tests. *Done when:* dummy real/synthetic rows
   produce identical-shape upstream manifests and no restricted path exists.
2. **Public/dummy CUDA preflight.** On approved public-data CUDA, resolve the
   official environment, build and verify the common DINOv2 initialization
   bridge, and run a tiny self-authored step for contrastive, MLM, and DINO/iBOT
   plus checkpoint/resume and a tiny generated lexical evaluator wiring test.
   No scientific metric is retained. *Stop* on weight mismatch, loss/config,
   architecture-sync, resume, or distributed/single-GPU inconsistency. *Done
   when:* strict pre-step sync, logs, resolved configs, hashes, and proof limits
   are reviewed.
3. **Governance and preregistration.** Approve governed CUDA, local ASR and
   translator; freeze \(C\), eligible child/session split, \(H/r\), nested
   real-only budgets, margins, seeds, fixed step, sampling, statistics,
   exposure reporting, and blinding. *Stop* if independent test children,
   permissions, or compute isolation are inadequate. *Done when:* signed
   lineage/accounting and analysis plans exist without opening synthetic scores.
4. **Common evaluation assets.** Using only \(C\) and public resources, freeze
   one Machine-DevBench Lexical asset; from evaluation-only children/sessions,
   freeze the real temporal-retrieval asset. Neither may steer generation.
   *Done when:* manifests, hashes, filters, overlap audits, and model-derived
   labeling are locked for every arm.
5. **Real-only readiness.** Run the preregistered real-only learning curve and
   validate the above-chance/positive-curve gate. *Stop* and keep synthetic
   results sealed if it fails. *Done when:* the gate decision and uncertainty
   report are frozen without protocol changes.
6. **Generator and learner datasets.** Freeze one generator after the separate
   public pilot, create the synthetic allocation without test-concept
   targeting, run identical audio→ASR→translation, and compile/hash all four
   arm manifests. *Stop* on leakage, hour/accounting mismatch, or QA selection
   informed by downstream tests. *Done when:* duration, exposure, lineage, and
   objective-step ledgers reconcile.
7. **Final four-arm evaluation.** Train all frozen arm×seed runs, select the
   common fixed-step checkpoints, evaluate once on the common lexical and
   held-out-real assets, then test Mixed equivalence to `Real-full` and
   superiority to `Real-small`; treat `Synthetic-full` parity as secondary.
   *Done when:* confidence intervals, exposure strata, real-transfer safeguard,
   failures, and conditional-\(C\) claim are reported.
8. **Separate blinded fidelity/cost evaluation.** With independent randomized
   samples and raters blinded to source/model, run the already frozen
   audiovisual fidelity protocol and report acceptance rate, retries, wall/GPU
   time, energy if available, storage, and cost. Its sample/order and judgments
   stay separate from learner test assets and checkpoint decisions. *Done
   when:* fidelity/cost results are linked by lineage but cannot alter the
   four-arm analysis.

## Bounded public/dummy preflight performed

At the evidence cut-off, a shallow OS-temporary checkout at the pinned commit
was inspected and removed. No weights, datasets, checkpoints, media, or
environments were downloaded. A self-authored manifest object with real and
synthetic variants was checked using only the Python standard library for
required keys, legal `source_kind`, equal field shape, relative frame paths,
numeric timestamps, and JSON round-trip; the temporary file was removed.

This proves only that the proposed adapter contract is syntactically coherent
and can project both source types to the same upstream field shape. Static
inspection also maps trainer/config/preprocessing/evaluation interfaces. It
does **not** prove imports, dependency resolution, model/tokenizer loading,
CUDA/MPS runtime compatibility, numerical correctness, DINO/iBOT training,
checkpoint resume, Machine-DevBench generation/evaluation, or any scientific
performance.

## Candidate matrix

Claims below refer to official repositories, model cards, or technical reports
as of the evidence cut-off. “Storage” is a planning estimate for weights plus
required encoders, not a quoted download size; caches and outputs would live
outside Git. Cost is a relative pilot estimate because provider prices and
optimized kernels change.

| Family | Public code / weights | License and research constraint | Official hardware path | Memory / practical clip | Continuation, keyframes, controls | Audio | Privacy, storage, expected pilot cost | Screen |
|---|---|---|---|---|---|---|---|---|
| **LTX-2 / current LTX-2.x** | Official inference/training code and weights are public; Diffusers/ComfyUI ecosystem support is documented. Official LTX Desktop runs LTX-2.3 22B locally | LTX-2 Community License, not Apache: use restrictions and a paid-license threshold for entities with ≥$10M annual revenue; institutional legal review required | Two distinct official paths: LTX Desktop local inference on Apple Silicon/macOS 13+ via MPS with ≥15 GB **free RAM at launch**, and Python training on Linux/CUDA. Intel Macs fall back to API-only mode. LTX Desktop is beta | Joint model is roughly 19–22B parameters depending release; official training guidance recommends 80 GB VRAM, with a 32 GB INT8 training config. Native joint clips are documented around 10 s; older LTX-Video has distinct long-shot modes. Desktop requires substantial model/output disk space; plan ~45–70 GB local model assets | Official multiple-keyframe conditioning, I2V, forward/backward extension, retake/video-to-video, camera/control LoRAs; LTX Desktop exposes T2V, I2V, A2V, retake, and LoRAs | Native synchronized audio/video and audio-conditioned modes. Exact German transcript reproduction is unproven and must be tested separately | Local MPS or CUDA video inference can preserve media privacy. Desktop recommends free cloud text encoding, which sends prompts to LTX; its documented fully local text encoder is clearly supported on Windows but not clearly on macOS. Mac throughput is unverified: expect no provider charge for local inference but substantial wall time/energy; retain the earlier planning range of roughly 1–3 high-memory CUDA GPU-hours per accepted scene only for the CUDA path | **Advance**, chiefly for keyframes/extension, official Mac feasibility, and a bounded comparison of joint versus modular audio; license, privacy, and reproducible parameter/seed capture remain gates |
| **Wan 2.2** | Official code and T2V-A14B, I2V-A14B, TI2V-5B, S2V-14B, and Animate weights are public | Repository and released models state Apache-2.0 | Official path is PyTorch/CUDA; no official MPS path | Official A14B examples call for ≥80 GB VRAM. TI2V-5B is the materially cheaper 720p/24 fps candidate; quantization/offload are possible but not the canonical claim. Nominal demonstrations are short clips, so 60 s requires planned segments | T2V/I2V in official code; first-frame image conditioning. No official arbitrary multi-keyframe or general continuation contract comparable to LTX; S2V adds audio/pose control but is a separate 14B model | T2V/TI2V are silent. S2V takes speech/audio (and optionally pose video) to drive a depicted speaker, not a general egocentric soundtrack generator | Local CUDA preserves privacy. A14B variants can require >50 GB each; TI2V-5B is a smaller asset. Moderate/high cost; budget roughly 0.5–2 high-memory GPU-hours per accepted scene with fixed retries | **Advance TI2V-5B only**; do not pilot A14B or S2V in this bounded screen |
| **HunyuanVideo 1.5** | Official code, checkpoints, Diffusers, ComfyUI and optimized paths are public | Tencent community model license must be reviewed; not represented here as Apache-2.0. Research use is available subject to its terms | Official requirement is Linux, NVIDIA CUDA, Python ≥3.10; no official MPS path | 8.3B DiT; official minimum is 14 GB VRAM with offload for 10 s 720p, with optional 1080p super-resolution. Plan ~20–35 GB assets | Official T2V and I2V. Sparse attention targets longer sequences, but no equally mature official multi-keyframe/forward-backward extension interface was verified | No native general soundtrack/speech generation in the base model | Local CUDA preserves privacy; lowest documented VRAM threshold of the four larger current families. Moderate storage and low/moderate pilot cost | **Exclude**: efficient, but less control evidence than the two selected families and the pilot cap is two |
| **CogVideoX / 1.5-5B** | Official SAT and Diffusers code and weights are public | Code and 2B weights are Apache-2.0; 5B/1.5 weights use the separate CogVideoX license and require review | Official CUDA-oriented path; quantization/offload documented. No official MPS support | 1.5-5B supports up to 10 s at 1360×768; official table reports 76 GB BF16 before memory-saving paths. Earlier 2B can run much smaller. Plan ~15–30 GB assets | T2V and I2V. Official arbitrary-resolution I2V exists; interpolation and richer keyframe workflows are community rather than the official scientific interface | Silent | Local inference preserves privacy. Moderate storage; moderate cost with offload, but older quality/control surface | **Exclude**: useful reproducibility fallback, but older and less controllable than selected candidates |

Sources: [LTX-2 repository](https://github.com/Lightricks/LTX-2),
[LTX-2 license](https://github.com/Lightricks/LTX-2/blob/main/LICENSE),
[LTX-Video repository](https://github.com/Lightricks/LTX-Video),
[LTX Desktop repository](https://github.com/Lightricks/LTX-Desktop),
[LTX trainer requirements](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-trainer/docs/quick-start.md),
[Wan 2.2 repository](https://github.com/Wan-Video/Wan2.2),
[Wan 2.2 license](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt),
[HunyuanVideo-1.5 repository](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5),
[HunyuanVideo-1.5 report](https://arxiv.org/abs/2511.18870),
[CogVideo repository](https://github.com/zai-org/CogVideo), and
[CogVideoX report](https://arxiv.org/abs/2408.06072).

### Interpretation of the screen

Apple Silicon is an officially supported **LTX-2.3 inference** target through
LTX Desktop, not an LTX training target and not an official Wan target. A Mac
feasibility preflight must record chip, unified memory, free RAM at launch,
macOS/Desktop/model revisions, model storage, whether MPS local mode actually
engaged, wall time, memory pressure/weight streaming, and every API-backed
feature used. “Local generation” is not equivalent to “offline”: the
recommended cloud text encoder sends prompts to LTX, and the current
documentation does not clearly promise a fully local text encoder on macOS.
Restricted prompts or media therefore remain prohibited until a network-trace
privacy preflight establishes a fully local path. Community MLX/MPS ports remain
non-canonical.

The public pilot may run LTX through pinned LTX Desktop/MPS only if it exposes
and records all frozen prompts, conditioning, seeds, inference settings,
attempts, and outputs needed by the manifest. Otherwise LTX must use the pinned
official/Diffusers CUDA runner. Wan still requires local or institutionally
controlled CUDA. Closed services may be tested with the same public prompts
only as a non-canonical quality ceiling, outside the two-candidate decision and
without influencing selection.

Sixty seconds is an episode composition target, not a single model-call claim.
The canonical runner produces 5–10 s planned segments, carries forward explicit
state and approved boundary frames where supported, generates German speech
separately, and uses FFmpeg to align, mix, and mux. An episode manifest records
plan revision, source/license identifiers, model and code revisions, weights,
precision, hardware, prompts, negative prompts, seeds, configs, boundary
frames, every attempt including rejected attempts, QA, timings, storage hashes,
and compute/cost.

ComfyUI is excluded as the scientific runner: graph serialization helps
exploration, but Python/CLI calls to pinned official code or Diffusers provide
clearer dependency locking, batch semantics, manifest capture, and testing.

## Evidence table

| Work | Decision-relevant finding | Limitation for this project |
|---|---|---|
| [EgoBabyVLM](https://arxiv.org/abs/2605.19130v1) and [pinned code](https://github.com/facebookresearch/egobabyvlm/tree/224621caf0628270b6115845ac75a65b984234a3) (Lin et al., 2026) | Defines CLIP+ `triple`, corpus-grounded Machine-DevBench, preprocessing, manifests, checkpoints, and evaluation interfaces | English/CUDA-oriented; generated benchmark imagery is not held-out real data; pinned `triple` example needs the initialization repair and CUDA verification described above |
| [SAYCam](https://pmc.ncbi.nlm.nih.gov/articles/PMC8412186/) (Sullivan et al., 2021) | Longitudinal child headcam video includes natural speech and transcriptions; establishes the naturalistic audiovisual regime | Three children and a different age/language distribution; not ChildLens |
| [Grounded language acquisition through the eyes and ears of a single child](https://doi.org/10.1126/science.adi1374) (Vong et al., 2024) | CVCL learns word–referent alignment from 61 h using paired frames and transcribed utterances; directly operationalizes the claim | Single child; learner sees transcripts, not waveform; reported evaluations do not prove synthetic transfer |
| [CVCL supplementary material](https://gwern.net/doc/ai/nn/cnn/2024-vong-supplement.pdf) | Defines utterance windows, frame extraction, DINO visual initialization, and contrastive variants needed for reproduction | Exact reproduction still depends on data preprocessing and vocabulary choices |
| [Robustness of grounded word learning](https://arxiv.org/abs/2507.14749) (Vong & Lake, 2025) | Extends the setup to automated transcripts and 500+ h across all SAYCam children; supports learner robustness beyond one child | Automated transcription adds noise; still English SAYCam and not a synthetic-data study |
| [BabyView dataset](https://arxiv.org/abs/2406.10447) (Long et al., 2024/25) | 493 h, high resolution, wide vertical field of view; visual and language transfer scales but remains hard on naturalistic data | Public paper only here; data are unavailable and cannot be an empirical target |
| [DevBench](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8d987b2981388c99c7eab6095d1d29fd-Abstract-Datasets_and_Benchmarks_Track.html) (Tan et al., 2024) | Seven lexical/syntactic/semantic tasks and human response patterns make a useful external developmental battery | Not a training method; task vocabulary coverage can invalidate results, as prior Nursery work found |
| [Learning Video Representations without Natural Videos](https://arxiv.org/abs/2410.24213) (Yu et al., 2024/26) | Procedural synthetic motion plus natural images closes 97.2% of the UCF101 pretraining gap and improves several OOD tests | Adult action benchmarks and deliberately simple synthetic processes; no language grounding or child-view fidelity |
| [Are Synthetic Data Useful for Egocentric Hand-Object Interaction Detection?](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08953.pdf) (Leonardi et al., 2024) | Synthetic egocentric HOI data can help real-data learning, motivating mixed curves and real-test evaluation | Detection/segmentation, mostly image-level, simulator domain differs from generative video |
| [Leveraging Synthetic Data for Enhancing Egocentric Hand-Object Interaction Detection](https://arxiv.org/abs/2603.29733) (Leonardi et al., 2026) | With 10% real data, aligned synthetic data improves overall AP on three real benchmarks; benefit tracks object/grasp/environment alignment | Synthetic-only remains poor; dense HOI labels and 3D assets give supervision unavailable here |
| [EgoInteract](https://arxiv.org/abs/2605.18214) (Leonardi et al., 2026) | Controlled camera, body/hand motion, object manipulation, and scene composition improve multiple real egocentric tasks | Simulator pipeline and dense annotations; not evidence that photorealistic prompt generators achieve the same control |
| [HandsOnWorld](https://arxiv.org/abs/2607.02075) (Chen et al., 2026) | Camera-disentangled 3D hand control addresses ego-motion/hand entanglement and evaluates hand control in diverse scenes | Very recent preprint; specialized hand trajectories are not required inputs for this branch |
| [EgoControl](https://openaccess.thecvf.com/content/CVPR2026/html/Pallotta_EgoControl_Controllable_Egocentric_Video_Generation_via_3D_Full-Body_Poses_CVPR_2026_paper.html) (Pallotta et al., 2026) | Explicit 3D pose couples head-camera dynamics and articulated body motion in future-frame generation | Requires pose sequences and context frames; quality evidence does not establish ChildLens fidelity |
| [Exo2Ego-V](https://proceedings.neurips.cc/paper_files/paper/2024/file/f5a8b5e5d007e66c929b971c2bc21d76-Paper-Conference.pdf) (Ponimatkin et al., 2024) | Viewpoint transformation is itself a difficult conditional-generation problem | Requires exocentric source video, outside the planned input contract |
| [E3C](https://e3c-videogen.github.io/) (2026) | Environmental memory and pose control target scene persistence across ego/exo generation | New specialized system; public evidence does not justify adding a third pilot candidate |
| [VBench](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf) (Huang et al., 2024) | Separates temporal quality, subject/background consistency, motion, aesthetics, and prompt fidelity | Standard prompts and aggregate scores are not a ChildLens distributional test |
| [VBench++](https://arxiv.org/abs/2411.13503) (Huang et al., 2024/25) | Extends diagnostic dimensions and trustworthiness coverage across generation settings | Automatic evaluators inherit model bias and can miss hand/contact failures |
| [VBench 2.0](https://arxiv.org/abs/2503.21755) (Zheng et al., 2025) | Intrinsic faithfulness expands evaluation toward physics, commonsense, anatomy, and compositionality | Broad benchmark; its mean score cannot validate a specific egocentric distribution |
| [VideoPhy](https://github.com/Hritikbansal/videophy) (Bansal et al., 2024/25) | Human-grounded semantic and physical-commonsense evaluation exposes failures hidden by visual quality | Prompt/action distribution is not child-centered and the autorater is not sufficient alone |
| [VideoPhy 2](https://videophy2.github.io/) (Bansal et al., 2026) | Fine-grained, action-centric physical rules support contact and transition diagnostics | Newly released; VLM-derived rules and ratings need human confirmation |
| [LTX-2 technical report](https://arxiv.org/abs/2601.03233) (Lightricks, 2026) | Joint audio/video cross-attention and temporal conditioning motivate a bounded native-audio diagnostic | Vendor-authored comparisons; does not establish exact German wording or egocentric fidelity |
| [HunyuanVideo-1.5 report](https://arxiv.org/abs/2511.18870) (Tencent, 2025) | 8.3B architecture and sparse attention reduce 10 s 720p cost, informing the efficiency comparison | Vendor evaluation and no required multi-keyframe/audio control |
| [CogVideoX](https://arxiv.org/abs/2408.06072) (Yang et al., 2024) | Public expert-transformer T2V baseline with reproducible code/weights | Older control surface and no native audio |

## Canonical architecture

1. A versioned JSON or YAML episode plan is the sole semantic source of truth.
   It declares persistent entities, licensed assets, scene state, camera
   behavior, actions, utterance text/speaker, word-level target times, visible
   referent intervals, segment boundaries, and prohibited content.
2. A deterministic compiler emits segment prompts, conditioning frames where
   supported, a TTS script/timeline, FFmpeg composition instructions, and a
   schema-validated manifest skeleton. Generated prompts are recorded, never
   silently edited.
3. The selected official model code or pinned Diffusers pipeline generates
   short silent segments. A persistent-state ledger records object attributes,
   location, hand occupancy, action pre/postconditions, and boundary frames.
4. A separately licensed, non-cloned German TTS voice renders frozen text.
   Store engine/version, voice license identifier, text, pronunciation
   overrides, requested timing, realized word times, and any time-stretch.
   Voice cloning and identifiable-child imitation are prohibited.
5. FFmpeg performs deterministic trim, approved crossfades, loudness control,
   ambience mixing, and muxing. It must not conceal failed action continuity.
6. Frozen automatic QA runs before human QA. Human raters are blinded to model
   and attempt number. Retry limits are fixed; failed attempts remain in the
   manifest so selective reporting is measurable.

The approximately 60 s episode is composed from an ordered plan of short
segments. Cross-segment identity is an empirical gate, not assumed solved by
prompt repetition. A later full study must freeze a single generator family
after the public pilot; it must not ensemble winners per scene.

## Frozen public/synthetic-only technical pilot

### Question and inputs

The pilot asks one question: **which of the two frozen model families can
execute the canonical modular pipeline with adequate egocentric interaction,
state continuity, camera motion, and timed word–referent alignment to justify a
governed downstream study?** It is not a leaderboard or a media-production run.

Use only self-authored plans, commercially reusable or CC0 reference images,
and synthetic TTS. No ChildLens/BabyView frames, derived prompts, statistics, or
restricted media are permitted. Use one neutral, licensed German adult voice.

Freeze four 5–10 s scenes, each with one target noun:

1. **Pick up:** child-height head camera; one hand reaches for a red cup,
   establishes contact, lifts it, and holds it; say “Das ist der Becher” while
   the cup is visible and stable.
2. **Transfer:** take a wooden block from the left hand with the right hand;
   preserve block color/shape before, during, and after transfer; name
   “der Bauklotz” during the post-transfer hold.
3. **Occlusion/persistence:** place a small toy car partly behind a box, move
   the head camera around the box, and reveal the same car; name “das Auto”
   within the reveal interval.
4. **Action transition/camera motion:** walk two steps toward a table, stop,
   grasp a spoon, and stir once in a bowl; name “der Löffel” after grasp and
   before stirring ends.

Each family receives exactly the same semantic plan. Model-specific prompt
syntax and conditioning are allowed only through the frozen compiler. Wan uses
TI2V-5B silent video. LTX uses its current public checkpoint in silent/modular
mode; on scene 1 only, it also gets one **diagnostic** joint-audio attempt with
the exact German line. That diagnostic does not count as a third candidate and
cannot improve LTX's modular score.

### Attempts, compute, and manifests

- Two families maximum; four scenes; three preregistered seeds per
  family/scene.
- At most two attempts per seed: the initial attempt and one retry using a
  single frozen corrective suffix selected from a predeclared taxonomy
  (`hand`, `identity`, `camera`, `transition`, `referent timing`, `safety`).
- Maximum: 48 modular video attempts (2 × 4 × 3 × 2), plus one LTX joint-audio
  diagnostic attempt per seed for scene 1 (3), and no substitutions.
- Generate at the closest common deliverable setting: 720p, 24 fps, 5–10 s.
  Record native resolution/fps and any deterministic conversion.
- All attempts, wall time, GPU type/hours, peak VRAM, energy if available,
  provider charge, output size, and QA are recorded. No attempt is deleted
  because it looks bad.

### Automatic diagnostics

Run the same frozen versions/configs on every attempt:

- schema and media conformance: duration, frame count, fps, resolution, audio
  presence, decode errors, manifest completeness;
- target-object detection/tracking confidence and embedding drift across the
  pre-action, contact, post-action, and occlusion/reveal intervals;
- hand count/anatomy flags, hand–object overlap/contact timing, and action
  pre/postcondition checks;
- camera-motion magnitude, discontinuity/shot-cut detection, optical-flow
  outliers, flicker, and frozen-frame ratio;
- transcript exact match from ASR, realized target-word onset, and overlap of
  the target word with the plan's visible-referent interval;
- selected disaggregated VBench/VBench 2.0-style dimensions: subject and
  background consistency, motion smoothness, temporal flicker, anatomy,
  commonsense, and compositional consistency;
- VideoPhy 2-style per-action physical-rule checks. No aggregate benchmark
  score is a pass criterion.

Automatic model outputs are diagnostics, not ground truth. Gates involving
contact, identity, timing, or anatomy require human confirmation.

### Human QA and gates

Use at least three adult raters per attempt, blinded to family, seed, and retry.
Randomize order and show no cherry-picked exemplars. Raters mark each item
pass/fail plus confidence:

1. one continuous egocentric shot with plausible head-camera motion;
2. no severe hand/anatomy defect;
3. intended contact and action completion are visually unambiguous;
4. target object retains identity and required attributes;
5. transition ordering matches the plan;
6. named referent is visible and unambiguous from 500 ms before target-word
   onset through 500 ms after target-word offset;
7. speech is exact, intelligible German with no extra lexical content;
8. no unsafe, sexualized, frightening, or identifiable-person content.

An attempt is a **scene pass** only when all eight items have majority pass and
no rater flags item 8. The family-level gate is:

- at least 2 of 3 seeds pass in each of the four scenes after the allowed retry;
- no scene has an identity, contact, or safety failure on all three seeds;
- manifest completeness is 100%.

Inter-rater agreement (Krippendorff's alpha with interval) is reported per
item; alpha is diagnostic rather than a gate at this pilot size. Disagreements
are not adjudicated after family identities are revealed.

### Stop and decision rules

Stop a family immediately for a safety-policy violation attributable to the
model/pipeline, an unresolvable license prohibition, inability to run its
approved official inference path (MPS or CUDA), failure to capture the frozen
manifest fields, or three consecutive invalid media outputs under the frozen
settings. Do not replace it.

If exactly one family passes every family-level gate, select it. If both pass,
select by this fixed lexicographic rule: (1) more seed-level scene passes before
retry, (2) higher total seed-level scene passes after retry, (3) fewer
identity/contact failures, (4) fewer retries used across passing attempts,
(5) lower model-asset storage footprint. Wall time, energy, provider charges,
and raw MPS/CUDA accelerator-hours are reported but not compared as equivalent.
If neither passes, the decision is **no-go**:
do not start a ChildLens-conditioned or downstream study, do not add a third
model, and revise the scientific premise/protocol in a new prospective review.
The joint-audio diagnostic can only support keeping modular audio; it cannot
change the selected family.

## Risks and unknowns

| Class | Concrete risk / unknown | Frozen mitigation or consequence |
|---|---|---|
| Engineering | Segment stitching can create cuts, duplicated actions, or state resets; model and Diffusers APIs change | Pin commits/weights, compile from one plan, hash boundary frames, retain every attempt, deterministic FFmpeg |
| Hardware / privacy | LTX Desktop officially supports local MPS inference, but free-RAM/throughput limits, beta stability, reproducible manifest capture, and fully local macOS text encoding are unresolved; Wan still needs CUDA | Public-only LTX Mac preflight plus approved MPS/CUDA pilot paths; record all API-backed features; later restricted prompts/media require a verified no-egress path |
| Generator control | Hands, contact, object identity, camera motion, transition order, and exact speech timing may fail independently | Four-scene gates, disaggregated failure labels, capped retries, no post-hoc candidate cycling |
| Distributional fidelity | A polished prompt video can omit child-view blur, occlusion, clutter, interaction density, and social/language statistics | Blinded real/synthetic study and predeclared distributional diagnostics are required later; benchmark scores are insufficient |
| Human evaluation | Raters can infer synthetic origin, disagree on contact, or be biased by audio quality | Blinding, randomized attempts, itemized judgments, multiple raters, agreement intervals, no cherry-picked media |
| Downstream learning | Synthetic artifacts can create shortcuts; augmentation gains may be compute gains; vocabulary coverage may be too small | Held-out-real testing, nested real fractions, matched steps, synthetic-only secondary arm, vocabulary/task gate, multiple learner seeds |
| Licensing / deployment | LTX and some weight licenses are not standard permissive licenses; TTS voices have separate output/use terms | Legal review before pilot, record every license/version, prohibit cloned voices, retain Wan TI2V as permissive alternative |
| Scientific interpretation | Success on ChildLens ages 3–5 would not establish infant learning or BabyView fidelity | Claims remain ChildLens-specific; BabyView requires separate access and governance |

## Deferred by design

This review does not implement the runner, choose a TTS vendor, set a monetary
budget, create ChildLens splits, inspect any recordings, run generation, or
download weights. Those choices require separate engineering, governance, and
resource approvals. A later protocol may fill in operational constants, but it
must preserve the learner, two-candidate cap, modular architecture, gates, stop
rules, and decision rule frozen here unless a new prospective review supersedes
this document through Git history.
