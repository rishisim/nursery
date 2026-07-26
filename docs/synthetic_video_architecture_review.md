# Synthetic-video architecture review and frozen public pilot

**Status:** frozen decision record

**Evidence cut-off:** 2026-07-26

**Scope:** public sources and public/synthetic-only pilot design; no ChildLens or
BabyView media were accessed.

## Decision

The primary downstream learner will be a **CVCL-style frame-plus-transcript
contrastive learner**, reproduced from the public method and adapted only where
needed for the study's frozen vocabulary and language. It consumes sampled
frames paired with timed, normalized transcripts; it does not consume audio
waveforms. The primary downstream claim is a real-data-efficiency claim:
synthetic augmentation reaches a predeclared held-out-real performance target
with less real training data than real-only training. Synthetic-only parity is
a stronger secondary target, not the success criterion.

This choice is preferable to raw audiovisual learning because CVCL directly
tests the intended grounded word-learning mechanism at tractable scale and has
both the original 61-hour result and a subsequent 500+ hour robustness study.
It is preferable to a BabyView vision/language baseline because BabyView data
are unavailable to this project and its reported baselines emphasize broad
visual transfer rather than the timed word–referent claim. DevBench is retained
only as an optional external evaluation battery after a vocabulary/task
coverage gate; it is not the learner or the primary endpoint.

Consequences are explicit:

- accurate transcripts and word timing relative to visible referents are
  required;
- realistic speech waveforms are required for the blinded audiovisual fidelity
  endpoint, but not for primary learner training;
- the canonical architecture is **modular video plus separately controlled
  licensed German TTS**, synchronized from one episode plan;
- native joint audio is diagnostic-only in the pilot and cannot replace the
  canonical TTS track unless it passes the same exact transcript/timing gates;
- the only later technical-pilot candidates are **Wan 2.2 TI2V-5B** and
  **LTX-2/LTX-2.x**. The frozen pilot compares at most these two families and
  cannot add a replacement after seeing results.

## Downstream study contract

The learner input unit is one transcript utterance paired with sampled frames
from its temporal window, following CVCL's dual-encoder contrastive setup.
Before any downstream run, freeze: tokenizer and German normalization,
vocabulary, utterance inclusion rules, frame sampling, window width, negative
sampling, initialization, optimization budget, seeds, and held-out-real splits.
Use the same learner code and optimization budget in every arm.

The later governed study must use nested real-data fractions (provisionally
`{5, 10, 25, 50, 100}%`), a fixed synthetic allocation rule declared before
training, and at least three learner seeds. Compare real-only, real+synthetic,
and synthetic-only on held-out real data. Define one target score from the
real-only learning curve without reference to synthetic results. The primary
estimand is the reduction in real examples/hours required to reach that target,
with seed-level uncertainty. Equal total optimizer steps and a second
equal-unique-pair analysis should separate data benefit from extra compute.
No ChildLens split, target, or statistic is defined in this public-only review.

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
