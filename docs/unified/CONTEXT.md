# Phase 2 Context: Unified Embodied Language Learner (`research/unified-architecture`)

This brief captures the complete scientific context, locked design decisions,
and architectural principles for Phase 2 of Nursery. Start here before writing
code or launching experiments on `research/unified-architecture`.

## 1. Where Phase 1 Ended (Completed Checkpoint)

Phase 1 (Steps 1–9, archived in [archive/phase1_embodied/](../../archive/phase1_embodied/))
established a clean **double dissociation** on 63–630 simulated GRAB +
EmbodiedGen clips (85 models, 5 seeds each, four-choice word matching under
weak caregiver speech; archived data in
[docs/results/noun_action_acquisition_curves/](../results/noun_action_acquisition_curves/)):

- **Gaze (`V+G` vs. `V`):** Lifts **nouns** by **+9.5 to +20.0 pp** at every
  clip count while adding ≈ 0 pp on actions (resolves *which object* is named).
- **Body (`V+G+B` vs. `V+G`):** Lifts **actions** by **+6.4 to +9.5 pp** at
  every clip count while adding ≈ 0 pp on nouns; reaches ~2× sample efficiency
  (`V+G+B` at 315 clips = 68.6% vs. `V` at 630 clips = 69.9%), with +7.8 to
  +12.4 pp on object-matched actions and ≈ 0 pp for shuffled body.
- **Physical property proof of concept (Step 7):** Body lifts `heavy`/`light`
  by **+46.0 pp** (+60.4 pp on nouns that straddle heavy and light instances).

## 2. Why Phase 2 Exists (Fixing Four Brittle Pieces)

Phase 1 proved the data hypothesis using a lightweight contrastive probe built
on [vlm-headcam](https://github.com/mcfrank/vlm-headcam) `RegionMIL`. However,
four architectural limitations prevent Phase 1 from scaling cleanly to full
language acquisition and fair VLM comparisons:

1. **Per-word `temporal_weight[vocab]` + `T[:, :128]` slice:** Fixed the Step 7
   noun drop, but stores a scalar per word ID, uses an arbitrary 128-d subspace
   slice, allows arbitrary sign flips across seeds, and cannot generalize to
   unseen words or compositional sentences.
2. **Post-hoc dev sweep of `body_weight`:** Sweeping a scalar on dev at
   evaluation time is not a developmentally plausible mechanism; modality
   trust should be learned end-to-end.
3. **Zero cross-modal interaction before final score addition:** Pure late sum
   (`score = visual + body`) meant removing body at test time dropped `V+G+B`
   straight back to `V+G` on actions—body acted as an online readout rather
   than shaping visual representations.
4. **Orderless bag-of-words text + adult web-pretrained vision:** Averaging
   word embeddings ignores word order and syntax, and `facebook/dinov2-base`
   carries web-scale visual priors rather than strictly infant visual
   experience (unlike [EgoBabyVLM](https://arxiv.org/html/2605.19130v1)).

## 3. Locked Architecture: Embodied Baby-CoCa (`nursery/unified/`)

One clean, unified model class runs `V`, `V+G`, and `V+G+B` (unavailable
modalities are simply masked out):

1. **Three Aligned 512-d Sensory Streams:**
   - **Static Vision + Steering Gaze:** Frozen DINO patch grid + CLS projected
     to 512-d, Soft-MIL pooling (`tau = 0.1`), 2D Gaussian gaze spatial prior
     (`sigma = 0.15`), and shared-projection foveal close-up token.
   - **Temporal Vision:** 11-second (±5 s) clip-bounded BiGRU (`128d`) with a
     clean `Linear(128→512)` projection (eliminates the `T[:, :128]` slice).
   - **Body Stream:** 5 anatomical/physical groups (contact, fingers, arm,
     load, skin) → per-group MLPs → clip-reset 11-second BiGRU (`128d`) →
     `Linear(128→512)`.
2. **Zero-Init Cross-Modal Sensory Fusion:** Packs active sensory summaries
   `[z_vis, z_temp, z_body]` and applies 1 lightweight Multi-Head Self-Attention
   block with a zero-initialized output residual (`z' = z + 0 · MHSA(LN(z))`),
   starting as exact identity at epoch 0 while allowing body and vision to
   shape each other during training.
3. **Compact Causal Text Transformer (from scratch):** A **2-layer, 4-head,
   256-d Causal Transformer** trained from scratch on each clip-count's
   Nursery utterances, projecting `[EOS]` to a normalized 512-d utterance
   embedding `T`.
4. **Utterance-Conditioned Modality Gate `g(T)`:** A small 2-layer MLP maps
   `T` to a masked softmax over active modalities initialized near
   `[0.96, 0.02, 0.02]` (`b_init = [2.0, -2.0, -2.0]`):
   `g(T) = softmax(MLP(T) + b_init)` on the 2-simplex `(static, temporal, body)`.
   Replaces both `temporal_weight[vocab]` and post-hoc `body_weight`, keeps
   weights non-negative and interpretable, and directly yields the **3D Sensory
   Routing Simplex**.
5. **Dual Contrastive + Generative Objective (Embodied Baby-CoCa):**
   `L = L_InfoNCE(scale * sum_m g_m(T) * s_m) + lambda_gen * L_LM(utterance | z'_vis, z'_temp, z'_body)`.
   - Contrastive branch runs the Text Transformer unconditionally (zero leakage)
     for 4-choice comprehension curves.
   - Generative branch runs the exact same Transformer weights conditioned on
     the fused sensory prefix tokens for minimal-pair syntax, Embodied VP-Swap,
     and generative caption fluency.

## 4. Locked Evaluation, Simulator & Fairness Contract

- **Simulator boundary:** Keep GRAB + EmbodiedGen rooms (do not reopen the
  archived HOIDiNi + InterMimic infant-physics rebuild for this paper). Build
  **Tier A compositional sentences** directly from existing GRAB events
  (action + object + property/hand + negation); treat kinematic 2-object
  spatial scenes as an optional Tier B.
- **Visual backbone ("1 Shared Eye"):** Build and run initial sweeps on the
  already-cached `facebook/dinov2-base` features; once Hugging Face access to
  [`mcxfrank/babyview-dino-vitb16`](https://huggingface.co/collections/mcxfrank/babyview-dino-models)
  (zero internet images, trained from scratch on 2,701h of BabyView) is
  approved via Databrary, re-embed the 630 clips and run the final zero-web-data
  comparison on that single shared eye.
- **Fair "Other architectures" baselines:** Benchmark video-only
  **`CLIP-Global`** (EgoBabyVLM `CLIP+` style) and **`BabyLLaVA-Nursery`**
  (generative prefix VLM) on the exact same Nursery clips, 2-layer Causal
  Transformer capacity, visual backbone, seeds (`0–4`), and 60-epoch best-dev
  selection—zero post-hoc dev weight tuning anywhere.
- **Separate curves per rung:** Always report **Nouns**, **Actions**, and
  **Compositional Sentences** separately, alongside **Embodied VP-Swap**
  (text-only physical semantics), **DevBench / Konkle transfer**, and
  **Wordbank AoA** (acquisition order + vocabulary growth acceleration).

## 5. Phase 2 Blueprint Index & Expected Figures

| Step | Blueprint | Core Deliverable & Expected Figure(s) |
|---|---|---|
| U1 | [step_u1_unified_learner.md](step_u1_unified_learner.md) | `nursery/unified/` Embodied Baby-CoCa + `g(T)`; parity curve vs. Step 9 + **3D Sensory Routing Simplex**. |
| U2 | [step_u2_compositional_sentences.md](step_u2_compositional_sentences.md) | Tier A compositional sentences + Embodied VP-Swap; 3-rung dissociation chart + text-only VP-Swap bar chart. |
| U3 | [step_u3_fair_baselines.md](step_u3_fair_baselines.md) | Fair `CLIP-Global` & `BabyLLaVA-Nursery` baselines; **flagship 4-curve acquisition plot** (`V+G+B`, `V+G`, `V`, `Other architectures`). |
| U4 | [step_u4_generative_and_grammar.md](step_u4_generative_and_grammar.md) | Minimal-pair grammar, generative fluency, body-removed cross-modal shaping; **3D Embedding Trajectory** across epochs. |
| U5 | [step_u5_babyview_backbone.md](step_u5_babyview_backbone.md) | Swap to `mcxfrank/babyview-dino-vitb16` (zero web data); final flagship curves + **3D Experience × Alignment × Accuracy** surface. |
| U6 | [step_u6_developmental_comparison.md](step_u6_developmental_comparison.md) | Wordbank AoA + DevBench transfer; **Model Mastery vs. Wordbank AoA** scatter + **Vocabulary Growth Acceleration** curve. |
