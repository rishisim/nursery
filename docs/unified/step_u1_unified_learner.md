# Step U1: Unified Embodied Learner (`Embodied Baby-CoCa` + Per-Word Unified Soft-MIL)

Status: **design blueprint updated for Per-Word Unified Soft-MIL (no gate MLP, no MHSA).**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Replace Phase 1's four brittle pieces (`temporal_weight[vocab]`, `T[:, :128]`
slice, post-hoc dev-tuned `body_weight`, and orderless bag-of-words lookup)
with one clean, unified architecture in `nursery/unified/` that:

1. Matches Step 9's held-out **Noun** and **Action** double dissociation across
   `V`, `V+G`, and `V+G+B` with **zero post-hoc dev weight sweeps**, **no
   separate modality-gate MLP**, and **no `b_init` tuning**.
2. Grounds words across vision, motion, and body using one single
   **Hierarchical Per-Word Soft-MIL** pooling rule—yielding interpretable
   posterior modality routing shares `(p_vis, p_temp, p_body)` on the 2-simplex.
3. Supports both contrastive comprehension (`L_InfoNCE`) and prefix-conditioned
   generative modeling (`L_LM`) through one shared 2-layer Causal Transformer.

## Module Layout (`nursery/unified/`)

- `nursery/unified/sensory.py`: Three aligned 512-d streams (`StaticVisionGaze`,
  `TemporalVideoStream`, `BodyStream`) reusing Phase 1's existing feature
  caches, gaze windows, and body store—all unit-normalized before scoring and
  LM prefixing (no `SensoryFusion` MHSA block).
- `nursery/unified/language.py`: Compact **2-layer, 4-head, 256-d Causal
  Transformer** trained from scratch on Nursery utterances, exposing
  `encode_tokens(words, counts)` for per-word normalized 512-d embeddings
  `(tokens_512, content_mask)` (with zero-init contextual residual `ctx_proj`)
  and `lm_loss(words, counts, sensory, active)` for prefix-conditioned causal LM.
- `nursery/unified/model.py`: `UnifiedEmbodiedLearner` combining the three
  sensory streams, `CausalLanguage`, **Hierarchical Per-Word Unified Soft-MIL**
  (`soft_pool` over active streams per word token with shared `tau_m = 0.1`,
  averaged over content words), contrastive InfoNCE + per-stream auxiliary
  InfoNCE (`aux_weight = 0.25`), and generative LM head (`lambda_gen = 0.2`,
  ablatable to `0.0`).
- `nursery/unified/train.py` & `nursery/unified/evaluate.py`: Clean training
  and evaluation runner logging per-word posterior Soft-MIL routing shares
  `(p_vis, p_temp, p_body)` on matched held-out clips, 4-choice accuracies, and
  all-extra-inputs-removed transfer.
- `tests/unified/`: Unit tests covering masking invariance (`V` / `V+G` /
  `V+G+B` exact equivalence when modalities are absent), uniform active-stream
  prior invariance (equal stream scores return exact score regardless of 1, 2,
  or 3 active streams), per-word vs. single-word scoring equivalence, zero
  text-prefix leakage into InfoNCE, gradient flow to all streams, and simplex
  normalization (`sum(p) == 1`, `p >= 0`).

## Verification Protocol (Cached `dinov2-base`)

1. **Local unit tests + 1-epoch smoke at 63 clips:** Verify forward/backward,
   checkpoint save/load, and exact posterior routing logging on CPU/MPS.
2. **630-clip 3-seed parity & ablation matrix (seeds 0–2, 60 epochs, best
   noun+action dev checkpoint):**
   - Arms: `V`, `V+G`, `V+G+B`, `V+G+B` shuffled body, and `V+G+B` with
     `lambda_gen = 0.0` (5 arms × 3 seeds = 15 runs; no seed-0 tuning loops).
3. **Full 5-point ladder (`63, 158, 315, 473, 630` clips, seeds 0–4)** once
   parity passes (or immediately combined with Step U2 sentences).

## Stop / Go Gate

- **Go** if `UnifiedEmbodiedLearner` (zero post-hoc `body_weight` sweep, zero
  gate MLP) reproduces the full double dissociation across 3 seeds: strong gaze
  lift on `V+G` nouns (~70%+ at 630), clear real-vs-shuffled body lift on
  `V+G+B` actions (with shuffled body near ~0 pp over `V+G`), and non-collapsed
  posterior routing `(p_vis, p_temp, p_body)` that separates nouns (high
  `p_vis`) from actions (elevated `p_temp` + `p_body`).
- **Stop & diagnose** if `lambda_gen` degrades 4-choice accuracy by > 2 pp or
  if posterior routing collapses to a single modality across all word types.

## Expected Figures

1. **2D Parity & Efficiency Curve:** `V`, `V+G`, `V+G+B` (unified, no dev
   sweep) overlaid against Step 9's legacy curves for Nouns and Actions.
2. **3D Sensory Routing Simplex:** Ternary / 3D simplex scatter of posterior
   Soft-MIL routing mass `(p_vis, p_temp, p_body)` per vocabulary word at 630
   clips—showing nouns clustering at the **Static Vision+Gaze** vertex, actions
   bridging **Temporal + Body**, and heavy/light properties pulling toward
   **Body**.
