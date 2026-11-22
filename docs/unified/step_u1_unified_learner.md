# Step U1: Unified Embodied Learner (`Embodied Baby-CoCa` + Modality Gate)

Status: **design blueprint, ready to build.**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Replace Phase 1's four brittle pieces (`temporal_weight[vocab]`, `T[:, :128]`
slice, post-hoc dev-tuned `body_weight`, and orderless bag-of-words lookup)
with one clean, unified architecture in `nursery/unified/` that:

1. Matches or beats Step 9's held-out **Noun** and **Action** accuracy across
   `V`, `V+G`, and `V+G+B` with **zero post-hoc dev weight sweeps**.
2. Learns an interpretable, utterance-conditioned **Modality Gate `g(T)`** on
   the 2-simplex `(static vision, temporal vision, body)`.
3. Supports both contrastive comprehension (`L_InfoNCE`) and prefix-conditioned
   generative modeling (`L_LM`) through one shared 2-layer Causal Transformer.

## Module Layout (`nursery/unified/`)

- `nursery/unified/sensory.py`: Three aligned 512-d streams (`StaticVisionGaze`,
  `TemporalVideoStream`, `BodyStream`) reusing Phase 1's existing feature
  caches, gaze windows, and body store—plus `SensoryFusion` (1-layer zero-init
  residual Multi-Head Self-Attention over `[z_vis, z_temp, z_body]`).
- `nursery/unified/language.py`: Compact **2-layer, 4-head, 256-d Causal
  Transformer** trained from scratch on Nursery utterances, with `[EOS]`
  projection to 512-d (`T`) and prefix-conditioned causal LM mode.
- `nursery/unified/gate.py`: `ModalityGate(T)` — 2-layer MLP + masked softmax
  over active modalities with `b_init = [2.0, -2.0, -2.0]` (starting near
  `[0.96, 0.02, 0.02]`).
- `nursery/unified/model.py`: `UnifiedEmbodiedLearner` combining sensory
  streams, fusion, text Transformer, `ModalityGate`, contrastive InfoNCE head,
  and generative LM head (`lambda_gen`, default `0.2`, ablatable to `0.0`).
- `nursery/unified/train.py` & `nursery/unified/evaluate.py`: Clean training
  and evaluation runner logging per-word `g(T)` simplex coordinates, 4-choice
  accuracies, and all-extra-inputs-removed transfer.
- `tests/unified/`: Unit tests covering masking invariance (`V` / `V+G` /
  `V+G+B` exact equivalence when modalities are absent), zero-init MHSA
  identity at step 0, zero text-prefix leakage into InfoNCE, gradient flow to
  all streams, and simplex normalization (`sum(g(T)) == 1`, `g(T) >= 0`).

## Verification Protocol (Cached `dinov2-base`)

1. **Local unit tests + 1-epoch smoke at 63 clips:** Verify forward/backward,
   checkpoint save/load, and exact `g(T)` logging on CPU/MPS.
2. **630-clip parity & ablation check (seeds 0–2, 60 epochs, best noun+action
   dev checkpoint):**
   - Arms: `V`, `V+G`, `V+G+B`, and `V+G+B` shuffled body.
   - Ablations on `V+G+B`: (a) `lambda_gen = 0.0` vs. `0.2`, (b) zero-init
     MHSA on vs. off.
3. **Full 5-point ladder (`63, 158, 315, 473, 630` clips, seeds 0–4)** once
   parity passes (or immediately combined with Step U2 sentences).

## Stop / Go Gate

- **Go** if `UnifiedEmbodiedLearner` with **zero post-hoc `body_weight` sweep**
  stays within ±1.5 pp of Step 9 on `V+G` nouns (~72.9% at 630) and matches or
  exceeds Step 9 on `V+G+B` actions (~76.5% at 630), while shuffled body stays
  near 0 pp and `g(T)` cleanly separates nouns (high `g_vis`) from actions
  (elevated `g_temp` + `g_body`).
- **Stop & diagnose** if `lambda_gen` degrades 4-choice accuracy by > 2 pp or
  if `g(T)` collapses to a single modality across all word types.

## Expected Figures

1. **2D Parity & Efficiency Curve:** `V`, `V+G`, `V+G+B` (unified, no dev
   sweep) overlaid against Step 9's legacy curves for Nouns and Actions.
2. **3D Sensory Routing Simplex (`g(T)`):** Ternary / 3D simplex scatter of
   learned `(g_vis, g_temp, g_body)` per vocabulary word at 630 clips—showing
   nouns clustering at the **Static Vision+Gaze** vertex, actions bridging
   **Temporal + Body**, and heavy/light properties pulling toward **Body**.
