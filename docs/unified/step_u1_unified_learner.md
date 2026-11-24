# Step U1: Unified Embodied Learner (`Embodied Baby-CoCa` + Per-Word Unified Soft-MIL)

Status: **completed and locked (`79c77f05`, `outputs/runs/unified_u1_softmil_v1/`).**
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

## Locked 15-Run Parity & Ablation Results (630 Clips, Seeds 0–2, 40 Epochs)

Run directory: `outputs/runs/unified_u1_softmil_v1/` (commit `79c77f05`; best
dev epochs ranged from 10 to 26 across all 15 runs).

| Arm | Noun (as trained) | Action (as trained) | Noun (extras removed) | Action (extras removed) | Mean Routing: Nouns (V / T / B) | Mean Routing: Actions (V / T / B) |
|---|---|---|---|---|---|---|
| `V` | 49.2 ± 3.3 | 64.2 ± 2.7 | – | – | .40 / .60 / .00 | .31 / .69 / .00 |
| `V+G` | 68.9 ± 0.7 | 66.4 ± 1.7 | – | – | .62 / .38 / .00 | .38 / .62 / .00 |
| `V+G+B` | **70.4 ± 1.7** | **73.5 ± 1.1** | 60.3 ± 2.2 | 65.2 ± 2.9 | .50 / .30 / .20 | .22 / .42 / .36 |
| `V+G+B` shuffled body | 68.0 ± 4.0 | 65.1 ± 2.1 | 59.4 ± 1.8 | 69.6 ± 0.5 | .54 / .31 / .15 | .30 / .56 / .15 |
| `V+G+B`, `lambda_gen = 0` | 66.1 ± 1.8 | 68.6 ± 4.6 | 56.0 ± 2.1 | 60.2 ± 6.0 | .42 / .38 / .19 | .19 / .51 / .31 |

**Per-word action routing (`V+G+B` vs. shuffled body, vis / temp / body):**
`offhand` (.13 / .37 / .50 vs. .21 / .56 / .23), `lift` (.13 / .46 / .41 vs.
.19 / .69 / .12), `pass` (.15 / .45 / .39 vs. .24 / .59 / .17), `inspect`
(.23 / .43 / .34 vs. .33 / .55 / .13), `drink` (.44 / .39 / .17 vs. .51 / .42 / .07).

**Go-Gate Verdict: PASS (Locked).**
- **Double dissociation exceeds Step 9 with zero crutches:** Gaze lifts nouns
  by **+19.7 pp** (Step 9: +16.0 pp); real body lifts actions by **+7.1 pp**
  over `V+G` and **+8.4 pp** over shuffled body (Step 9: +6.4 / +6.5 pp), while
  shuffled body matches `V+G` on both nouns and actions.
- **Generative LM (`lambda_gen = 0.2`) actively helps:** Turning it off costs
  −4.3 pp on nouns, −4.9 pp on actions, and quadruples action SD (±1.1% → ±4.6%).
- **Posterior routing is clean and non-collapsed:** Real body receives >2× the
  action routing mass of shuffled body across all 5 action words.
- **Caveats carried forward:** `V` nouns (49.2%) reflect dropping Step 9's
  per-word `temporal_weight[vocab]`; extras-removed `V+G+B` actions (65.2% vs.
  69.6% shuffled) reflect genuine reliance on real body during training (flag
  explicitly in Step U4).

## Expected Figures

1. **2D Parity & Efficiency Curve:** `V`, `V+G`, `V+G+B` (unified, no dev
   sweep) overlaid against Step 9's legacy curves for Nouns and Actions.
2. **3D Sensory Routing Simplex:** Ternary / 3D simplex scatter of posterior
   Soft-MIL routing mass `(p_vis, p_temp, p_body)` per vocabulary word at 630
   clips—showing nouns clustering at the **Static Vision+Gaze** vertex, actions
   bridging **Temporal + Body**, and heavy/light properties pulling toward
   **Body**.
