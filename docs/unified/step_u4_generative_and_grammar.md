# Step U4: Generative Fluency, Grammaticality & Cross-Modal Shaping

Status: **design blueprint, blocked on Step U3 go-gate.**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Evaluate the three capabilities that separate a full language-acquisition model
from a simple word-matching probe—using checkpoints already trained in Steps
U1–U3 (zero new training sweeps unless ablating cross-modal interaction):

1. **Minimal-pair grammaticality:** Score grammatical vs. ungrammatical
   Nursery-vocabulary minimal pairs (agreement, word order, negation,
   argument structure; BLiMP / Zorro style) via unconditional log-probability
   `P(grammatical) > P(ungrammatical)`.
2. **Generative caption fluency & grounding:** Generate greedy / low-temperature
   utterances conditioned on held-out test scenes (`V`, `V+G`, `V+G+B`, and
   `BabyLLaVA-Nursery`) and measure lexical precision/recall of the held object,
   active action, and physical property, plus perplexity on held-out caregiver
   transcripts.
3. **Cross-modal visual shaping (all-extra-inputs-removed):** Test whether
   joint training through the shared 512-d concept space and prefix-conditioned
   LM allows body signals during training to improve **video-only** action
   accuracy at test time (note: explicit `SensoryFusion` MHSA was ablated and
   dropped in U1 after hurting nouns without improving extras-removed actions).

## Metrics & Stop / Go Gate

- **Grammaticality:** Minimal-pair accuracy (%) across `V`, `V+G`, `V+G+B`,
  `BabyLLaVA-Nursery`, and a text-only LM baseline.
- **Grounded generation:** Object / action / property F1 in generated scene
  descriptions + held-out perplexity.
- **Cross-modal shaping:** `V+G+B (body removed)` minus `V+G (gaze removed)`
  on actions and compositional sentences (with optional U4 ablation if testing
  an explicit temporal-to-body alignment term).
- **Go:** Every evaluation runs deterministically from saved checkpoints with
  paired bootstrap CIs across test clips and seeds.

## Expected Figures

1. **Grammar & Grounded Generation Scorecard:** Grouped comparison of
   minimal-pair grammaticality, Embodied VP-Swap, and grounded caption F1 across
   `Text-only`, `BabyLLaVA-Nursery`, `V`, `V+G`, and `V+G+B`.
2. **3D Representation Trajectory (PCA / UMAP across epochs & clip counts):**
   Visualizing how `V+G+B` separates action concepts (`lift`, `pass`, `offhand`,
   `drink`, `inspect`) in shared embedding space by 158–315 clips while `V` and
   `CLIP-Global` remain entangled.
3. **Body-Removed Transfer Bar Plot:** Showing whether embodied training
   through the shared concept space and LM prefix lets `V+G+B` retain a visual
   action advantage even when body is turned off at test time.
