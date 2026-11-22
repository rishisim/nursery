# Current Nursery plan (`research/unified-architecture`)

Start with [docs/unified/CONTEXT.md](docs/unified/CONTEXT.md) for the full
scientific rationale, locked architecture, and fairness contract for Phase 2.

## Active objective

Show that embodied inputs (gaze + touch, hand pose, arm motion, and wrist load)
raise language acquisition beyond what video-only learners and standard VLM
architectures reach across amounts of independent training experience—from
nouns and actions to compositional sentences, grammaticality, generative
fluency, and developmental alignment with human infants.

Language acquisition is measured on separate rungs (never blended into one
score):

1. **Nouns:** object grounding (control rung, driven by steering gaze).
2. **Actions (and Step 7 heavy/light PoC):** embodied motor/physical words.
3. **Compositional sentences & physical semantics:** action + object +
   property/hand + negation contrasts, minimal-pair grammar, and text-only
   Embodied VP-Swap.

## Completed Phase 1 checkpoint (Steps 1–9)

Phase 1 is complete and archived in
[archive/phase1_embodied/](archive/phase1_embodied/) (with Step 9 curve data
and plots kept live in
[docs/results/noun_action_acquisition_curves/](docs/results/noun_action_acquisition_curves/)).
Across 85 models (63–630 clips, 5 seeds), Phase 1 established a clean **double
dissociation**: gaze lifts nouns by **+9.5 to +20.0 pp** (body ≈ 0), body lifts
actions by **+6.4 to +9.5 pp** (gaze ≈ 0; ~2× sample efficiency: `V+G+B` at 315
clips = 68.6% vs. `V` at 630 clips = 69.9%), shuffled body ≈ 0, and Step 7's
heavy/light proof of concept gained +46.0 pp from body.

## Phase 2 roadmap: Unified Architecture (`U1–U6`)

Phase 2 replaces Phase 1's four brittle pieces (`temporal_weight[vocab]`,
`T[:, :128]` slice, post-hoc dev-tuned `body_weight`, and orderless
bag-of-words text) with one clean **Embodied Baby-CoCa** architecture
(`nursery/unified/`) benchmarked fairly against EgoBabyVLM-style baselines and
upgraded to a zero-web-data infant visual backbone.

| Step | Blueprint | Focus & Expected Figure(s) | Status |
|---|---|---|---|
| **U1** | [Unified learner](docs/unified/step_u1_unified_learner.md) | Build `nursery/unified/` (3 aligned 512-d streams, zero-init cross-modal MHSA, 2L Causal Transformer, Modality Gate `g(T)`, dual InfoNCE + LM heads); verify parity vs. Step 9 with **zero post-hoc `body_weight` sweep**. *Figures: parity curve + 3D Sensory Routing Simplex.* | Ready to build |
| **U2** | [Compositional sentences & VP-Swap](docs/unified/step_u2_compositional_sentences.md) | Tier A compositional sentences from existing GRAB events (action + object + property/hand + negation) + text-only **Embodied VP-Swap**. *Figures: 3-rung dissociation + VP-Swap bar chart + updated 3D simplex.* | Blueprint locked |
| **U3** | [Fair baselines & flagship curves](docs/unified/step_u3_fair_baselines.md) | Fair video-only **`CLIP-Global`** and **`BabyLLaVA-Nursery`** (optional `Flamingo-Lite`) on identical clips, backbone, and 2L Transformer capacity. *Figures: flagship 4-curve plot (`V+G+B`, `V+G`, `V`, `Other architectures`) per rung.* | Blueprint locked |
| **U4** | [Grammar, fluency & shaping](docs/unified/step_u4_generative_and_grammar.md) | Minimal-pair grammaticality, grounded caption fluency, and body-removed visual transfer (MHSA-on vs. MHSA-off). *Figures: grammar/fluency scorecard + 3D Embedding Trajectory across epochs.* | Blueprint locked |
| **U5** | [BabyView visual backbone](docs/unified/step_u5_babyview_backbone.md) | Swap cached `dinov2-base` for Stanford's zero-web-data [`mcxfrank/babyview-dino-vitb16`](https://huggingface.co/collections/mcxfrank/babyview-dino-models) once HF access is approved; run final zero-web-data flagship curves. *Figures: final flagship curves + 3D Experience × Alignment × Accuracy surface.* | Pending HF approval |
| **U6** | [Developmental comparison](docs/unified/step_u6_developmental_comparison.md) | **Wordbank AoA** (acquisition order correlation + vocabulary growth acceleration) and **DevBench** (overlapping lexical + THINGS human similarity). *Figures: Mastery vs. Wordbank AoA scatter + Vocabulary Growth Acceleration curve.* | Blueprint locked |

## Decisions to keep consistent

- **One shared eye per comparison:** Never mix visual backbones within a curve.
  Build and run initial sweeps on cached `facebook/dinov2-base`; switch to
  `mcxfrank/babyview-dino-vitb16` as the single shared eye for the final
  zero-web-data curves (Step U5).
- **Zero post-hoc dev weight tuning:** Modality trust is learned end-to-end by
  the utterance-conditioned `ModalityGate g(T)` across all arms.
- **Simulator boundary:** Keep GRAB + EmbodiedGen rooms for this paper; do not
  reopen the archived HOIDiNi + InterMimic physics rebuild.
- **Fairness contract:** Every architecture (`V+G+B`, `V+G`, `V`,
  `CLIP-Global`, `BabyLLaVA-Nursery`) trains from scratch on the exact same
  nested clip prefixes (`63, 158, 315, 473, 630`), same 5 seeds (`0–4`), same
  2-layer Causal Transformer capacity, and same 60-epoch best-dev selection.
- **Controls & reporting:** Keep shuffled-body alignment controls at `158` and
  `630` clips; report separate curves per rung with 95% joint seed + clip CIs;
  log all-extra-inputs-removed and picture/DevBench transfer separately.

## Archived Phase 1 reference

Completed Phase 1 study protocols, one-off scripts, and script-specific tests
live in [archive/phase1_embodied/](archive/phase1_embodied/). Earlier studies
and motion-pipeline archives remain indexed in [archive/README.md](archive/README.md).
Retained datasets, features, and run outputs under `outputs/` stay untouched.
