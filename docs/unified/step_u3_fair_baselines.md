# Step U3: Fair "Other Architectures" Baselines & Flagship Curves

Status: **ready to build (Steps U1–U2 locked at `79c77f05` and `b69091fb`).**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Produce the **flagship 4-curve acquisition comparison** (`V+G+B`, `V+G`, `V`,
and standard video-only VLM architectures from
[EgoBabyVLM](https://arxiv.org/html/2605.19130v1)) across `63, 158, 315, 473,
630` clips, proving that embodied cues lift language acquisition beyond what
standard VLM paradigms achieve from video alone.

## Fair Baselines (`nursery/unified/baselines.py`)

All baselines share the **exact same** frozen visual feature cache, nested
clip prefixes (`63–630`), weak caregiver transcripts, 5 seeds (`0–4`),
2-layer 4-head 256-d Causal Transformer capacity, 60-epoch cap, and best-dev
checkpoint selection:

| Baseline | Architecture Family | Visual Input | Scoring on 4-Choice Panels |
|---|---|---|---|
| **`CLIP-Global`** | EgoBabyVLM `CLIP+` style dual-encoder (InfoNCE + interleaved LM loss) | Mean/CLS DINO frame embedding (no Soft-MIL patches, no gaze, no body) | Cosine similarity between frame CLS and utterance `[EOS]` embedding |
| **`BabyLLaVA-Nursery`** | EgoBabyVLM `LLaVA` style generative VLM (2-layer MLP projector → 2L Causal LM) | DINO patch tokens projected into prefix of the 2L Causal Transformer | Length-normalized caption log-likelihood `log P(text | video)` (matching EgoBabyVLM's LLaVA eval) |
| **`Flamingo-Lite`** *(optional)* | Temporal video cross-attention (Perceiver resampler over 11 frames → gated cross-attention into 2L LM) | 11-second DINO frame sequence | Caption log-likelihood `log P(text | 11s video)` |

*(Start with `CLIP-Global` + `BabyLLaVA-Nursery`; add `Flamingo-Lite` if compute
permits or if reviewers ask for a stronger temporal video-only VLM.)*

## Fairness Contract

- Zero web-pretrained text encoders or external caption data in any model.
- Identical train/dev/test splits, identical held-out `s8` split, identical
  4-choice panels across **Nouns**, **Actions**, and **Compositional Sentences**.
- Zero post-hoc dev weight tuning in any model.
- Shuffled-body alignment control retained at `158` and `630` clips.

## Stop / Go Gate

- Confirm `CLIP-Global` and `BabyLLaVA-Nursery` converge cleanly (no collapsed
  loss, above-chance learning by 315–630 clips) so they are genuinely fair,
  well-trained competitors rather than strawmen.
- Verify the full 5-point ladder reproduces clean CIs over both seeds and test
  clips.

## Expected Figures

1. **Flagship 4-Curve Acquisition Figure (3 panels: Nouns, Actions,
   Compositional Sentences):** Plotting **`V+G+B (ours)`**, **`V+G (ours)`**,
   **`V (ours)`**, and **`CLIP-Global` / `BabyLLaVA-Nursery`** across
   `63 → 630` clips with 95% confidence bands—directly realizing the target
   paper figure.
2. **Sample-Efficiency Multiplier Table/Plot:** Clips required by `V`,
   `CLIP-Global`, and `BabyLLaVA-Nursery` to match `V+G+B` at 158 and 315 clips.
