# Step U3: Fair "Other Architectures" Baselines & Flagship Curves

Status: **completed and locked** (`outputs/runs/unified_u3_curves/`, 135 runs,
5 seeds, 60-epoch cap, baseline code `efa9c672`). The last 19 runs used the
packed Juno launcher.
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

| Baseline | Architecture Family | Visual Input | Scoring |
|---|---|---|---|
| **`CLIP-Global`** | EgoBabyVLM `CLIP+` style dual-encoder (InfoNCE + interleaved LM loss) | Mean/CLS DINO frame embedding (no Soft-MIL patches, no gaze, no body) | Cosine similarity between frame CLS and utterance `[EOS]` embedding |
| **`BabyLLaVA-Nursery`** | EgoBabyVLM `LLaVA` style generative VLM (2-layer MLP projector → 2L Causal LM) | DINO patch tokens projected into prefix of the 2L Causal Transformer | PMI and length-normalized raw log-likelihood on sentence panels, as planned |

The optional `Flamingo-Lite` baseline was not built.

## Fairness Contract

- Zero web-pretrained text encoders or external caption data in any model.
- Identical train/dev/test splits, held-out `s8` split, and evaluation panels
  for **Nouns**, **Actions**, and **Compositional Sentences**.
- Zero post-hoc dev weight tuning in any model.
- Shuffled-body alignment control retained at `158` and `630` clips.

## 630-Clip Results

Five-seed mean accuracy (%), with 95% CIs shown for `V+G+B`. The hierarchical
bootstrap resamples seeds and then held-out trials or source clips. Raw-LL
uses seed means only. Source tables and the acquisition figure are in
`outputs/runs/unified_u3_curves/figures/`.

| Panel | V+G+B (95% CI) | Shuffled body | V+G | V | CLIP-Global | BabyLLaVA PMI* | BabyLLaVA Raw-LL |
|---|---:|---:|---:|---:|---:|---:|---:|
| U1 Nouns | 74.6 (72.8–76.5) | 72.7 | 73.3 | 56.7 | 47.4 | 42.8 | — |
| U1 Actions | 73.2 (71.0–75.3) | 66.0 | 70.0 | 66.1 | 52.0 | 62.3 | — |
| Action + Object | 71.5 (65.9–76.1) | 68.0 | 67.8 | 58.7 | 57.1 | 44.9 | 32.6 |
| Property + Object | 86.1 (83.7–88.5) | 67.0 | 66.9 | 57.3 | 53.4 | 46.9 | 35.2 |
| Straddling subset | 75.8 (67.5–84.2) | 58.3 | 54.2 | 52.5 | 47.5 | 35.8 | 35.8 |
| Negation | 81.7 (79.5–83.8) | 72.5 | 72.5 | 64.5 | 71.1 | 51.5 | 74.8 |

\* BabyLLaVA uses its as-trained scoring on U1. The separate bare-prompt U1
diagnostic gives 34.6% on nouns and 49.7% on actions at 630 clips.

## Findings

- **Sample efficiency.** To match `V+G+B` at 158 clips on nouns, `V` needs
  2.6× as many clips and `CLIP-Global` needs 3.3×. On Property + Object, all
  three video-only baselines miss the 158-clip target by 630 clips, a lower
  bound of roughly 4×. `V+G` reaches that target at 630 clips. The three
  video-only baselines also miss the 315-clip targets for Nouns, Action +
  Object, and Property + Object, a lower bound of 2×.
- **Shuffled body.** Shuffling drops Property + Object from 65.8% to 43.3%
  at 158 clips and from 86.1% to 67.0% at 630. Nouns move from 45.1% to 43.3%
  and from 74.6% to 72.7%, respectively.
- **Caveats.** On negation, `CLIP-Global` leads `V+G+B` at 158 and 315 clips
  and nearly ties it at 473 (72.4% vs. 73.2%, with overlapping CIs). At 630,
  BabyLLaVA's 74.8% Raw-LL vs. 51.5% PMI on negation suggests a text-prior
  effect. The 24-clip straddling subset has a roughly ±8-point CI for `V+G+B`.

## Stop / Go Gate

**PASS (Locked).** All 135 runs are present and the five-point ladder is
complete.
