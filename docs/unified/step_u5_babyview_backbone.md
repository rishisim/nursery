# Step U5: Zero-Web-Data Visual Backbone (`mcxfrank/babyview-dino-vitb16`)

Status: **design blueprint, pending Hugging Face approval for `mcxfrank/babyview-dino-vitb16`.**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Eliminate the last "adult web-pretrained vision" critique by replacing
`facebook/dinov2-base` with Stanford's
[`mcxfrank/babyview-dino-vitb16`](https://huggingface.co/collections/mcxfrank/babyview-dino-models)
(85.7M-parameter DINOv3 ViT-B/16 trained 100% from scratch on 2,701 hours /
~9.7M frames of real infant BabyView head-cam video, zero internet images).

## Governance & Data Handling

- **Databrary access:** Requires approved BabyView access on Databrary +
  Hugging Face access approval from `mcfrank@stanford.edu`.
- **No participant-level BabyView video enters Nursery runs:** Only the
  approved model weights are used to embed Nursery's existing synthetic GRAB +
  EmbodiedGen renders; record the exact model revision and SHA-256 provenance
  in `embedding.json` (never commit weights or caches to Git).

## Execution Protocol

1. **Embedder support (`nursery/learner/embed.py`):** Support ViT-B/16 patch
   geometry (`14×14 + CLS = 197` tokens at 224 px, or `16×16 + CLS = 257`
   tokens at 256 px) into a fresh ignored cache root
   (`outputs/features/babyview_vitb16/`).
2. **Domain-shift sanity check (630 clips, seeds 0–1):** Run `V`, `V+G`,
   `V+G+B`, and shuffled `B` to confirm BabyView-pretrained infant vision
   transfers cleanly to Nursery's rendered rooms.
3. **Final zero-web-data acquisition curves (`63, 158, 315, 473, 630` clips,
   seeds 0–4):** Re-run the exact Step U1–U3 unified learner and fair baselines
   (`CLIP-Global`, `BabyLLaVA-Nursery`) on the single shared
   `babyview-dino-vitb16` eye—producing the final publication-ready curves with
   **zero web pretraining anywhere in the pipeline**.

## Stop / Go Gate

- Verify embedding integrity (finite float16 arrays, exact frame alignment,
  valid patch grid) and confirm `V+G` noun gain and `V+G+B` action/sentence
  gains replicate cleanly on `babyview-dino-vitb16` (prediction: embodied gains
  remain as large or larger because real infant vision is noisier than
  web-scale DINOv2).

## Expected Figures

1. **Final Publication Flagship Curves (Zero Web Data):** The 4-curve plot
   (`V+G+B`, `V+G`, `V`, `CLIP-Global` / `BabyLLaVA-Nursery`) across Nouns,
   Actions, and Compositional Sentences on `mcxfrank/babyview-dino-vitb16`.
2. **3D Experience × Speech Alignment × Accuracy Surface:** Comparing the
   `V+G+B` surface floating above `V` across clip counts (`63 → 630`) and
   controlled speech-alignment degradation (matching EgoBabyVLM's alignment
   axis), paired with clean 2D contour slices.
