# Open/local temporal-grounding model landscape

As of 2026-07-22. Scores are author-reported and are not necessarily
apples-to-apples because original and repaired benchmark variants, zero-shot
and task-tuned settings, and metrics differ. A missing score is not treated as
a zero.

| Instrument | Temporal evidence | Relevant reported results | License | 32 GB Apple Silicon | CUDA |
|---|---|---|---|---|---|
| Qwen3-VL-8B-Instruct | General VLM with interleaved temporal RoPE and explicit text–timestamp alignment. | The 2026 TimeLens2 comparison reports Qwen3-VL-8B mIoU of 47.2 on Charades-TimeLens, 48.0 on ActivityNet-TimeLens, 59.4 on QVHighlights-TimeLens, and 5.3 on Ego4D-NLQ. | Apache-2.0 | Already demonstrated locally in this project in 4-bit form. | Straightforward in Transformers/vLLM. |
| Gemma 4 E4B | General edge VLM; Google documents video as frame sequences, at about 1 fps and up to about 60 seconds. | Google reports no Charades-STA, ActivityNet temporal grounding, VideoMME, MVBench, TempCompass, or EgoSchema score in the official model card. Its reported vision results are static/general (for example MMMU-Pro 52.6). | Apache-2.0 | Excellent fit and already ran locally, but the evidence here shows it is not a useful lag instrument. | Straightforward but scientifically unmotivated for another run. |
| **Grounded-VideoLLM Phi-3.5 4B** | Independent Phi-3.5 base plus an explicit temporal stream, discrete timestamp tokens, and staged fine-grained temporal-grounding training. | General checkpoint: Charades-STA 36.8 mIoU; ActivityNet grounding 36.1 mIoU; MVBench 59.4; Video-MME without subtitles 47.7. A benchmark-subset-tuned variant reports 49.4, 47.2, 60.0, and 48.1 respectively. | MIT | Parameter and weight size are memory-feasible, but the exact project runner is CUDA-oriented and no official MLX conversion was found. Treat M5 execution as an unvalidated port, not a promised run. | **Best exact implementation path.** Use the authors' BF16 runner on one NVIDIA GPU; 24 GB VRAM is the conservative target. |
| TimeLens2-4B | Very recent generalist temporal grounder trained to return interval sets with temporal-Wasserstein plus tIoU rewards. | 47.7 average mIoU over seven benchmarks; 57.7 Charades-TimeLens, 59.0 ActivityNet-TimeLens, 69.3 QVHighlights-TimeLens, and 18.6 Ego4D-NLQ. | Apache-2.0 | The 4B footprint is promising, but the released recipe assumes PyTorch/FlashAttention. A Mac port is plausible, not validated. | Strongest compact current temporal specialist in this comparison. |
| TRACE 8B | Mistral/VideoLLaMA2-derived causal event model that interleaves timestamps, salience, and captions. | Zero-shot: Charades-STA 38.7 mIoU, ActivityNet moment retrieval 39.0 mIoU, MVBench 48.1, VideoMME without subtitles 43.8. | Apache-2.0 | BF16 weights fit 32 GB only with limited headroom; custom video stack makes a Mac run high-risk. | Viable on a 24 GB+ NVIDIA GPU, but larger and weaker than the recommended option for this mini-test. |

## Recommendation

Test exactly one second instrument: **Grounded-VideoLLM
Phi-3.5-Vision-Instruct-4B, using the general checkpoint rather than the
Charades/ActivityNet-subset-tuned variant**.

It is not the highest-scoring current compact model—TimeLens2-4B is stronger on
the repaired temporal benchmarks—but TimeLens2 is a Qwen3-VL-4B derivative.
Using it would test specialized post-training more than it would provide an
independent instrument. Grounded-VideoLLM instead changes the language/vision
base and adds explicit temporal machinery while remaining small enough for a
bounded test. Its main risk is German: the base model card emphasizes English,
so the frozen public German canary must pass before any restricted inference.

The best CUDA choice and the recommended second instrument are therefore the
same model. The M5 32 GB machine is suitable for a no-download runtime
compatibility investigation, but the one-shot scientific run should use the
authors' CUDA path unless a public synthetic canary first proves an exact,
hash-pinned Mac port.

## Primary sources

- [Qwen3-VL-8B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [Gemma 4 official model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [TimeLens2 paper](https://arxiv.org/abs/2607.17423) and [TimeLens2-4B model card](https://huggingface.co/MCG-NJU/TimeLens2-4B)
- [Grounded-VideoLLM official repository](https://github.com/WHB139426/Grounded-Video-LLM) and [model card](https://huggingface.co/WHB139426/Grounded-Video-LLM)
- [TRACE model card](https://huggingface.co/Yongxin-Guo/trace)
- [Phi-3.5-Vision official model card](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
