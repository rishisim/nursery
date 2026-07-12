# EgoBabyVLM / Machine-DevBench integration record

Validated against `facebookresearch/egobabyvlm` commit
`224621caf0628270b6115845ac75a65b984234a3` (2026-06-23).

## Completed before BabyView access

- Downloaded and verified the public Machine-DevBench archive: 20 manifests,
  10 tasks, two image styles, and 3,721 total trials.
- Implemented a macOS/MPS runner that imports the official benchmark datasets,
  `MultiModalFeatureExtractor` protocol, frequency-bin merge, and result
  aggregator directly from the pinned repository, then applies the official
  lexical and grammatical trial equations. Only the Slurm/CUDA job launcher is
  replaced; source-file and manifest hashes are recorded in every result.
- Added a standalone Nursery checkpoint format containing model weights,
  tokenizer vocabulary, architecture, input dimensions, training arm, seed,
  and protocol digests.
- Added `NurseryMachineDevBenchExtractor`, which satisfies the official
  multimodal protocol. Because the public benchmark contains static images,
  its transparent compatibility rule repeats each image across the clip length
  expected by Nursery's video encoder.
- Completed every public trial with the synchronized provisional Nursery
  checkpoint and saved raw per-trial records plus official aggregate metrics.
- Added an exact vocabulary-coverage audit over all evaluated captions and
  benchmark target words.

## What the diagnostic run says

The provisional Nursery checkpoint scored 51.6 lexical, 17.1 grammatical, and
34.4 overall. These numbers are **not** a substantive model result. Its 48-word
training vocabulary (special tokens excluded) covers only 13 of 1,414 unique
Machine-DevBench target words (0.9%; 2.6% trial-weighted). Many benchmark words
therefore collapse to the same unknown token, producing ties and invalidating
scientific interpretation.

That failure mode is useful preparation: the evaluation integration is ready,
and the vocabulary audit makes the post-access prerequisite explicit. BabyView
calibration must expand the learner's lexical experience before fixed
Machine-DevBench can be treated as a secondary language-acquisition outcome.
The action-first held-out-composition grounding test remains the primary outcome
because it is controlled and corpus-aligned by construction.

## Off-the-shelf reference check

The same local runner evaluated the official OpenCLIP extractor with OpenAI
`ViT-L-14-quickgelu` weights on every trial. It produced 86.27 lexical, 70.29
grammatical, and 78.28 overall. The published CLIP-L row is 87.3, 70.4, and
78.8, respectively. The overall difference is 0.52 percentage points and the
grammatical difference is 0.11 points; the lexical difference is 1.03 points.
This is a close platform-level reproduction, with the remaining difference
plausibly attributable to the leaderboard's unspecified exact CLIP-L weight
tag and/or evaluation environment.

## Reproduce locally

```bash
PYTHONPATH=.external/egobabyvlm .venv/bin/python \
  -m scripts.eval_data.download_machine_devbench \
  --cache-dir .external/cache/machine_devbench

PYTHONPATH=. .venv/bin/python scripts/train_nursery_checkpoint.py \
  --episodes data/grounding_provisional/examples.jsonl \
  --out output/machine_devbench/nursery_sync_seed11.pt \
  --arm synchronized --seed 11 --alignment weak --epochs 8 --device auto

PYTHONPATH=. .venv/bin/python scripts/run_machine_devbench.py \
  --model nursery \
  --checkpoint output/machine_devbench/nursery_sync_seed11.pt \
  --out output/machine_devbench/nursery_full.json --device auto

PYTHONPATH=. .venv/bin/python scripts/audit_machine_devbench_coverage.py \
  --checkpoint output/machine_devbench/nursery_sync_seed11.pt

PYTHONPATH=. .venv/bin/python scripts/run_machine_devbench.py \
  --model openclip --openclip-model ViT-L-14-quickgelu \
  --openclip-pretrained openai \
  --out output/machine_devbench/openclip_l_full.json --device auto
```

## Scientific and leaderboard boundary

The released benchmark was constructed from vocabulary shared across BabyView,
Ego4D, HowTo, and COCO-MC. Regenerating a corpus-grounded version for Nursery
would require the official multi-GPU prompt-generation, image-generation, and
filtering pipeline; it is separate from running the fixed public evaluator.

Synthetic training is not eligible for the official BabyView challenge track,
which permits BabyView 2025.1 only and disallows external image, video, text, or
audio data for encoder pretraining or fine-tuning. Nursery is a complementary
causal study, not a leaderboard submission.

## Primary references

- Official repository: https://github.com/facebookresearch/egobabyvlm
- Evaluation data guide: https://github.com/facebookresearch/egobabyvlm/blob/main/docs/eval_data.md
- Benchmark generation: https://github.com/facebookresearch/egobabyvlm/tree/main/apps/benchmark_creation
- Paper: https://arxiv.org/abs/2605.19130
- Submission rules: https://facebookresearch.github.io/egobabyvlm/submit.html
