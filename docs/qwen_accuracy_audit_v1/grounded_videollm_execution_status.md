# Grounded-VideoLLM one-shot execution status

## Outcome

After storage was cleared, the exact public Phi artifacts were downloaded and
verified and the one permitted public canary was started. The model and
checkpoint loaded, but the first synthetic case failed before producing an
answer because PyTorch 2.1.2 does not implement the model's InternVideo2
`Conv3D` operation on MPS.

The test therefore stopped before any restricted quarantine access. This is a
runtime failure of the frozen Mac port, not evidence that Grounded-VideoLLM
performs worse than, equal to, or better than Qwen3-VL.

Three separately frozen prospective Mac-runtime follow-ups were then executed;
none overwrites the original result:

- v2 upgraded to PyTorch 2.13. The exact production-shaped MPS `Conv3D`
  passed and agreed with a CPU reference, but the full model stopped before an
  answer on an fp16/fp32 projector-bias mismatch.
- v3 normalized all floating model state to fp16. It stopped before an answer
  on a deeper MPS matrix-multiplication accumulator/destination dtype
  assertion.
- v4 used the conservative device split: InternVideo2 on CPU float32; the
  spatial tower and learned projectors on MPS float32; Phi on MPS float16.
  This completed an end-to-end forward pass and generated one unscored
  synthetic response. The frozen runner then failed while constructing its
  private receipt because it contained the Python typo `true` instead of
  `True`. The response was not printed or retained, so schema validity is not
  recoverable and the smoke cannot be admitted as a pass.
- After the user explicitly authorized a new prospective attempt, v5 retained
  the exact v4 model/runtime and corrected only that receipt boolean. The
  unscored smoke passed. The unchanged eight-case German public canary then
  produced 8/8 schema-valid answers but only 4/8 correct referential-plus-lag
  decisions, below the frozen 6/8 minimum. It called all eight cases present,
  including the null and ambiguous controls, and selected lead in six cases.

## Gates checked

- The official Hugging Face repository was pinned to
  `b97f0e826a0b6c292ca54d9c02c483c479952501`.
- The official GitHub inference code was pinned to
  `e26da4e4b681357fd911d5ace3467f031af29208`.
- The exact Phi path requires 21 files totaling 27,529,808,010 bytes
  (25.64 GiB), before dependencies, caches, or generated outputs.
- After storage was cleared, the volume had 124.84 GiB free and a projected
  99.20 GiB after weights, so the governing 50-GiB floor passed.
- The authors' runner is CUDA-oriented, uses a 96-frame InternVideo2-1B
  temporal stream, and recommends a 24-GiB NVIDIA GPU. There is no official
  MPS runner or public prevalidated M5 port.
- The model card reports both a general Phi model and an asterisked model
  trained with Charades-STA/ActivityNet subsets. The public repository exposes
  only one generically named Phi SFT checkpoint, so the frozen test's required
  general variant cannot be identified unambiguously.
- The official public GitHub source was shallow-cloned for compatibility
  inspection before a per-file SHA-256 manifest was frozen. No weights or
  restricted data were accessed, but this public-only preflight deviation also
  prevents activation under the protocol's stricter artifact-resolution rule.

A prospective clarification, frozen before model output, bound the test to the
single checkpoint used by the authors' official Phi inference script while
labeling its general-versus-benchmark-subset status undocumented. It permits no
benchmark-cleanliness claim.

All 21 model files passed checksum verification. The public-only port used
eager attention, float16, PyTorch 2.1.2, and OS network denial. It reached the
InternVideo2 temporal patch embedding on the first synthetic case and stopped
with `RuntimeError: Conv3D is not supported on MPS`.

## What was not tested

- Grounded-VideoLLM was tested on all eight public canary cases, but not on the
  27 frozen ChildLens windows because the public gate failed.
- No paired ChildLens Qwen comparison, restricted defensibility result, or
  lexical-accuracy result exists for Grounded-VideoLLM.
- No restricted frames, ASR, timestamps, identifiers, paths, or predictions
  were opened, transmitted, or exported.
- Qwen was not rerun; no acquisition or causal endpoint was touched.

## Decision

Do **not** invest in the larger human-grounded ChildLens study. The user’s
condition—Grounded-VideoLLM first showing enough promise to justify a full
human-grounded study—was not met. The model failed the public semantic gate
despite perfect schema compliance.

No remote/cloud run is proposed because restricted ChildLens material may not
leave owner-controlled local storage.

The v5 result establishes that this M5 can execute the exact released model
end to end with whole-InternVideo2 CPU execution. The remaining failure is
scientific, not technical: the model over-aligned the public controls and
recovered only 4/8 oracle referential-plus-lag decisions. The frozen rule
therefore stops the experiment before restricted inference.
