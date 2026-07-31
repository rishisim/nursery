# Public-only synthetic-video qualitative pilot

**Status:** frozen preview **COMPLETE**; all eight media outputs are valid, with
qualitative failures retained and formal human QA pending

**Canonical protocol:** `configs/synthetic_video_public_pilot.json`

This pilot makes the proposed episode-plan architecture visible before
governed ChildLens work can run. It produces eight short clips: the same four
self-authored scenes rendered once by Wan 2.2 TI2V-5B and once by LTX-2.3. The
clips are an engineering preview only. They cannot select the final generator,
support an equivalence claim, tune a ChildLens-facing prompt, or replace the
larger frozen formal pilot.

## Boundary

The only semantic inputs are the four episode plans committed in the canonical
config. Cloud jobs may receive those plans, pinned public code and weights, the
public Piper German voice, and the resulting generated artifacts.

The following may never enter the cloud job or its output repository:

- ChildLens or BabyView video, audio, frames, or transcripts;
- filenames, identifiers, vocabulary, statistics, prompts, embeddings, or
  other derivatives from either restricted corpus;
- governed paths, calibration decisions, or restricted-data tuning; and
- API-generated prompt expansions.

The Hugging Face output dataset is private by default. This is a precaution,
not a statement that the generated media are restricted ChildLens data.

## Frozen preview

Each family receives the same semantic scene and seed (`314159`):

1. pick up and hold a matte red cup;
2. transfer one blue wooden block from the left hand to the right;
3. partly occlude and reveal the same yellow toy car; and
4. walk toward a table, grasp a silver spoon, and stir once.

Every native clip is 121 frames at 24 fps and is normalized to 1280 × 704.
Wan produces silent video. LTX produces joint audio-video, but its native audio
is stripped from every modular comparison clip. The same pinned Piper German
voice is then placed on the episode timeline and muxed into both families.
LTX's native audio remains eligible only for the separately prespecified
scene-1 diagnostic; it is not used in this eight-clip preview.

There are no preview retries. A visibly bad clip remains part of the result.
The formal profile retains three frozen seeds and at most one taxonomy-bound
retry, but this preview does not authorize executing that profile.

## Local validation and compilation

All generated files live below the ignored `runs/` root.

```bash
python scripts/run_synthetic_video_public_pilot.py validate

python scripts/run_synthetic_video_public_pilot.py compile \
  --profile preview \
  --run-id public-preview-20260730
```

Compilation records exact family prompts, prompt hashes, planned output paths,
and blank blinded QA records. It does not download a model or run inference.
To inspect the exact official command without executing it:

```bash
python scripts/run_synthetic_video_public_pilot.py show-model-command \
  --work-order runs/synthetic_video_public_pilot/public-preview-20260730/work_order.json \
  --attempt-id wan__pick_up__s314159__a1__modular \
  --source-root /path/to/pinned/Wan2.2 \
  --weights-root /path/to/pinned/Wan2.2-TI2V-5B \
  --raw-output /tmp/raw.mp4
```

## Cloud execution design

The first cloud target is Hugging Face Jobs because both model families publish
their official weights on the Hub and the job can persist every attempt to one
private dataset repository. The worker is
`scripts/run_synthetic_video_hf_job.py`.

Execution is staged:

1. one CPU-only job clones the exact Nursery commit, verifies the protocol
   hash, creates the private output dataset, and proves persistence without
   inference;
2. one Wan job runs all four Wan attempts and uploads each completed attempt
   immediately;
3. one LTX job runs all four LTX attempts and uploads each completed attempt
   immediately; and
4. the two family roots are downloaded under one ignored run root and compiled
   into the local side-by-side gallery.

The final CPU-only cloud preflight passed at Nursery inference commit
`9c2983355f7ecc41981661cb0dfac0bfa0f6f9d2`. Hugging Face Job
`6a6bfb08b36a6516e96a35eb` checked out that exact commit, matched protocol hash
`956a76f…0a316`, compiled zero inference attempts, verified the private output
dataset, and persisted its control records. The compact record is
`results/synthetic_video_public_pilot_preflight.json`.

The frozen GPU flavor was one Hugging Face `a100-large` (A100 80 GB), at the
frozen price of $2.50 per hour. The approved ceiling was $20. The successful
Wan job (`6a6bfb6323ed89c748ec8c95`) ran for 2,037.65 seconds and the successful
LTX job (`6a6c03d8b36a6516e96a362f`) ran for 609.10 seconds. Their computed
cost is $1.838. Including a conservative four-minute upper bound for two Wan
setup-only failures, total GPU cost remained below $2.005. Each completed
attempt was uploaded before the next one began.

The two failed Wan setup jobs reached no inference. They exposed a strict
FlashAttention import and then an incomplete/eager upstream import surface.
The canonical worker now applies two narrow, hash-recorded runtime source
adaptations: it exposes only the prespecified TI2V runner, and binds the
upstream attention dispatcher so its SDPA fallback is reachable. The final
environment record contains the original and patched source hashes.

The LTX job requires that the authenticated Hugging Face account has already
accepted Google's gated Gemma license for
`google/gemma-3-12b-it-qat-q4_0-unquantized`. The LTX Community License and the
Piper/voice lineage restrictions remain applicable. This repository treats the
preview as non-commercial research and does not grant broader product rights.

Generate the exact two paid job specifications without launching anything:

```bash
python scripts/run_synthetic_video_public_pilot.py cloud-plan \
  --run-id public-preview-20260730
```

The command is bound to the passed preflight's Nursery commit and current
protocol hash. It emits both the structured Hugging Face Jobs arguments and
copyable shell previews, along with the $10-per-family and $20-total ceilings.
It never submits a job.

Official references:

- [Wan 2.2 repository](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 TI2V-5B weights](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [LTX-2 repository](https://github.com/Lightricks/LTX-2)
- [LTX-2.3 weights](https://huggingface.co/Lightricks/LTX-2.3)
- [Piper](https://github.com/OHF-Voice/piper1-gpl)
- [Hugging Face Jobs pricing](https://huggingface.co/docs/hub/en/jobs-pricing)

## Persistence and local viewing

The remote layout is:

```text
<run-id>/
  families/
    wan/
      work_order.json
      environment.json
      run_status.json
      attempts/...
    ltx/
      work_order.json
      environment.json
      run_status.json
      attempts/...
```

After downloading that run root, build the comparison page:

```bash
python scripts/run_synthetic_video_public_pilot.py gallery \
  --run-root runs/synthetic_video_public_pilot/public-preview-20260730
```

For the completed run, the command creates
`runs/synthetic_video_public_pilot/public-preview-20260730/gallery/index.html`.
It places Wan and LTX clips side by side for each scene and links to the blank
QA record for each attempt. The complete generated run remains ignored by Git
and is also retained in the private dataset
`rishisim/nursery-synthetic-video-public-pilot`.

## Review

The preview reviewer answers the eight frozen pass/fail/cannot-judge items:
continuous egocentric shot, anatomy, contact/action completion, identity,
transition order, referent timing, exact speech, and safety. Free-form notes
may describe failure modes, but they cannot become prompt edits or a generator
selection rule.

The initial unblinded assistant screen retained several important failures.
LTX followed the first-person action structure more closely, but all four
clips contain pseudo-subtitle text. Its cup finishes tilted and its spoon
scene duplicates the referent. Wan repeatedly drifts into a third-person view;
its transfer is incomplete and its spoon scene shows a child's face. These
observations are diagnostic only and do not replace the eight pending blinded
human QA records.

The compact execution, integrity, cost, and screening record is
`results/synthetic_video_public_pilot_preview.json`. A polished subset must
never be presented as the pilot.
