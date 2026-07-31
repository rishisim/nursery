# LTX-2.3 versus Seedance 2.0 public quality ceiling

**Status:** frozen and locally validated; Seedance credential and explicit
$6.068 API-charge approval are pending

**Canonical protocol:** `configs/synthetic_video_quality_ceiling.json`

This comparison asks whether hosted Seedance 2.0 materially improves the four
task-specific scenes over the already-generated LTX-2.3 baseline. It is a
scientifically distinct follow-on to the completed Wan/LTX public preview. It
does not replace or mutate that run: the four scenes, compiled LTX prompts,
LTX media, and German audio remain the shared source of truth.

The comparison may support a recommendation to pursue Seedance access and
legal/governance clearance. It does not authorize using Seedance output as
learner training data.

## Why fal

The frozen route is fal's authorized standard endpoint:
`bytedance/seedance-2.0/text-to-video`.

Direct BytePlus access is excluded for this workflow. Its published
Seedance-specific terms state both that the service is unavailable in the
United States and that outputs may not be used to develop or train other
models absent a separate agreement. fal marks its Seedance endpoint for
commercial use and makes it globally accessible, but fal also incorporates
applicable third-party terms. Written provider and institutional confirmation
is therefore still required before any learner sees these outputs.

## Frozen comparison

Seedance receives the exact compiled LTX prompt for each of the four existing
scenes, unchanged:

1. pick up and hold the red cup;
2. transfer the blue cube between hands;
3. partly occlude and reveal the yellow car; and
4. approach the table, grasp the spoon, and stir once.

The standard, not fast or mini, endpoint is used at 720p, 16:9, high bitrate,
and five seconds. Native Seedance audio is disabled. The current live fal
input schema does not expose a caller-supplied seed, so no seed is sent and
the provider-returned seed is recorded. The LTX baseline remains seed
`314159`; the inability to match candidate and baseline seeds is an explicit
closed-endpoint comparability limitation.

Provider and protocol retries are both zero. The API request explicitly
disables automatic retries and request-payload history, makes the output
private to the calling fal account, and expires the provider copy after one
day.

Each downloaded Seedance video is center-cropped from 1280×720 to 1280×704,
normalized to 121 frames at 24 fps, and muxed with the exact AAC audio stream
from its paired LTX final. This makes the visible model output the only
intentional family difference.

The four Seedance calls produce 20 seconds total. At the frozen fal rate of
$0.3034 per generated second, the maximum generation charge is:

```text
4 scenes × 5 seconds × $0.3034/second = $6.068
```

No replacement request is authorized if a call fails or is moderated.

## No-cost setup

All generated plans and media live under ignored `runs/` roots.

```bash
python scripts/run_synthetic_video_quality_ceiling.py validate

python scripts/run_synthetic_video_quality_ceiling.py compile \
  --run-id quality-ceiling-20260731

python scripts/run_synthetic_video_quality_ceiling.py plan \
  --run-id quality-ceiling-20260731
```

`validate` checks the quality-ceiling protocol against the exact completed
public-pilot hash. `compile` writes four immutable requests, eight family-free
QA records, and a separate blinding key without contacting fal. `plan`
re-verifies every local LTX hash and reports credential/spend readiness.

## Paid execution gate

The runner reads `FAL_KEY` from the environment and never writes or prints it.
After the key is installed and the exact ceiling is explicitly approved:

```bash
python scripts/run_synthetic_video_quality_ceiling.py run \
  --run-id quality-ceiling-20260731 \
  --approved-spend-usd 6.068
```

Requests run sequentially. The submission ID and resumable status URLs are
persisted before polling, and one completed private output is downloaded,
normalized, hashed, and recorded before the next request is submitted. An
interrupted invocation resumes the existing provider request rather than
creating a duplicate charge. The runner rejects approvals above or below the
frozen `$6.068` ceiling and rejects any modified work order before contacting
the provider.

## Blinded review

After the candidate outputs are present:

```bash
python scripts/run_synthetic_video_quality_ceiling.py gallery \
  --run-id quality-ceiling-20260731
```

The gallery uses opaque media symlinks and labels cards only as Clip A or Clip
B. Family order is independently randomized per scene. The mapping lives in
`blinding_key.json`, which must remain unopened until all eight QA records are
frozen.

The review uses the same eight items as the completed preview: continuous
egocentric shot, anatomy, contact/action completion, identity, transition
order, referent timing, exact speech, and safety. All four Seedance attempts
remain in the result; no polished subset may be substituted.

## Decision boundary

A clear Seedance visual win means “pursue terms and governance clearance,” not
“begin Seedance training-data generation.” The full-project gate remains:

1. written confirmation that the intended non-competing learner-training use
   is permitted;
2. institutional approval for the selected provider and data-processing
   terms;
3. an acceptable dataset-scale cost and failure-rate projection; and
4. a frozen complete dataset before learner training begins.

Until all four pass, Seedance is a public-only qualitative ceiling and LTX
remains the reproducible sensitivity baseline.
