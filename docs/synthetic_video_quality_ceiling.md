# LTX-2.3 versus Seedance 2.0 public quality ceiling

**Status:** frozen execution authorized; one unseen private-ACL transport
diagnostic is preserved, and the four-call comparison uses an expiring public
delivery path

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
disables automatic retries and request-payload history and expires the
provider copy after one day. Because fal's Seedance partner endpoint denied
the documented owner-token, signed-URL, and ACL-management paths for the first
private output, the canonical public-only comparison now downloads immediately
from the provider's unguessable public URL. The inputs and outputs contain no
restricted data, and the URL itself remains only in the ignored run record.

Each downloaded Seedance video is center-cropped from 1280×720 to 1280×704,
normalized to 121 frames at 24 fps, and muxed with the exact AAC audio stream
from its paired LTX final. This makes the visible model output the only
intentional family difference.

The four Seedance calls produce 20 seconds total. At the frozen fal rate of
$0.3034 per generated second, the maximum generation charge is:

```text
4 scenes × 5 seconds × $0.3034/second = $6.068
```

The inaccessible first output was generated but never downloaded, viewed, or
quality-screened. It is therefore excluded as a quality-blind infrastructure
failure, not a rejected sample. Its estimated charge is `$1.517`, so the
maximum expected total after generating the four comparison clips is:

```text
$1.517 inaccessible diagnostic + $6.068 comparison = $7.585
```

This remains below the user's `$10` authorization. The provider invoice is not
available to the runner, so both figures remain frozen-rate estimates. No
replacement beyond this single documented transport recovery is authorized if
a new call fails or is moderated.

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
persisted before polling, and each completed expiring public output is
downloaded, normalized, hashed, and recorded before the next request is
submitted. An interrupted invocation resumes the existing provider request
rather than creating a duplicate charge. The runner rejects approvals above or
below the frozen `$6.068` ceiling and rejects any modified work order before
contacting the provider. Paid execution also requires a clean Git worktree and
records the exact 40-character adapter commit in the approval, run status, and
every candidate attempt record. The prior inaccessible request is stored
separately under the ignored diagnostic root and cannot be resumed into the
canonical four-clip comparison.

## Blinded review

After the candidate outputs are present:

```bash
python scripts/run_synthetic_video_quality_ceiling.py gallery \
  --run-id quality-ceiling-20260731
```

The gallery uses opaque hard-linked media aliases (or copies when hard links
are unavailable) and labels cards only as Clip A or Clip B, so local link
targets cannot disclose the family. Family order is independently randomized
per scene. The mapping lives in mode-`0600` `blinding_key.json`, which must
remain unopened until all eight QA records are frozen.

The review uses the same eight items as the completed preview: continuous
egocentric shot, anatomy, contact/action completion, identity, transition
order, referent timing, exact speech, and safety. All four Seedance attempts
remain in the result; no polished subset may be substituted.

The qualitative screening rule is frozen before Seedance output is viewed.
Speech is reported only as an identity control because each candidate final
contains the exact paired LTX audio stream. A material Seedance visual win
requires all four candidate files to be technically valid, at least four more
visual passes in total, at least two scene wins and no scene losses, no increase
in critical visual failures, no safety failure, and equal judgeable-item
denominators within every scene. Anything weaker retains LTX as the
reproducible baseline; a technical or judgeability shortfall is inconclusive.

Before unblinding, validate the eight edited QA records:

```bash
python scripts/run_synthetic_video_quality_ceiling.py review-status \
  --run-id quality-ceiling-20260731
```

Once this reports `READY_TO_UNBLIND`, freeze and finalize the comparison:

```bash
python scripts/run_synthetic_video_quality_ceiling.py finalize-review \
  --run-id quality-ceiling-20260731
```

The finalizer hashes and freezes the complete family-free QA bundle before
opening `blinding_key.json`. It then re-verifies all eight media records and
hashes, computes per-family and per-scene adherence, technical validity,
canonical generated-attempt failure rate, overall provider-generation failure
rate, end-to-end artifact failure rate including the inaccessible diagnostic,
and the frozen-price cost estimate. It writes `review/review_summary.json` plus
`review/recommendation.md` under the ignored run root. The provider invoice
remains a separate billing source and is never inferred as an observed value.

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
