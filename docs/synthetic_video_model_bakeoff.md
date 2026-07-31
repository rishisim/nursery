# Gemini Omni Flash and MiniMax H3 public model bakeoff

**Status:** frozen and compiled without API calls; blocked at the new
credential and `$4.66` spend gate.

**Canonical protocol:** `configs/synthetic_video_model_bakeoff.json`

**Canonical runner:** `scripts/run_synthetic_video_model_bakeoff.py`

This is a scientifically distinct extension of the completed LTX-2.3 versus
Seedance 2.0 screen. It asks whether Gemini Omni Flash or MiniMax H3 shows a
clear task-specific advantage over LTX and whether either is competitive with
Seedance on the same four public-only scenes.

The completed two-family protocol and result are immutable inputs. LTX and
Seedance will not be regenerated. Their eight final videos, the four exact
compiled LTX prompts, and the paired LTX audio are the shared source of truth.
The extension adds exactly eight new calls: four to Gemini and four to
MiniMax.

No ChildLens/BabyView input or derivative is permitted. The comparison may
inform qualitative model selection, but it does not authorize using either
provider's output as learner training data.

## Frozen providers and requests

Gemini uses Google's official paid preview Interactions API:

```text
model: gemini-omni-flash-preview
endpoint: POST https://generativelanguage.googleapis.com/v1beta/interactions
duration: 5 seconds
aspect ratio: 16:9
delivery: URI
store: false
background: false
stream: false
credential: GEMINI_API_KEY
```

Google documents Gemini Omni Flash output as 720p at 24 fps. The model
generates native audio, but that audio is discarded.

MiniMax uses the official H3 V2 API:

```text
model: MiniMax-H3
endpoint: POST https://api.minimax.io/v2/video_generation
duration: 5 seconds
resolution: 2K
aspect ratio: 16:9
credential: MINIMAX_API_KEY
```

The public MiniMax H3 schema currently exposes only 2K. Its less expensive
768p tier is documented as closed beta, so it is not assumed available. H3 is
asynchronous: the runner persists the task ID before polling the official
query endpoint and downloads the time-limited result immediately.

Neither documented model-specific request exposes a supported input seed.
No seed is sent. Each provider receives the exact compiled LTX prompt with no
provider-specific wording change, one attempt per scene, and no automatic or
protocol retry.

## Presentation control

Every new output is normalized through the same shared delivery path:

1. discard native provider audio;
2. scale and center-crop to 1280×704;
3. normalize to 24 fps and 121 frames; and
4. stream-copy the exact AAC payload from the paired LTX final.

The 2K H3 generation and 720p Gemini generation are therefore displayed at
the same comparison resolution, although native generation resolution remains
a model/provider confound and will be reported.

The gallery contains all four families for every scene: 16 clips total. Card
order is independently blinded per scene, and opaque media filenames prevent
the local path from revealing the family.

## Review rule

The primary task-adherence instrument remains unchanged:

- continuous egocentric shot;
- anatomy;
- contact/action completion;
- identity;
- transition order;
- referent timing; and
- safety.

Speech remains an excluded identity control because all four family finals
use the same paired audio. Two prospectively frozen secondary presentation
items now record unwanted text/branding and malformed frame composition. This
captures the pseudo-caption and border artifacts that were visible in the
first comparison but absent from its primary score.

For each new family, the finalizer reports:

- primary and presentation pass counts;
- critical and safety failures;
- scene wins, losses, and ties versus LTX;
- scene wins, losses, and ties versus Seedance;
- whether it clears the original material-advantage threshold over LTX; and
- whether it remains within the frozen competitiveness band around Seedance.

This remains a one-rater, four-scene exploratory screen, not a formal model
ranking.

## Cost gate

Google prices Gemini Omni Flash video output at 5,792 tokens per second of
720p video and `$17.50` per million video-output tokens. Twenty requested
seconds therefore have `$2.0272` of video-output cost. The protocol reserves
8,000 input tokens at `$1.50` per million and a small duration/billing margin,
freezing Gemini's maximum expected charge at `$2.06`.

MiniMax lists H3 2K at `$0.13` per generated second:

```text
4 scenes × 5 seconds × $0.13/second = $2.60
```

The exact new execution gate is therefore:

```text
$2.06 Gemini maximum expected charge
+ $2.60 MiniMax maximum expected charge
= $4.66 maximum expected new charge
```

Actual invoices remain provider billing records and are not inferred by the
runner. A failed or moderated call may be billed according to provider policy.
No replacement or retry is authorized.

## No-cost setup

The generated work order is retained under the ignored run root
`runs/synthetic_video_model_bakeoff/model-bakeoff-20260731`.

```bash
python3 scripts/run_synthetic_video_model_bakeoff.py validate

python3 scripts/run_synthetic_video_model_bakeoff.py compile \
  --run-id model-bakeoff-20260731

python3 scripts/run_synthetic_video_model_bakeoff.py plan \
  --run-id model-bakeoff-20260731
```

`compile` writes eight immutable requests, 16 family-free QA templates, and a
mode-`0600` blinding key without contacting either provider. `plan`
hash-verifies all four LTX and all four Seedance finals and reports only
credential presence, never credential values.

At the current gate, both `GEMINI_API_KEY` and `MINIMAX_API_KEY` are missing,
paid execution is not authorized in the frozen config, and no new provider
request has been sent.

## Paid execution and review

After both credentials are present and the user separately approves the new
`$4.66` ceiling, the protocol authorization is committed before execution.
The clean-commit runner then uses:

```bash
python3 scripts/run_synthetic_video_model_bakeoff.py run \
  --run-id model-bakeoff-20260731 \
  --approved-spend-usd 4.66
```

After all eight finals validate:

```bash
python3 scripts/run_synthetic_video_model_bakeoff.py gallery \
  --run-id model-bakeoff-20260731

python3 scripts/run_synthetic_video_model_bakeoff.py review-status \
  --run-id model-bakeoff-20260731

python3 scripts/run_synthetic_video_model_bakeoff.py finalize-review \
  --run-id model-bakeoff-20260731
```

The finalizer freezes and hashes every complete family-free QA file before it
opens the separate family key. Full run media and provider records remain
ignored. Only a compact aggregate result and decision record may be retained
in Git.

## Official references

- [Gemini Omni Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-omni-flash)
- [Gemini Omni Flash video guide](https://ai.google.dev/gemini-api/docs/omni)
- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [MiniMax H3 video guide](https://platform.minimax.io/docs/guides/video-generation)
- [MiniMax H3 V2 OpenAPI schema](https://platform.minimax.io/docs/api-reference/video/generation/api/v2-video-generation.json)
- [MiniMax pay-as-you-go pricing](https://platform.minimax.io/docs/guides/pricing-paygo)
