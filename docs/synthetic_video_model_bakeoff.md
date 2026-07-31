# Gemini Omni Flash and MiniMax H3 public model bakeoff

**Status:** execution authorized after both credentials were installed. The
user approved up to `$6.00`; the immutable runner remains hard-limited to the
eight-call `$4.66` maximum expected plan.

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
delivery: inline
store: false
background: false
stream: false
credential: GEMINI_API_KEY
```

Google documents Gemini Omni Flash output as 720p at 24 fps. The model
generates native audio, but that audio is discarded.

The REST payload encodes the duration as `"5s"`. Google's live Interactions
OpenAPI marks this field as `google-duration`, whose ProtoJSON representation
requires the trailing `s`. Delivery remains inline because the live endpoint
requires `store: true` for URI delivery, which conflicts with this protocol's
request-storage boundary.

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
runner. A failed or moderated call after provider acceptance may be billed
according to provider policy. No replacement or quality retry is authorized
after an accepted provider submission.

A request rejected at schema validation before an interaction/task ID or media
exists is retained as a transport diagnostic and does not count as one of the
eight generated attempts. Such a correction must update the canonical request,
pass validation, and run from a clean Git commit; provider billing remains
unknown.

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

Both `GEMINI_API_KEY` and `MINIMAX_API_KEY` are supplied from a mode-`0600`
user-owned environment file. Credential values are neither printed nor
persisted in the run. Paid execution is authorized up to `$6.00`, while the
runner accepts only the frozen `$4.66` plan value.

## Paid execution and review

With both credentials present and the user's `$6.00` ceiling recorded, the
clean-commit runner still uses the narrower frozen plan value:

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
