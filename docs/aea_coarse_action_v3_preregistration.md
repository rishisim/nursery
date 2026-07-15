# AEA terminal coarse-action and IMU diagnostic preregistration v3

**Frozen:** 2026-07-15T00:50:15Z  
**Protocol ID:** `aea-coarse-action-v3`  
**Scientific role:** adult, partly scripted sensor-format analogue. This is not developmental evidence and is not BabyView-like.  
**Status:** terminal development diagnostic; no confirmatory effect test is authorized.

## Motivation, questions, and ordering

V2 produced the prespecified non-severe `REVISE` result: observable-action agreement was 81.9% (kappa 0.777), but simplified modeling support was only 38/72 and the agency interface missed its gate. That result authorizes exactly one ontology/interface repair. V3 uses that authorization once, prospectively, to ask two separate questions:

1. Is the fixed distinction between gross wearer motion and goal-directed object/body interaction learnable from video and predictable from synchronized head IMU across held-out performed-event groups?
2. Independently of sensor learnability, does AEA natural speech align with the visible wearer's action often and reliably enough to remain a language-grounding analogue?

This preregistration, the machine-readable protocol, and the codebook are frozen before any v1/v2 completed pass label or rationale is read, before any dense image is reviewed under the v3 interface, and before any v3 annotation, split, capacity, language, IMU, or acquisition outcome is seen. The permitted order is:

1. verify the frozen v2 dense manifest and zero-reserve receipt by hash and metadata only;
2. reblind exactly the same 72 development rows and reference exactly the same 31-frame evidence without opening raw VRS or IMU;
3. run two separately randomized, context-isolated, mutually blinded model-assisted passes, each with a sealed video-only stage before transcript reveal;
4. compare the passes once, without adjudication, and compute the coarse annotation/support gate and separate language-alignment gate;
5. if the coarse gate passes, solve one deterministic constrained group fold and donor assignment;
6. if annotation and split/donor gates pass, run the frozen same-row video action-head capacity control;
7. if annotation, split/donor, and capacity pass, run the fixed IMU diagnostic;
8. apply the terminal route decisions and prospective acquisition rule once.

No post-outcome relabeling, adjudication, ontology change, subset search, fold retry, feature change, threshold change, extra seed, or model rerun is permitted. Language-gate failure does not prevent the sensor diagnostic when the sensor stage gates pass, and sensor results cannot rescue the language gate.

## Immutable sample, evidence, and reserve exclusion

V3 contains exactly the 72 rows in `output/aea_visible_action_v2/dense_clip_manifest.json`, SHA-256 `772b194c145aa305775a1feade279c33ae0f38fedfb9e7ccdb945cc03f82077e`, sample digest `1dfc09d2490bb80c355dab2b326bceaf2eeff82080068dbc316b7fb4a6d04384`, and sorted example-ID digest `a7868ca5999049852e3f47c1f7207adac2bcea057bdea5fe2f3d1f56c91fe3e3`. No row may be added, removed, selected, or substituted. All 18 development event groups remain represented.

Each row uses the existing 31 upright RGB frames at the centers of equal bins in the exact six-second window. V3 references the immutable v2 files and does not rematerialize RGB. The v2 access receipt, SHA-256 `2f20f459426292945a119b83c2b87b1d59de0694f60fa2f636cc2d2811b72a4b`, records 2,232 development RGB queries and zero reserve RGB/IMU access. V3 must verify every referenced frame and contact sheet is beneath the v2 dense-evidence root and matches the manifest; it must not construct a reserve media path.

The five reserve groups remain forbidden: `loc1_script4_seq4`, `loc2_script5_seq6`, `loc3_script2_seq3`, `loc4_script3_seq4`, and `loc5_script4_seq3`. The monolithic 247-row source, reserve RGB, reserve IMU, signed URLs, raw VRS, audio, and new downloads are forbidden. Development IMU may be opened only by the conditional IMU runner after all sensor stage gates pass.

## Single coarse ontology repair

The primary visible wearer action in the six-second evidence receives exactly one label:

1. `gross_body_motion`: wearer locomotion, a posture transition, or deliberate body/head orientation without direct object manipulation;
2. `object_or_body_interaction`: reaching/grasping, carrying/placing, operating/state change, food/material handling, cleaning/grooming, or another goal-directed manipulation of an object or body surface;
3. `no_goal_directed_visible_action`: no defensible goal-directed wearer action is visible;
4. `uncertain`: image quality, occlusion, temporal ambiguity, or competing evidence prevents a defensible label.

Only the first two labels are modeled. There are no subtypes and no post-hoc endpoint subsets. Looking or head motion is `gross_body_motion` only when it is deliberate orientation evident from the sequence; static possession, talking, listening, and incidental camera jitter are not goal-directed actions. When manipulation and gross motion overlap, direct manipulation takes precedence. Otherwise select the action spanning the temporal midpoint; if none spans it, use the nearest evidence interval, then the longer interval, then the ontology order above.

Visible-action labeling is video-only. The stage-1 packet excludes transcript and anchored verb. Transcript semantics may never be used to infer the visible action, confidence, evidence interval, or rationale. Those fields are sealed before stage 2.

## Simplified language interface

After the visible label is sealed, stage 2 reveals the anchored ASR verb and transcript and records:

- `wearer_action`: the anchored expression refers to a concrete action by the wearer;
- `nonwearer_or_nonliteral`: the expression refers to another person, media/phone narration, or a figurative/non-action usage;
- `unclear`: the referent cannot be resolved defensibly.

Temporal relation is separate: `aligned`, `before`, `after`, `none`, or `unclear`. `aligned` means a corresponding visible wearer action overlaps the centered anchor between frames 15 and 16; `before` or `after` means the corresponding action is visibly confined to that side within the six-second window; `none` means no corresponding visible wearer action occurs in the window. Each stage has its own `high`, `medium`, or `low` confidence and rationale of at most 25 words. Evidence frames range from 0 through 30 and may be null only for `no_goal_directed_visible_action` or `uncertain`.

## Blinded model-assisted passes

V3 derives new opaque blind IDs using `aea-coarse-action-v3-blind|example_id`. Pass A and pass B use independent SHA-256 order salts `aea-coarse-action-v3-pass-a` and `aea-coarse-action-v3-pass-b`. Packets expose no example ID, sequence, event group, location, v1 audit membership, v1/v2 label, other pass, agreement, support, or model outcome.

Each pass is a context-isolated model-assisted run. It must not read any v1/v2 completed labels or rationales or the other v3 pass. These are model-assisted development labels, not human annotations and not human inter-rater reliability. Both passes must first complete all 72 video-only rows; only then may that pass receive its transcript-stage packet. No disagreement is adjudicated and the interface is not tuned after comparison.

## Frozen coarse annotation/support gate

A modeled consensus row requires exact agreement on one of the two modeled actions and `medium` or `high` visible-action confidence in both passes. Report confusion matrices, exact agreement, unweighted Cohen's kappa, confidence-weighted exact agreement, Wilson intervals, and consensus support by class, event group, and location.

The coarse annotation/support gate passes only if all conditions hold:

- both passes contain 72 unique valid rows, with sealed stage ordering and complete rationales;
- at least 85% of rows are judgeable in each pass (`uncertain` is not judgeable; `no_goal_directed_visible_action` is);
- coarse-action exact agreement is at least 0.70 and unweighted kappa is at least 0.50;
- at least 60% of all 72 rows form modeled consensus;
- both modeled labels independently have at least eight consensus windows spanning at least four event groups.

Any miss is a terminal coarse-route failure. It is not another authorization to revise the ontology.

## Frozen language-alignment viability gate

Referent reliability requires exact agreement at least 0.75 and unweighted Cohen's kappa at least 0.50. Temporal agreement is reported but is not allowed to block the sensor route.

An ASR anchor counts as consensus language-aligned only when, independently in both passes: (a) `asr_referent` is `wearer_action`; (b) the visible-action label is one of the two modeled actions; and (c) temporal relation is `aligned`, `before`, or `after`. The denominator is always all 72 rows, including uncertain and no-action rows.

AEA remains viable for language grounding only if all conditions hold:

- at least 18/72 rows (25%) meet the consensus definition;
- neither pass has fewer than 15/72 `wearer_action` labels (the integer form of the 20% minimum);
- referent exact agreement and kappa both pass their frozen reliability thresholds.

Failure yields `STOP_LANGUAGE_ROUTE` regardless of video/IMU results. It forbids acquisition for language grounding and cannot be weakened based on v2's low alignment.

## Frozen folds and donors

Only medium/high-confidence exact consensus rows from the two modeled classes enter modeling. Both labels must meet the global support gate. Exactly one three-fold assignment partitions all 18 performed-event groups, six whole groups per fold, with no sequence or event-group leakage. The existing deterministic SciPy MILP machinery is used once. For each label and fold it requires at least two test windows from two test groups and at least four training windows from three training groups. It lexicographically minimizes label imbalance, location imbalance, and the salted tie objective `aea-coarse-action-v3-fold|group|fold`.

Training-side whole-window donor bijections use seeds 6201, 6202, and 6203. IMU-diagnostic test-side bijections use seeds 6301, 6302, and 6303. Every donor must be different-row and different-event-group; the largest-group half-side theorem and deterministic bipartite matching must agree. Every required map must be complete. Infeasibility is persisted from the single solver run and controls are never relaxed. A primary motor-withheld grounding evaluation would not require test-side shuffled IMU, but no such primary grounding run is authorized in v3.

## Frozen video capacity control

Only after the coarse annotation and split/donor gates pass, a supervised coarse action head is trained and evaluated on the same consensus rows. It is an optimization/capacity control, not held-out generalization evidence, and uses no natural transcript or IMU.

The budget is unchanged from v2: 12 evenly sampled indices from the 31 frames, 64-pixel RGB, `VideoEncoder(hidden_dim=64, embedding_dim=64)` plus one linear classifier, batch size 16, AdamW learning rate 0.001, 120 epochs, inverse-frequency class weights, and seeds 7201, 7202, and 7203. There is no augmentation, early stopping, tuning, or rerun.

The gate requires mean training balanced accuracy at least 0.90, every seed at least 0.85, and mean proportional cross-entropy loss reduction at least 0.50. Failure yields `STOP_AEA_ROUTE`. Natural transcripts are not trained as a capacity substitute; the language gate above is the terminal language diagnostic.

## Frozen IMU diagnostic

Only after coarse annotation, split/donor, and capacity gates pass, the two modeled actions are predicted from synchronized six-axis head IMU. V3 reuses the label-independent v1/v2 129-dimensional summary features, training-fold standardization, and class-balanced L2 logistic regression (`C=1`, `max_iter=5000`). No alternative feature, classifier, subset, fold, or threshold is tried.

Report fold train and group-held-out balanced accuracy, per-class accuracy, macro chance (`1/2`), an event-group bootstrap (2,000 draws, seed 8201), within-fold label-permutation chance (2,000 draws, seed 8203), synchronized versus each paired whole-window shuffled test donor, a paired event-group bootstrap, and 10,000 event-group sign-flip randomizations (seed 8205).

The IMU gate requires all of:

- synchronized held-out balanced accuracy at least macro chance + 0.10;
- its 95% event-group bootstrap lower bound above macro chance;
- chance-permutation p-value at most 0.05;
- synchronized-minus-mean-donor balanced accuracy at least 0.05;
- paired bootstrap lower bound above zero;
- paired randomization p-value at most 0.05;
- mean train-minus-held-out balanced-accuracy gap at most 0.35.

A valid null, contrary estimate, or severe overfit is terminal `STOP_AEA_ROUTE`; threshold proximity cannot override the rule.

## Terminal scientific and formal decisions

Integrity failure maps to formal `REVISE` and authorizes only implementation repair without new evidence. Otherwise precedence is:

1. Any coarse annotation/support, split/donor, capacity, or IMU failure yields scientific `STOP_AEA_ROUTE` and formal `STOP`.
2. If every sensor gate passes but the language gate fails, language is `STOP_LANGUAGE_ROUTE`, the limited sensor conclusion is `RETAIN_SENSOR_ANALOG_ONLY`, and the formal field is `REVISE`. This permits only limited adult sensor/action calibration evidence; it is not language-grounding evidence.
3. Only if every sensor and language gate passes is the conclusion `GO_TO_TWO_HUMAN_ANNOTATORS` and the formal field `GO`.

No decision authorizes reserve access, the locked experiment, downloads, license acceptance, outreach, or a scientific effect claim.

## Prospective storage rule

V3 uses the already sanitized v2 release-metadata summary (SHA-256 `8b3e5a5dd50a4176b6cf5c316935dd2349dd2c3df658c372ecd708f0e35ae39f`) and current free-space metadata; it does not reopen a signed manifest. Additional recordings may be recommended only if coarse annotation and capacity pass, the relevant route remains scientifically viable under the decision rules above, and the sole certified remaining bottleneck is event-group support or inferential power with synchronized-minus-donor point estimate at least +0.05, synchronized accuracy at least chance + 0.10, and train-minus-held-out gap at most 0.35. Valid nulls, negative/subthreshold estimates, severe overfit, language failure for a language-grounding purpose, or any ontology/capacity failure forbid acquisition.

If and only if that conjunction passes, the prior metadata-only limits remain: at most 12 new recordings, annotations plus main VRS only, at most 40 GiB declared, and at least 150 GiB projected free. Otherwise the exact recommendation is `no_additional_acquisition`, with no bounded plan and no storage use.

## Artifacts and prohibitions

All v3 artifacts are new files under `docs/aea_coarse_action_v3_*` or `output/aea_coarse_action_v3/`. Required artifacts include the frozen protocol/codebook and receipts, reblinded fixed manifest and stage packets, two raw passes and isolation receipts, agreement/consensus and language reports, reserve-access receipt, split/donor result, capacity and conditional IMU results or explicit not-run receipts, acquisition decision, canonical JSON, exact provenance, and a concise scientific report.

V1, v2, and smoke evidence must remain untouched. No commit, push, deletion, prospective-reserve inspection, locked run, Professor Frank contact, license acceptance, recording download, signed-URL exposure, or auxiliary task is authorized.
