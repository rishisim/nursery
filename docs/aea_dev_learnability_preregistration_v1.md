# AEA development learnability preregistration v1

**Frozen:** 2026-07-14T23:06:30Z  
**Protocol ID:** `aea-dev-learnability-v1`  
**Source examples SHA-256:** `7bda9e9b02da94b95a9199e02d14db8dd6d6059cda6389673d92d930ce95a98e`  
**Scientific role:** adult, partly scripted sensor-format analogue. This work is not developmental evidence and is not BabyView-like.

## Purpose and prospective status

This bounded iteration asks, before the locked five-fold/seven-seed/eight-epoch experiment, whether the ASR-anchored action task is visually valid and learnable on development data and whether synchronized six-axis head IMU contains paired action information. The only permitted terminal recommendation is GO, REVISE, or STOP under the rules below.

The earlier reduced infrastructure smoke used all 247 records. Therefore the reserve below is **prospective from this iteration onward**, not pristine in an absolute sense. No audit judgment, newly trained model output, or development endpoint was examined before this file and its machine-readable companion were created. Development estimates remain exploratory. Any later result on the reserve must be explicitly labeled prospective confirmation and must use a separately frozen confirmation analysis; this iteration will not inspect reserve pixels, IMU, labels beyond already available lexical metadata, or outcomes.

Endpoint, subset, threshold, budget, or mapping selection after outcomes is prohibited. Null and contrary results are valid.

## Frozen group-safe partition

The partitioning unit is `event_group = location + script + performed sequence`; concurrent recordings/views share this value and therefore cannot cross partitions. All windows from a recording and all concurrent views stay together.

Exactly one event group per location is assigned to the reserve. The selected five-group combination minimizes the following metadata-only score over the Cartesian product of one group per location:

`abs(Nreserve - round(0.20 * 247))/247 + 0.5 * L1(action_distribution_reserve, action_distribution_all) + 0.5 * common_action_undercoverage`,

where `common_action_undercoverage` is the mean shortfall from two reserve windows for actions having at least ten windows overall. Ties are broken by SHA-256 of `aea-dev-confirm-v1|<sorted group ids>`. Only event IDs and lexical-label support enter this rule; no pixels, IMU values, audit judgments, or model outputs enter it.

Frozen reserve event groups:

- `loc1_script4_seq4`
- `loc2_script5_seq6`
- `loc3_script2_seq3`
- `loc4_script3_seq4`
- `loc5_script4_seq3`

The reserve contains 55 windows, five performed-event groups, and six recordings. Development contains the other 192 windows, 18 performed-event groups, and 22 recordings. The partition manifest must record every example ID, group, partition, source hash, and algorithm version. Any mismatch is a hard failure.

## Frozen label mappings and support rules

The fine label is the existing canonical ASR lexical `action_verb`; it is not a human action annotation. A fine action is eligible for a modeled endpoint only when development has at least six windows across at least three performed-event groups. This support filter is outcome-independent and is applied once globally before fold construction.

The coarse mapping is fixed as follows:

- `locomotion_transport`: walk, bring, take, get, move, remove, pick, grab
- `object_state_manipulation`: put, set, open, close, turn, press, stack, fold, cut
- `food_drink_preparation`: make, pour, cook, serve, eat, drink
- `cleaning_grooming`: wash, clean, wipe, brush, vacuum
- `media_leisure`: play, read, watch

A coarse class is eligible only with at least six development windows across at least three performed-event groups.

The prospectively semantic high-motion set is fixed as `walk, bring, take, get, move, remove, vacuum, wash, clean, wipe`. It reflects expected sustained locomotor/transport/cleaning behavior, not observed IMU magnitude or test performance. Within this set, the same six-window/three-group fine-label support rule applies. No data-derived “high motion” threshold may be created.

## Development-only visual audit

The audit sample size is 48 development windows. Sampling is deterministic and support-aware:

1. Sort eligible development windows within each fine action by SHA-256 of `aea-audit-v1|example_id`.
2. Repeatedly select one window from the currently least-sampled action, breaking ties by total action support then action name, while preferring the candidate from the currently least-sampled performed-event group; continue to 48.
3. No reserve record is eligible. The pre-label manifest is written before viewing any sampled frames.

All eight available frames spanning the six-second window must be inspected as a temporal strip; a single frame is insufficient unless files are missing, in which case the label is `not_visually_judgeable`. Labels are:

- `clear_match`: visible wearer action clearly instantiates the spoken verb;
- `plausible_or_ambiguous`: visible motion is compatible but timing, object, agency, or verb sense is ambiguous;
- `mismatch`: visible wearer action contradicts or does not instantiate the lexical anchor;
- `not_visually_judgeable`: occlusion, absent hands/objects, insufficient temporal evidence, or corrupt/missing frames prevents judgment.

Every row retains example ID, action, group, transcript, frame paths, category, and concise rationale. Report action/group/location coverage; category counts; the judgeable rate; `clear_match` rate; `clear_match + plausible_or_ambiguous` rate; and `mismatch` rate, each with two-sided 95% Wilson intervals. Audit gates are: at least 75% judgeable; lower Wilson bound for clear-or-plausible at least 0.60; and upper Wilson bound for mismatch at most 0.25. High-motion and coarse audit summaries are secondary and use the same fixed labels without relabeling.

## Development-only video-language learnability

The model is the repository video-text contrastive learner trained with motor null/masked. Prediction uses RGB and action-text prompts; motor is absent. Fine action-balanced 2AFC accuracy is primary; coarse action-balanced 2AFC and the semantic high-motion fine endpoint are secondary. Chance is 0.50 for 2AFC.

Splits are five deterministic `StratifiedGroupKFold` folds with `shuffle=True`, `random_state=41031`, stratifying by the endpoint label and grouping by performed-event group. Global eligibility is fixed before splitting. Every example appears in one held-out fold; concurrent views cannot cross folds. Training and held-out predictions are both retained per action and fold.

Training is frozen at three seeds (`3101, 3102, 3103`), 30 epochs, eight frames, image size 64, hidden dimension 64, embedding dimension 48, batch size 24, learning rate 0.001, and no early stopping or outcome-directed rerun. Initialization, order, updates, and inputs are recorded. The primary fine endpoint passes when mean held-out 2AFC is at least 0.60, its 95% event-group bootstrap lower bound exceeds 0.50, and its chance-permutation p-value is at most 0.05. Secondary endpoints use the same gate and are not interchangeable after outcomes.

Capacity/optimization sanity is a predeclared easier positive control: the same architecture is fit for 120 epochs with seed 3199 to a deterministic support-balanced 64-window development subset and evaluated on that same subset. It passes only if 2AFC is at least 0.90 and final video-text loss is at least 20% below first-epoch loss. This control is evidence of capacity/optimization only, not generalization.

Intervals use 2,000 event-group cluster bootstrap replicates (`seed=42101`). Chance calibration uses 2,000 within-fold label permutations (`seed=42103`) applied to saved predictions, preserving fold sizes and class counts; p-values use the plus-one correction. No reserve result is computed.

## Development-only head-IMU information test

Each complete six-axis trajectory is summarized without label-informed selection using, per axis: mean, standard deviation, minimum, maximum, median, 10th/25th/75th/90th percentiles, RMS, mean absolute value, mean square, and first-difference mean/standard deviation/RMS; plus all 15 axis-pair Pearson correlations and FFT energy proportions in fixed bands 0–0.5, 0.5–2, 2–5, and 5–25 Hz. Features are standardized within each training fold.

The classifier is multinomial logistic regression (`C=1.0`, L2, `class_weight=balanced`, maximum 5,000 iterations). Fine action-balanced accuracy is primary; coarse and semantic-high-motion fine accuracies are secondary. It uses the identical five folds and endpoint eligibility rules as the grounding analysis.

For donor seeds `5101, 5102, 5103`, synchronized and shuffled conditions are paired on identical labels/folds. Shuffling is performed separately inside each training or test side as a bijection of whole-window IMU trajectories; donors must be in the same split side, must not be self, and must come from a different performed-event group. If a valid derangement is impossible, that fold/endpoint fails rather than weakening the rule. Donor maps, self-match rate, same-group rate, pairing digests, and feature dimensions are retained.

The IMU gate requires synchronized minus shuffled action-balanced accuracy of at least +0.05, a 95% paired event-group bootstrap lower bound above zero, a paired randomization p-value at most 0.05, and synchronized accuracy itself above chance with p at most 0.05. Two thousand paired cluster-bootstrap replicates use seed 52101; 10,000 event-group sign-flip/randomization draws use seed 52103; chance calibration uses 2,000 within-fold label permutations with seed 52105. All p-values use plus-one correction.

## Leakage, pairing, manipulation, and reporting checks

Hard checks are: source hash match; zero development/reserve example, sequence, or event-group overlap; complete concurrent-view grouping; each development row held out exactly once; no fold group overlap; zero shuffled self or same-event-group donors; whole-window donor bijections; identical paired folds/labels/features; finite features; training budgets exactly matched; no motor access in grounding evaluation; and no reserve file access by the development runner. Optimization histories, train-versus-held-out results, per-action support/performance, fold assignments, donor maps, permutation results, and bootstrap summaries are mandatory outputs.

New code receives unit tests for deterministic partitioning, reserve exclusion, fold leakage, donor derangement, fixed semantic mappings, interval determinism, and decision-rule boundaries. The full repository suite and `git diff --check` must pass.

## Frozen decision rule

- **GO** only if the visual-audit gates, capacity control, fine grounding gate, fine IMU gate, and every hard check pass. GO authorizes only a separately preregistered run on the prospective reserve; it does not itself authorize the expensive locked experiment or external outreach.
- **REVISE** if GO fails but all hard checks and capacity control pass and either (a) both coarse grounding and coarse IMU gates pass with the audit gate, or (b) both semantic-high-motion grounding and IMU gates pass with the corresponding audit gate. REVISE also applies when the capacity control fails, because this iteration then cannot distinguish task failure from optimization failure. It authorizes a new prospective protocol/annotation or coarse-label design, never post-hoc endpoint substitution and never reserve peeking.
- **STOP** when hard checks and capacity control pass but the audit gate fails or neither the fine nor either predeclared alternative jointly passes grounding and IMU gates. STOP means do not run the locked experiment and do not recommend contacting Professor Frank from this evidence.

If a hard integrity check fails, the scientific endpoints are invalid and the recommendation is REVISE for implementation repair, not a scientific GO or STOP. Thresholds are applied mechanically; proximity to a threshold does not change the label.

