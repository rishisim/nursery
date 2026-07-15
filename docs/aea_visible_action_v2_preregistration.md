# AEA visible-action rescue preregistration v2

**Frozen:** 2026-07-14T23:54:45Z  
**Protocol ID:** `aea-visible-action-v2`  
**Scientific role:** adult, partly scripted sensor-format analogue. This is not developmental evidence and is not BabyView-like.  
**Status:** development/protocol feasibility only; no confirmatory effect test is authorized.

## Question and ordering

This bounded iteration asks whether AEA can support a stable, observable wearer-action label space and a donor-feasible evaluation, or whether the current AEA route should be abandoned for the proposed language-grounding effect. All model-assisted labels and all estimates are development diagnostics. They cannot support a scientific effect claim.

This document and `output/aea_visible_action_v2/preregistered_protocol.json` are frozen before any new development clip, v2 annotation, split result, capacity result, or IMU result is viewed. The permitted order is:

1. validate the frozen v1 development materialization and reserve boundary without opening media;
2. select 72 development rows and persist their blinded order;
3. materialize dense RGB evidence only for those rows;
4. complete two independently ordered, mutually blinded model-assisted passes;
5. compute agreement, consensus support, and the annotation gate once;
6. construct the constrained folds and donor maps once;
7. run the action-label capacity control only if annotation and split gates pass;
8. run the separate transcript control only if the action-label capacity control passes;
9. run the IMU diagnostic only if annotation, split, and action-label capacity gates pass;
10. apply the mechanical decision and acquisition rules once.

No relabeling, ontology change, threshold change, subset search, fold relaxation, extra seed, or endpoint substitution is permitted after results.

## Immutable source and reserve exclusion

The only row source is the v1 development-only JSONL at `output/aea_dev_learnability_v1/development_examples.jsonl`, SHA-256 `7d6e23a4589a179449ed05bd093be7acbdda75bef451261ebf594d8b343354f0`. Its membership must equal the 192 development entries in the v1 partition manifest, SHA-256 `fc7b9299c01d09eb20119060134e51c048146c8dfb52a4e76726437d51783794`. The monolithic 247-row JSONL is permitted only for a hash comparison in preflight and must not be passed to a v2 media or model runner.

The five v1 prospective-reserve event groups remain excluded:

- `loc1_script4_seq4`
- `loc2_script5_seq6`
- `loc3_script2_seq3`
- `loc4_script3_seq4`
- `loc5_script4_seq3`

The reserve contains 55 windows in five event groups. V2 code must reject any selected ID, sequence, or event group that is not an exact development-manifest member before constructing a media path. A preflight receipt must be written before media access. The dense packet may open RGB from selected development VRS files only; it may not open IMU. The capacity runner may open only materialized dense v2 RGB. The IMU runner, if gate-authorized, may open only the consensus-retained development IMU paths. Every stage records counts of development RGB/IMU files opened and asserts zero reserve RGB/IMU files opened. No signed URL may be loaded, printed, or copied.

## Frozen observable-action codebook

The annotation target is the primary observable action of the camera wearer in the exact six-second ASR-anchored window. It is independent of whether the transcript describes that action. Prefer an action spanning the anchor (the temporal midpoint); otherwise choose the action whose evidence interval is closest to the anchor, then the longer interval, then the ontology order below. Static possession, looking, listening, and talking alone are not actions.

Action labels are mutually exclusive:

1. `locomotion_posture`: translational walking or a clear sit/stand/body-posture transition;
2. `reach_grasp`: reaching to and acquiring or deliberately taking hold of an object;
3. `transport_place`: carrying, bringing, repositioning, putting, setting, or releasing an object;
4. `state_change_operate`: opening, closing, turning, pressing, switching, or operating a container/device;
5. `food_material_handling`: pouring, stirring, cutting, preparing, eating, drinking, or analogous material transformation;
6. `clean_groom`: wiping, washing, cleaning, brushing, vacuuming, or grooming;
7. `other_observable`: a clear goal-directed wearer action not covered above;
8. `none_visible`: no goal-directed wearer action is visibly completed or underway;
9. `uncertain`: image quality, occlusion, temporal ambiguity, or competing actions prevent a defensible primary label.

`other_observable`, `none_visible`, and `uncertain` are reported but never modeled. If evidence is insufficient, use `uncertain`, not a semantic guess from speech. If only another person acts, label the wearer's own visible action (often `none_visible`) and record ASR agency separately.

Each pass also labels:

- `asr_refers_to_visible_wearer_action`: `yes`, `no`, or `unclear`;
- `agency`: `wearer`, `other_person`, `narrated_media_or_phone`, `figurative_or_nonaction`, or `unclear`;
- `temporal_alignment`: `aligned_within_window`, `action_precedes_anchor`, `action_follows_anchor`, `no_corresponding_action`, or `unclear`;
- `confidence`: `high`, `medium`, or `low`;
- an evidence frame interval from 0 through 30 (nullable only for `none_visible` or `uncertain`), and a rationale of at most 25 words.

Agency is the referent/source of the anchored verb, not whoever is most salient in the image. Temporal alignment is relative to the centered ASR anchor: `aligned_within_window` means the corresponding wearer action overlaps the anchor or crosses it; otherwise record which side contains the nearest corresponding action. Use `no_corresponding_action` when none occurs in the six-second evidence.

## Deterministic sample and dense evidence

The sample has exactly 72 development windows. It includes all 48 IDs in the frozen v1 audit prelabel manifest (ID-set digest `7c9654522a290d109349ba5bc3cf1365f2f994de8cf01449ab7b087788d14171`) but never exposes their v1 labels or rationales. The other 24 rows fill every development event group to exactly four sampled rows. Within each deficient group, candidates are chosen by the lowest current sample count of their ASR action, then lowest total development support for that ASR action, then SHA-256 of `aea-visible-action-v2-sample|example_id`, then example ID. Groups are processed by SHA-256 of `aea-visible-action-v2-group|event_group`, then group ID. No pixel, IMU, v1 audit label, or outcome enters selection.

Each item receives an opaque blind ID assigned by SHA-256 of `aea-visible-action-v2-blind|example_id`. Review packets expose only blind ID, anchored verb, transcript, the 31-frame evidence, and timing guidance. Pass A is sorted by SHA-256 of `aea-visible-action-v2-pass-a|blind_id`; pass B uses `aea-visible-action-v2-pass-b|blind_id`. Neither packet exposes example ID, group, location, v1 audit membership, v1 label, the other pass, agreement, support, or model output.

Dense evidence is 31 upright RGB frames queried at the centers of 31 equal-duration bins spanning the exact existing six-second window. It is substantially denser than the prior eight-frame strip. A contact sheet must preserve all frames in temporal order with indices and a visible anchor marker between frames 15 and 16. Missing/corrupt evidence forces `uncertain`; it is never replaced by an existing sparse strip. No audio, IMU, or reserve media is materialized.

## Two model-assisted passes and agreement gate

The two passes must be produced in separate context-isolated model runs, in their separately randomized orders, without access to v1 labels or to each other. These are **MODEL-ASSISTED DEVELOPMENT LABELS**, not human annotations and not human inter-rater reliability. No adjudication is allowed. A modeling consensus exists only when both passes give the same one of the six modeled action labels (`locomotion_posture` through `clean_groom`) and both confidences are `medium` or `high`.

Report confusion matrices, raw exact agreement and unweighted Cohen's kappa for action, agency, temporal alignment, and ASR-referent label; confidence-weighted and unweighted results; Wilson intervals for judgeability, modeled-consensus yield, `asr_refers=yes`, and uncertainty; and support by action, event group, and location.

The annotation gate passes only if all conditions hold:

- both passes contain 72 unique valid rows and complete evidence/rationales;
- judgeable in both passes is at least 85%;
- action exact agreement is at least 0.70 and action kappa at least 0.50;
- agency exact agreement is at least 0.75 and agency kappa at least 0.55;
- temporal exact agreement is at least 0.65 and temporal kappa at least 0.40;
- ASR-referent exact agreement is at least 0.75 and kappa at least 0.50;
- at least 60% of all sampled rows form modeled-action consensus;
- at least two modeled labels each have at least eight consensus windows across at least four event groups.

If modeled-consensus yield is below 0.40, fewer than two labels have at least six rows across three groups, or action exact agreement is below 0.50, the observable structure is unsuitable and the terminal decision is `STOP`. An annotation-gate miss above all three severe-failure bounds is `REVISE` for one ontology/interface change only. No modeling runs after either annotation-gate failure.

## Frozen support, folds, and donors

Only exact, medium-or-high confidence modeled-action consensus rows are eligible. A label is retained only with at least eight windows across at least four performed-event groups. At least two labels must be retained. There is one endpoint: the fixed six-label observable-action endpoint; no post-hoc binary or merged endpoint is allowed.

Three folds partition all 18 development event groups, exactly six groups per fold. A deterministic constrained integer program uses only frozen consensus labels, event groups, locations, and counts. It requires every retained label to contribute at least two test rows from at least two groups per fold and at least four training rows from at least three groups per fold. It minimizes, lexicographically, total absolute deviation from per-fold label counts, location counts, and the SHA-256 tie-break objective `aea-visible-action-v2-fold|group|fold`. The solver runs once with deterministic settings. An infeasible certified status is reported as mathematical infeasibility; controls are not relaxed and a second split is not tried.

For every fold and donor seed `6201, 6202, 6203`, the **training intervention** requires a split-local bijection of whole-window IMU trajectories, with no self donor and no same-event-group donor. Matching is deterministic from SHA-256 order and must be complete. Since primary grounding evaluation withholds IMU, no test-side shuffle is required for that estimand.

For the separate IMU-only diagnostic, test IMU is an evaluated modality, so paired test-side donor maps are relevant and additionally required for seeds `6301, 6302, 6303`. A complete different-event-group bijection exists exactly when the largest group occupies at most half of that side; both this bound and bipartite matching are checked. Donor maps, hashes, self/same-group rates, fold membership, and infeasibility evidence are persisted. No donor reuse or event-group relaxation is allowed.

## Capacity stage gate and transcript diagnostic

The basic capacity control asks whether the repository video encoder can represent the frozen observable-action labels without natural-transcript noise. It uses a supervised action head on 12 deterministic frames (indices from an even sample of the 31 dense frames), 64-pixel RGB, hidden dimension 64, batch size 16, learning rate 0.001, 120 epochs, and seeds `7201, 7202, 7203`. All eligible consensus rows are used; inverse-frequency class weighting is frozen. Evaluation on the same rows is explicitly an optimization/capacity control, not generalization evidence. The gate passes only if mean training balanced accuracy is at least 0.90, every seed is at least 0.85, and mean proportional cross-entropy loss reduction is at least 0.50.

If that gate passes, the same video encoder budget is run once with seed `7299` using the original natural transcripts and the fixed action prompts for action-balanced training-set 2AFC. This transcript-based control is reported separately and is not allowed to substitute for the action-head gate. Its descriptive reference is 0.50 chance and 0.90 capacity; it does not change the v2 decision because transcript noise is the diagnosed target of repair.

Expensive held-out grounding is not run in v2. V2 is a protocol/capacity feasibility iteration and model-assisted labels cannot support the claimed effect.

## Conditional IMU diagnostic

Only if annotation, split/donor, and action-head capacity gates pass, fixed 129-dimensional six-axis summary features and train-fold standardization are used with class-balanced multinomial logistic regression (`C=1`, L2, 5,000 iterations). Report train and group-held-out balanced accuracy, per-class results, event-group bootstrap intervals (2,000 draws, seed `8201`), within-fold label-permutation chance (2,000 draws, seed `8203`), and the paired synchronized-minus-test-donor diagnostic (10,000 paired event-group randomizations, seed `8205`). No subset is chased.

The IMU viability gate requires all of: synchronized held-out balanced accuracy at least macro chance plus 0.10; its 95% event-group bootstrap lower bound above macro chance; permutation p at most 0.05; synchronized minus paired donor accuracy at least 0.05 with bootstrap lower bound above zero and randomization p at most 0.05; and mean train-minus-held-out gap at most 0.35. A valid null or overfit result is `STOP` for the current IMU-mediated AEA route.

## Mechanical decision

Precedence is fixed:

1. Reserve access, source mismatch, leakage, post-freeze mutation, or fabricated/incomplete annotation is an integrity failure: `REVISE`, authorizing only implementation repair without new viewing.
2. A severe annotation failure defined above is `STOP`.
3. A non-severe annotation threshold miss is `REVISE`, authorizing only one prospectively frozen ontology/interface repair.
4. Annotation pass plus certified split infeasibility is `REVISE` only when the capacity gate can be evaluated on supported labels and passes; otherwise `STOP` for insufficient observable structure.
5. Annotation and split pass but action-head capacity fails: `STOP`.
6. All prior gates pass but the valid IMU viability gate fails: `STOP`.
7. `GO` requires stable annotation, retained support, a valid donor-feasible design, passing action-head capacity, and passing conditional IMU viability. It authorizes only a genuine two-independent-human annotation round. It never authorizes reserve access, the locked experiment, downloads, licenses, or outreach.

The transcript diagnostic is always interpreted separately. Any model-assisted result remains development evidence only.

## Prospective acquisition rule

Additional recordings are recommended only if the annotation and action-head capacity gates pass and the sole demonstrated bottleneck is either (a) certified label/event-group support infeasibility or (b) an IMU estimate at least +0.05 whose interval crosses zero while train-minus-held-out gap is at most 0.35 and effective held-out support is below 12 event groups. A negative/subthreshold point estimate, severe overfit, ontology failure, or capacity failure forbids expansion.

If and only if that gate passes, a future acquisition plan may select at most 12 recordings not in the existing 40-recording plan, annotations plus main VRS only, with declared total size at most 40 GiB and a projected free-space floor of 150 GiB. Using only the safe release manifest (dataset/release, sequence IDs, location/script/sequence/recording, component names, declared byte sizes; signed URLs discarded), candidates are ranked to add new event groups in the most underrepresented location-by-script cells, then smallest declared bytes, then SHA-256 of `aea-visible-action-v2-acquire|sequence_id`. Concurrent recordings for a selected event group are taken together and count toward both limits. The first ranked prefix satisfying the limits is the exact recommendation; no file is downloaded in v2.

If the gate does not pass, the recommendation is exactly `no additional acquisition`; the available 198 GiB is not an argument to spend it.

## Required artifacts and limits

All v2 artifacts go under `output/aea_visible_action_v2/` or new v2-named source/test paths. Prior outputs are immutable. Required outputs are the protocol and freeze receipt, codebook, selection and dense-evidence manifests, reserve-access receipts, two raw passes, agreement/support report, constrained split/donor result, gated capacity/transcript/IMU diagnostics or explicit not-run receipts, safe release metadata summary, acquisition decision, blinded human packet, machine-readable decision, exact commands/provenance, and a concise scientific report. Full tests and `git diff --check` must pass. No commit, push, deletion, overwrite of v1/smoke outputs, license acceptance, signed-URL exposure, reserve access, locked run, or external contact is permitted.
