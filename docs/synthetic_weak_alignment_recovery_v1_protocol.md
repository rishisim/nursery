# Frozen development protocol: synthetic weak-alignment recovery v1

Protocol ID: `synthetic-weak-alignment-recovery-v1`  
Status: **FROZEN BEFORE ANY OUTCOME-PRODUCING DEVELOPMENT RUN**  
Phase: development only  
Freeze date: 2026-07-15

## Scope and claim boundary

The narrow claim is recovery of transferable lexical/action meanings from weakly aligned synthetic egocentric episode bags. It is not complete language acquisition, timestamp agreement, infant evidence, BabyView equivalence, or a result about raw vision/audio representation learning. Canonical but noisy action-observation dimensions isolate lexical assignment from perceptual feature learning.

The causal contrast is synchronized train-only side information versus a split-local, candidate-count-matched, lexical-group-safe shuffled control. Time-shifted, structurally absent, and distribution-matched uninformative arms are secondary matched controls. Side information can select an event or a null/no-visible-referent state during training. No side field, encoder, or state is accepted by final evaluation.

AEA v3 is terminally stopped and is not reopened. Machine-DevBench is excluded because current exact target vocabulary coverage is 13/1,414; it remains a secondary integration diagnostic only.

## Hypotheses and estimands

H1 (sole primary): for `latent_mil_cross_occurrence`, synchronized training improves macro 6-way action-word compatibility on independent new visible action instances and object-action compositions absent from training, relative to group-safe shuffled training.

The primary estimand is

`mean_pair[accuracy_synchronized - accuracy_shuffled]`,

where a pair is one synthetic corpus seed crossed with one model-initialization seed. The metric is `heldout_composition_action_6way_macro_accuracy`, macro-averaged over the 12 preassigned scoring lexical types in panels 0 and 1. Those types receive training exposures because lexical recovery without exposure is undefined; they are held out from calibration/controls and scored only on independent tokens and new action instances. Six truly zero-exposure lexical types form a negative control and should remain at 1/6 chance.

Secondary estimands are synchronized minus time-shifted, absent, and uninformative; combined learner minus each component/ablation within synchronized training; the matched noun/non-action lift; and synchronized-minus-shuffled lexical mapping within each factor level. These are descriptive and do not create additional discoveries.

## Synthetic population and balanced factor design

Each corpus randomizes three balanced six-word verb panels over six action concepts, six scene-noun words over six object concepts, and one held-out object-action composition per action. Panels 0 and 1 are preassigned primary scoring lexemes; panel 2 is an anchor panel. Word surfaces contain no semantic action substring, and mappings change with corpus seed.

Every training item is a temporally extended bag with 2 or 5 candidate events plus an explicit null state. An utterance can lead, align with, or lag its referent; it can also have no visible referent. Candidate action observations can be visible/reliable or uninformative. A global scene observation supports the noun/non-action control independently of event selection.

The independently manipulated and recorded factors are:

| Factor | Frozen levels |
| --- | --- |
| Speech/action lag | -0.28, 0.00, +0.28 normalized episode time |
| Grounded-utterance rate | 0.45, 0.80, with realized no-referent outcomes recorded |
| Referential ambiguity | 2, 5 candidate events |
| Action visibility rate | 0.55, 0.90 |
| Word occurrence count | 8, 16 |
| Side-modality informativeness | 0.00, 0.75, 0.95 probability of intentionally placing the largest cue on the true event/null, otherwise exchangeable |

The non-repetition grid has 3 x 2 x 2 x 2 x 3 = 72 cells. Nine low-repetition lexemes jointly receive exactly one complete grid (9 x 8 = 72), and nine high-repetition lexemes jointly receive exactly two complete grids (9 x 16 = 144). Thus the other five factors are exactly balanced within both repetition levels. There are 216 training episodes per corpus. This is a complete balanced fractional allocation over lexical types, not an outcome-adaptive sample.

## Side-modality conditions and fairness

All arms share the same base episode IDs, text, event observations, scene observations, split, data order, lexical initialization, update count, and corpus/model seeds.

- `synchronized`: the episode's own score vector over candidate events plus null.
- `shuffled`: an exact score-vector bijection within train split and candidate count, with no self donor and no donor from the same lexical episode group.
- `time_shifted`: the own vector cyclically shifted one position, including the null position.
- `absent`: the side key is omitted entirely. There is no constant tensor or distinctive learned missingness encoder.
- `uninformative`: a deterministic within-episode random permutation of the own vector, preserving its exact distribution while destroying intended alignment.

The score multiset is identical across every present-modality condition. The absent arm performs the same eight lexical update passes; only the auxiliary term is structurally unavailable.

## Learners and positive controls

1. `exact_window`: select the candidate event nearest utterance time, never null, then pool selected observations by word. This retains the repository's exact-window assumption in symbolic form.
2. `latent_mil_single_occurrence`: latent event/null selection but only the first occurrence of each word. This removes cross-occurrence aggregation.
3. `cross_situational_uniform`: aggregate all candidate observations over repeated word occurrences without latent selection or null.
4. `latent_mil_cross_occurrence`: the primary combined learner. Eight fixed soft assignment/update passes jointly use current lexical prototypes, a weak temporal prior, and side scores when present; the posterior includes null.
5. `latent_mil_cross_no_null`: the combined learner with only the null state removed.
6. `oracle_alignment`: physically separate ground-truth event indices select grounded instances and skip no-referent utterances. This is the strong-alignment/base-learnability positive control.

All learner hyperparameters are fixed in `configs/synthetic_weak_alignment_recovery_v1.yaml`; no development outcome may tune them.

## Evaluation and controls

Final evaluation uses a frozen `EvaluationItem` schema containing only action observations, scene observations, lexical candidate strings, answer indices, and split/composition metadata. A frozen `LexiconModel` contains only action prototypes, scene prototypes, learner name, and model seed. The evaluator rejects any other item type and audits field names and source for auxiliary modality or encoder access.

Primary test items are independent generator draws, use new visible action instances, and use only object-action compositions excluded from training. The matched scene-noun endpoint checks whether a side effect is action-selective. A zero-exposure verb panel checks label/surface leakage. Direct observation decoding, oracle alignment, model-visible allowlists, physically separate oracle records, randomized mappings, donor bijections, and train/evaluation hash disjointness are mandatory controls.

## Development and confirmation seeds

Development corpus seeds: `1103, 2207, 3301, 4409`.  
Development model seeds: `17, 29, 43`.  
Untouched confirmation corpus seeds: `91009, 92009, 93001, 94007`.  
Untouched confirmation model seeds: `101, 103, 107`.

Every generation, training, evaluation, raw-record read, and summary entry point rejects a reserved seed unless a future explicit user authorization file validates against both this freeze receipt and the reserve manifest. No such authorization exists in this task. The reserve manifest contains identifiers only and no outcomes.

## Inference and multiplicity

The primary point estimate averages the 12 paired corpus/model differences. The 95% percentile interval uses 5,000 paired hierarchical bootstrap replicates: sample corpus seeds with replacement, then sample the three model seeds with replacement within each sampled corpus. Episodes/windows are never resampled as independent inferential units.

There is exactly one primary metric/contrast at alpha 0.05, so no primary multiplicity adjustment is required. All other contrasts and factor strata are labeled secondary/descriptive; their intervals are not used to claim additional discoveries.

## Frozen GO / REVISE / STOP rule

GO for a later separately frozen synthetic confirmation requires all of:

- every freeze, generation, manipulation, pairing, leakage, reserve, learnability, and final-modality audit passes;
- synchronized-minus-shuffled primary lift at least +5 absolute percentage points and hierarchical 95% CI strictly above zero;
- at least 9 of 12 corpus/model pairs have positive differences;
- synchronized point accuracy exceeds shuffled, time-shifted, absent, and uninformative;
- action lift minus noun-control lift is at least +3 points;
- each informative side-informativeness stratum has positive synchronized-minus-shuffled lexical mapping; and
- the informativeness-0 absolute lift is at most 10 points.

STOP if any validity or positive-control gate fails, or if the primary 95% CI is entirely below zero. Otherwise the recommendation is REVISE. A GO recommendation does not authorize confirmation access.

## Manipulation and learnability gates

For informative episodes, synchronized event/null top-1 accuracy must exceed mean chance by at least 0.35 and beat each present matched control by at least 0.30. At configured informativeness 0, synchronized absolute accuracy-minus-chance must be at most 0.08. Oracle primary accuracy must be at least 0.80; direct action and scene observation decoding at least 0.95; noun-control accuracy at least 0.95. Cross-corpus surface-only accuracy must be at most 0.35, and the zero-exposure control cannot exceed 1/6 chance by more than 0.02.

## Primary-source rationale

Sources were checked before any outcome run on 2026-07-15. Smith and Yu (2008, [doi](https://doi.org/10.1016/j.cognition.2007.06.010)), Siskind (1996, [doi](https://doi.org/10.1016/S0010-0277(96)00728-7)), and Fazly, Alishahi, and Stevenson (2010, [doi](https://doi.org/10.1111/j.1551-6709.2010.01104.x)) motivate repeated ambiguous cross-situational evidence, noise, and probabilistic alignment. Dietterich, Lathrop, and Lozano-Perez (1997, [doi](https://doi.org/10.1016/S0004-3702(96)00034-3)) and Wang et al. (CVPR 2017, [open paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_UntrimmedNets_for_Weakly_CVPR_2017_paper.html)) justify bag-level multiple-instance selection in untrimmed data. Lopez-Paz et al. (ICLR 2016, [arXiv](https://arxiv.org/abs/1511.03643)) motivates a strict train-only privileged-information boundary. Efron (1979, [Project Euclid](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full)) motivates bootstrap uncertainty, applied here at corpus/model level. Lake and Baroni (ICML 2018, [PMLR](https://proceedings.mlr.press/v80/lake18a.html)) motivates a held-out-composition test. The current EgoBabyVLM paper (Lin et al., 2026, [arXiv](https://arxiv.org/abs/2605.19130)) supplies the weak-alignment/benchmark context, without changing the local coverage exclusion.

## Authoritative repository boundaries and stale text

Newer artifacts supersede two stale statements without deleting them. `docs/frank_preaccess_experimental_plan.md` says Machine-DevBench was not reproduced and mentions 24 tests; newer artifacts record completed reproduction/integration, and the parent audit reports 69 passing tests. README AEA acquisition commands are superseded by `output/aea_coarse_action_v3/terminal_decision.json`, which authorizes no further AEA language-grounding, sensor-side-channel development, or acquisition. This protocol preserves and flags both contradictions.
