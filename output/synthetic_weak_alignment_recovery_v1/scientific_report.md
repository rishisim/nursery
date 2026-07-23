# Synthetic weak-alignment recovery study v1

## Terminal recommendation: GO

This development-only symbolic study asks whether lexical/action meanings can be recovered from temporally extended, weakly aligned bags and transferred to new visible action instances in held-out object-action compositions. It does **not** test complete language acquisition or timestamp agreement.

## Primary finding

For the frozen combined latent-MIL + cross-occurrence learner, synchronized minus group-safe shuffled side information was +60.16 pp (paired hierarchical 95% CI +57.55 pp to +64.58 pp; 12/12 corpus/model pairs positive).

The primary condition means were:

| Training-time side condition | Held-out action 6-way accuracy |
| --- | ---: |
| synchronized | 95.57% |
| shuffled | 35.42% |
| time_shifted | 2.08% |
| absent | 78.47% |
| uninformative | 24.22% |

Side information was structurally absent from the final evaluator and serialized lexical model. The final test used only learned lexical prototypes and new action/scene observations.

## Learner comparison

| Learner | Synchronized | Shuffled | Absent |
| --- | ---: | ---: | ---: |
| exact_window | 75.00% | 75.00% | 75.00% |
| latent_mil_single_occurrence | 33.07% | 21.44% | 40.36% |
| cross_situational_uniform | 32.03% | 32.03% | 32.03% |
| latent_mil_cross_occurrence | 95.57% | 35.42% | 78.47% |
| latent_mil_cross_no_null | 77.08% | 43.75% | 77.60% |
| oracle_alignment | 100.00% | 100.00% | 100.00% |

`exact_window` selects the event nearest the utterance and has no null option. `latent_mil_single_occurrence` isolates latent selection without repetition; `cross_situational_uniform` aggregates repetitions without latent selection; `latent_mil_cross_no_null` removes only the null state; and `oracle_alignment` is the strong-alignment positive control.

## Manipulations, validity, and controls

- protocol_freeze: **PASS**
- manipulation_checks: **PASS**
- pairing_fairness_checks: **PASS**
- leakage_shortcut_checks: **PASS**
- learnability_controls: **PASS**
- modality_withholding_audit: **PASS**
- confirmation_reserve_guard: **PASS**

When configured informative, synchronized cues identified the true event or null 87.8% of the time versus a 25.0% chance reference. At configured informativeness 0, the corresponding rate was 27.1% versus 25.0% chance.

Oracle alignment reached 100.0% on the primary endpoint. The zero-exposure lexical-type control was 16.7% (6-way chance 16.7%).

## Factor sensitivity

Each secondary analysis refit the frozen primary learner within one level of one manipulated factor and retained corpus/model pairing. These are descriptive; no multiplicity-adjusted claims are made.

- action_visibility_rate — synchronized minus shuffled lexical mapping: 0.55: +20.14 pp, 0.9: +56.25 pp
- candidate_event_count — synchronized minus shuffled lexical mapping: 2: +43.75 pp, 5: +47.92 pp
- grounded_utterance_rate — synchronized minus shuffled lexical mapping: 0.45: +41.67 pp, 0.8: +31.25 pp
- side_informativeness — synchronized minus shuffled lexical mapping: 0.0: +7.45 pp, 0.75: +64.20 pp, 0.95: +61.55 pp
- speech_action_lag — synchronized minus shuffled lexical mapping: -0.28: +36.11 pp, 0.0: +23.30 pp, 0.28: +28.03 pp
- word_occurrence_count — synchronized minus shuffled lexical mapping: 8: +58.54 pp, 16: +36.16 pp

## Estimands and uncertainty

The sole primary estimand is held-out-composition action 6-way macro accuracy for the combined learner under synchronized training minus group-safe shuffled training. The interval resamples four corpus seeds and then three model seeds within each sampled corpus; episodes and windows are never treated as independent inferential units. All other condition, component, noun-control, and factor-stratified contrasts are secondary/descriptive.

## Decision rule outcome

- GO gate `all_validity_and_control_audits_pass`: pass
- GO gate `minimum_primary_lift`: pass
- GO gate `primary_ci_above_zero`: pass
- GO gate `positive_pair_count`: pass
- GO gate `synchronized_beats_all_matched_controls`: pass
- GO gate `action_selective_over_noun_control`: pass
- GO gate `informative_strata_positive`: pass
- GO gate `configured_uninformative_stratum_near_zero`: pass

Recommendation: **GO** — all frozen development validity, positive-control, effect, selectivity, and sensitivity gates passed. This artifact does not authorize the reserved confirmation seeds; they remain inaccessible without a separate, future explicit authorization.

## Current primary sources (accessed 2026-07-15)

- [Infants rapidly learn word-referent mappings via cross-situational statistics](https://doi.org/10.1016/j.cognition.2007.06.010) — Cognition 106(3), 1558-1568 (2008), doi:10.1016/j.cognition.2007.06.010. Justification: Motivates repeated ambiguous occurrences as the evidence unit for cross-situational lexical recovery.
- [A computational study of cross-situational techniques for learning word-to-meaning mappings](https://doi.org/10.1016/S0010-0277(96)00728-7) — Cognition 61(1-2), 39-91 (1996), doi:10.1016/S0010-0277(96)00728-7. Justification: Supports treating noisy multiword input and referential uncertainty as explicit computational problems rather than exact timestamp labels.
- [A Probabilistic Computational Model of Cross-Situational Word Learning](https://doi.org/10.1111/j.1551-6709.2010.01104.x) — Cognitive Science 34(6), 1017-1063 (2010), doi:10.1111/j.1551-6709.2010.01104.x. Justification: Justifies probabilistic alignment jointly with accumulation of word-meaning evidence across situations.
- [Solving the multiple instance problem with axis-parallel rectangles](https://doi.org/10.1016/S0004-3702(96)00034-3) — Artificial Intelligence 89(1-2), 31-71 (1997), doi:10.1016/S0004-3702(96)00034-3. Justification: Provides the canonical multiple-instance framing in which one ambiguous bag contains several candidate instances.
- [UntrimmedNets for Weakly Supervised Action Recognition and Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_UntrimmedNets_for_Weakly_CVPR_2017_paper.html) — Proceedings of CVPR, 4325-4334 (2017). Justification: Supports separating bag-level classification from latent temporal/event selection in untrimmed weakly supervised streams.
- [Unifying distillation and privileged information](https://arxiv.org/abs/1511.03643) — International Conference on Learning Representations (2016), arXiv:1511.03643. Justification: Justifies the train-only privileged-information boundary: synchronized explanations may aid learning but are unavailable at test.
- [Bootstrap Methods: Another Look at the Jackknife](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) — The Annals of Statistics 7(1), 1-26 (1979), doi:10.1214/aos/1176344552. Justification: Supports bootstrap uncertainty; this protocol applies it only to paired corpus/model-seed units.
- [Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks](https://proceedings.mlr.press/v80/lake18a.html) — Proceedings of ICML, PMLR 80, 2873-2882 (2018). Justification: Motivates an explicit held-out-composition test rather than interpreting in-distribution accuracy as transferable meaning.
- [EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data](https://arxiv.org/abs/2605.19130) — arXiv:2605.19130 (2026). Justification: Provides the current weak-alignment and Machine-DevBench context; Nursery excludes the benchmark here because its measured target vocabulary coverage is inadequate.

## Relationship to prior repository evidence

The original exact-window pilot remains a negative infrastructure result (synchronized minus shuffled -20.37 percentage points, 95% CI [-33.80, -1.39]). AEA v3 remains a terminal STOP: only 7/72 audited windows had consensus natural-speech/action alignment, so this study does not reopen AEA language, sensor, or acquisition work. AEA is an adult, partly scripted sensor-format analogue, not developmental or BabyView-like evidence. Machine-DevBench remains secondary and was not rerun because exact target-word coverage is only 13/1,414.

Two stale-document contradictions are preserved rather than silently repaired: `docs/frank_preaccess_experimental_plan.md` says Machine-DevBench was not reproduced and mentions 24 tests, whereas newer artifacts document completed reproduction/integration and the parent audit found 69 passing tests; README AEA acquisition instructions are superseded by the v3 terminal STOP artifact.

## Limitations

- This is a controlled symbolic feature study, not raw-pixel or raw-audio learning and not infant evidence.
- Canonical perceptual action dimensions are assumed learnable; the study isolates lexical assignment and transfer.
- Development seeds are used for method diagnosis only. No confirmation outcome was generated, read, evaluated, or summarized.
- The fixed fractional design balances the complete non-repetition factor grid within both repetition levels, but only four corpus seeds support uncertainty estimation.
- A positive result would establish only synthetic lexical/action grounding under this generator and learner family.
