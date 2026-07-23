# Synthetic sensor-to-event robustness study v2

## Terminal recommendation: REVISE

This frozen development-only experiment tests whether synchronized raw synthetic six-axis IMU, proprioceptive state, and contact streams add training value for weak action-language grounding. The learner never receives an oracle target-event pointer, and final evaluation structurally accepts only learned lexical prototypes plus cue-free action/object observations.

Recommendation rationale: the study remained interpretable, but at least one frozen GO gate did not pass.

## Co-primary causal findings

| Contrast (combined learner, held-out composition endpoint) | Corpus-level lift | 95% t CI | One-sided t p | Positive corpora | Exact sign p | Sign-flip p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| synchronized − absent | +38.02 pp | [+18.95 pp, +57.10 pp] | 0.000258643 | 11/20 | 0.000488281 | 0.000488281 |
| synchronized − shuffled | +38.02 pp | [+18.95 pp, +57.10 pp] | 0.000258643 | 11/20 | 0.000488281 | 0.000488281 |

Both contrasts are co-primary under a predeclared intersection-union rule: both must pass their full effect, interval, one-sided test, and sign gates. Model seeds are stochastic algorithmic replicates averaged inside each of 20 independent corpus seeds.

## Condition and time-shift characterization

| Raw-sensor training condition | Primary accuracy | Difference from absence |
| --- | ---: | ---: |
| synchronized | 38.02% | +38.02 pp |
| shuffled | 0.00% | +0.00 pp |
| shifted_m16 | 0.00% | +0.00 pp |
| shifted_m8 | 0.00% | +0.00 pp |
| shifted_p8 | 0.00% | +0.00 pp |
| shifted_p16 | 0.00% | +0.00 pp |
| absent | 0.00% | +0.00 pp |
| uninformative | 0.00% | +0.00 pp |

Signed offsets were analyzed separately, rather than collapsed into a single adversarial condition. Shuffled and uninformative channels are safety controls; the frozen reliability mechanism may set token-level trust to zero and additionally applies training-time cue dropout.

## Detector calibration and boundary capacity

The fixed detector was trained only on independent generic calibration episodes with wearer-activity and boundary supervision. It had no lexical, referent, groundedness, or randomized-mapping targets and was reused unchanged in every sensor arm.

| Held-out generic calibration measure | Result | Gate |
| --- | ---: | ---: |
| Informative activity precision | 82.14% | ≥ 70.00% |
| Informative activity recall | 91.15% | ≥ 70.00% |
| Informative boundary F1 | 90.08% | ≥ 60.00% |
| Informative candidate-owner AUC | 0.921 | ≥ 0.78 |
| Zero-information candidate-owner AUC | 0.488 | within ±0.12 of .50 |

## Learners and transfer endpoints

| Learner | Synchronized | Shuffled | Absent |
| --- | ---: | ---: | ---: |
| exact_window_symbolic | 0.00% | 0.00% | 0.00% |
| sensor_latent_single_occurrence | 28.65% | 1.11% | 2.78% |
| cross_occurrence_no_sensor | 0.00% | 0.00% | 0.00% |
| sensor_latent_cross_occurrence | 38.02% | 0.00% | 0.00% |
| sensor_latent_cross_no_null | 35.28% | 0.00% | 0.00% |
| structural_absence | 0.00% | 0.00% | 0.00% |
| oracle_event_alignment_upper | 100.00% | 100.00% | 100.00% |
| v1_pointer_style_upper | 100.00% | 100.00% | 100.00% |

`exact_window_symbolic` is the existing nearest-window assumption. The two upper controls use oracle event alignment and the v1 pointer-style score only to establish learnability; neither is the primary learner.

| Cue-free synchronized endpoint (primary learner) | Accuracy |
| --- | ---: |
| Seen lexical components, independent tokens | 63.82% |
| New action instances | 38.02% |
| Seen action combinations | 38.62% |
| Held-out object-action compositions | 38.02% |
| Structured primitive × manner held-out concept | 35.00% |
| Matched noun/non-action task | 28.54% |
| Truly zero-exposure word panel | 16.67% |

The structured concept panel withholds one primitive × manner combination while exposing both components elsewhere. It does not claim recovery of an arbitrary, completely unexposed word meaning; that stronger leakage control is the zero-exposure panel and remains a chance benchmark.

## Informativeness and matched selectivity controls

| Sensor informativeness | Sync − absent | Sync − shuffled |
| ---: | ---: | ---: |
| 0.0 | +0.00 pp | +0.00 pp |
| 0.9 | +68.40 pp | +68.40 pp |
| 1.0 | +85.83 pp | +85.83 pp |

Matched noun lift was +0.62 pp versus absence and +0.62 pp versus shuffled. Nouns use the same ambiguous bags, latent selection, cross-occurrence updates, reliability fusion, and final evaluator; only causal ownership is made irrelevant.

## Audits and validation

- `calibration_boundary`: **PASS**
- `confirmation_reserve_guard`: **PASS**
- `detector_validation`: **PASS**
- `evaluation_firewall`: **PASS**
- `leakage`: **PASS**
- `learnability`: **PASS**
- `pairing_fairness`: **PASS**
- `protocol_freeze`: **PASS**
- `stochastic_model_replicates`: **PASS**

An independent validation pass recomputed record counts, per-condition means, and both co-primary corpus effects directly from raw run records. Exact tables are used instead of a chart because the inferential unit, gates, and audit values are more inspectable in tabular form.

## Inference and estimand definitions

The independent unit is the generated corpus seed. Each corpus effect first averages the two stochastic model seeds and then subtracts matched conditions. Primary uncertainty uses Student t inference over 20 corpus effects (19 degrees of freedom), with exact sign and exhaustive sign-flip p-values as corroboration. A generic percentile bootstrap is not used. The synchronized-minus-absence contrast establishes positive added value; synchronized-minus-shuffled establishes synchronization specificity.

Factor-stratified refits are secondary and descriptive; no additional discovery claim or multiplicity-adjusted factor inference is made.

## Primary sources

- [Deep Convolutional and LSTM Recurrent Neural Networks for Multimodal Wearable Activity Recognition](https://doi.org/10.3390/s16010115) — Sensors 16(1):115 (2016), doi:10.3390/s16010115. Supports learning temporal representations directly from minimally processed multimodal wearable streams rather than supplying candidate-wise oracle scores.
- [Hierarchical Span-Based Conditional Random Fields for Labeling and Segmenting Events in Wearable Sensor Data Streams](https://proceedings.mlr.press/v48/adams16.html) — Proceedings of ICML, PMLR 48:334-343 (2016). Establishes event detection and temporal segmentation as joint problems over continuous wearable sensor streams.
- [AutoLoc: Weakly-supervised Temporal Action Localization in Untrimmed Videos](https://openaccess.thecvf.com/content_ECCV_2018/html/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.html) — Proceedings of ECCV, 154-171 (2018). Motivates separating weak bag-level supervision from explicit temporal-boundary prediction in untrimmed sequences.
- [ModDrop: Adaptive Multi-Modal Gesture Recognition](https://doi.org/10.1109/TPAMI.2015.2461544) — IEEE TPAMI 38(8):1692-1706 (2016), doi:10.1109/TPAMI.2015.2461544. Supports channel dropout during fusion so a learner remains useful when a modality is missing or corrupted.
- [EmbraceNet: A robust deep learning architecture for multimodal classification](https://www.sciencedirect.com/science/article/pii/S1566253517308242) — Information Fusion 51:259-270 (2019). Motivates bounded, availability-aware fusion rather than forcing every present modality into every prediction.
- [A new learning paradigm: Learning using privileged information](https://doi.org/10.1016/j.neunet.2009.06.042) — Neural Networks 22(5-6):544-557 (2009), doi:10.1016/j.neunet.2009.06.042. Supports a strict training-only information boundary: auxiliary explanations may shape learned state but are unavailable at evaluation.
- [A Probabilistic Computational Model of Cross-Situational Word Learning](https://doi.org/10.1111/j.1551-6709.2010.01104.x) — Cognitive Science 34(6):1017-1063 (2010), doi:10.1111/j.1551-6709.2010.01104.x. Supports probabilistic alignment jointly with accumulation of word-meaning evidence across ambiguous situations.
- [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://research.google/pubs/measuring-compositional-generalization-a-comprehensive-method-on-realistic-data/) — ICLR (2020). Motivates holding out combinations while retaining exposure to their atomic components, and explicitly separating atom from compound generalization.
- [Inference with Few Heterogeneous Clusters](https://www.princeton.edu/~umueller/BehrensFisher.pdf) — Review of Economics and Statistics 98(1):83-96 (2016), doi:10.1162/REST_a_00545. Supports t-based inference over a small number of independent group-level estimates; v2 therefore reduces to one averaged effect per corpus seed before inference.
- [Randomization Tests Under an Approximate Symmetry Assumption](https://doi.org/10.3982/ECTA13081) — Econometrica 85(3):1013-1030 (2017), doi:10.3982/ECTA13081. Supports sign-flip randomization evidence when group-level estimates are symmetric or approximately symmetric; v2 reports it as corroboration rather than replacing the corpus-level estimate.
- [Multiparameter Hypothesis Testing and Acceptance Sampling](https://www.jstor.org/stable/1267823) — Technometrics 24(4):295-300 (1982). Supports an intersection-union decision in which the global alternative is established only when every co-primary component rejects at the nominal level, without alpha splitting.

## Limitations and scope

- This is a synthetic feature-level study, not BabyView/ChildLens evidence, infant evidence, raw-pixel/audio learning, or ecological validation.
- Visual action/object observations are generated categorical feature vectors; direct capacity controls establish their learnability but do not validate a perceptual front end.
- Detector supervision comes from a separate synthetic generic calibration distribution; transfer to real sensors remains untested.
- Development evidence may motivate only a later separately authorized v2 confirmation. It does not authorize or reveal any v1 or v2 confirmation outcome.
- AEA remains a terminal STOP; Machine-DevBench remains secondary at 13/1,414 coverage; BabyView and ChildLens access remain pending.

Recommendation: **REVISE**. Confirmation remains unauthorized.
