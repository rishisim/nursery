# Frozen development protocol: synthetic sensor-to-event robustness v2

Protocol ID: `synthetic-sensor-event-robustness-v2`  
Phase: development only  
Status at receipt creation: **FROZEN BEFORE ANY V2 OUTCOME-PRODUCING RUN**  
Freeze date: 2026-07-15

## Scientific question and claim boundary

This study asks whether synchronized raw synthetic sensor streams can add training value for weak action-language grounding by helping a separately calibrated component rank wearer-caused candidate events and boundaries. It does not expose a target event pointer, action label, word identity, grounded flag, or randomized lexical mapping in model-visible sensor data. It is a controlled synthetic representation study, not infant evidence, raw-pixel/audio learning, timestamp agreement, or ecological validation.

AEA remains a terminal STOP. Machine-DevBench remains excluded from primary inference because exact target coverage is 13/1,414. BabyView and ChildLens access remain pending. V1 confirmation corpus seeds `91009, 92009, 93001, 94007` and model seeds `101, 103, 107` remain untouched.

## Co-primary hypotheses and estimands

The primary learner is `sensor_latent_cross_occurrence`; the primary endpoint is cue-free 6-way macro accuracy on independent action instances whose object-action compositions are excluded from training. Each action concept is a primitive x manner combination, and one combination per corpus is withheld from training while both components remain exposed in other combinations.

Two co-primary corpus-level estimands are required:

1. synchronized raw sensors minus structurally absent sensors;
2. synchronized raw sensors minus group-safe shuffled raw sequences.

For each corpus, the two model-seed results are averaged within condition before subtraction. The 20 corpus effects are the only independent empirical units. The global positive conclusion is an intersection-union test: both one-sided corpus-level t tests must have p < .05, both two-sided 95% t intervals must lie above zero, both point lifts must be at least +5 percentage points, and at least 15/20 corpus effects must be positive. No alpha splitting is applied because the alternative requires both component alternatives.

## Population, raw streams, and oracle separation

An episode contains 64 samples, three or five randomly placed, nonoverlapping, temporally extended candidate visual events, and an utterance that can lead, align, lag, be irrelevant, or have no visible referent. Candidates include wearer-caused and visually plausible nonwearer/environmental events. Action bags include a lexically systematic but nonwearer visual distractor in 60% of grounded episodes; no-referent action bags contain repeated visually plausible distractors. This deliberately makes uniform cross-occurrence and exact-window assumptions nontrivial while leaving the raw detector to recover only ownership/boundary evidence. Model-visible raw streams contain a six-axis IMU-like signal, two proprioceptive/action-state channels, one touch/contact channel, timestamps, and availability masks. Dropout is represented by zeroed raw samples plus availability, not by semantic codes.

The following factors receive independent, exactly marginal-balanced random permutations within action and matched noun families: speech/action lag, configured grounded rate, candidate count, visibility, lexical repetition, sensor informativeness, sensor SNR, sensor dropout, false-positive activity, and wearer-event prevalence. Realized values and referent status are recorded separately. At sensor informativeness zero, a candidate-count-matched number of activity pulses is generated independently of both event ownership and lexical content; this removes both within-episode and between-episode ownership leakage while retaining the pulse/noise family.

Raw sensors have no categorical action field, lexical field, target index, grounded flag, oracle score, or mapping-dependent encoding. Event ownership, true boundaries, timepoint activity, referents, action/object identities, and randomized lexical mappings live only in physically separate oracle records.

## Independent generic calibration detector

The sensor-to-event component is fit once on generic calibration seeds `52021, 52027, 52051, 52057` and evaluated on held-out generic seeds `52103, 52121`. Calibration examples contain raw streams, generic candidate intervals, and separate wearer-activity/boundary supervision, but no words, randomized mappings, lexical targets, or referent targets. Fixed logistic timepoint and boundary heads operate on local energy, derivative, contact, and availability features. Candidate evidence combines predicted activity inside a visible interval with predicted onset/offset evidence; null evidence rises when no candidate has predicted wearer activity.

The fitted detector is frozen in memory and used unchanged for every present-channel condition. Held-out informative calibration must reach timepoint precision and recall >= .70, boundary F1 >= .60, and candidate-owner AUC >= .78. The zero-information candidate-owner AUC must remain within .12 of .50.

## Matched sensor conditions

All conditions share episode IDs, text, visual candidates, candidate order, lexical mapping, initialization, training order, update passes, and evaluation items.

- `synchronized`: the episode's own raw sequence.
- `shuffled`: a whole-sequence bijection within candidate count with no self donor and no donor sharing the lexical episode group.
- `shifted_m16`, `shifted_m8`, `shifted_p8`, `shifted_p16`: all raw channel values and availability masks cyclically shifted by the declared samples while timestamps and visual events stay fixed.
- `absent`: the raw sensor field is omitted.
- `uninformative`: four-sample raw blocks are deterministically permuted as joint rows, preserving every channel value and availability row while destroying event timing.

Time-shift results are reported by signed offset. They are characterization conditions, not independent discoveries.

## Reliability-aware fusion and learners

The primary learner first fits a sensor-free latent cross-occurrence prototype. It separately builds a sensor-ranked prototype from detector event/null posteriors, weighting episodes by detector overlap quality. Reliability is estimated without oracle labels at the action-component or noun family level from two observed quantities: mean detector overlap quality and the mean absolute change in cross-occurrence semantic concentration relative to uniform candidate weighting. Sigmoid gates are centered at .55 (scale .015) and .055 (scale .008), respectively. A product below .18 rejects the channel and exactly returns the sensor-free prototype; an accepted channel receives frozen weight .80. Candidate updates are independently dropped with probability .30. This identical accept-or-ignore fusion applies to every present condition and to action and noun slots; it is not keyed to the condition name or oracle family relevance.

Learners are:

1. `exact_window_symbolic`: existing nearest-utterance candidate assumption, no null.
2. `sensor_latent_single_occurrence`: sensor-derived event/null selection using only the first occurrence of each lexical component.
3. `cross_occurrence_no_sensor`: repeated ambiguous evidence with latent event/null selection, no sensor evidence.
4. `sensor_latent_cross_occurrence`: co-primary combined learner.
5. `sensor_latent_cross_no_null`: combined learner with only the null state removed.
6. `structural_absence`: same repeated latent machinery with the channel disabled by construction.
7. `oracle_event_alignment_upper`: separate oracle event alignment, positive control only.
8. `v1_pointer_style_upper`: an oracle-derived candidate score vector, positive control only and never represented as raw sensing.

The oracle and pointer controls cannot contribute to the primary contrast.

## Matched noun selectivity and composition

Noun utterances use the same episode bags, systematic ambiguous distractors, event candidates, null option, detector outputs, posterior code, trust estimation, dropout, and update machinery. Their target candidate and no-referent status are sampled independently of wearer ownership, so raw sensors are causally irrelevant but not bypassed by code. The synchronized noun lift relative to both absent and shuffled must be within +/-5 points empirically.

Action concepts are three primitives x two manners. Corpus-specific random surface words name primitives and manners. One primitive-manner combination is absent from action training, but every primitive and manner word appears in other combinations. Structured evaluation composes those exposed components; it does not claim recovery of an arbitrary unexposed word meaning. A six-word zero-exposure panel remains a 1/6 chance/leakage control.

Endpoints distinguish seen lexical components on independent tokens, new action instances, held-out object-action compositions, the structured held-out combination, matched nouns, and truly zero-exposure words.

## Evaluation firewall

The final evaluator accepts only a frozen lexical model containing token prototypes and cue-free evaluation items containing visual action/object observations, candidate words/phrases, answers, and endpoint roles. It rejects dictionaries and any schema or model field containing sensor, detector, IMU, proprioception, touch, cue, side score, or encoder state. Raw streams, calibration heads, derived event evidence, trust traces, and side encoders cannot cross this boundary.

## Factor sensitivity and zero-information requirement

The primary learner is refit separately within every level of every manipulated factor under synchronized, shuffled, and absent conditions using four fixed passes. These analyses are secondary/descriptive. The informativeness-zero synchronized lift relative to both absent and shuffled must have absolute point magnitude <=5 points. Informative levels should show positive added value.

## Inference, uncertainty, and multiplicity

There are 20 fresh independent development corpus seeds. Two model seeds create real variation through initialization and cue-dropout masks, but are averaged within corpus and never counted as empirical replications. Primary point estimates and Student-t intervals use the 20 corpus effects (df=19); generic percentile bootstrap inference is forbidden. Exact sign counts and exhaustive 2^20 sign-flip p-values are reported as corroboration under the symmetry rationale of Canay, Romano, and Shaikh (2017), not as a substitute for the corpus-level estimand.

Berger's intersection-union formulation requires both co-primary component tests to pass at alpha .05. Secondary endpoint, ablation, factor, noun, and time-shift results are descriptive gates and do not create additional discoveries.

## Frozen GO / REVISE / STOP rule

GO toward a later, separately authorized and separately frozen v2 confirmation requires all of the following:

- both co-primary components pass the intersection-union requirements;
- all integrity, detector, leakage, pairing, reserve, learnability, and evaluation-firewall audits pass;
- synchronized primary accuracy >=72%, seen-combination accuracy >=72%, structured held-out concept accuracy >=62%, and seen component accuracy >=80%;
- synchronized exceeds exact-window, single-occurrence, no-sensor cross-occurrence, and no-null ablations by the configured margins;
- shuffled and uninformative primary accuracy are each no more than 5 points below absence;
- sensor-informativeness zero has no synchronized advantage beyond 5 points;
- both matched noun lifts are within 5 points of zero;
- oracle accuracy >=90%, pointer-upper accuracy >=80%, direct action/object observation capacity >=95%, and the zero-exposure panel is no more than 3 points above 1/6 chance.

STOP if an integrity, leakage, reserve, evaluation-firewall, detector, or learnability positive-control gate fails, or if both co-primary 95% intervals are entirely below zero. Otherwise the recommendation is REVISE. No outcome in this study authorizes confirmation.

## Seeds and preservation

Development corpus seeds are `12011, 12037, 12049, 12071, 12097, 12101, 12113, 12119, 12143, 12149, 12157, 12161, 12163, 12197, 12203, 12211, 12227, 12239, 12241, 12251`; model seeds are `211, 223`. V2 confirmation corpus seeds `96103, 96137, 96149, 96157`, model seeds `809, 811, 821`, and calibration seeds `96931, 96959` are identifiers only and must be blocked at generation, calibration, training, evaluation, reading, and summarization. Fixture-only seeds are disjoint and may support implementation smoke checks but are not study outcomes.
