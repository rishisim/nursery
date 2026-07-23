# Child-only scientific and data-boundary contract

Version: `child-only-scientific-contract-v1.0.0`  
Stage: construction only  
Current selected corpus: none  
Current data profile: `NONSCIENTIFIC_CONSTRUCTION_FIXTURE`  
Scientific acquisition outcome authorized: no

## Scientific question

Under realistically weak speech–event alignment, do synchronized training-only embodied cues improve held-out word–object and word–action grounding beyond identical vision-language experience with matched shuffled cues, when all embodied cues are unavailable at evaluation?

This is a narrow synthetic causal test of lexical/object and lexical/action grounding. It cannot establish infant learning, BabyView equivalence, ecological validity, clinical/educational efficacy, or that synthetic data can replace child data.

## Nonnegotiable data boundary

The scientific study is child-data-only. Aria Everyday Activities (AEA) is historical, immutable, and outside the ancestry of this study. No AEA empirical data, aggregates, distributions, priors, language, IMU statistics, samples, features, tokenizers, checkpoints, weights, or results may enter calibration, generation, training, evaluation, model selection, or interpretation. Existing AEA files are neither read nor modified by the new namespace. The current contract supersedes the older suggestion that AEA might calibrate provisional stress-test ranges.

Only one child corpus may be selected for a scientific study instance:

- A BabyView instance may use only that BabyView instance for empirical calibration, tokenizer training, real-data training, and corpus-grounded vocabulary.
- A ChildLens instance may use only that ChildLens instance for those roles.
- The other corpus may be studied later only as a separately initialized external replication with a new study instance, fresh corpus-local tokenizer, and fresh model construction. Samples, empirical distributions, learned weights, vocabularies, caches, and checkpoints never cross the boundary.
- “Apples-to-apples” means identical architecture, procedure, causal arms, evaluation, and compute policy with corpus-namespaced fresh initialization—not pooling or transfer.

The machine-readable fail-closed implementation is [provenance_policy_v1.json](./provenance_policy_v1.json). Unknown source families, missing ancestry, cross-corpus ancestry, adult-data markers, parent checkpoints, and reused tokenizer/model artifact IDs are rejected before data loading.

## Pre-access and fixture rule

Restricted child access is not currently available and no corpus is selected. Every numeric fixture is therefore labeled exactly `NONSCIENTIFIC_CONSTRUCTION_FIXTURE`. Fixture numbers are arbitrary software-test values, are not estimates, and cannot support ecological or acquisition claims. A fixture cannot become scientific evidence by relabeling; all post-access artifacts must be regenerated from a selected child-corpus instance under a separately frozen outcome protocol.

## Modality and learner boundary

- Audio and TTS are deferred in v1.
- The primary learner is a compact temporal CLIP+-style model: small bidirectional BERT-like text tower, compact spatial ViT plus temporal Transformer, shared contrastive embedding, and no pretrained weights.
- The tokenizer and every scientific model initialize from scratch within the selected corpus instance. External pretrained models are permitted only in a quarantined, explicitly non-scientific diagnostic and can never be an ancestor of scientific artifacts.
- A small raw IMU/proprioception/contact temporal encoder and candidate-event/null aligner are training-only.
- Primary evaluation accepts a typed vision/text batch only. The exported evaluation module contains no side encoder, aligner, detector, oracle, or corpus-metadata state.
- Hidden object state, event truth, utterance–event targets, counterfactuals, detector annotations, and scoring labels stay physically separate from model-visible records.

## Causal and adaptation boundary

The five later arms are the strong-alignment ceiling, weak VL baseline, weak+synchronized side, weak+matched whole-episode shuffled side, and weak+within-episode time-shifted side. Every weak arm shares RGB/text data, splits, tokenizer, model architecture, initialization receipt, example order, optimizer, updates, padding/batch shapes, stopping rule, and compute budget. The first side construct combines IMU, proprioception, and contact; cue-specific ablations are a separately frozen branch only if a predeclared manipulation check says the combined construct is informative.

Construction-validity and software-correctness defects may be repaired now. In a later outcome task, once corpus instance, calibration payload, data, arms, endpoints, thresholds, seeds, and compute are frozen, no adaptation to effect direction is permitted.

## Terminal interpretation

`CHILD_ONLY_PROTOTYPE_CONSTRUCTION_READY` means all construction gates pass, no restricted child record or AEA-derived information entered the namespace, and no scientific acquisition comparison ran. It authorizes only a later, separate post-access freeze-and-outcome task. `CHILD_ONLY_PROTOTYPE_STOP` means an evidence-backed construction blocker makes the scoped prototype indefensible.
