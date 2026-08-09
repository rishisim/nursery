# Synthetic development launch v2 limitations

This package is synthetic. Even a later `DEVELOPMENT_PASS` would show only that the frozen synthetic sensor stream assists the frozen weak lexical/action grounding learner on held-out synthetic prompts. It would not establish infant learning, ecological validity, causal effects in BabyView/AEA/ChildLens data, or generalization to real children, homes, sensors, or language.

The corpus generator, event geometry, observation vectors, sensor process, null mechanism, detector, and learner are deliberately constructed. Exact balance and reproducibility make the software test interpretable; they do not make the synthetic distribution naturalistic.

The learner has useful but bounded detector capacity and must learn both target presence and absence. The null controls demonstrate dependence on target-absent training and visible absence evidence, not a psychologically realistic absence-learning process.

The primary inference is paired across 40 synthetic corpus seeds. Model replicates quantify algorithmic stochasticity but are averaged within corpus and are not independent experimental units. Bootstrap and sign-test behavior is frozen, including conservative failure on degenerate zero or insufficient-nonzero distributions.

The confirmation reserve is only reserved. This package cannot execute it, and development results—positive or negative—cannot be used to alter confirmation gates.

