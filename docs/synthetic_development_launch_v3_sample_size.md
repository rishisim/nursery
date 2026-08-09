# Frozen development sample size

The development cohort uses 40 independent corpus seeds and three stochastic model replicates per corpus. Replicates are averaged within corpus before inference.

Forty corpora are retained because they exceed the program's normal minimum, permit non-degenerate paired-distribution diagnostics, and make the exact sign test capable of rejecting a null direction without treating model replicates as independent. Three model seeds are the smallest frozen count that demonstrates genuine stochastic replication while keeping the independent unit at corpus seed.

No development or confirmation outcome informed this size. The package-sealing fixture and excluded rehearsal use separate `310xxx` identifiers and suppress scientific decisions. If the realized development distribution is all zero, below threshold, or has fewer than eight nonzero corpus effects, the frozen terminal is `DEVELOPMENT_NO_GO`; the package does not manufacture a confidence interval.
