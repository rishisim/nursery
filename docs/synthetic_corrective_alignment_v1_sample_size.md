# Corrective alignment v1 sample size

The frozen development population contains 100 independent corpus seeds and three paired stochastic model replicates per corpus. Corpus, not model fit or evaluation prompt, is the inferential unit.

The planning alternative is a mean strict-top-1 gain of 0.20 over the stronger per-corpus absent/shuffled comparator, against a practically meaningful bound of 0.10, with paired-corpus SD 0.30 and one-sided alpha 0.05. The normal approximation gives

\[
\Pr\left(\bar d-t_{.95,99}s/\sqrt{100}>0.10\right)\approx 0.952.
\]

Requiring both co-primary endpoints gives a conservative conjunctive-power lower bound of `1 - 2*(1-0.952) = 0.904` by the union bound. This calculation uses no construction or development outcome. There is no adaptive extension, seed replacement, or optional stopping. Missing corpora fail completeness rather than reducing the denominator.

The +0.10 bound is tied to categorical recovery: it represents ten additional unique correct mappings per 100 balanced held-out units, far larger and scientifically different from the closed protocol's 0.005 probability-gain threshold.

