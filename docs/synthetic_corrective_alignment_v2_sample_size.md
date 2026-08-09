# Corrective alignment v2 sample size

The frozen development population contains 100 independent corpus seeds and three paired stochastic model replicates per corpus. Corpus, not model fit or evaluation prompt, is the inferential unit.

The planning alternative is a mean strict-top-1 gain of 0.20 over the stronger per-corpus absent/shuffled comparator, against a practically meaningful bound of 0.10, with paired-corpus SD 0.30 and one-sided alpha 0.05. The 0.20 alternative is a design target, not a construction estimate: it corresponds to correcting 20 additional held-out decisions per 100 balanced decisions and leaves a full 0.10 separation above the minimum useful effect. The SD 0.30 planning value likewise is not estimated from any fixture. It permits corpus-to-corpus variation 1.5 times the target mean and three times the 0.10 alternative-to-bound separation. This is substantial heterogeneity for a bounded top-1 contrast, although it is not a worst-case bound. With 100 corpora it yields a planning standard error of 0.03, small enough to distinguish the 0.20 alternative from the 0.10 decision boundary.

Under the prospective normal corpus-effect planning model, the statistic for testing the +0.10 boundary has a noncentral Student-t distribution with 99 degrees of freedom and noncentrality δ = `(0.20 - 0.10)*sqrt(100)/0.30 = 3.333`. Therefore

\[
\Pr\left(T_{99,\delta=3.333}>t_{.95,99}\right)=0.9521,
\]

which is prospectively rounded to `0.952` in the configuration. Requiring both co-primary endpoints gives a conservative conjunctive-power lower bound of `1 - 2*(1-0.952) = 0.904` by the union bound. This calculation uses no construction or development outcome. There is no adaptive extension, seed replacement, or optional stopping. Missing corpora fail completeness rather than reducing the denominator.

The frozen sensitivity calculation keeps the 0.20 mean and varies the paired-corpus SD without changing sample size or the +0.10 boundary:

| Paired-corpus SD | Standard error at 100 corpora | Noncentral-t per-endpoint power | Union-bound conjunctive power |
|---:|---:|---:|---:|
| 0.20 | 0.020 | 0.9996 | 0.9991 |
| 0.30 | 0.030 | 0.952 | 0.904 |
| 0.40 | 0.040 | 0.799 | 0.598 |
| 0.50 | 0.050 | 0.634 | 0.267 |

Thus the advertised 0.904 conjunctive value is explicitly conditional on the 0.30 planning SD, not a guarantee. If the fixed synthetic population is more heterogeneous, the study remains a valid fixed-sample estimate and test but has lower power; it is not extended or re-seeded. Following Morris, White, and Crowther (2019), these target values and the sensitivity range are prospective operating assumptions rather than values tuned on simulated scientific outcomes.

The +0.10 bound is tied to categorical recovery: it represents ten additional unique correct mappings per 100 balanced held-out units, far larger and scientifically different from the closed protocol's 0.005 probability-gain threshold.

Corpus seeds are independent, identically distributed draws from the one frozen generator. Model replicates are paired nuisance replicates and are averaged within corpus before inference. Because the resulting corpus contrasts are bounded, the primary Student-t interval is justified as a studentized central-limit approximation at 100 independent corpora; the 20,000-replicate paired-corpus bootstrap is a sensitivity analysis. Ibragimov and Müller (2016) motivates group-first aggregation under heterogeneity but is not asserted to provide direct finite-sample coverage for these corpus contrasts.
