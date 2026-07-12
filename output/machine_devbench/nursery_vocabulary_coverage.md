# Nursery × Machine-DevBench vocabulary coverage

- Benchmark: 20 official manifests / 3721 target-word trials.
- Nursery vocabulary: 48 tokens (special tokens excluded).
- Unique benchmark target coverage: 13/1414 (0.9%).
- Trial-weighted target coverage: 2.6%.
- Unique evaluated-caption token coverage: 16/1599 (1.0%).
- Occurrence-weighted evaluated-caption coverage: 29.9%.

| Task | Unique targets | Covered | Unique coverage | Trial-weighted |
|---|---:|---:|---:|---:|
| gram_comparatives | 63 | 2 | 3.2% | 3.9% |
| gram_counting | 27 | 0 | 0.0% | 0.0% |
| gram_embedded_relative | 23 | 2 | 8.7% | 4.7% |
| gram_negation | 44 | 2 | 4.5% | 5.4% |
| gram_order_matters | 12 | 3 | 25.0% | 31.1% |
| gram_prepositions | 205 | 0 | 0.0% | 0.0% |
| gram_subject_adjective | 91 | 2 | 2.2% | 2.4% |
| gram_subject_verb | 32 | 0 | 0.0% | 0.0% |
| lex_adjectives | 239 | 5 | 2.1% | 2.5% |
| lex_nouns | 1009 | 5 | 0.5% | 0.4% |

Interpretation: Exact-token coverage is a compatibility diagnostic, not an evaluation score. Low target coverage predicts UNK collisions and makes the provisional Nursery checkpoint unsuitable for substantive Machine-DevBench conclusions until its training vocabulary and experience distribution are expanded.
