# Step U6: Developmental Comparison (Wordbank AoA & DevBench)

Status: **design blueprint, blocked on Step U3 / U5 completed curves.**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Connect Nursery's simulated acquisition curves directly to **real human infant
language development**, answering two questions central to developmental
cognitive science:

1. **Order & rate (Wordbank):** Does embodied experience (`V+G+B`) make the
   *order* and *growth rate / acceleration* of word acquisition match human
   infants in [Wordbank](https://langcog.github.io/wordbank-book/)
   ([Frank et al., 2016](https://www.cambridge.org/core/journals/journal-of-child-language/article/wordbank-an-open-repository-for-developmental-vocabulary-data/977D930531B5318CA976CD8582D9F401))
   more closely than video-only models (`V`, `CLIP-Global`, `BabyLLaVA`)?
2. **Transfer & human-like error patterns (DevBench):** On the overlapping
   subsets of [DevBench](https://arxiv.org/abs/2406.10215) (Looking-While-Listening,
   Visual Vocabulary, and THINGS representational similarity), does `V+G+B`
   achieve higher accuracy and closer model–human similarity than video-only
   baselines?

## Evaluation Protocol

1. **Wordbank Age-of-Acquisition (AoA) matching:**
   - Map all overlapping Nursery nouns, action verbs, and physical properties
     to English MacArthur-Bates CDI comprehension and production AoA (in
     months) from Wordbank.
   - For each model (`V`, `V+G`, `V+G+B`, `CLIP-Global`, `BabyLLaVA-Nursery`),
     compute each word's **mastery clip count** (interpolated clip count where
     per-word 4-choice accuracy crosses 75%) and compute Spearman/Pearson
     correlation with Wordbank AoA (with bootstrap 95% CIs over words and
     seeds).
   - Compute **vocabulary growth curves** (number of words mastered ≥ 75% vs.
     training clips) and test slope / early acceleration, directly connecting
     to [Frank (2026)](https://arxiv.org/pdf/2608.17120).
2. **DevBench & Konkle transfer:**
   - Evaluate saved checkpoints (all extra inputs removed, vision + text only)
     on overlapping DevBench lexical trials (LWL, VV), THINGS representational
     similarity (Spearman `r` against human judgments), and Konkle 60-object
     transfer.

## Stop / Go Gate

- Report exact vocabulary overlap counts with Wordbank and DevBench up front
  (never silently drop non-overlapping words without stating coverage).
- Confirm all correlations report both word-level and seed-level uncertainty.

## Expected Figures

1. **Model Mastery vs. Human Wordbank AoA Scatter:** Plotting each word's
   model mastery clip count (`V` vs. `V+G+B`) against human infant AoA
   (months), colored by word class (nouns, actions, properties).
2. **Vocabulary Growth & Acceleration Curve:** Total words mastered (≥ 75%)
   across `63 → 630` clips for `V+G+B`, `V+G`, `V`, `CLIP-Global`, and
   `BabyLLaVA-Nursery`.
3. **DevBench Accuracy & Model–Human Similarity Comparison:** Side-by-side
   transfer readout alongside Konkle picture accuracy.
