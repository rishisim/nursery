# Step 9 acquisition curves

Verified all 85 metrics and completion records against the frozen queue and protocol hash. Each run completed 60 epochs with per-word temporal weights and noun+action dev selection. Recomputing selection from dev histories and body-weight sweeps reproduced every selected epoch and weight. All numeric values in the metrics are finite. Saved predictions, trial credits and aggregate accuracies agree with the frozen panels.

Accuracy is mean ± sample SD (%) over seeds 0–4. Gains are percentage points with paired joint percentile 95% intervals (20,000 draws, RNG seed 1707 restarted at each clip count). As in Step 7, each draw resamples five seed IDs and the observed number of held-out source clips with replacement, sharing both draws across arms. All trials in each sampled clip are kept. Pooled action panels weight trials equally. Gaze = V+G − V, body = V+G+B − V+G, total = V+G+B − V.

## Reading the curves

Aligned body improves pooled action accuracy at every clip count, including 63 clips. The gain also holds for object-matched actions at every amount. Gaze improves noun accuracy throughout the ladder, and every noun body-gain interval includes zero.

From 473 to 630 clips, action accuracy rises by 5.60 pp for V and 2.12 pp for V+G+B. Video-only action accuracy is still rising at the largest budget, so the observed modality gain leaves the ceiling question open. The next step toward testing a ceiling is measuring these arms beyond 630 clips. The low-data action results leave Step 8's wordless-pretraining trigger unmet.

## Nouns, clips_test

556 trials across 115 source clips.

| Clips | V | V+G | V+G+B | Gaze | Body | Total |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | 28.62 ± 1.34 | 38.08 ± 1.05 | 37.33 ± 1.60 | +9.46 [6.24, 12.74] | -0.76 [-3.22, 1.48] | +8.71 [5.35, 12.05] |
| 158 | 37.79 ± 1.43 | 53.29 ± 2.30 | 53.00 ± 1.43 | +15.50 [10.81, 19.96] | -0.29 [-3.24, 2.54] | +15.22 [10.47, 19.76] |
| 315 | 45.22 ± 2.40 | 65.25 ± 2.38 | 63.56 ± 1.82 | +20.04 [15.13, 25.28] | -1.69 [-5.52, 1.47] | +18.35 [13.91, 22.89] |
| 473 | 53.96 ± 1.67 | 69.82 ± 1.23 | 70.14 ± 1.52 | +15.86 [11.88, 19.97] | +0.32 [-1.91, 2.77] | +16.19 [12.01, 20.46] |
| 630 | 56.91 ± 1.60 | 72.95 ± 1.18 | 72.48 ± 1.03 | +16.04 [11.84, 20.33] | -0.47 [-2.66, 1.82] | +15.58 [11.59, 19.65] |

## Actions, test + s8 pooled

1,925 trials across 223 source clips.

| Clips | V | V+G | V+G+B | Gaze | Body | Total |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | 37.03 ± 3.42 | 36.09 ± 1.91 | 45.59 ± 1.49 | -0.95 [-4.43, 2.36] | +9.51 [5.51, 13.67] | +8.56 [3.99, 13.00] |
| 158 | 51.26 ± 2.49 | 48.81 ± 2.85 | 55.86 ± 3.31 | -2.45 [-5.98, 0.90] | +7.05 [3.84, 10.77] | +4.60 [0.70, 8.68] |
| 315 | 59.06 ± 1.77 | 59.27 ± 1.26 | 68.55 ± 1.98 | +0.21 [-2.83, 3.32] | +9.28 [6.09, 12.68] | +9.49 [6.49, 12.64] |
| 473 | 64.34 ± 1.27 | 65.34 ± 1.96 | 74.42 ± 2.31 | +1.00 [-2.73, 4.55] | +9.08 [6.27, 12.23] | +10.08 [6.73, 13.61] |
| 630 | 69.94 ± 2.09 | 70.12 ± 1.52 | 76.54 ± 2.24 | +0.18 [-3.54, 3.62] | +6.42 [4.21, 9.04] | +6.60 [3.29, 9.98] |

## Shuffled body

Gain is shuffled V+G+B minus V+G.

| Clips | Rung | V+G+B shuffled | Shuffled body gain |
| --- | --- | --- | --- |
| 158 | Noun | 52.86 ± 1.91 | -0.43 [-1.87, 0.95] |
| 158 | Action | 49.64 ± 2.43 | +0.83 [-0.39, 2.10] |
| 630 | Noun | 72.12 ± 2.27 | -0.83 [-2.38, 0.63] |
| 630 | Action | 70.04 ± 1.54 | -0.08 [-2.02, 1.87] |

## Dev-selected body weights

| Clips | Arm | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | V+G+B | 0.70 | 0.70 | 0.70 | 0.90 | 0.80 |
| 158 | V+G+B | 0.30 | 0.30 | 0.35 | 0.50 | 0.45 |
| 158 | V+G+B shuffled | 0.15 | 0.10 | 0.00 | 0.05 | 0.05 |
| 315 | V+G+B | 0.45 | 0.45 | 0.35 | 0.25 | 0.55 |
| 473 | V+G+B | 0.50 | 0.30 | 0.30 | 0.40 | 0.50 |
| 630 | V+G+B | 0.15 | 0.40 | 0.65 | 0.35 | 0.45 |
| 630 | V+G+B shuffled | 0.00 | 0.25 | 0.00 | 0.00 | 0.40 |

## Selected epochs

Epochs 56–60 flag possible under-training. Counts include the shuffled arm where queued.

| Clips | Models | Selected epoch range | Late selections |
| --- | --- | --- | --- |
| 63 | 15 | 5–49 | 0 |
| 158 | 20 | 14–55 | 0 |
| 315 | 15 | 13–34 | 0 |
| 473 | 15 | 12–26 | 0 |
| 630 | 20 | 10–20 | 0 |

## Action halves and object-matched actions

| Clips | Panel | Trials | Source clips | V | V+G | V+G+B | V+G+B shuffled | Body gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 63 | Test | 1049 | 103 | 38.78 ± 3.47 | 38.22 ± 3.38 | 48.54 ± 1.65 | — | +10.31 [4.43, 16.19] |
| 63 | s8 | 876 | 120 | 34.95 ± 3.93 | 33.53 ± 1.07 | 42.07 ± 2.59 | — | +8.54 [3.33, 14.27] |
| 63 | Object-matched pooled | 637 | 80 | 38.27 ± 3.29 | 36.60 ± 2.32 | 47.90 ± 2.50 | — | +11.30 [5.88, 16.91] |
| 63 | Object-matched test | 235 | 26 | 44.87 ± 4.72 | 40.70 ± 3.25 | 54.91 ± 5.09 | — | +14.21 [3.94, 24.68] |
| 63 | Object-matched s8 | 402 | 54 | 34.40 ± 3.61 | 34.20 ± 2.43 | 43.81 ± 3.76 | — | +9.60 [2.93, 16.99] |
| 158 | Test | 1049 | 103 | 56.43 ± 1.93 | 54.81 ± 3.45 | 58.76 ± 4.17 | 55.54 ± 2.67 | +3.95 [-0.68, 9.28] |
| 158 | s8 | 876 | 120 | 45.07 ± 3.19 | 41.62 ± 3.00 | 52.40 ± 3.65 | 42.58 ± 2.90 | +10.78 [6.38, 15.24] |
| 158 | Object-matched pooled | 637 | 80 | 52.68 ± 3.98 | 46.84 ± 3.01 | 55.79 ± 4.18 | 46.94 ± 2.37 | +8.95 [4.95, 13.47] |
| 158 | Object-matched test | 235 | 26 | 59.83 ± 5.23 | 52.77 ± 4.65 | 59.66 ± 5.49 | 52.34 ± 2.89 | +6.89 [0.48, 14.50] |
| 158 | Object-matched s8 | 402 | 54 | 48.51 ± 3.85 | 43.38 ± 2.54 | 53.53 ± 3.93 | 43.78 ± 2.80 | +10.15 [5.13, 15.34] |
| 315 | Test | 1049 | 103 | 65.45 ± 1.60 | 66.41 ± 1.04 | 71.27 ± 2.02 | — | +4.86 [1.09, 8.71] |
| 315 | s8 | 876 | 120 | 51.42 ± 2.48 | 50.73 ± 3.17 | 65.30 ± 2.35 | — | +14.57 [9.97, 19.42] |
| 315 | Object-matched pooled | 637 | 80 | 57.11 ± 4.04 | 52.43 ± 2.89 | 64.43 ± 3.25 | — | +11.99 [6.84, 17.32] |
| 315 | Object-matched test | 235 | 26 | 66.21 ± 3.96 | 60.26 ± 3.15 | 68.77 ± 4.26 | — | +8.51 [-0.65, 18.21] |
| 315 | Object-matched s8 | 402 | 54 | 51.79 ± 4.36 | 47.86 ± 3.24 | 61.89 ± 3.51 | — | +14.03 [8.55, 19.38] |
| 473 | Test | 1049 | 103 | 71.00 ± 2.06 | 72.05 ± 1.85 | 77.39 ± 1.39 | — | +5.34 [2.41, 8.53] |
| 473 | s8 | 876 | 120 | 56.37 ± 2.82 | 57.31 ± 3.34 | 70.87 ± 4.82 | — | +13.56 [9.16, 18.50] |
| 473 | Object-matched pooled | 637 | 80 | 59.59 ± 2.47 | 58.81 ± 2.63 | 71.21 ± 3.05 | — | +12.40 [6.99, 18.54] |
| 473 | Object-matched test | 235 | 26 | 64.77 ± 4.93 | 64.77 ± 4.14 | 73.79 ± 2.01 | — | +9.02 [1.46, 17.54] |
| 473 | Object-matched s8 | 402 | 54 | 56.57 ± 4.05 | 55.32 ± 2.73 | 69.70 ± 5.28 | — | +14.38 [7.89, 21.83] |
| 630 | Test | 1049 | 103 | 75.46 ± 2.31 | 76.78 ± 1.15 | 80.95 ± 1.24 | 76.47 ± 1.04 | +4.18 [1.81, 6.92] |
| 630 | s8 | 876 | 120 | 63.33 ± 2.82 | 62.15 ± 2.27 | 71.26 ± 3.60 | 62.33 ± 2.20 | +9.11 [5.52, 13.07] |
| 630 | Object-matched pooled | 637 | 80 | 66.09 ± 2.69 | 64.05 ± 2.69 | 71.87 ± 2.16 | 63.70 ± 3.07 | +7.82 [3.73, 12.47] |
| 630 | Object-matched test | 235 | 26 | 74.64 ± 5.10 | 72.26 ± 3.96 | 78.47 ± 1.40 | 71.83 ± 2.92 | +6.21 [-0.17, 13.83] |
| 630 | Object-matched s8 | 402 | 54 | 61.09 ± 3.70 | 59.25 ± 3.17 | 68.01 ± 3.37 | 58.96 ± 3.37 | +8.76 [3.94, 13.96] |

## Body gain by action word

At 63 clips, drink has three training utterances and is out of vocabulary. Its frozen trials receive 25% expected credit in every arm.

| Clips | drink | inspect | lift | offhand | pass |
| --- | --- | --- | --- | --- | --- |
| 63 | +0.00 [0.00, 0.00] | -6.87 [-15.59, 1.18] | +14.96 [7.88, 22.07] | +9.92 [-4.10, 24.43] | +14.66 [7.81, 21.13] |
| 158 | -1.53 [-13.59, 8.31] | -1.29 [-7.88, 5.52] | +13.64 [7.05, 21.40] | +6.89 [-3.52, 18.21] | +8.06 [3.50, 12.82] |
| 315 | -3.18 [-12.03, 4.80] | +0.25 [-7.83, 7.22] | +15.54 [8.44, 23.73] | +16.30 [6.07, 27.01] | +10.53 [6.62, 14.68] |
| 473 | +3.41 [-4.97, 13.49] | +4.48 [-1.92, 11.41] | +13.37 [8.53, 19.09] | +18.99 [8.36, 30.86] | +7.91 [3.66, 12.37] |
| 630 | +0.94 [-7.07, 9.31] | +2.58 [-2.63, 8.47] | +11.90 [7.45, 17.17] | +6.72 [-2.11, 16.06] | +5.57 [3.11, 8.51] |

## All extra inputs removed at evaluation

| Clips | Rung | V | V+G | V+G+B | V+G+B shuffled |
| --- | --- | --- | --- | --- | --- |
| 63 | Noun | 28.62 ± 1.34 | 32.69 ± 1.27 | 32.58 ± 0.84 | — |
| 63 | Action | 37.03 ± 3.42 | 38.14 ± 1.71 | 36.85 ± 1.81 | — |
| 158 | Noun | 37.79 ± 1.43 | 41.85 ± 3.02 | 41.10 ± 0.40 | 40.85 ± 0.85 |
| 158 | Action | 51.26 ± 2.49 | 51.42 ± 1.65 | 50.24 ± 2.28 | 52.85 ± 2.03 |
| 315 | Noun | 45.22 ± 2.40 | 49.39 ± 1.94 | 50.04 ± 3.23 | — |
| 315 | Action | 59.06 ± 1.77 | 60.14 ± 1.47 | 58.26 ± 3.01 | — |
| 473 | Noun | 53.96 ± 1.67 | 57.55 ± 1.63 | 59.14 ± 0.35 | — |
| 473 | Action | 64.34 ± 1.27 | 65.38 ± 1.71 | 61.15 ± 3.20 | — |
| 630 | Noun | 56.91 ± 1.60 | 60.61 ± 1.65 | 60.40 ± 1.54 | 61.40 ± 2.27 |
| 630 | Action | 69.94 ± 2.09 | 70.74 ± 1.92 | 67.11 ± 1.24 | 70.75 ± 1.83 |

## Picture transfer

Rendered and Konkle panels use the selected checkpoint with picture inputs only.

| Clips | Panel | Trials | V | V+G | V+G+B | V+G+B shuffled |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | Rendered | 7800 | 28.01 ± 1.17 | 33.22 ± 1.05 | 33.22 ± 1.59 | — |
| 63 | Konkle | 6000 | 28.85 ± 1.34 | 36.00 ± 1.14 | 36.38 ± 0.66 | — |
| 158 | Rendered | 7800 | 43.29 ± 2.26 | 51.82 ± 3.38 | 50.58 ± 2.07 | 51.03 ± 1.71 |
| 158 | Konkle | 6000 | 43.57 ± 0.88 | 54.57 ± 4.94 | 53.95 ± 2.46 | 53.64 ± 3.10 |
| 315 | Rendered | 7800 | 57.87 ± 3.18 | 65.76 ± 3.37 | 65.84 ± 3.41 | — |
| 315 | Konkle | 6000 | 58.46 ± 3.36 | 70.58 ± 2.40 | 70.44 ± 2.23 | — |
| 473 | Rendered | 7800 | 67.88 ± 1.26 | 73.63 ± 1.52 | 74.64 ± 1.48 | — |
| 473 | Konkle | 6000 | 64.45 ± 2.76 | 75.10 ± 2.43 | 76.60 ± 2.29 | — |
| 630 | Rendered | 7800 | 72.27 ± 3.53 | 79.01 ± 2.98 | 79.14 ± 2.85 | 79.45 ± 2.68 |
| 630 | Konkle | 6000 | 70.29 ± 2.67 | 80.73 ± 2.49 | 81.13 ± 1.87 | 81.12 ± 2.16 |

## 630 clips versus published Step 7

The held-out panels, five seeds and 630 training clips match Step 7. Step 9 changes temporal initialization and dev selection, which limits attribution to either change alone. [Step 7 published table](../../../../docs/step_7_word_types.md#completed-results--19-november-2026)

| Rung | Study | V | V+G | V+G+B | V+G+B shuffled |
| --- | --- | --- | --- | --- | --- |
| Noun | Step 7 | 50.65 | 65.65 | 66.55 | 65.65 |
| Noun | Step 9 | 56.91 ± 1.60 | 72.95 ± 1.18 | 72.48 ± 1.03 | 72.12 ± 2.27 |
| Noun | Δ pp | +6.26 | +7.30 | +5.93 | +6.47 |
| Action | Step 7 | 65.64 | 66.45 | 77.87 | 65.79 |
| Action | Step 9 | 69.94 ± 2.09 | 70.12 ± 1.52 | 76.54 ± 2.24 | 70.04 ± 1.54 |
| Action | Δ pp | +4.30 | +3.67 | -1.33 | +4.25 |

| Rung | Study | Gaze | Body | Total |
| --- | --- | --- | --- | --- |
| Noun | Step 7 | +15.00 [11.11, 19.05] | +0.90 [-2.11, 3.85] | +15.90 [11.73, 20.44] |
| Noun | Step 9 | +16.04 [11.84, 20.33] | -0.47 [-2.66, 1.82] | +15.58 [11.59, 19.65] |
| Noun | Δ pp | +1.04 | -1.37 | -0.32 |
| Action | Step 7 | +0.81 [-2.34, 3.83] | +11.42 [8.57, 14.62] | +12.23 [8.96, 15.61] |
| Action | Step 9 | +0.18 [-3.54, 3.62] | +6.42 [4.21, 9.04] | +6.60 [3.29, 9.98] |
| Action | Δ pp | -0.63 | -5.00 | -5.63 |

## Noun curve versus Step 6

Both studies use the same noun test panel. Step 6 used static video, different speech and dev selection, and three seeds. Differences are Step 9 minus Step 6 and are descriptive. [Step 6 published curve](../../../../docs/step_6_language_acquisition_curves.md#completed-results--17-november-2026)

| Clips | Step 6 V | Step 6 V+G | Step 6 V+G+B | Δ V (pp) | Δ V+G (pp) | Δ V+G+B (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| 63 | 29.45 ± 0.36 | 39.40 ± 0.73 | 39.34 ± 0.31 | -0.83 | -1.32 | -2.01 |
| 158 | 37.14 ± 0.54 | 54.95 ± 0.18 | 54.71 ± 1.26 | +0.65 | -1.65 | -1.70 |
| 315 | 48.44 ± 1.51 | 66.85 ± 1.63 | 66.19 ± 1.09 | -3.23 | -1.59 | -2.63 |
| 473 | 58.51 ± 1.46 | 72.06 ± 0.52 | 72.78 ± 0.45 | -4.56 | -2.24 | -2.64 |
| 630 | 63.67 ± 1.18 | 75.60 ± 1.08 | 76.26 ± 0.48 | -6.76 | -2.65 | -3.78 |

Published Step 6 gain intervals resampled source clips with seeds held fixed.

| Clips | Step 6 Gaze | Step 6 Body | Step 6 Total |
| --- | --- | --- | --- |
| 63 | +9.95 [6.05, 14.11] | -0.06 [-0.73, 0.67] | +9.89 [6.12, 13.99] |
| 158 | +17.81 [13.72, 21.74] | -0.24 [-1.49, 1.00] | +17.57 [13.46, 21.54] |
| 315 | +18.41 [14.19, 22.69] | -0.66 [-1.83, 0.54] | +17.75 [13.73, 21.95] |
| 473 | +13.55 [10.42, 16.81] | +0.72 [-0.00, 1.53] | +14.27 [11.04, 17.59] |
| 630 | +11.93 [8.67, 15.40] | +0.66 [-0.00, 1.33] | +12.59 [9.37, 16.03] |

