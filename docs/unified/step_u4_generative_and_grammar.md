# Step U4: Generative Fluency, Grammaticality & Cross-Modal Shaping

Status: **completed and locked** (`outputs/runs/unified_u4_eval/`, 30 locked
630-clip U3 checkpoints plus a five-seed Text-Only LM). Evaluation code
`e2231d08` and `4f17ad0b`. Context: [CONTEXT.md](CONTEXT.md).

## Prompted Generation

Five-seed mean accuracy (%) ± sample SD. Parentheses give held-out clip counts.
Open-vocab uses the model's top token. In-category ranks only the task's
candidate tokens. Property scores are identical under both rankings.

| Arm | Object open (297) | Object in-cat (297) | Action in-cat (204) | Action open (204) | Property (203) | Straddling (24) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V+G+B | 26.5 ± 6.9 | 26.6 ± 7.0 | 72.0 ± 2.6 | 25.2 ± 9.7 | 93.5 ± 3.0 | 91.7 ± 6.6 |
| Shuffled body | 17.5 ± 7.2 | 17.5 ± 7.2 | 60.2 ± 3.2 | 21.1 ± 6.0 | 67.1 ± 4.0 | 52.5 ± 4.8 |
| V+G | 14.7 ± 9.2 | 14.7 ± 9.2 | 64.4 ± 5.0 | 9.7 ± 5.0 | 66.9 ± 4.3 | 54.2 ± 5.9 |
| V | 3.2 ± 0.9 | 3.2 ± 0.9 | 62.3 ± 1.7 | 19.8 ± 6.1 | 64.8 ± 2.0 | 51.7 ± 2.3 |
| CLIP-Global | 6.4 ± 2.4 | 6.4 ± 2.4 | 47.5 ± 3.1 | 0.2 ± 0.4 | 63.5 ± 0.0 | 50.0 ± 0.0 |
| BabyLLaVA-Nursery | 18.9 ± 3.2 | 18.9 ± 3.2 | 63.8 ± 2.1 | 26.1 ± 13.4 | 67.3 ± 5.6 | 58.3 ± 8.8 |
| Text-Only LM | 1.7 ± 0.0 | 1.7 ± 0.0 | 44.6 ± 0.0 | 0.0 ± 0.0 | 63.5 ± 0.0 | 50.0 ± 0.0 |

The prompts were checked against immediate next tokens in the training speech.

- **Object** uses `look at the`. A noun follows in 240 of 242 occurrences.
- **Action** uses `let's` (tokenized `let s`). It occurs 879 times, followed by
  `lift` 103 times, `pass` 173 times, `drink` 33 times, and `inspect` 64 times.
  The literal `offhand` never follows `let s`, so the single-token action task
  excludes its 19 clips. `switch` is a frequent continuation (295).
- **Property** uses `that feels`, followed by `heavy` 126 times and `light`
  77 times. The straddling subset covers bell, bottle, doll, and rock, each
  with both mass labels.

Matched seed-0 examples show the open-vocab top token.

| Task | Clip | Target | V+G+B | BabyLLaVA-Nursery | V |
| --- | --- | --- | --- | --- | --- |
| Object | `apple_offhand_1` | apple | apple | balloon | sandwich |
| Action | `banana_lift` | lift | switch | switch | find |
| Property | `cubesmall_inspect_1` | light | light | light | heavy |
| Property straddling | `waterbottle_pour_1` | heavy | heavy | light | light |

`V+G+B` leads on property completion (93.5%, 91.7% on the
straddling subset). The gap between in-category and open-vocab action accuracy
shows how often other continuations outrank the intended verb.

## Held-Out Perplexity

Perplexity uses the grammatical, scene-matched reference from each frozen U2
trial. Lower is better. Values are five-seed mean ± SD. Text-Only LM perplexity
is unconditional.

| Arm | Perplexity ± SD | Conditioning |
| --- | ---: | --- |
| V+G+B | 48.9 ± 5.2 | Scene |
| Shuffled body | 49.2 ± 6.1 | Scene |
| V+G | 50.2 ± 6.5 | Scene |
| V | 55.2 ± 5.1 | Scene |
| CLIP-Global | 43.9 ± 8.9 | Scene |
| BabyLLaVA-Nursery | 65.3 ± 9.2 | Scene |
| Text-Only LM | 106.4 ± 14.4 | None |

Unprompted greedy captions collapsed to caregiver chatter such as “oh wow
there we go” across all six scene-conditioned arms. Their mean lexical F1
peaked at 8.8%. The Text-Only LM run saved no greedy captions or lexical F1.
Prompted completion is the generation readout.

[Generation and grammar scorecard](../../outputs/runs/unified_u4_eval/figures/u4_generation_grammar_scorecard.png)

## Body-Removed Transfer

Five-seed mean accuracy (%) ± SD. U1 Nouns uses test. All other panels pool
test + S8. Body removal keeps gaze. Removing all extras gives video-only input.

| Arm | U1 Nouns | U1 Actions | Action + Object | Property + Object | Negation |
| --- | ---: | ---: | ---: | ---: | ---: |
| V+G+B as trained | 74.6 ± 1.0 | 73.2 ± 1.3 | 71.5 ± 5.6 | 86.1 ± 1.3 | 81.7 ± 1.0 |
| V+G+B body removed | 74.3 ± 1.7 | 66.4 ± 2.1 | 66.9 ± 4.0 | 66.4 ± 2.4 | 76.2 ± 1.7 |
| V+G+B all extras removed (video only) | 65.1 ± 1.2 | 65.5 ± 3.7 | 64.1 ± 4.7 | 59.4 ± 1.9 | 66.9 ± 6.0 |
| V+G | 73.3 ± 2.9 | 70.0 ± 1.2 | 67.8 ± 4.1 | 66.9 ± 1.8 | 72.5 ± 5.4 |
| V | 56.7 ± 3.9 | 66.1 ± 3.0 | 58.7 ± 4.5 | 57.3 ± 5.4 | 64.5 ± 4.3 |
| Shuffled body | 72.7 ± 1.3 | 66.0 ± 3.2 | 68.0 ± 3.6 | 67.0 ± 1.6 | 72.5 ± 3.5 |
| CLIP-Global | 47.4 ± 3.0 | 52.0 ± 1.7 | 57.1 ± 3.3 | 53.4 ± 4.2 | 71.1 ± 2.7 |
| BabyLLaVA-Nursery | 42.8 ± 2.9 | 62.3 ± 2.7 | 44.9 ± 2.9 | 46.9 ± 5.9 | 51.5 ± 5.2 |

Removing body drops Property + Object from 86.1% to 66.4%. On U1 Actions,
video-only `V+G+B` scores 65.5% and `V` scores 66.1%.

[Body-removed transfer](../../outputs/runs/unified_u4_eval/figures/u4_body_removed_transfer.png)

## Nursery-BLiMP

Unconditional minimal-pair accuracy (%) ± SD over five seeds. Counts are in
parentheses. Familiar/unseen counts are 69/48 for agreement, 32/0 for word
order, 11/32 for negation syntax, and 21/0 for verb argument.

| Arm | Overall (213) | Agreement (117) | Word order (32) | Negation syntax (43) | Verb argument (21) | Familiar frame, novel pair (133) | Unseen frame (80) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V+G+B | 81.9 ± 1.8 | 87.9 ± 2.0 | 25.6 ± 12.4 | 100.0 ± 0.0 | 97.1 ± 4.3 | 77.1 ± 3.1 | 89.8 ± 0.6 |
| Shuffled body | 79.7 ± 1.7 | 87.4 ± 0.4 | 11.9 ± 10.5 | 100.0 ± 0.0 | 99.0 ± 2.1 | 73.5 ± 2.8 | 90.0 ± 0.0 |
| V+G | 83.4 ± 2.9 | 87.7 ± 1.1 | 35.6 ± 19.0 | 100.0 ± 0.0 | 98.1 ± 2.6 | 79.2 ± 4.7 | 90.2 ± 0.6 |
| V | 81.1 ± 4.5 | 87.4 ± 0.4 | 25.0 ± 26.4 | 100.0 ± 0.0 | 93.3 ± 9.9 | 75.8 ± 7.2 | 90.0 ± 0.0 |
| CLIP-Global | 88.9 ± 3.5 | 87.2 ± 0.0 | 73.1 ± 23.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 88.3 ± 5.5 | 90.0 ± 0.0 |
| BabyLLaVA-Nursery | 88.7 ± 2.8 | 88.5 ± 2.2 | 68.8 ± 16.7 | 100.0 ± 0.0 | 97.1 ± 6.4 | 87.7 ± 4.0 | 90.5 ± 2.3 |
| Text-Only LM | 89.5 ± 4.0 | 89.4 ± 1.8 | 70.0 ± 32.2 | 100.0 ± 0.0 | 98.1 ± 2.6 | 89.0 ± 6.7 | 90.2 ± 1.9 |

Word order has only 32 pairs and is noisy across seeds. It drives the overall
between-arm spread. Removing that bucket leaves a narrow 91.0–92.9% range
across arms. The word `these` is absent from training and the evaluated pairs.

## Action Embeddings

`pooled_sensory` is the mean of active sensory prefix tokens, averaged over
the labeled moments in each of 223 held-out action clips. The figure
fits one PCA basis per arm on its pooled seed-0 embeddings across all four
training sizes. Lines connect class centroids across independently trained
checkpoints. The coordinates can shift with training size, so separation is
measured within each checkpoint in the original space. Between/within is the
mean cosine distance between classes divided by that within classes, using
distinct clip pairs. The 630-clip aggregate reports five-seed mean ± SD.

| Arm | Training clips | Seeds | Cosine silhouette | Between / within |
| --- | ---: | --- | ---: | ---: |
| V+G+B | 63 | 0 | 0.069 | 1.148 |
| V+G+B | 158 | 0 | 0.087 | 1.186 |
| V+G+B | 315 | 0 | 0.070 | 1.186 |
| V+G+B | 630 | 0 | 0.135 | 1.309 |
| V+G+B | 630 | 0–4 | 0.130 ± 0.005 | 1.297 ± 0.024 |
| V | 63 | 0 | 0.106 | 1.370 |
| V | 158 | 0 | 0.094 | 1.296 |
| V | 315 | 0 | 0.102 | 1.359 |
| V | 630 | 0 | 0.132 | 1.470 |
| V | 630 | 0–4 | 0.139 ± 0.004 | 1.456 ± 0.020 |
| CLIP-Global | 63 | 0 | 0.040 | 1.093 |
| CLIP-Global | 158 | 0 | 0.049 | 1.108 |
| CLIP-Global | 315 | 0 | 0.060 | 1.124 |
| CLIP-Global | 630 | 0 | 0.093 | 1.311 |
| CLIP-Global | 630 | 0–4 | 0.092 ± 0.006 | 1.298 ± 0.062 |

At 630 clips, `V+G+B` separates actions better than CLIP-Global by silhouette.
`V` has the higher five-seed mean on both separation measures.

[3D action trajectories](../../outputs/runs/unified_u4_eval/figures/u4_action_trajectory_3d.png) ·
[Action separation data](../../outputs/runs/unified_u4_eval/figures/u4_action_separation.json)

## Stop / Go Gate

**PASS (Locked).** The saved evaluations give U5 a reference for the backbone
swap. Check whether the BabyView backbone retains the property grounding gain
and the U3 acquisition gains under the same splits and scoring. The U4
video-only action and grammar results give no basis to expect gains on those
measures.
