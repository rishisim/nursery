# Step U2: Compositional Sentences & Embodied VP-Swap

Status: **completed and locked (`b69091fb`, `outputs/runs/unified_u2_v1/`).**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Move Nursery beyond isolated word grounding to genuine **compositional sentence
comprehension** and **text-only physical semantics**, using the exact same 630
GRAB + EmbodiedGen clips so curves remain directly comparable with Nouns and
Actions.

## 1. Locked Tier A Design (Audit-Driven Scope)

Based on the read-only GRAB data audit, Hand/Role (`left` vs. `right`) was
dropped because it is 100% confounded with `offhand` (8/8 test, 11/11 s8; ~17 pp
MDE without `offhand`). Whole-clip `"not holding the X"` had 0 held-out clips,
so negation pivoted to the well-powered 2-object contrast (`295` held-out
clips with held `X` + visible unheld `Y`).

All training speech was rebuilt in `nursery/unified/u2_data.py` using short,
natural CHILDES-style caregiver frames (mean 5.16 tokens, 29.54 utt/clip,
`sigma = 1.0 s` jitter) preserving Step 9's exact alignment ratios
(16.3 / 19.1 / 64.5% over all utterances; 33.1 / 38.0 / 28.8% within named
utterances), with **strictly disjoint train vs. test carrier templates** and
frozen SHA-256 provenance in `outputs/runs/unified_u2_data/frozen/`:

- **Panel A · Action + Object (`206` clips: 92 test, 114 s8; 4-choice, 25% chance):**
  Minimal-pair clip retrieval over the 5 Step 9 actions (`bell`–`drink` excluded),
  with hard foils swapping only action, only object, or both.
- **Panel B · Property + Object (`203` clips: 93 test, 110 s8; 129 heavy / 74 light; 4-choice, 25% chance):**
  Minimal-pair clip retrieval for `heavy`/`light`, plus a separate 24-clip
  straddling-noun subset (4 nouns attested at both masses).
- **Panel C · 2-Object Negation (`295` clips: 133 test, 162 s8; 2-choice, 50% chance):**
  `"holding the {X}, not the {Y}"` vs. `"holding the {Y}, not the {X}"` on the
  same clip—identical bag of words so `bow_only` is mathematically forced to
  50.0% chance (`1e-6` exact-tie tolerance).
- **Embodied Mass VP-Swap (unconditional `sensory=None`):**
  12 always-heavy + 12 always-light train nouns split into **seen** (8 H + 8 L,
  64 quads) and **held-out verbal pairs** (4 H + 4 L, 16 quads; zero `(noun,
  heavy/light)` co-occurrences in training speech), trained with `p_uncond = 0.25`
  sensory prefix dropout and scored via EgoBabyVLM quads (`logP(S1) > logP(S1')`
  and `logP(S2) > logP(S2')`, 25% chance) and bias-free swap margin (50% chance).

## 2. Locked 18-Run 3-Seed Results (630 Clips, Seeds 0–2, 40 Epochs)

Run directory: `outputs/runs/unified_u2_v1/` (code commits `376685a2` / `b69091fb`).

| Metric | V | V+G | **V+G+B** | Shuffled B | Bag-of-Words | Text-Only LM |
|---|---|---|---|---|---|---|
| U1 Nouns | 59.0 ± 3.1 | 71.9 ± 2.4 | **73.9 ± 0.5** | 72.1 ± 0.5 | 75.5 ± 1.0 | — |
| U1 Actions | 65.2 ± 1.4 | 70.0 ± 1.6 | **73.1 ± 1.4** | 65.4 ± 4.3 | 73.4 ± 2.5 | — |
| Nouns, extras removed | — | — | 65.3 ± 1.4 | 64.5 ± 2.0 | 65.5 ± 3.1 | — |
| Actions, extras removed | — | — | 66.3 ± 4.6 | 68.4 ± 2.3 | 67.5 ± 2.0 | — |
| **A · Action + Object (206)** | 61.0 ± 4.4 | 68.3 ± 4.9 | **73.6 ± 2.7** | 70.4 ± 1.3 | 70.1 ± 2.0 | — |
| **B · Property + Object (203)** | 58.1 ± 6.4 | 66.0 ± 1.8 | **85.7 ± 1.3** | 67.7 ± 0.8 | 83.4 ± 2.8 | — |
| **B · same-noun subset (24)** | 50.0 ± 4.2 | 55.6 ± 6.4 | **75.0 ± 4.2** | 58.3 ± 4.2 | 80.6 ± 4.8 | — |
| **C · Negation (295)** | 66.1 ± 5.0 | 69.4 ± 3.7 | **82.1 ± 1.1** | 74.4 ± 2.3 | **50.0 ± 0.0** | — |
| VP-Swap quad, seen (25%) | 0.0 ± 0.0 | 13.0 ± 22.6 | 30.2 ± 18.3 | 10.4 ± 9.5 | 30.7 ± 17.3 | 29.2 ± 26.0 |
| VP-Swap quad, held-out (25%) | 0.0 ± 0.0 | 8.3 ± 14.4 | 29.2 ± 40.2 | 8.3 ± 14.4 | 16.7 ± 28.9 | 8.3 ± 14.4 |
| VP-Swap margin, seen (50%) | 49.5 ± 20.8 | 59.4 ± 13.9 | 63.0 ± 29.5 | 57.3 ± 3.9 | 85.4 ± 7.0 | 93.8 ± 4.1 |
| VP-Swap margin, held-out (50%) | 52.1 ± 28.2 | 60.4 ± 14.4 | 81.2 ± 18.8 | 47.9 ± 29.5 | 41.7 ± 37.7 | 60.4 ± 3.6 |

**Constituent word routing inside test sentences (% vis / temp / body, 3-seed mean):**
- `light`: **14 / 10 / 76** (`V+G+B`) vs. `74 / 20 / 7` (Shuffled B)
- `heavy`: **21 / 21 / 58** vs. `50 / 37 / 14`
- `lift`: **12 / 46 / 42** vs. `12 / 80 / 8`
- `pass`: **15 / 39 / 46** vs. `28 / 56 / 16`
- `drink`: **56 / 33 / 12** vs. `57 / 40 / 4`
- `not`: **27 / 42 / 31** vs. `24 / 53 / 23`
- `holding`: **23 / 45 / 33** vs. `25 / 40 / 35`

**Go-Gate Verdict: PASS (Locked).**
- **Real body drives compositional grounding:** `V+G+B` beats shuffled body by
  **+18.0 pp** on Property+Object (+16.7 pp on the same-noun subset), **+7.7 pp**
  on Negation, **+7.7 pp** on U1 Actions, and **+3.2 pp** on Action+Object,
  while improving U1 Nouns (`73.9 ± 0.5%` vs. U1's `70.4 ± 1.7%`).
- **Word order is strictly necessary for negation:** Parameter-matched `bow_only`
  collapses from **82.1 ± 1.1%** to **50.0 ± 0.0%** exact chance (−32.1 pp).
- **VP-Swap caveat (exploratory only):** With 16 held-out quads across 3 seeds,
  variance is too high (`29.2 ± 40.2%` quad; `81.2 ± 18.8%` held-out margin vs.
  `60.4 ± 3.6%` text-only LM and `47.9 ± 29.5%` shuffled body) to make a strong
  claim; report candidly as an underpowered exploratory signal.

## Expected Figures

1. **3-Rung Dissociation Chart:** Gaze gain (`V+G − V`) and Body gain
   (`V+G+B − Shuffled B`) across **Nouns**, **Actions**, **Action+Object**,
   **Property+Object**, and **Negation**.
2. **Updated 3D Sensory Routing Simplex:** Per-word `(p_vis, p_temp, p_body)`
   for nouns, manipulation verbs, mass adjectives, and negation in `V+G+B` vs.
   `Shuffled B` (`outputs/runs/unified_u2_v1/routing_simplex_3d.html`).
