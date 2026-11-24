# Step U2: Compositional Sentences & Embodied VP-Swap

Status: **ready to build (Step U1 locked at `79c77f05`).**
Context: [CONTEXT.md](CONTEXT.md).

## Objective

Move Nursery beyond isolated word grounding to genuine **compositional sentence
comprehension** and **text-only physical semantics**, using the exact same 630
GRAB + EmbodiedGen clips so curves remain directly comparable with Nouns and
Actions.

## 1. Tier A Compositional Sentences (Zero New Rendering)

Generate weakly aligned caregiver sentences directly from GRAB event metadata
(preserving natural timing jitter, nonheld chatter, and absent referents):

- **Action + Object binding:** e.g., *"lift the cup"* vs. foils that swap only
  the action (*"pass the cup"*) or only the object (*"lift the bowl"*).
- **Property + Object binding:** e.g., *"the heavy cup"* vs. *"the light cup"*
  (using persistent instance masses from Step 7).
- **Action + Hand / Role binding:** e.g., *"hold the mug with the left hand"*
  vs. *"right hand"* (directly testing proprioceptive laterality + syntax).
- **Negation:** e.g., *"not holding the cup"* vs. *"holding the cup"*.

**Evaluation format:** Same four-choice forced-choice protocol (chance 25%) on
held-out source clips and held-out subject `s8`, where **hard minimal-pair
foils** match either the noun or the verb so an orderless bag-of-words model
cannot win by spotting a single keyword.

*(Optional Tier B, deferred unless Tier A completes ahead of schedule: scripted
kinematic 2-object Blender renders inside existing EmbodiedGen rooms for
spatial preposition swaps such as `"cup in bowl"` vs. `"bowl in cup"`, without
reopening InterMimic/HOIDiNi.)*

## 2. Embodied VP-Swap (Text-Only Physical Semantics)

Adapted from [EgoBabyVLM](https://arxiv.org/html/2605.19130v1)'s Visual
Property Swap (Appendix C.4.2) to test whether **embodied multimodal training
improves the text encoder itself even when vision and body are absent at test
time**:

- Construct paired true/swapped sentences over Nursery's training vocabulary
  covering **mass** (*"the hammer is heavy"* / *"the cup is light"* swapped to
  *"the cup is heavy"* / *"the hammer is light"*) and **action affordance**
  (*"drink from the mug"* / *"pass the..."*).
- Score zero-shot using the Causal Text Transformer's unconditional sentence
  log-probability: credit only when both true sentences outscore both swapped
  counterparts (`P(S1) > P(S1')` and `P(S2) > P(S2')`, chance 25% per quad /
  50% per pair).

## Metrics & Stop / Go Gate

- **Primary metrics:** Held-out 4-choice accuracy on **Compositional Sentences**
  (overall + per contrast type: action-swap, object-swap, property-swap,
  hand-swap, negation) across `V`, `V+G`, `V+G+B`, and shuffled `B`; plus
  **Embodied VP-Swap** accuracy on text-only `V` vs. `V+G+B` vs. unimodal
  text-only LM baseline.
- **Bag-of-words sanity check:** Confirm an orderless bag-of-words ablation
  drops sharply on minimal-pair role/negation contrasts while the 2-layer
  Causal Transformer succeeds.
- **Go** if panels pass power sizing (~3 pp MDE across held-out clips) and
  `V+G+B` beats `V` and shuffled `B` on compositional sentences without
  regressing Nouns or Actions.

## Expected Figures

1. **3-Rung Dissociation Chart:** Gaze gain (`V+G − V`) and Body gain
   (`V+G+B − V+G`) side by side across **Nouns**, **Actions**, and
   **Compositional Sentences** (with 95% joint seed + clip CIs).
2. **Embodied VP-Swap Bar Chart:** Text-only physical semantics accuracy for
   unimodal LM vs. `V` vs. `V+G` vs. `V+G+B` (testing whether embodied training
   breaks EgoBabyVLM's zero-unimodal-transfer finding).
3. **Updated 3D Sensory Routing Simplex:** Showing where multi-word sentences
   (and their constituent nouns, verbs, and properties) sit in posterior
   Soft-MIL routing space `(p_vis, p_temp, p_body)`.
