# Blinded AEA visible-action annotation codebook

Protocol: `aea-visible-action-v2`

This packet concerns adult, partly scripted AEA recordings. It is a sensor-format analogue, not developmental evidence and not BabyView-like. Your labels are model-assisted development judgments, not human annotations or human reliability evidence.

## What to inspect

For every opaque item, inspect all 31 frames in temporal order. The ASR anchor is centered between frames 15 and 16. Use the transcript only to answer the ASR-referent, agency, and alignment fields. Choose the observable wearer action from pixels, not from the transcript. Do not inspect repository manifests, v1 labels, another pass, group/location metadata, results, or model outputs.

## Primary observable wearer action

Choose exactly one:

- `locomotion_posture`: walking/translation or a clear sit, stand, or posture transition.
- `reach_grasp`: reaching to and acquiring or deliberately taking hold of an object.
- `transport_place`: carrying, bringing, repositioning, putting, setting, or releasing an object.
- `state_change_operate`: opening, closing, turning, pressing, switching, or operating a container/device.
- `food_material_handling`: pouring, stirring, cutting, preparing, eating, drinking, or similar material transformation.
- `clean_groom`: wiping, washing, cleaning, brushing, vacuuming, or grooming.
- `other_observable`: a clear goal-directed wearer action outside the six categories.
- `none_visible`: no goal-directed wearer action is visibly underway or completed; passive looking/listening/talking or static holding alone belongs here.
- `uncertain`: occlusion, image quality, timing, or competing actions prevents a defensible choice.

Prefer an action spanning the centered anchor. If none spans it, choose the nearest action interval; then the longer interval; then the first applicable ontology entry above. Label the camera wearer's action even when somebody else or a phone is the source of speech.

## ASR relationship

`asr_refers_to_visible_wearer_action`:

- `yes`: the anchored verb describes the wearer's visible action in this window.
- `no`: it clearly refers elsewhere or no matching wearer action occurs.
- `unclear`: referent or visible match cannot be determined.

`agency` identifies the anchored verb's source/referent:

- `wearer`
- `other_person`
- `narrated_media_or_phone`
- `figurative_or_nonaction`
- `unclear`

`temporal_alignment`:

- `aligned_within_window`: corresponding wearer action overlaps/crosses the centered anchor.
- `action_precedes_anchor`: nearest corresponding wearer action is before the anchor.
- `action_follows_anchor`: nearest corresponding wearer action is after the anchor.
- `no_corresponding_action`: no corresponding wearer action occurs in the six seconds.
- `unclear`

## Evidence, confidence, rationale

Use `high`, `medium`, or `low` confidence. For observable actions other than `none_visible` or `uncertain`, provide inclusive `evidence_frame_start` and `evidence_frame_end` between 0 and 30. For `none_visible` or `uncertain`, both may be null. Give a concrete rationale of at most 25 words, naming visible changes and any transcript-source evidence without guessing intent.

Return one JSON object per item with exactly these fields:

```json
{
  "blind_id": "VA2-...",
  "asr_refers_to_visible_wearer_action": "yes|no|unclear",
  "agency": "wearer|other_person|narrated_media_or_phone|figurative_or_nonaction|unclear",
  "temporal_alignment": "aligned_within_window|action_precedes_anchor|action_follows_anchor|no_corresponding_action|unclear",
  "observable_action": "one frozen ontology label",
  "confidence": "high|medium|low",
  "evidence_frame_start": 0,
  "evidence_frame_end": 30,
  "rationale": "25 words maximum"
}
```
