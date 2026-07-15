# AEA coarse-action v3 model-assisted codebook

These are **model-assisted development labels**, not human annotations or human inter-rater reliability. AEA is an adult, partly scripted sensor-format analogue, not developmental evidence and not BabyView-like.

## Mandatory two-stage order

Complete and seal every stage-1 row before opening the stage-2 packet. Stage 1 contains images and timing only; it deliberately omits transcript and anchored verb. Never use transcript semantics, an expected script, or a guessed ASR action to infer visible action.

Do not read v1/v2 completed labels or rationales, the other v3 pass, example IDs, groups, locations, agreement, support, capacity, or IMU outcomes. Do not adjudicate or revise labels after comparison.

## Stage 1: visible action from video only

Choose exactly one primary action of the camera wearer in the six-second, 31-frame sequence:

- `gross_body_motion`: wearer locomotion, a posture transition, or deliberate body/head orientation without direct object manipulation.
- `object_or_body_interaction`: reaching/grasping, carrying/placing, operating/state change, food/material handling, cleaning/grooming, or another goal-directed manipulation of an object or body surface.
- `no_goal_directed_visible_action`: no defensible goal-directed wearer action is visible.
- `uncertain`: occlusion, image quality, timing ambiguity, or competing evidence prevents a defensible label.

Direct object/body manipulation takes precedence when it overlaps gross movement. Otherwise select the action spanning the temporal midpoint between frames 15 and 16; if none spans it, use the nearest evidence interval, then the longer interval, then the label order above.

Deliberate looking/head orientation can be gross motion; incidental camera jitter cannot. Static holding, talking, listening, and another person's action are not wearer actions. If only another person acts, usually choose `no_goal_directed_visible_action`.

Record `visible_confidence` (`high`, `medium`, or `low`), inclusive evidence frames 0–30, and a `visible_rationale` of 1–25 words. Evidence frames may both be null only for `no_goal_directed_visible_action` or `uncertain`.

## Stage 2: ASR referent and timing

After stage 1 is sealed, use the revealed anchored verb/transcript and the already reviewed visual evidence to label:

- `wearer_action`: the anchored expression refers to a concrete action by the wearer.
- `nonwearer_or_nonliteral`: it refers to another person, media/phone narration, or a figurative/non-action usage.
- `unclear`: the referent cannot be resolved defensibly.

Then label `temporal_relation`:

- `aligned`: a corresponding visible wearer action overlaps the centered anchor between frames 15 and 16;
- `before`: the corresponding action is visible within the window but confined before the anchor;
- `after`: the corresponding action is visible within the window but confined after the anchor;
- `none`: no corresponding visible wearer action occurs in the six-second window;
- `unclear`: the relation cannot be resolved.

Referent and timing do not alter the sealed visible label. Record `language_confidence` (`high`, `medium`, or `low`) and a `language_rationale` of 1–25 words. Be literal and conservative: topical similarity is not action reference, and a verb in speech is not evidence that the wearer performed it.
