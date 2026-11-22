# Integrated Unity–MuJoCo embodied episode gate

Decision: **NO-GO at ordered Stage C**. Stages A and B pass. The bounded tactile
repair did not produce a stable unassisted grasp, and the head-view camera did
not contain the physical event. Stages D and E were therefore not accepted.

## Ordered evidence

### A — static digital-twin registration: PASS

The exact weighted MPFB avatar and `BakeMesh(useScale=true)` path from visual
audition commit `537d30c` were reused. Unity exported the retained rest hierarchy
once; MuJoCo body positions and fixed transforms were derived from it, including
collapsed twist/deformation offsets. Visible bones are driven only by
`restLocalRotation * mapped axis-angle`.

- MuJoCo→Unity rest landmark round trip: mean `0.000000181 m`, maximum
  `0.000000294 m` (frozen maximum `0.010 m`).
- Unity registration: mean `0.000729 m`, maximum `0.003746 m`.
- 52 isolated proximal/middle/distal/wrist sweep frames: maximum segment-length
  change `1.58e-7 m`; maximum mesh-bounds ratio `1.0572`; no forced positions or
  double scale.
- Only the actual weighted skin contributes final avatar pixels. Green capsules
  are QA-only and excluded from head RGB.

### B — dynamic no-object registration: PASS

An immutable 240 Hz MuJoCo trace drove 660 Unity frames at exact 8:1. The
22-second head and external-overlay videos show articulated torso/head/arm,
wrist, and all three joints of all five digits without duplicate skin, proxy
pixels, detachment, or drift. The same trace state drives skin and overlay.

### C — physical touch/grasp/lift/place/release: NO-GO

The 55 mm, 55 g free `red_toy_001` remained dynamics-controlled. There were no
attachments, equality constraints, external forces, direct object pose writes,
or Unity dynamics.

Pre-freeze iterations first established that stronger open-loop closure was not
sufficient: it reached all five named digits plus palm and up to four contacts
simultaneously, but squeezed the cube laterally and lifted only `0.01499 m`.
That result was not accepted.

The decisive bounded repair used deterministic tactile impedance:

- thumb+index+middle had to persist together for `0.20 s` before lift;
- each digit stopped at contact and regulated a `1.0 N` target with correction
  limited to `4 deg/s`;
- contact-free closure was `24 deg/s`;
- signed contact-force imbalance could recenter the shoulder actuator by at
  most `3 deg` at gain `0.002 deg/(N·step)`;
- failure to qualify by 11 s triggered a visible actuator-only reopen at
  `36 deg/s`; lift was never commanded without qualification.

The feedback run contacted thumb and index but never middle simultaneously,
never satisfied the dwell, entered `retry_open` at `11.0042 s`, and lifted only
`0.01107 m`. The cube escaped laterally. This is a controller/physical NO-GO,
not a registration or renderer incompatibility.

The fixed camera mount replay is numerically rigid (sampled translation error
`<=1.41e-7 m`, rotation error `0 deg`), but the neutral face-forward basis and
available head motion look away from the interaction envelope. The target is
absent from the head view at touch. Thus visual credibility also fails even
independently of the grasp.

## Downstream gates

Stage D was not accepted: failed Stage C traces were rendered only as diagnostic
evidence, not promoted to a shared-capture PASS. Stage E was not run because
robustness cells cannot repair a failed nominal controller and would misstate
the ordered protocol. Event speech was not generated.

## Ignored artifacts and reproducibility

All complete runs remain ignored under
`runs/embodied_simulation/integrated_unity_mujoco/`. Key artifacts are:

- `stage_a_derived/registration_overlay.png`, `registration_qa.json`, and
  `local_rotation_sweeps.json`;
- `stage_b/head_view.mp4`, `external_qa.mp4`, and dense timelines;
- `stage_c_feedback/authoritative_trace.npz`, `model.xml`,
  `render_trace.json`, and `manifest_registration_qa.json`;
- `stage_c_feedback_render/head_view.mp4`, `external_qa.mp4`, RGB/depth/ID
  frames, timelines, and `unity_render_qa.json`.

Runtime: Unity `6000.0.80f1` ARM64/Metal; MuJoCo `3.3.7`; Python `3.12`;
physics `240 Hz`; render `30 Hz`; exactly 5280 steps and 660 frames. Public/CC0
or synthetic assets only. No restricted ChildLens media was accessed, and no
claim of infant training, age matching, ChildLens calibration, or human
validation is made.

The only simulation entry point is manifest-derived and requires an explicit
mode:

```text
.venv-integrated/bin/python -m babyworld_lite.childlens_engine_bakeoff.unity_mujoco_gate \
  --output <ignored-output> --rest-manifest <mpfb_rest_manifest.json> --registration
.venv-integrated/bin/python -m babyworld_lite.childlens_engine_bakeoff.unity_mujoco_gate \
  --output <ignored-output> --rest-manifest <mpfb_rest_manifest.json> --manipulation
```

There is no default or hard-coded alternate embodiment protocol.
