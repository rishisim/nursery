# ChildLens media-grounding environment terminal report

**Decision: `NO_GO_MEDIA_GROUNDING_ENV`**

The environment is not ready for a ChildLens-calibrated cue-lift experiment.
The frozen rendering/physics gate failed under the predetermined two-backend
policy. The final multimodal cue-lift experiment was not launched.

TDW v1.13.0 was the preregistered primary engine. Its isolated controller
installed only after the one allowed compatibility repair, then failed before
launch because its controller imported the removed `pkg_resources` interface.
The repair allowance was exhausted, so no dependency shopping followed.

The predetermined BlenderProc 2.8.0 / Blender 4.2.1 arm64 fallback did render a
24-frame Apple-Metal scene with RGB, depth, instance/class segmentation,
timestamps, rigid-body motion, and pose state. It did not pass the complete
acceptance gate: the prototype's contact field was a post-hoc object-height
threshold, not an engine collision/contact record, and the scene emitted no
joint state. Those deficiencies directly violate the frozen rule that contact,
proprioception, and IMU must originate from one auditable physical state and
clock. Treating the threshold as physical touch would recreate the invalid
metadata-as-calibration error this task was intended to eliminate.

As an independent check of the starting scaffold, 180 central episodes were
measured from their actual 1-fps RGB pixels with the V5 grayscale estimator.
Mean motion was 0.0073664, versus the ChildLens development mean 0.1185 and
90% interval [0.1095, 0.1273]. Scene-change rate was zero in every episode,
versus 0.131 and [0.1069, 0.1558]. Declared target-valued motion,
persistence, and scene-change metadata were not used.

The selected German route remains fixed local macOS Anna (`de_DE`) speech with
no voice cloning. The intended learner remains the official egobabyvlm checkout
at commit `224621caf0628270b6115845ac75a65b984234a3`; its Apple-local
environment was installed, but learner and oracle controls were not promoted
to passed after the earlier fatal engine gate. ChildLens remained the sole
empirical source, the 22 locked participants were not accessed, and no AEA or
BabyView material was used.

Reproduction:

```text
python3 scripts/doctor_childlens_media_grounding.py --json
pytest -q tests/test_childlens_media_grounding_environment.py
```

The next decision is whether to authorize one protocol amendment for the
existing Blender fallback: require collision geometry/manifold-derived contact
and an articulated body whose joint state is read directly from Blender's
physics state. That would be a new bounded implementation attempt, not a
reinterpretation of this failed gate.
