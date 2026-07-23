# Child-only episode bundle schema v1

Schema version: `child-only-episode-v1.0.0`  
Audio/TTS: deferred  
Construction fixture label: `NONSCIENTIFIC_CONSTRUCTION_FIXTURE`

The executable closed-schema validator is [schema.py](../../babyworld_lite/child_only_v1/schema.py). Unknown keys, undeclared units/coordinate frames, unsafe paths, non-finite values, malformed dimensions, nonmonotonic clocks, missing digests, symlinks, unexpected files, and mismatched episode inventories fail closed.

## Physical bundle layout

```text
bundle/
  bundle_manifest.json
  fixture_settings.json
  model_visible/
    episodes.jsonl
    frames/<opaque-episode-id>/<frame-index>.png
  instrumentation/
    object_states.jsonl
  hidden_oracle/
    events.jsonl
```

The three roots are distinct directories and their JSONL records are distinct files. `model_visible/episodes.jsonl` has no pointer to either hidden root. The bundle manifest contains content hashes and root roles for offline integrity validation, but it is never a learner input. Model-visible and hidden inventories must contain exactly the same opaque episode IDs.

## Model-visible episode

Each episode declares the schema/contract versions, construction profile, opaque episode/study IDs, split, monotonic-nanosecond timebase, and duration.

- `frame_stream`: actual RGB image paths, timestamps, SHA-256, pixel width/height, three-channel sRGB, `uint8`, and PNG/JPEG encoding. Paths are relative and confined to `model_visible/`.
- `utterances`: opaque utterance ID, onset/offset nanoseconds, text, speaker role, language tag, and validity. No audio/TTS field exists.
- `head_imu`: declared right-handed head coordinate frame; timestamped three-axis acceleration in m/s², three-axis angular velocity in rad/s, and validity.
- `proprioception`: joint names, coordinate frame, timestamped joint position in rad, joint velocity in rad/s, end-effector position in m, normalized quaternion, and validity.
- `contact_touch`: sensor names, coordinate frame, timestamped contact binary, normal force in N, pressure in Pa, slip velocity in m/s, vibration in m/s², and validity.
- `continuous_motor`: noncategorical channel names with one unit per channel and timestamped continuous values plus validity. Action/event/label-bearing channel names are rejected.

Irregular sampling is allowed; timestamps and validity flags are authoritative. No loader may infer synchrony from row number alone.

## Separate instrumentation

`instrumentation/object_states.jsonl` contains the requested complete object-state trajectories: opaque object IDs with timestamped position in m, normalized quaternion, linear velocity in m/s, angular velocity in rad/s, and visible fraction. It is generation instrumentation, not a primary-evaluation input. It remains physically separate so it cannot enter a permissive batch by accident.

## Hidden oracle/event truth

`hidden_oracle/events.jsonl` contains event intervals/types, participant object IDs, causal parents, nullable utterance-to-event truth, candidate-event sets, null reasons, and counterfactual truth. Later detector-derived annotations and scoring targets belong here (or in another hidden-role root), never in model-visible data.

The learner loader validates only the model-visible schema. Oracle validation, trial construction, and scoring use separately typed offline paths. Primary evaluation receives only already-built vision and text tensors; even labels remain outside the model call.
