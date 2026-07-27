# Nursery / BabyWorld

This branch is the clean scientific starting point for the Embodied Simulation
research program.

The program asks whether embodied side modalities available only during
training—action state, proprioception, contact/touch, IMU-like signals, and
object state—improve transferable lexical and language grounding when
naturalistic vision and speech are weakly aligned.

No simulator platform or implementation is currently selected. The next
authorized engineering study should compare viable rich-scene substrates using
public or synthetic inputs and attempt one deterministic 15–20 second episode:

`look → reach → grasp → inspect → manipulate → release`

The camera must arise from the embodied agent. Physics and contact must remain
authoritative, and the episode must provide synchronized RGB, depth,
segmentation, camera, action, proprioception, contact/touch, IMU-like, and
object-state streams.

See [Scientific baseline](docs/embodied_simulation_baseline.md) before defining
or implementing a protocol. The full pre-reset repository remains recoverable
from Git commit `7571c0e`.

## Current status

- Active implementation: none
- Selected simulator: none
- Authorized experiment: none
- Empirical child-data source: ChildLens only
- Restricted-media execution: local only
- Viewer–view calibration: stopped at the local CUDA/privacy gate

Generated datasets, runs, media, caches, checkpoints, and reports do not belong
in Git. Only compact frozen configurations, aggregate results, provenance
manifests, and concise decision records may be committed.
