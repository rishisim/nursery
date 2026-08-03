# Unity visual-shell audition

Decision: **NO-GO**.

The corrected visual-feasibility question was whether one actual Unity-rendered,
furnished scene with a skinned child-proportioned body, articulated five-finger
hand, and strictly body-derived frozen head-camera mount could produce a
7-second look/reorient → reach → touch → withdraw clip plainly stronger than
the accepted MIMo footage. It did not.

Unity 6000.0.80f1 (`2dfd32957da2`) ran natively on arm64 through Metal with an
assigned Unity Personal entitlement and the Built-in Render Pipeline. The
avatar was the retained MPFB/MakeHuman age-minimum rigged output (CC0 core
graphical assets); the room used Kenney Furniture Kit objects (CC0). Exact
pins, hashes, camera mount, and comparison hashes are frozen in
`configs/embodied_simulation_unity_visual_audition.json`.

The Unity clip is actual 960×540, 30 fps, 210-frame H.264 output. Its camera is
parented to the imported head beneath the avatar root/spine/neck chain and uses
one frozen local mount only. It has no target lock or independent animation.
The optical origin is 0.3165 m beyond the frozen 0.16 m radial head envelope.
Nevertheless, imported self-geometry and furniture scale dominate the frame:
the room does not read coherently, the anatomically articulated hand is not
readable, touch cannot be seen, and prolonged self-occlusion makes the motion
visually negligible. The comparable MIMo 6–13 s window plainly retains room,
target, embodiment, and reach readability. Unity therefore fails the combined
visual comparison rather than merely lacking texture or lighting polish.

All Unity motion, CCD IK, finger closure, and touch are kinematic and
nonphysical; the target is static. The clip is not simulator truth, a physical
episode, a training example, a side-stream example, calibration, human
validation, or causal evidence. No contact, proprioception, IMU, object-truth,
or ChildLens stream was generated. Restricted ChildLens media was not accessed.

Reproduction used the official Hub CLI to install the native Editor, Blender
only to export the existing CC0 rig to FBX, and Unity Editor batch mode as the
sole renderer:

```text
Unity Hub -- --headless install --version 6000.0.80f1 --architecture arm64
Unity -batchmode -force-metal -projectPath <ignored-project> \
  -executeMethod AuditionBuilder.Render -logFile <ignored-run>/unity_render.log
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p <clip>.mp4
```

The generated project, caches, imported assets, PNG frames, videos, timelines,
and logs remain in ignored `runs/embodied_simulation/unity_visual_audition/`.
The smallest evidence-backed alternative is to stop this Unity scaffold. A
future visual direction would need a preassembled first-person child rig and
room authored at one common scale with a verified face-forward camera socket;
that is a materially different visible-shell architecture and was not launched.
