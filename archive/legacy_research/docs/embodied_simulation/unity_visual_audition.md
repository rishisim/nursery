# Unity visual-shell audition

Decision: **PASS** after one bounded repair of the same canonical audition.

The visual-feasibility question was whether one Unity-rendered furnished scene
with a skinned child-proportioned body, articulated five-finger hand, and
strictly body-derived frozen head-camera mount could produce a 7-second
look/reorient → reach → touch → withdraw clip plainly stronger than the accepted
MIMo footage. The repaired clip does. This is a visual-shell result only.

Unity 6000.0.80f1 (`2dfd32957da2`) ran natively on arm64/Metal with the Built-in
Render Pipeline. The avatar is the retained MPFB/MakeHuman age-minimum rigged
output (CC0); the room uses Kenney Furniture Kit objects (CC0). One Unity unit
is one meter. Each Kenney OBJ is normalized from measured raw
`Renderer.bounds` to an explicit metric dimension; the original many-meters-
tall furniture defect is gone.

The 1.133 m eye camera is constructed once from the rest head-weighted mesh and
fixed room/root basis, 0.032 m beyond visible head geometry, then frozen under
the head. It uses a 62° vertical FOV and 0.03 m near plane. There is no target
lock or independent camera animation. The toy is visible at every sampled phase
and is exactly 85% of the measured 0.495 m articulated reach.

The FBX audit proves the driven upper-arm, lower-arm, wrist, and finger objects
are exact transforms in `SkinnedMeshRenderer.bones`, with nonzero weights.
Unity's manual `Camera.Render` path did not display scripted skinning directly,
so the final camera renders the same full weighted child mesh baked per frame
by `BakeMesh(useScale=true)`, with the importer transform applied exactly once.
Only the duplicate source draw is excluded. The capsule diagnostic is disabled
and contributes no final pixels. At contact, 1,883 right-arm-weighted vertices
move by 0.413 m on average, the baked arm bound comes within 2.1 mm of the
target, and the fingertip is 2.1 cm from the toy center (inside its radius).

Against the same actual MIMo 6–13 s window, Unity is sparser but materially
stronger on the combined embodied-view evidence: stable head view, clean
horizon/roll, smooth attached forearm, readable five-finger anatomy, visible
approach/contact/withdraw, and no torso takeover or clipping. MIMo retains
richer room detail, but its body occlusion and coarse hand are visibly weaker.

All Unity motion, CCD IK, finger closure, and touch are kinematic and
nonphysical; the target is static. The clip is not simulator truth, a physical
episode, a training example, calibration, human validation, or causal evidence.
No contact, proprioception, IMU, object-truth, or ChildLens stream was generated.
Restricted ChildLens media was not accessed.

Official Unity 6 API rationale: model axis conversion belongs in
`ModelImporter`; camera clipping and viewport depth are world units;
`WorldToViewportPoint` supplies visibility checks; deformed bounds come from
`SkinnedMeshRenderer.BakeMesh`; and the culling mask excludes only the duplicate
source draw. Reproduction remains batch-driven:

```text
Unity Hub -- --headless install --version 6000.0.80f1 --architecture arm64
Unity -batchmode -force-metal -projectPath <ignored-project> \
  -executeMethod AuditionBuilder.Render -logFile <ignored-run>/unity_render.log
ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p <clip>.mp4
```

Generated assets and evidence remain under ignored
`runs/embodied_simulation/unity_visual_audition/`. The only recommended next
step is a separately authorized gate registering this exact visible rig and
camera to the existing MuJoCo trace/physics contract. This PASS does not
authorize or imply that integration.
