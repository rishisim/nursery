# Integrated Dexterous Embodiment Gate Decision

## Decision

**NO-GO at ordered Gate A (2026-08-05).** Two sealed, eligible Unity-native
microcell attempts completed the frozen 24 s / 5,760-step / 720-frame clock.
The first failed physical capture; the only authorized controller/contact repair
was then frozen and the second also failed. Attempt 2 produced no simultaneous
right-thumb plus two-non-thumb force-bearing row, no qualified dwell, no lift,
and no release at the required destination. The target was pushed from the
table and first registered floor support at 8.6375 s. A third controller/contact
repair is outside the bounded protocol, so Gates B--F stop unexecuted.

The decision is limited to public/synthetic engineering qualification. It makes no infant-trained, age-matched, ChildLens-calibrated, human-validated, biological-torque, or real-child claim. No restricted ChildLens media or restricted external drive was accessed.

## Current Gate-A freeze and retained attempts

- Engine/authority: Unity 6000.0.80f1 ARM64/Metal, Unity/PhysX sole physics
  authority, Unity sole renderer, 240 Hz manual physics and 30 Hz render with
  exact 8:1 mapping.
- Canonical entry point:
  `python -m babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate`.
- Executed config SHA-256:
  `3001b37bcaae9f9616f719431e8f56d96886adbb283a3efe9d9e1f83886c83d6`.
- Episode A contract SHA-256:
  `eb8fc293718bba694952b24a901835ca42f98eb5a1b9976343b43151374de66c`.
- Attempt-2 source-audit receipt SHA-256:
  `281670aaf1e245ad5b1ec02d8e77e5efba60681477a3cd6fe55ecf0146b091a0`;
  source-set SHA-256:
  `2baec75c7b7b75169e329a1700f90aab0ee6ef53931d19291e42fe9d4472e5c9`.
- Retained ignored evidence roots:
  `runs/embodied_simulation/procedural_scene_gate/episodes/A_playroom_red_toy/physical_attempt_01_failure`
  and
  `runs/embodied_simulation/procedural_scene_gate/episodes/A_playroom_red_toy/physical_attempt_02_failure`.

Both traces contain 5,760 post-`Physics.Simulate` rows over 24 s and map to 720
render frames. All six head/external/overlay videos are H.264, 1920x1080, 30
fps, 720 frames, 24 s, pass full ffmpeg decode, and were inspected in complete
0.5 s and 1 s contact sheets. Attempt-1 and attempt-2 trace SHA-256 values are
`2ddcbdbd062b0b1bdff9397c8fd839480b20b8e5a14311ea683f0dad13c54fc1`
and `7c0b6f57f53e21c40006b82877731c476d170ca9d09cf32d31670f5352d5a002`.

## Force-bearing result

| Measure | Attempt 1 | Attempt 2 | Frozen requirement |
|---|---:|---:|---:|
| Simultaneous right thumb + at least two non-thumb eligible nonzero-impulse rows | 0 | 0 | >0.30 s continuously |
| Right force-bearing qualification | false | false | true |
| Left non-little force-bearing support qualification | false | false | >0.25 s |
| Raw lift above initial center | 0 m | 0 m | >0.10 m after qualification |
| Commanded opening | 19.0042 s | 19.0042 s | required after supported manipulation |
| Free release at `shallow_bin` | false | false | true |
| Empty assistance ledger | 0 entries | 0 entries | 0 entries |

Attempt 1 developed only index force evidence: its index-only eligible impulse
interval lasted 526 steps / 2.1917 s (steps 3064--3589), with maximum impulse
`4.2621e-5 N s`. The thumb's closest separation was 5.8949 mm; middle and ring
never contacted. The object never lifted, and a late left-hand impact swept it
off the table. The single permitted repair removed the premature index-only
palm latch and required complete thumb-plus-two-digit geometry before latch,
with bounded radial thumb/middle preshape changes.

Attempt 2 still had only index force evidence. Across the full trace the thumb
closest separation was 1.6033 mm with zero eligible impulse, the middle closest
separation was 8.3450 mm with zero impulse, and no row contained thumb plus two
non-thumb eligible impulses. The index reached `0.0162103 N s`, but index-only
contact cannot qualify. Maximum finger/object penetration was 5.9321 mm, above
the frozen 2 mm limit. The object center never rose above its initial
0.6483297 m, first contacted the floor at step 2072 / 8.6375 s, and ended away
from the required shallow bin. Opening therefore was not a release from capture.

## Authority and truth limits

Both runtime receipts report zero post-initialization target pose writes,
velocity writes, external force/torque calls, attachment/joint changes,
parenting changes, and kinematic changes. Runtime accounting passed, the
assistance ledgers are empty, and the mandatory source audits passed. These
receipts support a clean no-assistance result, but their zero counters are not
claimed as universal independent proof beyond their declared coverage.

The compliant-drive telemetry channel is **invalid and excluded from
acceptance**. All 172,800 attempt-2 `compliant_joint_states` rows report the
inactive `angularXDrive` (spring 0, damper 0, maximum force `FLT_MAX`), while
the executed joints use `rotationDriveMode = Slerp` and `slerpDrive` spring
2.4, damper 0.075, maximum force 1.15. No physical conclusion is inferred from
that channel, and it was not repaired or reinterpreted after the terminal
physical failure.

## Registration and camera failures

Neither run qualifies anti-clipping or registration. Attempt 1 reports 144.279
mm maximum skin/collider error, 18.648 mm maximum nonadjacent self penetration,
15.731 mm post-solver maximum, and 2,038 incomplete prospective sweep intervals.
Attempt 2 reports 85.368 mm, 18.542 mm, 18.981 mm, and 942 respectively, plus
the failed 5.932 mm finger/object penetration bound. Each Unity log contains
38,934 nonconvex `Physics.ClosestPoint` errors. These are recorded failures,
not Gate-B evidence.

The target center and both palm inputs are outside the head-camera frustum in
720/720 frames of each run, and the registered contact projection contains zero
visible records. Full head videos show an unreadable interaction while the
clean external and labeled overlay streams expose the physical failure. The
prospective geometric FOV receipt is therefore not accepted as a camera or
event-visibility pass.

## Ordered stopping result

| Gate | Status | Consequence |
|---|---|---|
| A -- compliant free-object microcell | **FAIL / NO-GO** | Two eligible attempts exhausted the one-repair limit. |
| B -- anatomy, anti-clipping, three garments | Not executed | Existing failed registration diagnostics are not qualification sweeps. |
| C -- body-derived camera and contact-free motion | Not executed | Head streams are retained failure evidence only. |
| D -- polished furnished bimanual episode | Not generated | Gate A prerequisite failed. |
| E -- three rich no-retuning cells | Not generated | Gate A prerequisite failed. |
| F -- robustness, replay/rerender, multimodal QA | Not generated | Gate A prerequisite failed. |

This result is specific to the tested compliant-hand/controller configuration.
It does not generalize to Unity, PhysX, or the wider project.

## Preserved prior corrected Stage-D decision

The record below is retained verbatim as the corrected historical baseline from
the preceding procedural scene-gate run. Its `STAGE_D_NO_GO_PROMOTION_VETO`
label and metrics remain historical evidence only; they are not reused as
current Gate-A acceptance evidence.

**Historical decision: NO-GO.** One Unity-native implementation owned the avatar, garments, full-body controller, PhysX target, room compiler, head camera, synchronized capture, and truth recorder. Its final frozen primary Stage-D run failed genuine free-object capture and lift, failed a clean commanded release, and did not provide a usable child-head view of the interaction. These were multiple hard promotion failures, not one localized bounded repair.

## Final freeze and eligible run

- Canonical entry point: `python -m babyworld_lite.childlens_engine_bakeoff.procedural_scene_gate`
- Engine: Unity 6000.0.80f1 ARM64/Metal; Unity/PhysX is the sole physical authority and Unity is the sole renderer.
- Clock: 240 Hz physics, 30 Hz render, exact 8:1 integer mapping; 3,840 trace rows and 480 render frames over 16 seconds.
- Final config SHA-256: `3e24522b0cda7bce99f50d07fe94004a1ce1ea53ecd876bfb0600881fa085519`.
- Final primary EpisodeSpec SHA-256: `507677658a2df717d222856221abe7c66277bc4cc1ac0f6badf52c6be2ff4570` (`sage_living_corner__sunset_play`).
- Final trace SHA-256: `9bf8f001acae001d32aa69b4e085460ea8429421940e5c299fe8c8065845abec`.
- Local ignored evidence root: `runs/embodied_simulation/procedural_scene_gate/episodes/sage_living_corner__sunset_play/bimanual_cell`.
- Encoded evidence: clean head, clean external, and labeled external contact/collider overlay; each is 1920x1080, 30 fps, 480 frames, 16 seconds, and passes full ffmpeg decode.

All earlier Stage-D traces are development-only. In particular, the 35 g object, the outcome-informed 120 g change, the contact-evidence correction, the shared aperture-aware reach-band change, the analytic collider change, and controller repairs each invalidated prior acceptance hashes. No earlier dwell, lift, seed, garment, or mass-only result contributes to this decision's PASS or robustness evidence. The complete receipt chain is preserved in `procedural_scene_gate_freeze_amendment.json`.

The final compiler freezes one generic aperture-aware target band for every room seed: 0.34-0.38 m from the shoulder midpoint, with a 0.025 m lateral bias toward the right shoulder. There is no seed-specific code or retuning. The 120 g mass for the exact 55 mm sphere implies 1.3775 g/cm3 for 87.113 cm3; that is physically plausible for a solid or filled polymer/elastomer toy, without claiming a specific manufactured product.

## Ordered gate result

| Gate | Scientific status | Evidence |
|---|---|---|
| A | Implemented and qualified at the interface level | Frozen schemas/config/spec/source hashes and one-state bindings. The authority receipt's zero object-write/force/joint counters are passive hard-coded fields, not runtime detectors; they are not used as independent proof. |
| B | Development qualification only | Three garment sweeps reported approximately 2.843 mm maximum skin/collider error and zero measured garment/body penetration, but these pre-final runs are not reused as final acceptance evidence. The final primary trace sampled 2.103 mm skin/collider error and zero penetration for its three sunset garments. |
| C | Development qualification only; final integrated visual check fails | The neutral mount is 0 degrees from face forward with positive measured clearance (minimum 40.695 mm in the per-step trace and 42.455 mm in captured-frame qualification), but the support table occludes most of the final head interaction view. |
| D | **FAIL / independent promotion veto** | The 120 g free target never lifts after qualification; current opposing contacts are not maintained; a clean commanded free release is absent. |
| E | Not run by ordered-gate rule | No eligible multi-seed/multi-garment hero matrix was generated after Stage D failed. Compiler specs exist for three rooms by three garments, but specs are not episode outcomes. |
| F | Partial diagnostics only; not promotable | Full trace and three qualification videos exist and decode densely. Final registered depth/semantic/instance hero capture, fresh-process replay, same-trace rerender, and multi-seed robustness were not attempted after the Stage-D prerequisite failed. |

## Corrected Stage-D evidence

The pre-outcome evidence definition is measured separation at the completed physics step of at most 0.5 mm. Positive rows inside that shell are geometric proximity, not proof of force. Impulse and nonzero-force-equivalent semantics are reported separately. Speculative rows above 0.5 mm remain auditable raw truth and never increment dwell.

| Measure | Final result | Requirement | Result |
|---|---:|---:|---|
| Right thumb + at least two non-thumb geometric dwell | 1.6875 s | >0.25 s | geometric condition only |
| Same right-hand condition with simultaneous nonzero impulse | 0 s | separately reported | no force-supported dwell |
| Meaningful opposing left geometric dwell, simultaneous with right geometry | 0.3125 s | meaningful stable support | geometric condition only |
| Meaningful opposing left nonzero-impulse dwell | 0 s | separately reported | no force-supported dwell |
| Controller `carry_contacts_maintained` telemetry | 16 steps / 0.06667 s | maintained through carry | **FAIL**; consumes the prior-step contact set |
| Literal same-row post-qualification opposing geometry | 15 steps / 0.0625 s | maintained through carry | **FAIL** |
| Lift after bimanual qualification | 0 m | >0.08 m | **FAIL** |
| Object turn during `BimanualTurn`, support-bound for all 792 steps | 57.0641 degrees | >20 degrees after lift | ineligible because lift failed |
| Commanded free release | hand contact persists after opening command | free release and settle | **FAIL** |
| Minimum hand/object separation | -0.808 mm | penetration <=3 mm | within penetration bound |
| Maximum hand/object impulse | 0.07174 N s | reported separately | measured, not a dwell substitute |
| Passive assistance ledger | 0 entries | 0 | empty as recorded; not independent runtime proof |

The right historical counter and left historical counter become true, but qualification at step `s` consumes the prior step's contact set. `carry_contacts_maintained` is true for steps 1922-1937 (16 steps), while literal same-row opposing geometry strictly after qualification exists only for steps 1923-1937 (15 steps). The telemetry value is therefore not described as current-step carry evidence. The hands then hold rather than synthesizing an object trajectory, proving that the configured hand/controller/target system does not achieve the required physical episode.

Before qualification, the live `interactionAnchorWorld` follows the free target Rigidbody center of mass and drives only bounded kinematic hand waypoints; the anchor is then latched. This is disclosed scene-conditioned controller target tracking, not object assistance: source audit found no direct post-initialization object pose/velocity write, force, torque, parenting, or joint-construction API in the canonical controller/compiler/recorder path. The executed zero counters and empty assistance ledger are passive receipts and cannot independently detect uninstrumented assistance.

## Dense visual audit and direct comparison

All three final videos were inspected as dense phase contact sheets in addition to full decode.

- Clean external: the weighted child is clothed, both arms move, anatomy remains attached, and the overlay is separated from hero pixels. The target remains on the support throughout; no lift, held bimanual inspection, placement, or visibly free release occurs. The room contains catalog furniture, but the hero interaction still reads as sparse and support-table dominated rather than a polished furnished-room episode.
- Clean head: initial scan and final gaze expose the room, but from reach through release the table fills most of the image. Touch, capture, left assistance, turn, placement, and release are not reliably visible. This is a hard event-visibility failure even though the optical mount itself is neutral and clear of the head/clothing.
- Overlay: colliders and contacts are confined to the labeled diagnostic stream. It corroborates hand proximity but cannot substitute for visible hero contact, and no completed contact-to-visible-skin projection product was promoted.

Compared directly with the preserved Unity visual audition, this implementation adds a weighted clothed full body, room context, expanded synchronized body/hand/contact/object trace coverage, and a neutral head mount, but loses the audition's clean target-centric view and does not complete the interaction. Registered depth/semantic/instance hero capture, fresh-process replay/rerender, and robustness remain absent. Compared with the corrected failed bimanual clip, it fixes clothing, full-body/head trace coverage, palm/per-digit rotation/state, object identity, and the fixed approximately 50-degree optical pitch; however, its head view is more occluded and its free-object sequence still fails. The integrated result therefore does not plainly improve over both references as required for PASS.

## Why promotion stops here

The primary hard NO-GO condition is failed unassisted coherent interaction. The visual event-visibility failure and missing eligible generality/replay products are additional independent defects. Reaching PASS would require a materially different hand-object control/contact architecture and camera/embodiment layout qualification, followed by a new prospective freeze and complete no-retuning rerun. That is not one localized repair, so `PROMISING-BUT-ONE-BOUNDED-REPAIR` is not scientifically defensible.

The repository retains only canonical source, frozen config, tests, license/hash manifests, this amendment, and this decision. Complete traces, frames, videos, Unity projects, imported assets, logs, and QA reports remain under the ignored run root.
