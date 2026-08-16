# InteractMove to InterMimic: Stage 1–3 protocol

This is the canonical protocol for the preparation boundary between a reviewed
human activity request, an EmbodiedGen scene, and later InteractMove and
InterMimic work. Its machine-readable authority is
`configs/interactmove_intermimic.json`; the validators and compilers in
`babyworld_lite/interactmove_intermimic` define the exact accepted schemas.

Stages 1–3 are deliberately narrow. They specify an activity, request and
ingest an externally generated scene, and compile a deterministic
`SceneBundle`. They do not run InteractMove, run InterMimic, retarget a body,
infer contact, execute physics, or record RGB. A valid Stage 3 bundle therefore
does not establish that the activity is physically successful.

## Frozen conventions and source

The protocol uses metres, kilograms, radians, seconds, a right-handed world
with +Z up, `xyzw` canonical quaternions, `T_parent_child` transform semantics,
and gravity `[0, 0, -9.81]` m/s². The actor is an adult SMPL-X human. Nursery,
not EmbodiedGen, chooses and records the human's target-relative initial pose.
An EmbodiedGen robot pose may help generate a layout, but it is never the
authoritative humanoid pose.

EmbodiedGen is pinned to v2.0.1 at commit
`9b333554254af196bace88c1a171a3bf047fa09c`, native profile
`embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw`. The following pinned primary sources
define the upstream facts used here:

- [LayoutInfo fields](https://github.com/HorizonRobotics/EmbodiedGen/blob/9b333554254af196bace88c1a171a3bf047fa09c/embodied_gen/utils/enum.py#L164-L184)
- [layout and URDF loading in SAPIEN](https://github.com/HorizonRobotics/EmbodiedGen/blob/9b333554254af196bace88c1a171a3bf047fa09c/embodied_gen/utils/simulation.py)
- [layout-generation output contract](https://github.com/HorizonRobotics/EmbodiedGen/blob/9b333554254af196bace88c1a171a3bf047fa09c/docs/documentation/tutorials/layout_gen.md)
- [simulator and URDF guidance](https://github.com/HorizonRobotics/EmbodiedGen/blob/9b333554254af196bace88c1a171a3bf047fa09c/docs/documentation/tutorials/any_simulators.md)
- [affordance annotation and grasp format](https://github.com/HorizonRobotics/EmbodiedGen/blob/9b333554254af196bace88c1a171a3bf047fa09c/docs/documentation/tutorials/affordance.md)

`LayoutInfo` is not self-describing: it does not declare units, handedness,
world-up, or quaternion order. The profile above is consequently part of the
input contract, not a value inferred from `layout.json`.

## Stage 1: reviewed ActivitySpec

A person first writes or reviews the activity. The prompt identifies the
desired action and target before any scene exists; the scene generator does
not get to redefine the target or success criteria. `ActivitySpec` is strict
JSON with no unknown or duplicate keys.

| Field | Exact content |
| --- | --- |
| `schema`, `schema_version`, `protocol_id` | `InteractMoveInterMimicActivitySpec`, integer `1`, and `interactmove_intermimic` |
| `activity_id`, `prompt` | A lowercase path-safe activity ID and a non-empty reviewed prompt |
| `actor` | `body_model="SMPL-X"`, `morphology="adult"`, `gender`, ten finite `betas`, `initial_stance`, and `initial_placement` |
| `actor.initial_placement` | `method="target_relative"`, three-vector `target_relative_position_m`, `face_target=true`, and `selected_by="nursery"` |
| `target_request` | `description`, `category`, an exact prompt substring in `prompt_span`, and `resolution_rule` |
| `target_request.resolution_rule` | Either `sole_manipulated_instance` with `expected_source_node_key=null`, or `exact_source_node_key` with a non-empty source key |
| `relations` | Non-empty `start` and `final` lists. Each entry has `subject="target"`, a supported predicate, `object_description`, and `description` |
| `phases` | Ordered, uniquely named, contiguous half-open frame spans covering the complete activity; each records `description` and canonical `expected_target_contact_hands` |
| `timing` | Exactly the configured 10.0 s at 30 fps: 300 frames sampled on `[0,duration)`, with `sampling="half_open_[0,duration)"` |
| `hand_use` | `one_hand` with exactly `left` or `right`, or `two_hand` with `['left','right']`; its hands equal the union declared by the phases |
| `seeds` | Unsigned 32-bit integers named `master`, `embodiedgen_image`, `embodiedgen_asset`, `embodiedgen_layout`, `interactmove`, `intermimic`, and `render` |
| `success_criteria` | Unique criteria with `id`, `metric`, `operator`, finite or Boolean `threshold`, optional `unit`, and `evaluation_window` |

The success list must cover `target_instance_correct`,
`phase_order_fraction`, `intended_contact_fraction`,
`stable_grasp_fraction`, `max_penetration_m`, `balance_success`, and
`final_relation_satisfied`. These are prospective Stage 4+ criteria, not
measurements made in Stage 1.

Acceptance produces the reviewed ActivitySpec and its SHA-256 over compact,
sorted-key UTF-8 canonical JSON. JSON NaN, infinity, duplicate keys, missing
keys, and unknown keys fail closed.

## Stage 2: external EmbodiedGen request and result

`prepare-embodiedgen-request` projects the accepted ActivitySpec without
altering its prompt. Its output is the following `EmbodiedGenRequest`.

| Field | Exact content |
| --- | --- |
| `schema`, `schema_version`, `protocol_id` | `EmbodiedGenRequest`, integer `1`, and `interactmove_intermimic` |
| `activity_id`, `activity_spec_sha256`, `prompt` | Exact Stage 1 bindings |
| `upstream` | Pinned `version` and `commit` |
| `native_profile` | `embodiedgen-v2.0.1-sapien-rh-zup-m-xyzw` |
| `background_list` | Ordered, non-empty, unique external background references |
| `seeds` | Stage 1 image, asset, and layout seeds |
| `retry_limits` | `image=4`, `asset=3`, `pipeline=2` |
| `render_insert_robot` | `false` |
| `native_robot_pose_role` | Records that a native robot pose may be present for layout placement but is not humanoid authority |
| `expected_outputs` | Required and optional paths listed below |

EmbodiedGen runs outside this repository. The required returned tree is
`layout.json`, one `asset3d/**/result/*.urdf` package for every declared
non-background scene asset, every visual and collision mesh referenced by
those URDFs, and at least one background representation:
`background/mesh_model.ply` or `background/gs_model.ply`. Each background
file is individually optional because the official public example layouts can
be Gaussian-only. `affordance_annot.json` and `mesh_part_seg.glb` are optional.

Ingestion is strict and offline. `layout.json` must contain the pinned
`LayoutInfo` members `tree`, `relation`, `objs_desc`, `objs_mapping`, `assets`,
and `position`; `quality` is optional opaque source metadata. Within
`relation`, released layouts may additionally carry the non-empty source
metadata strings `task` and `task_desc`; other unknown keys remain errors. Each
non-background asset mapping must resolve to one
regular, non-symlink URDF below the declared scene root. URDF references must
likewise be traversal-free relative paths to regular visual and collision mesh
files. Every parsed link, joint, visual, collision, local `xyz`, local `rpy`,
mesh `scale`, and source file hash is retained. Missing required members,
unsafe paths, inconsistent node maps, invalid or non-finite transforms,
duplicate identities, missing target candidates, and absent referenced files
are errors. Source asset names and values are never guessed.

Robot-oriented reachability from EmbodiedGen is accepted as sufficient scene
placement evidence. There is **no additional humanoid accessibility,
reachability, or navigation gate** in this protocol. Nursery still supplies
the target-relative initial humanoid pose from Stage 1; a native robot pose
must not replace it.

## Stage 3: canonical SceneBundle

`compile_scene_bundle` combines the validated ActivitySpec,
EmbodiedGenRequest, `layout.json`, and the returned files into one
`SceneBundle`. The bundle is a deterministic description and manifest, not a
copy of the source assets.

| SceneBundle area | Required machine information |
| --- | --- |
| Identity | SceneBundle schema/version and protocol ID; activity ID; deterministic scene, asset, and instance IDs plus indexed geometry records |
| Bindings | Canonical ActivitySpec and EmbodiedGenRequest SHA-256 values; pinned upstream version, commit, and native profile; layout and every admitted source-file SHA-256 |
| Conventions | Metres, kilograms, radians, seconds, right-handed +Z-up world, `xyzw`, `T_parent_child`, and configured gravity |
| Target | Original `target_request`, resolution method, resolved source node key/category, exact category-match status, and resolved canonical instance ID. Resolution is either the sole manipulated instance or the exact reviewed source key; ambiguity fails |
| Assets | Canonical asset ID, source node/category, URDF path and hash, links, joints, visual and collision geometry records, local transforms, scale, and geometry hashes |
| Instances | Canonical instance ID and asset ID, source node/role, canonical world translation/quaternion and `T_world_instance`, plus static/dynamic body policy |
| Physics | Only source-provided mass, inertia, friction, restitution, and related values, each with provenance. A missing value is JSON `null` with missing-source provenance; it is never replaced with a default |
| Affordances | Optional source part/grasp records and hashes. Upstream grasp orientation is read as `wxyz`, reordered once to canonical `xyzw`, normalized with its conversion recorded, and retained in its declared asset/link-local layer |
| Environment | Background visual mesh capability, optional 3DGS capability, URDF visual/collision capability, explicit ground-plane capability at `z=0`, and settling status/requirement |
| Receipt | Normalizations performed, rejected guesses (none permitted), canonical bundle SHA-256, and the reserved settling state, which is `not_run` with no receipt in the base Stage 3 bundle |

### Coordinate layers

The layers must not be collapsed or silently reinterpreted:

1. `layout.json` positions are world poses in the pinned EmbodiedGen native
   profile: right-handed, +Z up, metres, quaternion `xyzw`.
2. A URDF link tree contributes parent-to-joint transforms, and each visual or
   collision contributes its own link-to-geometry `xyz` in metres, `rpy` in
   radians, and three-axis mesh scale. For rigid no-joint assets, canonical
   geometry world transforms are composed in declared order; source geometry
   is not baked or moved.
3. EmbodiedGen affordance grasp orientations use upstream `wxyz`. The compiler
   explicitly converts them to `xyzw`; it does not relabel the four numbers.

Canonical quaternion sign and normalization rules make semantically identical
poses hash identically. IDs derive from source node identity or admitted file
content rather than traversal order. File records, asset maps, and instance
records are sorted; source-order link and geometry arrays remain lossless. The
semantic hash is SHA-256 over compact sorted-key
UTF-8 JSON with the self-hash field excluded; the pretty artifact ends in one
newline. Repeating compilation from identical bytes must produce identical
IDs, JSON, and hashes.

The background `mesh_model.ply` is visual/reference scene context, not
collision geometry. A `gs_model.ply` is render-only and must never be sampled
as a collision surface. A Gaussian-only package remains a lossless valid
SceneBundle but reports `reference_mesh_ready=false` and cannot pass the
InteractMove full-scene point-cloud gate. The explicit ground plane gives floor
contact at `z=0`, but it does not make furniture, walls, or other background
surfaces collidable. Object collision exists only where admitted URDF collision
geometry says it does. Consequently, a scene with a ground plane can still lack
wall or furniture collision.

Physics values that EmbodiedGen did not provide remain `null` with provenance.
No mass, inertia, material, friction, restitution, joint property, transform,
or collision shape may be guessed to make an import succeed.

For rigid URDF assets with no joints, each instance also records explicit
`T_world_geometry = T_world_instance @ T_link_geometry` matrices while keeping
mesh scale separate. Articulated link world transforms are not guessed; their
presence creates a Stage 4 conversion blocker until a joint-state-aware
resolver is implemented.

### Offline and deferred checks

Stages 1–3 may check JSON/URDF syntax and schemas, exact keys, finite values,
quaternion norms, transform composition, graph and target consistency, path
containment, regular-file and no-symlink policy, file hashes, deterministic
IDs, canonical serialization, declared geometry, and declared environment
capabilities. These are pure offline checks and require no GPU, simulator,
checkpoint, mesh rendering, or downloaded fixture.

Actual URDF acceptance by the chosen simulator, collision-shape construction,
joint behavior, mass/inertia behavior, ground contact, interpenetration,
settling, stability, humanoid placement in physics, and replay are deferred.
A later simulator may produce a settling receipt bound to the SceneBundle hash,
simulator/version, gravity, time step, step count, and post-settle poses. Stage
3 may bind such a receipt when supplied, but it does not create or validate the
underlying simulation by itself. Because downstream physics requires settling,
an absent receipt is represented as pending, not as a pass.

## Commands

The preferred entry point is:

```sh
python -m babyworld_lite.interactmove_intermimic validate-activity \
  activity.json --print-digest
python -m babyworld_lite.interactmove_intermimic prepare-embodiedgen-request \
  activity.json --background-list '["/external/backgrounds/scene_001"]' \
  --output request.json
python -m babyworld_lite.interactmove_intermimic compile-scene-bundle \
  activity.json --request request.json \
  --scene-root /external/embodiedgen/task_0000 --output scene_bundle.json
python -m babyworld_lite.interactmove_intermimic validate-scene-bundle \
  scene_bundle.json --print-digest
```

The equivalent thin entry point is
`python scripts/run_interactmove_intermimic.py` followed by the same subcommand
and arguments. Both output-producing commands refuse to overwrite an existing
output unless `--overwrite` is supplied explicitly.

## Juno GPU execution

Juno is the frozen remote execution target. The canonical wrappers are:

```sh
ssh juno sbatch \
  /scratch/juno/dal503972/interactmove_intermimic/remote_scripts/\
interactmove_intermimic_juno_qualify.sbatch
ssh juno sbatch \
  /scratch/juno/dal503972/interactmove_intermimic/remote_scripts/\
interactmove_intermimic_juno_setup.sbatch
ssh juno sbatch \
  /scratch/juno/dal503972/interactmove_intermimic/remote_scripts/\
interactmove_intermimic_juno_scene_canary.sbatch
```

The first wrapper qualifies a Juno H100 MIG with CUDA/PyTorch. The second
creates the dedicated Python 3.10/CUDA 12.6 environment from pinned
EmbodiedGen commit `9b333554254af196bace88c1a171a3bf047fa09c` on an
H200. The third downloads official dataset commit
`58258b50a0fc95034f2f3cc03b332ec6f72b91fd`, runs its `task_0000` layout for
10 seconds at 200 Hz, records 300 RGB frames at 30 FPS, and never calls
`load_mani_skill_robot`. It retains the source package and canary result under
`/work/dal503972/interactmove_intermimic`; caches, logs, and staging remain
under the matching `/scratch/juno/dal503972` root.

The canary is intentionally narrower than a qualified settling receipt. It
uses EmbodiedGen's released SAPIEN importer, which does not apply the source
URDF mass, clips source friction, and supplies restitution `0.05`. Its receipt
records those facts, the full rigid-body rollout, and that no humanoid,
InterMimic controller, contact trace, or Gaussian background compositing was
present. A successful canary proves GPU import/dynamics/RGB rendering only.

The retained Juno qualification receipt is
`/work/dal503972/interactmove_intermimic/compact_records/`
`juno_qualification_328046.json`. The qualified EmbodiedGen environment
receipt is `embodiedgen_setup_328178.json`; its exact conda/pip manifest is
`/work/dal503972/interactmove_intermimic/manifests/`
`embodiedgen_environment_328178.txt` with SHA-256
`9f3ef4bd905f67841e32347fbdb4ad4d015fac957e0463bb977ad8d3628a1f3a`.
The resolved compatibility set includes Python `3.10.20`, PyTorch
`2.8.0+cu126`, NumPy `1.26.4`, FlashAttention `2.8.2`, xFormers
`0.0.32.post2`, SAPIEN `3.0.0b1`, and ManiSkill `3.0.0b21`. The retained
official-scene canary is
`/work/dal503972/interactmove_intermimic/simulation_canaries/`
`official_task_0000/`: SAPIEN `3.0.0b1`, 2,000 physics steps at 200 Hz, 300
H.264 frames at 30 FPS, zero static desk translation drift, and zero final
recorded body speeds. The video SHA-256 is
`8d159e1f2aa94b7963ca606c4ae4ef57eae0343688d53d08028f75998f04c5f8`.
The corresponding retrospective Stage 2–3 bundle is
`scene_b81d5c384badc17d99272a4203e85c6a`; it resolves the exact target `green
cube` but fails the InteractMove input gate because the official package is
Gaussian-only and the historical generation request is unavailable. This is
an honest capability result, not a failed scene import.

Fresh prompt-to-scene generation additionally requires the owner-only Juno
configuration
`/work/dal503972/interactmove_intermimic/frozen_configs/gpt_config.yaml` with
mode `0600`. The file supplies the public OpenAI endpoint, a project API key,
null Azure API version, and model `gpt-5.6-luna`; its secret value is never
copied into Git, Slurm scripts, logs, manifests, or receipts. The canonical
Nursery adapter preserves EmbodiedGen's existing
`query(text_prompt, image_base64, system_role, params) -> plain text`
interface while translating calls to OpenAI Responses `input_text` and
`input_image` blocks. It uses `store=false`, reasoning effort `none`, exact
OpenAI SDK `3.1.0`, actual PNG/JPEG/WEBP/GIF media types, and the Responses
`max_output_tokens` field. Legacy Chat sampling and penalty fields are ignored;
unknown parameters fail closed. Responses are not globally forced to JSON
because several released EmbodiedGen quality gates require literal plain text
such as `YES` or `NO`.

The historical environment manifest `embodiedgen_environment_328178.txt`
records OpenAI SDK `1.58.1`, which has no Responses resource. It remains an
immutable record of that earlier qualification and is not evidence for the
GPT-5.6 transport. The canonical setup wrapper overlays exactly `openai==3.1.0`,
runs `pip check`, proves the Responses resource exists, and writes a new
job-bound manifest whenever the environment is requalified. The public Hugging
Face token already used for downloads is also not copied into receipts.
EmbodiedGen's released `sim-cli` loads a Franka even when
`insert_robot=false`; therefore it must not be used as evidence of a robot-free
simulation. Nursery's robot-free canary calls only the released scene importer
and renderer.

## Artifact policy and Stage 4 handoff

Disposable execution belongs under `runs/interactmove_intermimic/`, which must
remain ignored and untracked. Tests use framework-managed temporary
directories. Do not commit generated scenes, URDF/mesh packages, 3DGS files,
checkpoints, SceneBundle runs, simulator caches, recordings, or logs. Real
scenes, model/checkpoint material, and complete runs require durable external
storage; the worktree must never hold their only copy. Git may retain only the
canonical source/config/protocol, focused tests, small aggregate decisions, and
small manifests pointing to external immutable artifacts.

Stage 4 receives exactly:

- the reviewed ActivitySpec and canonical hash;
- its EmbodiedGenRequest and canonical hash;
- the validated SceneBundle and canonical hash;
- the immutable external scene root or content-addressed artifact locator whose
  files match the bundle manifest;
- the Nursery-selected SMPL-X initial placement, 10-second frame clock,
  ordered phases, contact-hand declarations, seeds, and prospective success
  criteria; and
- when physics execution is authorized, the simulator/import configuration and
  a SceneBundle-bound settling receipt, or an explicit instruction to create
  that receipt before inference.

Only after that handoff may a separately specified stage convert scene
geometry for InteractMove, run InteractMove inference, reconstruct or complete
contacts and SMPL-X hands, adapt a reference to InterMimic, execute physics, or
render observations. Those later results cannot retroactively change the
reviewed target, initial human pose, phases, seeds, or gates.
