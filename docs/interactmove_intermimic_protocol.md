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
interactmove_intermimic_juno_generate.sbatch
ssh juno sbatch \
  /scratch/juno/dal503972/interactmove_intermimic/remote_scripts/\
interactmove_intermimic_juno_scene_canary.sbatch
```

The first wrapper qualifies a Juno H100 MIG with CUDA/PyTorch. The second
creates the dedicated Python 3.10/CUDA 12.6 environment from pinned
EmbodiedGen commit `9b333554254af196bace88c1a171a3bf047fa09c` on an
H200 and qualifies the frozen SAM 3D backend. The third performs the fresh,
robot-free prompt-to-scene generation. The fourth downloads official dataset
commit `58258b50a0fc95034f2f3cc03b332ec6f72b91fd`, runs its `task_0000` layout for
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
validates the full transitive OpenAI dependency closure, proves the Responses
resource exists, and writes a new job-bound manifest whenever the environment
is requalified. It also records the global `pip check` result. That broader
check currently reports three pre-existing upstream conflicts unrelated to the
OpenAI closure (FlashAttention/einops, Plyfile/NumPy, and Dash/Pydantic); they
remain visible in the manifest but do not misclassify a valid Responses overlay
as failed. The public Hugging Face token already used for downloads is also not
copied into receipts.

The active image-to-3D backend is EmbodiedGen's released `SAM3D` path. Source
is its `HochCC/sam-3d-objects` submodule at commit
`01417d16fb5cc762a60f370c1bf7f59d603ddfaf`. The gated checkpoint is
`facebook/sam-3d-objects` revision
`2e73555018d2741ccd486e56c24fac41155a1dc6`; both source and weights declare
the SAM License. Authenticated access has been verified for the Juno account.
Setup resumes only the official Hugging Face staging directory, validates the
exact 21-file selection (13,105,863,526 bytes), hashes every file, and publishes
the checkpoint atomically before a one-step-per-stage H200 mesh canary. Partial
ModelScope fragments from the earlier blocked attempt are not admitted or
relabelled. Generation rechecks the complete manifest and binds the snapshot
through the upstream-supported `image3d_model=SAM3D` path.

SAM 3D qualification passed in Slurm job `328529` on an NVIDIA H200 NVL. Its
setup receipt is
`/work/dal503972/interactmove_intermimic/compact_records/embodiedgen_setup_328529.json`,
SHA-256
`515161cafe21ae49eee0370a7137135f00b4c827e8d5f96d49e623370ef8d15f`.
The exact checkpoint manifest SHA-256 is
`e3042c85afe94e104d6253f5821f772e057a8d34779b060e744495cf12fe5410`;
the environment manifest SHA-256 is
`5d818c05fc84a4efde43bf1ac8cea4515af9cf9c1b37f482bcc4a4116c57323f`.
The inference canary produced a nonempty 400,172-vertex, 800,416-face mesh.

Fresh job `328320` completed the GPT layout and accepted one SD3.5 conditioning
image before SAM 3D access blocked asset generation. The canonical SAM 3D run
resumes those irreplaceable stages. The retained scene tree is SHA-256
`9e13a26bb74bbda2d35ee65e37fdba14d1876d0be59a5608df48725256782eb2`;
the recovered layout draft rerenders to those exact bytes and has SHA-256
`f3a2125ffce069c251b4db636d5ec1d2da4397ed79414b0ab3b5b8a5caaf7c19`.
The accepted `table.png` and raw image hashes are respectively
`317b980184bba52c5acb92de0e69d7992ba01dec7746fe5711dca246a80367ef`
and `21291ddb429eb8a31fc367a6e390f53033ffe8efff3a09ad89676ac3c6853cf2`.
The resume hook copies those exact bytes only when the source node, full asset
prompt, initial seed, prior receipt, layout, and all hashes match.

Continuation job `328381` completed a TRELLIS result package for each of the
table, mug, plate, and spoon before the released final QA retried the table.
That checker had combined four camera renders into one 2-by-2 image without
telling the VLM they were views of the same asset, so every response incorrectly
counted the camera views as multiple object instances. The partial job is not
admitted on those responses alone. Its compact receipt SHA-256 is
`0842642b0e10a5ddee297d2188accf3ac3ac95042764551f9c3ce3dd90408d23`.
The SAM 3D continuation validates every file against that receipt but admits
only its four accepted SD3.5 conditioning images. No TRELLIS mesh, URDF,
render, or result package is copied into the SAM 3D scene. Every asset is
regenerated by SAM 3D and must pass a corrected four-view exact-`YES` gate;
the mug must additionally pass the exact-one-handle gate.

Requalification job `328392` admitted the resumed plate but correctly rejected
the resumed mug because its mesh had a malformed vertical side protrusion. Its
compact receipt SHA-256 is
`6a5160b0f5e0a3bef4ae4a2a51831426adfbcb76b65d0b6b944c32bfef41ef3c`.
The canonical protocol therefore reuses the hash-bound accepted SD3.5 mug image
from job `328381` but not that job's mug result package. It regenerates only the
mug's TRELLIS geometry and applies the same corrected exact-`YES` multiview
gate. Table, plate, and spoon result packages remain eligible only after that
gate is repeated in the publishing run. This preserves earlier accepted stages
without admitting the known malformed geometry.

Generation job `328413` proved that the corrected generic gate could still
produce a false positive: direct inspection of its four target renders showed a
mirrored second mug handle. That package and its scene-only canary are retained
only under the disposable superseded-run root and are not the shared scene. The
canonical target gate now asks a separate exact-output multiview question that
requires exactly one connected handle. It preserves the same prompt and
hash-bound SD3.5 image while trying only the frozen TRELLIS retry seeds.

The previous admitted TRELLIS generation is Slurm job `328463`. The first permitted retry
seed, `33936`, passed the upstream check, corrected generic four-view check, and
single-handle target check exactly. Direct four-view inspection also confirmed
one handle and an open cup interior. Its durable package is
`/work/dal503972/interactmove_intermimic/scene_packages/red-mug-mouth-return/`.
The generation receipt SHA-256 is
`f507543ca408c143b3a538854bd5b6efec2b251f81945948b007a92d61cfe72a`;
`layout.json` is
`400c833dbfc22393800254f998277a3861baa0b5e6aceb7f146affeeca5453b8`.
The receipt proves `render_insert_robot=false`, `robot_actor_loaded=false`, and
that the released `sim-cli` was not invoked.

SceneBundle compilation and the bounded scene-only canary passed in Slurm job
`328468`. The pretty JSON file SHA-256 is
`9ad409fa58d04dc8faabf802a71f2225968f8341841a5ece3265d03cbe76bba6`;
its documented canonical JSON digest is
`6606445cc1d6561bfdb7b5411182841a315f54778cfa180d6a403a0542de4c0a`,
and its bundle ID is `scene_88520fa08789071d24de23598dbc4add`.
SAPIEN `3.0.0b1` ran 2,000 physics steps at 200 Hz and wrote a 300-frame,
30-FPS, 10-second video. The video SHA-256 is
`d7b3d77a7e80ac5a636f403024bb062617b1bc4db7d76aac9e1e6596c4ea9574`.

This is a canary, not a qualified settling receipt. `interactmove_scene_input_ready`
and `dynamic_settle_ready` remain false because the selected background has no
reference mesh/room collision, and the dynamic-object source URDFs omit
restitution. The canary intentionally exposes the released importer's fallback
behavior: it does not apply URDF mass, uses hard-coded restitution `0.05`, and
does not record contacts. No InteractMove motion or InterMimic execution has run.

The previous TRELLIS handoff, SHA-256
`5f6a1730bc1bb7c57c62a8f238bafabc5fe04592e2078097901093f5e0f151b9`,
is retained under the durable `.superseded/328463` record root. It is not the
active shared scene.

The admitted SAM 3D generation is Slurm job `328538`. It reused only the four
hash-bound SD3.5 conditioning images from job `328381` and generated fresh
SAM 3D geometry, collision proxies, URDFs, and renders for the plate, red mug,
spoon, and table. Plate, spoon, and table passed with asset seed `2026081502`.
The mug rejected seeds `2026081502` and `33936` at the strict geometry gates,
then seed `62468` passed the upstream, corrected four-view, and exact-one-handle
checks. Direct inspection confirms one connected handle and an open cup
interior. The generation receipt SHA-256 is
`beaeccc61e87e74636171d48b963d01563e107c906f166e8490b32e808f9d3ac`;
`layout.json` SHA-256 is
`8062b9ce0608d5851c8ae0ede7a21626fd8dcc99320d06462804c32490dcd207`.
No robot actor was inserted and released `sim-cli` was not invoked.

SceneBundle validation and the bounded scene-only SAPIEN canary passed in
Slurm job `328542`. The bundle ID is
`scene_bc22206bb607d780cc8b415f8d48f584`, its pretty-file SHA-256 is
`4901e5de136bda18a1f843853894b45cde76121b7e50d5354445afe74b8a859e`,
and its semantic digest is
`72d955b5f1bc4ecd63853744ae73b66da5b4182b72348fb25f5a76538f7df5ca`.
The target is source node `red mug`, instance
`egv2_028a54da21c1691ae18ea225`; its support is source node `table`, instance
`egv2_0d4fc4a78d3706edccafb665`. SAPIEN `3.0.0b1` ran 2,000 steps at 200 Hz
and wrote 300 H.264 frames at 30 FPS for exactly 10 seconds. The video SHA-256
is `c10c9e9c9a6e338344db58b7e93e33a43ccec89adf5ad9af38280b79ae716ac5`.

The active fair-comparison handoff is
`/work/dal503972/interactmove_intermimic/compact_records/shared_scene_red-mug-mouth-return.json`,
SHA-256
`9436c6cb6a267d9352cb5a2785374ccde302c0394657cc29c23cb76f0dd69466`.
It binds all 165 scene files (166,385,332 bytes) by relative path, byte size,
and SHA-256; the canonical inventory digest is
`5a35c99a439fd77b22893d1fb78c461a0171dc81a552b0e61cdd20f9fc18ed0d`.
The HOIDiNi protocol must consume this package directly or verify an exact
durable copy against that inventory; it must not independently sample a scene.

This remains a scene-only canary, not a qualified settling receipt. The bundle
is not yet InteractMove-input-ready: the selected Gaussian background has no
reference mesh or room collision, and the dynamic plate, mug, and spoon URDFs
omit restitution. The released SAPIEN importer supplies restitution `0.05`,
does not apply URDF mass, and records no contacts. No InteractMove motion or
InterMimic execution has run.

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
