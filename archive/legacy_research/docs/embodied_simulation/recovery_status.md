# Embodied artifact recovery status

Status date: 2026-08-09 (America/Phoenix).

## Incident boundary

On 2026-08-08, 17 auxiliary Codex worktrees were removed with
`git worktree remove --force`, followed by `git worktree prune`. The incident
session reported approximately 49 GB removed. That total included unique
ignored research runs and media, but also duplicated Unity editors, Unity
project caches/builds, Python environments, dependency checkouts, and dense
reproducible frame directories.

The committed scientific lineage was not lost:

- The August 5 reference-hand baseline is commit `f6451c4` on
  `research/embodied`; before this recovery record,
  `origin/embodied-simulation` also pointed there.
- The divergent Stage-C reconstruction remains at
  `origin/embodied-simulation-stage-c-reconstruction`, commit `d413506`.

## Exact restoration completed

The August 5 object-free reference-hand qualification has been regenerated
under ignored
`runs/embodied_simulation/reference_hand_microcell/qualification/`. Every
recorded canonical artifact is byte-identical to the hash frozen at `f6451c4`:

| Artifact | SHA-256 |
| --- | --- |
| `preflight.json` | `6299b1c11a13dbbfdb12fc19a5e5a5c06846f8ba97f4f20dca43b81b37d4c058` |
| `dof_manifest.json` | `4b961d6dc6b4eecf58be3a8b41d397d9679a692082fecfdac4089638411b208b` |
| `trace.json` | `429ad19266b72485d0b233f08fa2da106334dccc4e37425187884f02352c9cc3` |
| `qualification_metrics.json` | `1354b2f7e6b21c8a85e63d68951e4d24744a74b2cbd19960105e265a285efeb5` |
| `reset_ledger.json` | `ac0e135e52dd4aef5ac7c253c248787087bfd5c287e75e2d38bda7608021acd6` |
| `dof_sweep_diagnostic.mp4` | `580182169491b40d45372d98656874f3690a59156df999478805126dba3205bf` |

The restored media is 1920x1080, 30 fps, 1,536 frames, and 51.2 seconds. The
scientific result remains `HAND_QUALIFICATION_NO_GO`; recovery does not change
the decision or consume a physical attempt.

The exact public toolchain inputs needed by the embodied workflows are also
restored under ignored `.external/` roots:

- MIMo commit `040b0ae4914cbfb26afdf830aa81775b90922f3f`;
- MolmoSpaces commit `c2f1b583f087e1d3994e1377574843b759d9d0f8`;
- MPFB commit `f4f4f1ffa8203585730a7ce433b66738777ba168`;
- Unity `6000.0.80f1` arm64, revision `2dfd32957da2`;
- Ultraleap commit `833d82e7333a5f37ebc0844d02431acf74f35d24`,
  including the exact archive and binary asset hashes frozen in
  `configs/embodied_simulation_reference_hand_microcell.json`;
- the Kenney Furniture Kit archive with SHA-256
  `68afa4e6dc8a53942379fb47f1e84ec735d46b77bb1c1ceb968e245693dde067`.

## Exact Juno recovery completed

After VPN access was restored, the recorded Phase 3 path
`/work/dal503972/embodied_phase3` on `juno.hpcre.utdallas.edu` was reachable.
The scientific inputs, source/config snapshot, logs, and appearance run were
copied into ignored
`runs/embodied_simulation/recovery/juno_embodied_phase3/`.

The recovered scope contains 243 files totaling 273,361,097 bytes, including
108 MP4 files. A fresh remote SHA-256 manifest matched the local copy for all
243 files. The local verification manifest is
`juno_recovery_sha256.txt`; its SHA-256 is
`3f6560fc3a3de25f1fcd2fffc7ea11b268aa04c72ea0a58ede50574fb94513ef`.
Reproducible model weights, environments, caches, runtimes, and public source
checkouts were deliberately left on Juno rather than duplicating approximately
58 GB of model weights alone.

This remote path contains the July 31 Phase 3 egocentric appearance corpus. It
does not contain the later Unity visual-audition, native, anatomical, bimanual,
procedural, or August 5 reference-hand runs.

## Exact complementary visual recovery completed

The two complementary Unity visual experiments that led into the August 5
reference-hand reset have been reconstructed under ignored
`runs/embodied_simulation/recovery/complementary_visuals/`. The render staging
was recovered from the original August 2-3 Codex session records and rerun with
Unity `6000.0.80f1` arm64/Metal. These are byte-identical recoveries, not merely
visually similar rerenders:

| Artifact | SHA-256 | Validation |
| --- | --- | --- |
| Unity visual-audition movie | `5039aeaf0abe2f261ce09feef37e960c4d4c71abd1ab4c3d1b2da0095d8fb76c` | frozen hash match; 960x540, 210 frames, 7.0 s; full decode |
| Unity visual-audition diagnostics | `48851e6bc7a670979448b88322e44051d59e1c1264164058a6a42377dc077cd0` | frozen hash match |
| Unity visual-audition dense timeline | `077ae6e31ec8b0a02829d7975fcd0376eed516ebc71c4768cd6b200b1d742926` | frozen hash match |
| Bimanual Stage-D head view | `c133699038a10c12e0ca1ed07b4ef41e1765f8cf731302fdb320370226a21ae6` | frozen hash match; 960x540, 1,681 frames, 56.033333 s; full decode |
| Bimanual Stage-D external view | `8bf8fb480e6d7a2044717cc7d3c26e95360a212e5172e39dd4d8e6d1eea23859` | frozen hash match; 960x540, 1,681 frames, 56.033333 s; full decode |
| Bimanual Stage-D dense timeline | `c584e009b29262ed6648004f115835d7cc45379776ca210476b56047a9a781a2` | frozen hash match |
| Bimanual event-audio mux | `7e3932fa6e0d280ff09b40b5075fc7697ea57febdaaf20812e818c622c20d0fe` | frozen hash match; H.264/AAC; full decode |

The bimanual rerender used the surviving exact authority trace
(`7fe3777fe03806984378e9695371e885db31672d88d3df01e6fe4f99b0bc30e7`)
and contact record
(`5820f755ce45e25a4e191e46729495ee563a5d44363ee8de1a9983de09e4986d`).
For the visual audition, `frame_0120.png` is also byte-identical to the
historical session-cached frame. Although the regenerated FBX container has a
different nondeterministic container hash, the recovered flat OBJ/material
staging reproduces the imported scene, diagnostics, representative frame,
timeline, and encoded movie exactly.

The ignored machine-readable provenance and verification record is
`runs/embodied_simulation/recovery/complementary_visuals/recovery_receipt.json`.
The only missing part of the visual-audition comparison bundle is the Phase 6
MIMo source clip and the two derivatives made from it (`mimo_reference_6_13.mp4`
and `side_by_side.mp4`). Their source did not survive locally or on Juno.

## Derivative evidence recovered

Codex session records retained 137 viewed-image records representing 126 unique
cached blobs and 69 original paths. They are stored by content hash under
ignored `runs/embodied_simulation/recovery/session_image_blobs/`, with the
machine-readable inventory at
`runs/embodied_simulation/recovery/session_image_cache_manifest.json`.

These bytes are exact as cached by the viewer, but they may be resized or
re-encoded derivatives of the deleted originals. They must not be substituted
for an original artifact merely because their visual content is useful.

## Regenerated but not byte-identical

The MPFB weighted child `.blend` can be regenerated from the recovered pinned
source, and a compatible FBX can be exported. Blender embeds nondeterministic
IDs/timing in these containers, so the regenerated files do not match the
historical container hashes:

| Artifact | Historical SHA-256 | Current regenerated SHA-256 |
| --- | --- | --- |
| MPFB source `.blend` | `38765a776be116b3ce1c3c49cf253be8d25946170041e7bcf54b77fec72b5c3b` | `d26284152f284b729689a60ea9087ce028b6108e9f41c94968875a52846c7295` |
| exported `child.fbx` | `b766981d9d3504cea220c0d72ad8aa56cbd80453e910fc76dc8c8814fbd980de` | `90206af9a95d25f151c00deb321aadaff7be2af32ae92eaf9fce129ae9e2a16b` |

The procedural and earlier Unity gates intentionally continue to reject this
FBX as a byte-identical replacement. Their frozen asset hash has not been
weakened.

## Still missing locally

- Original ignored Phase 1-2 and Phase 4-6 run trees and their complete
  intermediate media; the exact Phase 3 appearance subset on Juno is restored
  as described above;
- the Phase 6 accepted episode
  `runs/embodied_simulation/phase_6/final/accepted_episode.mp4` (historical
  SHA-256
  `93885e1f6cdf9eaed34ab32d8795f489e67ee6541cc7795765a0330aa0f378c3`),
  needed to recreate the Unity-versus-MIMo side-by-side comparison;
- complete original Unity integrated, native, anatomical, and procedural-gate
  run trees, plus noncanonical intermediates from the visual-audition and
  bimanual runs; their session-cached visual derivatives remain available;
- the historical MPFB `.blend` and `child.fbx` containers with their frozen
  hashes;
- superseded development attempts, caches, environments, and duplicated
  editor/project data that contributed to the reported 49 GB.

Local recovery checks found no matching Trash contents, Time Machine
destination, Data-volume APFS snapshot, open deleted-file handle, or surviving
local duplicate. The Juno Phase 3 path has now been recovered and verified, but
it does not contain the later Unity outputs. The historical external volume
`EOS_202603` is not mounted and remains the outstanding source that may contain
additional originals.

Generated recovery outputs remain ignored and are not committed as repository
source. This document is the compact retained recovery/decision record.
