# Embodied artifact recovery status

Status date: 2026-08-08 (America/Phoenix).

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
- original Unity visual-audition, integrated, native, anatomical, bimanual,
  and procedural-gate run trees, except for session-cached visual derivatives;
- the 56.033-second bimanual Stage-D clean external video
  `baseline_external.mp4` (historical SHA-256
  `8bf8fb480e6d7a2044717cc7d3c26e95360a212e5172e39dd4d8e6d1eea23859`)
  and Stage-F audio mux (historical SHA-256
  `7e3932fa6e0d280ff09b40b5075fc7697ea57febdaaf20812e818c622c20d0fe`);
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
