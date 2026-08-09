# ChildLens v1.2 immutable baseline and quarantine/resource re-admission

Observation date: 2026-07-21  
Status: `PASS_QUARANTINE_AND_RESOURCE_SNAPSHOT`  
Media acquired by this workstream: no

## Scope

This receipt preserves the complete ChildLens feasibility v1 and v1.1 trees as immutable historical evidence and rechecks the existing restricted quarantine using filesystem metadata only. It does not read restricted file content, expose the restricted root or descendant names, access Keeper or email, acquire media, launch an annotation instrument, or train/evaluate a learner.

This is a time-scoped storage/control admission. It does not independently authorize media transfer. Transfer still requires a frozen-selection/release match plus a passing native-transfer controller or the predeclared conservative-bound amendment and cumulative-byte controller.

## Immutable baselines

The canonical artifact-set encoding is UTF-8. Each complete line is:

```text
SHA256␠␠repository-relative-PATH\n
```

Lines are lexicographically sorted as complete lines, paths use POSIX repository-relative notation, and the SHA-256 of the complete serialization is the artifact-set digest.

| Historical set | Files | Artifact-set digest | Result |
| --- | ---: | --- | --- |
| v1 | 23 | `35ba9acbba0fc11fc3419c88ea57d0721e08486c6d37595812ce6c77ce994abf` | matches the v1/v1.1 baseline |
| v1.1 | 40 | `3acd969804d71979ba8071e2446e8ee1ceb3194fc90dcfda802a99e296b7bf48` | frozen as the v1.2 handoff baseline |

The v1.1 set consists of every regular file under `docs/childlens_feasibility_v1_1` and `output/childlens_feasibility_v1_1`, plus top-level `scripts/*v1_1*` and `tests/*v1_1*`. Neither historical namespace was edited. Both historical terminal decisions remain preserved.

## Quarantine control receipt

Exactly one existing quarantine candidate was discovered from its no-index sentinel without serializing its location. The aggregate checks passed:

- canonical root remains outside the repository, on the repository's local encrypted data volume;
- FileVault is on;
- the root and all 10 directories are `0700`;
- all 201 regular files are `0600` after one owner-only mode normalization from `0644` to `0600`;
- ownership mismatches, symlinks, ACL-bearing paths, and ACL-check failures are all zero;
- the exact root is excluded from Time Machine;
- the root-level Spotlight no-index sentinel is present;
- no configured cloud-sync path component was detected; and
- Git tracking from the repository is impossible because the root is outside it.

The one mode normalization changed filesystem permission metadata only. The preflight did not open or alter the file's content and did not emit its name.

The signed-agreement review/delete deadline remains 2027-07-31, 375 days after this observation. That deadline is an active lifecycle control, not a future permission extension.

## Time-scoped resource receipt

The quarantine occupied 20,451,328 allocated bytes at observation. Shared-volume free space was 131,236,425,728 bytes (122.223446 GiB). After reserving the frozen 50-GiB free-space floor and respecting the 73-GiB namespace cap, the tighter maximum additional allocation was 77,549,334,528 bytes (72.223446 GiB). The full 20-GiB raw cap therefore fits this snapshot.

The controller must nevertheless recompute integer-byte capacity before every selected object. Admission remains fail-closed if any of these predicates fails:

```text
cumulative_unique_raw_bytes < 20 GiB before the next write
projected_namespace_peak <= 73 GiB
projected_free_space_after_peak >= 50 GiB
one complete selected object in flight at most
no full archive and no duplicate full-resolution copy
```

This workstream does not activate the optional conservative display-size amendment and does not change the frozen 15-item selection or any scientific gate.

## Privacy disposition

Only nonidentifying counts, byte totals, control booleans, dates, and digests were exported. The restricted path, descendant names, source filenames, participant/session identifiers, transcript or lexical text, frames, exact media times, and restricted hashes remain unexported. No external service was used.

## Reproducibility

- `scripts/validate_childlens_immutability_v1_2.py` recomputes both historical artifact sets and fails on mutation.
- `scripts/preflight_childlens_quarantine_v1_2.py` performs metadata-only discovery, access-control, backup/indexing, retention, and resource checks. It has an explicit owner-only mode-repair option and never serializes the discovered root.
- `tests/test_validate_childlens_immutability_v1_2.py` tests set membership and canonical-order stability.
- `tests/test_preflight_childlens_quarantine_v1_2.py` tests path/payload suppression, fail-closed symlink/nonunique-root behavior, Time Machine plist parsing, and mode normalization.

Machine-readable receipts are in `output/childlens_feasibility_v1_2/immutability_baseline_receipt.json` and `output/childlens_feasibility_v1_2/quarantine_resource_readmission_receipt.json`.
