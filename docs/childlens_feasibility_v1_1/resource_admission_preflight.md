# ChildLens feasibility v1.1: resource and quarantine admission preflight

Date: 2026-07-21  
Observation time: 2026-07-21T16:58:32-07:00  
Status: `HOST_CAPACITY_CONDITIONAL; QUARANTINE_NOT_ADMITTED`  
Scientific learner outcome run: no

Supersession note: this artifact is the earlier read-only capacity snapshot. Its `QUARANTINE_NOT_ADMITTED` state was later superseded for metadata and annotation shards only by `output/childlens_feasibility_v1_1/quarantine_admission_receipt.json`. Video transfer remains not admitted because exact selected remote bytes are unresolved.

## Scope and result

This is a read-only host-capacity and quarantine-design preflight. It did not access Keeper, email, release metadata, corpus media, annotations, transcripts, identifiers, or any restricted payload. It did not download data, create the quarantine, launch a measurement instrument, or train a learner. ChildLens feasibility v1 remains unchanged.

At the observation time, the host can accommodate the literal v1 low (41.4 GiB peak) and base (69.4 GiB peak) planning envelopes under the 73 GiB namespace cap and 50 GiB projected-free-space floor. The base envelope has only 2.93 GiB of floor headroom, so this is not execution admission. The v1 high envelope (90.0 GiB peak) fails both controls.

At this preflight snapshot, acquisition remained prohibited until G1, release binding, exact selected object sizes, derivative/scratch bounds, and quarantine controls passed. The candidate root did not yet exist, its parent was included in Time Machine scope, and volume indexing was enabled. Later evidence resolved G1 and admitted an exact hidden root for metadata and annotation shards only; it did not resolve exact media bytes or admit video transfer. FileVault alone is not compliant-retention proof.

This workstream makes no terminal ChildLens feasibility decision.

## Search-scope incident

A repository-wide text search intended to locate the active PyTorch runtime was too broad and returned two nonempirical environment lines from an AEA provenance receipt: a software-version field and an accelerator-availability field. No AEA data, aggregate, identifier, result, empirical code path, tokenizer, checkpoint, weight, or learner artifact was opened or used. The reported host runtime and MPS values were independently re-probed from the repository virtual environment and system state. The broad match nevertheless violated the stricter no-read boundary and is recorded here as a procedural search-scope incident; subsequent probes used fixed, ChildLens-relevant paths only.

## Current read-only host receipt

| Resource | Observation | Admission interpretation |
| --- | ---: | --- |
| Shared data-volume free space | 128,274,540 KiB = 122.332 GiB | Time-varying; remeasure immediately before each shard and each expansion phase |
| Data-volume utilization | 87% | Low/base arithmetic can pass, but shared-volume contention is material |
| Unified memory | 34,359,738,368 bytes = 32 GiB | Does not authorize worker count by itself |
| Memory-pressure report | 87% system-wide free; no swap configured or used | Snapshot only; worker RSS must be benchmarked |
| Logical / physical CPUs | 10 / 10 | Controller starts at two workers; hard maximum four |
| Local runtime | Python 3.12.13; PyTorch 2.13.0 | Existing repository virtual environment, read-only probe |
| MPS | built and available | At most one accelerator process globally; no trainer is authorized here |
| CUDA | unavailable; zero devices | No local CUDA parallelism |
| Storage encryption | FileVault on | Necessary context, not sufficient quarantine approval |
| Parent backup state | included in Time Machine scope | Exact restricted root must be evaluated and, if terms require, excluded before payload creation |
| Volume content indexing | enabled | Exact restricted root needs a documented no-index control or an explicit terms determination before payload creation |

The 122.332 GiB free-space value includes all files already present on the shared volume. Other processes may consume it after this receipt. The public v1/v1.1 report trees occupied approximately 296 KiB before these two resource-preflight artifacts were added; a fresh admission calculation must measure them again rather than rely on that number.

## Proposed restricted quarantine

The proposed top-level root is:

`[RESTRICTED_ROOT_OUTSIDE_REPOSITORY]`

It is hidden, outside `/Users/rishisim/Documents/research/nursery`, and would be on the same device as the repository (`st_dev = 16777230`). It was not created in this preflight. No path below the top-level root may be written into repository artifacts or logs.

After G1 passes and before any restricted payload is acquired, a local operator must create and validate the root with these fail-closed properties:

1. canonical realpath is outside the repository and remains outside it after resolving every existing ancestor;
2. the root and payload descendants are not symlinks, mount redirects, cloud-synced folders, or Git worktrees;
3. root and directories are owner-only (`0700`), files are owner-only (`0600`), process umask is `077`, and no ACL grants another principal access;
4. the precise root's backup and content-indexing disposition conforms to the resolved terms; parent-level state is not accepted as proof;
5. dedicated instrument weights, raw objects, derivatives, restricted manifests, annotations, temporary files, and quarantine logs all count toward the 73 GiB project namespace even if a tool would otherwise cache them elsewhere;
6. raw objects are content-addressed internally, while repository-visible receipts contain only counts, sizes allowed by the terms, and digests—not internal paths, filenames, identifiers, transcripts, or exact media times; and
7. a fresh volume and namespace measurement passes the equations below.

Recommended internal classes are `raw`, `derived`, `restricted_manifest`, `annotations`, `scratch`, `instruments`, and `restricted_receipts`. These labels are a storage design, not authorization to create or populate them.

## Exact admission equations

All enforcement uses integer bytes. Decimal display values are informational only.

Let:

- `G = 1,073,741,824` bytes per GiB;
- `F0` be `df`-reported free bytes on the shared volume immediately before the phase;
- `N0` be allocated bytes already charged to the complete v1.1 namespace, including dedicated caches outside the root;
- `R = sum(size_i)` over unique selected raw acquisition objects, using exact remote sizes;
- `D`, `S`, `Q`, and `T` be conservative byte upper bounds for restricted derivatives, scratch/in-flight transfer, annotations/receipts, and dedicated instruments/environment;
- `C` be the frozen contingency allocation in bytes;
- `Delta` be the maximum additional bytes that can coexist above the current on-disk state during the proposed phase; and
- `Npeak = N0 + Delta`, `Fafter = F0 - Delta`.

The phase passes only when every predicate is true:

```text
R <= 20 * G
Npeak <= 73 * G
Fafter >= 50 * G
Delta <= min((73 * G) - N0, F0 - (50 * G))
all selected sizes are exact and nonnegative
D, S, Q, T, C and downloader duplication behavior have finite upper bounds
```

At this observation, `F0 = 131,353,128,960` bytes. If `N0 = 0`, the free-space floor permits at most 72.332 GiB of additional allocation, making the free-space floor slightly tighter than the 73 GiB namespace cap. Actual `N0` must be measured at admission time.

For a from-scratch planning envelope, the v1 projection used:

```text
P = T + R + D + S + Q + C
T = 8 GiB
S = 2 * R
C = 20% of the pre-contingency subtotal
```

V1 described `D` as 25% of raw and displayed rounded component allowances. To reproduce the frozen v1 table conservatively, admission uses its literal tabulated component values rather than recomputing away rounding:

| Envelope | Literal components `T + R + D + S + Q + C` (GiB) | Peak (GiB) | Projected free now (GiB) | Namespace headroom (GiB) | Free-floor headroom (GiB) | Capacity result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Low | `8 + 8 + 2 + 16 + 0.5 + 6.9` | 41.4 | 80.932 | 31.6 | 30.932 | pass at snapshot |
| Base | `8 + 15 + 3.8 + 30 + 1 + 11.6` | 69.4 | 52.932 | 3.6 | 2.932 | conditional pass; narrow margin |
| High | `8 + 20 + 5 + 40 + 2 + 15` | 90.0 | 32.332 | -17.0 | -17.668 | fail |

These are nonempirical envelopes. Selected media bytes, duration, codecs, and expansion were not inspected. Before transfer, replace `R` with the exact content-blind selected-object sum and replace every other component with a frozen measured or conservative upper bound. Never lower an upper bound after observing lexical content or apparent grounding quality.

## Selective, deduplicated shard acquisition

If upstream gates pass, acquisition must remain metadata-first and serial by shard:

1. Freeze the expected selected-object inventory and exact byte sizes before transfer. An unknown-size object is ineligible until resolved without content inspection.
2. Define a release-local acquisition key from immutable remote revision evidence plus remote object identity, size, and ETag/checksum when exposed. Deduplicate only within this bound ChildLens release; never inspect or reuse AEA, BabyView, or unrelated caches.
3. Sum unique raw sizes for `R`. Reserve the downloader's worst-case in-flight behavior in `S`; if it may hold both a complete partial and final object, charge both until atomic promotion is proven.
4. Permit exactly one resumable shard in flight. Resume the same partial object; do not create parallel copies or speculative downloads.
5. Verify byte count and local SHA-256 inside quarantine, then atomically promote on the same filesystem. A duplicate hash becomes an internal reference to the existing object, not another stored raw copy.
6. Delete only disposable temporary material under a predeclared retention rule, then remeasure `F0` and `N0` before admitting the next shard. Do not credit a deletion until it has completed and `df`/namespace usage reflects it.
7. Decode bounded windows on demand. Full-frame materialization and full-archive acquisition are prohibited.

The controller must abort before a shard begins if another shard is in flight, the release/selection digest changes, a selected size is unknown, downloader duplication is unbounded, or any storage predicate fails.

## CPU, memory, and accelerator controls

Preprocessing begins with two controller-owned CPU workers. Every worker and numeric library receives one thread. Four workers are allowed only after a fixed, content-blind benchmark shows adequate resident-memory, memory-pressure, and I/O headroom.

For measured peak resident-memory values, a candidate worker count `W` must satisfy:

```text
2 <= W <= 4
W * worker_peak_RSS + controller_peak_RSS + instrument_peak_RSS + 8 GiB_memory_reserve <= 32 GiB
```

The 8 GiB memory reserve is a conservative operational reserve for this preflight, not a scientific parameter. Abort or reduce to two workers on swapping, sustained memory-pressure warnings, repeated page-out growth, I/O saturation, nondeterministic failures, or MPS contention. Hardware core count never overrides this bound.

At most one MPS process may run globally. In v1.1 it may be a fixed measurement instrument only after upstream authorization; learner training and causal-arm execution are prohibited. CPU preprocessing must not overlap an MPS job if the benchmark shows unified-memory or I/O contention. CUDA/Slurm scheduling is outside this local pilot and does not alter the frozen causal controls.

## Fail-closed abort conditions

Do not create or populate the quarantine, start transfer, or expand a derivative when any of these conditions holds:

- G1 is not `PASS` for the specific action and derivative class;
- immutable release binding or deterministic restricted-manifest digest is absent or changes;
- the selected inventory violates the frozen 12--18-video, participant-group, stratum, or hash-order rules;
- any selected raw byte size, duplicate key, codec-driven expansion bound, temporary-file bound, or dedicated cache location is unknown;
- `R > 20 GiB`, `Npeak > 73 GiB`, or `Fafter < 50 GiB` using fresh integer-byte measurements;
- full archive acquisition, all-frame materialization, more than one in-flight shard, or an untracked cache outside the counted namespace would occur;
- the canonical quarantine realpath is in the repository, resolves through a symlink, is group/world accessible, has an unapproved ACL, or enters a cloud-synced location;
- backup/indexing disposition is unresolved under the actual terms, including the currently observed parent inclusion/indexing state;
- the raw object or a restricted derivative would enter Git, a report artifact, tool commentary, an external service, or a nonlocal instrument;
- a worker count would exceed four, nested numeric threads exceed one, memory/I/O gates fail, or a second MPS process would overlap; or
- a learner, causal arm, or acquisition effect would be trained, evaluated, or inspected.

On abort, retain only what the resolved terms require or permit, do not improvise deletion, and issue an opaque engineering receipt with no restricted filenames, identifiers, transcript text, or exact media timestamps.

## Preflight disposition

- Host storage: `PASS_LOW; CONDITIONAL_PASS_BASE; FAIL_HIGH` at the recorded snapshot.
- CPU/memory: `CONDITIONAL_PASS_AT_2_WORKERS`; four workers require a benchmark.
- MPS: `AVAILABLE_BUT_SINGLE_PROCESS_ONLY`; no training is authorized.
- Quarantine: `NOT_ADMITTED_NOT_CREATED`; exact-root access, backup, indexing, and G1 checks remain.
- Transfer: `NOT_ADMITTED`; exact selected bytes and bounded expansion are absent from this workstream.

No ChildLens, AEA, or BabyView empirical material informed these capacity calculations. The incidental AEA provenance search match disclosed above was not used.
