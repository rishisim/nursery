# EgoBabyVLM BabyView CLIP+ Machine-DevBench reproduction

## Scope and success target

This workspace has exactly one scientific target: the published **BabyView CLIP+**
row for **Machine-DevBench** in EgoBabyVLM (arXiv:2605.19130v1). It does not
cover the paper's vision or language evaluation families, other models, other
training corpora, benchmark generation, or general challenge participation.

The target percentages (mean ± standard deviation over three runs) are lexical
**53.4 ± 0.7**, grammatical **53.8 ± 2.4**, and overall **53.6 ± 1.5**. Each
Machine-DevBench trial is a two-alternative contrastive choice, so chance is
**50.0%**. Overall is the equal-weight arithmetic mean of the lexical and
grammatical subgroup scores, not a trial-count-weighted average.

This is initially a **faithful independent reproduction**, meaning a fresh run
that follows the published scientific constraints using the pinned released
code and data but may require documented resolutions where the publication does
not uniquely specify an operational choice. It is not yet an **exact replay**,
which would require the authors' exact manifests, splits, seed list, selected
checkpoints, preprocessing artifacts and dependency revisions. Any resolution
is recorded only in `spec.json`; it must never be silently guessed.

## Phase gates

- **Phase 0 — frozen specification (complete):** official sources and immutable
  pins are recorded; scope, data policy, unresolved variables and metrics are
  explicit; durable Juno storage is isolated from disposable worktrees;
  `verify_phase0.py` passes.
- **Phase 1 — source/data access and audit (pending):** obtain authorized
  BabyView access without redistribution, inspect upstream code at the pin, and
  resolve or formally register every blocking variable. No training.
- **Phase 2 — preprocessing/manifests (pending):** create the authorized 2025.1
  manifests and derived inputs under ignored roots, with provenance checks.
- **Phase 3 — training (pending):** execute three from-scratch BabyView CLIP+
  runs with frozen seeds/configuration and retain only compact curated records.
- **Phase 4 — Machine-DevBench evaluation/report (pending):** verify the official
  archive, evaluate only lexical and grammatical tasks, aggregate identically,
  and compare with the target row.

Phase 0 is complete only while this command succeeds from this directory:

```sh
python3 verify_phase0.py
```

## Data and initialization policy

BabyView contains sensitive recordings of children. Release 2025.1 is hosted by
Databrary under controlled access: follow its participant/privacy terms, keep
raw and derived materials in access-controlled ignored storage, and do not
redistribute video, audio, transcripts, frames, manifests containing sensitive
identifiers, or other dataset-derived binaries through Git or public artifact
stores.

The scientific run may use only authorized BabyView 2025.1 visual/audio-derived
training inputs allowed by the protocol. **Public/web-pretrained weights may not
initialize any scientific-run encoder or other learned component.** Off-the-shelf
models may be used only for explicitly documented preprocessing permitted by
the paper (for example transcription/filtering), never as initialization of the
trained CLIP+ model. BabyView evaluation inputs are forbidden; evaluation uses
only the pinned official Machine-DevBench release asset.

## Canonical records

`spec.json` is the single machine-readable source of truth for the frozen
protocol and its assumptions/deviations ledger. Curated configs, compact
aggregate result tables, concise decision reports, and this protocol remain
trackable.

## Local project and Juno storage

The complete Git project remains on the laptop at
`/Users/rishisim/Documents/research/nursery`. Code, configs, checksums,
non-sensitive compact manifests, aggregate results, and decision reports are
committed there so Codex can work normally.

Worktrees are disposable code checkouts; they are never data or run-storage
locations. Large, sensitive, or generated material uses two applicant-private
Juno roots:

```text
Durable: /work/dal503972/egobabyvlm_clip_plus
Scratch: /scratch/juno/dal503972/egobabyvlm_clip_plus
```

The durable root holds manifests, promoted checkpoints, compact run records,
aggregate results, decision reports, and important logs. The larger scratch
root holds authorized raw/derived data, downloads, evaluation data, active
runs, outputs, artifacts, caches, and rendered reports. Scratch is not treated
as a backup: every irreplaceable run output must be checksum-verified and
promoted to the durable root before its scratch run may be removed.

Initialize or verify the Juno layout from the laptop with:

```sh
python3 storage.py init-juno
python3 storage.py check-juno
```

Resolve paths without hard-coding them in later scripts:

```sh
python3 storage.py root durable
python3 storage.py path durable checkpoints
python3 storage.py path scratch runs
```

The environment variables `EGOBABYVLM_REPRO_DURABLE_ROOT` and
`EGOBABYVLM_REPRO_SCRATCH_ROOT` may override the frozen Juno paths. If Juno is
unavailable, jobs must stop rather than fall back to a laptop or worktree path.
Juno protects against worktree deletion, but its backup/snapshot policy must be
verified before it is treated as the sole copy of an irreplaceable artifact.

Before archiving or deleting a linked worktree, run from that worktree:

```sh
python3 reproductions/egobabyvlm_clip_plus/storage.py safe-to-remove
```

The guard verifies the Juno roots first, then refuses to approve the primary
checkout, any modified/untracked files, or any unexpected ignored files. Only
disposable caches and logs are allowed inside a linked worktree. Never use
forced worktree removal.

## Official sources and provenance

- [EgoBabyVLM paper, arXiv:2605.19130v1](https://arxiv.org/abs/2605.19130v1)
  (submitted 2026-05-18; target row and experimental description).
- [Official facebookresearch/egobabyvlm repository](https://github.com/facebookresearch/egobabyvlm)
  (code pinned in `spec.json` to commit `224621caf0628270b6115845ac75a65b984234a3`).
- [Official Eval-Data release](https://github.com/facebookresearch/egobabyvlm/releases/tag/Eval-Data)
  (MachineDevBench archive metadata and digest).
- [Official BabyView 2025.1 Databrary page](https://databrary.org/volume/1882)
  and [original BabyView paper](https://arxiv.org/abs/2406.10447) (release,
  access, privacy, and dataset provenance).
