# Repository agent instructions

## Canonical project and worktrees

- The canonical Git project stays on the laptop at
  `/Users/rishisim/Documents/research/nursery`.
- Treat every linked Codex/Git worktree as disposable. Never place the only copy
  of data, downloads, checkpoints, run outputs, predictions, reports, or other
  research artifacts inside a linked worktree.
- Before archiving or deleting a linked worktree, run the reproduction's
  `safe-to-remove` guard. Never use forced worktree removal.
- Commit and push source code, frozen configs, checksums, non-sensitive compact
  manifests, aggregate result tables, and concise decision reports before
  removing a worktree.

## EgoBabyVLM CLIP+ storage

- Read `reproductions/egobabyvlm_clip_plus/spec.json` and its README before
  working on that reproduction.
- Resolve paths with `reproductions/egobabyvlm_clip_plus/storage.py`; do not
  hard-code a worktree path or silently choose a local fallback.
- Keep durable restricted records and promoted checkpoints under the
  applicant-private Juno `/work` root defined in the spec.
- Keep bulky reproducible inputs, active runs, intermediates, and caches under
  the applicant-private Juno `/scratch` root defined in the spec.
- Juno scratch is not a backup. Before removing a scratch run, promote every
  irreplaceable checkpoint, manifest, resolved config, metric record, and
  important log to the durable root and verify its checksum.
- If Juno is unavailable or its owner-only marker/mode check fails, stop. Do not
  write restricted or irreplaceable artifacts into the laptop checkout or a
  linked worktree as a fallback.

## Sensitive and generated material

- Never commit or redistribute BabyView video, audio, transcripts, frames,
  identifiers, sensitive manifests, model checkpoints, full predictions, or
  complete run directories.
- Keep temporary caches and disposable logs ignored. Important logs belong in
  the durable Juno run record.
- Retain in Git only intentionally curated, compact, non-sensitive scientific
  records. Large or sensitive artifacts remain on governed Juno storage and are
  referenced by opaque run IDs and checksums.

## Completed-task cleanup

- Do not retain one-off phase scripts, audit helpers, verification scripts, or
  task-specific test files after their immediate purpose is complete.
- Before completing work, delete task-created temporary executable scaffolding
  and remove its documentation/config references. Retain only canonical
  production code and durable scientific records needed by the active workflow.
- Add permanent tests only when they protect an ongoing production behavior;
  otherwise run temporary tests from framework-managed or OS temporary storage
  and remove them before committing.
