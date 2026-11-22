# Repository instructions

## Project goal

Nursery is a long-term build, not a race to a paper. The goal: show that extra
embodied modalities (gaze, contact/touch, hand pose, arm motion, and wrist
load) raise the slope and ceiling of language acquisition beyond what
video-only learners and standard VLM architectures reach across amounts of
training experience. This extends the [EgoBabyVLM](https://arxiv.org/html/2605.19130v1)
line of work.

- **Active branch & context:** Read [docs/unified/CONTEXT.md](docs/unified/CONTEXT.md)
  and [PLAN.md](PLAN.md) before modifying code or launching runs on
  `research/unified-architecture`.
- **Active code layout:** New Phase 2 unified architecture code belongs in
  `nursery/unified/` with tests in `tests/unified/`. Keep Phase 1's
  `nursery/learner/` and existing core `tests/` intact as reusable data
  infrastructure and regression reference.
- The core comparison is the gain in language acquisition: `V+G+B` and `V+G`
  versus `V` and fair video-only baselines (`CLIP-Global`, `BabyLLaVA-Nursery`)
  on identical clips, splits, seeds, and visual backbones.
- Every experiment is a step toward that goal. Report its result, then say what
  it means for the next step of the build. Do not turn a weak intermediate
  result into a standalone paper framing or stopping point unless asked.

## Experiments using controlled BabyView data

- Before a run that reads Databrary recordings, participant-level derivatives,
  or gated BabyView model weights (`mcxfrank/babyview-dino-vitb16`), flag the
  controlled inputs in the run plan. Verify personnel and storage against the
  project 15815 Data Management and Sharing Plan in
  `outputs/exports/databrary/`. Use access-restricted UTD Juno storage for
  controlled files, including manifests and per-clip outputs.
- Use scratch only for active processing. Verify required outputs in restricted
  persistent storage before clearing temporary copies, and verify backup and
  restore coverage before relying on that storage for unique records. Follow
  the repository's approval rule before deleting existing data.
- Rishi reviews access and storage quarterly during active processing, at
  onboarding or departure, before release, and at closeout. Record the review
  in restricted project storage. Check releases for identifying material, file
  manifests and credentials.

## Scope

- The immediate user request is the full scope. Background, long-term goals,
  and discovered problems do not authorize additional work.
- Make the smallest change that satisfies the stated result. Preserve existing
  architecture and conventions.
- Do not add abstractions, schemas, dependencies, workflows, fallback systems,
  future phases, or unrelated cleanup unless explicitly requested.
- If the request cannot be completed without materially expanding scope, stop
  and ask first. State the minimum expansion and affected files.
- Stop when the requested result is verified. Report adjacent issues as
  deferred; do not fix them.

## Repository hygiene

- Keep one canonical implementation, config, and protocol; use Git history
  instead of versioned copies.
- Keep runs, generated artifacts, downloads, caches, logs, checkpoints,
  datasets, and temporary files out of Git and in ignored or temporary roots.
- Write new generated files and runs under `outputs/`; use system temporary
  storage for scratch work. Keep existing dataset, asset and dependency paths
  unchanged.
- Name new study roots descriptively (for example,
  `outputs/runs/unified_u1_parity/`) and identify conditions and seeds
  explicitly. See the README for retained inputs and `archive/` for historical
  Phase 1 and earlier outputs.
- Ignored does not mean disposable: preserve saved runs, datasets and licensed
  assets unless their removal is explicitly authorized.
- Preserve unrelated and pre-existing user changes. Never delete user data or
  runs without approval.
- Before finishing, run focused validation, inspect `git status --short`, and
  remove only task-created temporary files.
- Commit and push each verified, appropriately scoped change when the branch
  and GitHub remote are clear.
