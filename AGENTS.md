# Repository instructions

## Project goal

Nursery is a long-term build, not a race to a paper. The goal: show that extra
modalities (contact/touch, proprioception, and others) raise the ceiling on
language acquisition beyond what video alone can reach. Video-only learners
plateau; extra modalities should lift that plateau. This extends the
EgoBabyVLM line of work, and the simulator exists to supply those modalities.

- The core comparison is the gain in language acquisition: video-only versus
  video plus extra modalities given to the learner as inputs, across amounts of
  training data.
- Every experiment is a step toward that goal. Report its result, then say what
  it means for the next step of the build.
- Do not turn a subgoal's result into a standalone analysis, paper framing, or
  stopping point unless the user asks. A null or weak result is a design
  signal, not an ending.

## Scope

- The immediate user request is the full scope. Background, long-term goals, and discovered problems do not authorize additional work.
- Make the smallest change that satisfies the stated result. Preserve existing architecture and conventions.
- Do not add abstractions, schemas, dependencies, workflows, fallback systems, future phases, or unrelated cleanup unless explicitly requested.
- If the request cannot be completed without materially expanding scope, stop and ask first. State the minimum expansion and affected files.
- Stop when the requested result is verified. Report adjacent issues as deferred; do not fix them.

## Repository hygiene

- Keep one canonical implementation, config, and protocol; use Git history instead of versioned copies.
- Keep runs, generated artifacts, downloads, caches, logs, checkpoints, datasets, and temporary files out of Git and in ignored or temporary roots.
- Write new generated files and runs under `outputs/`; use system temporary storage for scratch work. Keep existing dataset, asset and dependency paths unchanged.
- Name new study roots descriptively (for example, `outputs/runs/acquisition_curves/`) and identify conditions and seeds explicitly. See the README for retained inputs and the archive index for relocated historical outputs.
- Ignored does not mean disposable: preserve saved runs, datasets and licensed assets unless their removal is explicitly authorized.
- Preserve unrelated and pre-existing user changes. Never delete user data or runs without approval.
- Before finishing, run focused validation, inspect `git status --short`, and remove only task-created temporary files.
- Commit and push each verified, appropriately scoped change when the branch and GitHub remote are clear.
