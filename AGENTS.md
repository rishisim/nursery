# Repository instructions

## Scope

- The immediate user request is the full scope. Background, long-term goals, and discovered problems do not authorize additional work.
- Make the smallest change that satisfies the stated result. Preserve existing architecture and conventions.
- Do not add abstractions, schemas, dependencies, workflows, fallback systems, future phases, or unrelated cleanup unless explicitly requested.
- If the request cannot be completed without materially expanding scope, stop and ask first. State the minimum expansion and affected files.
- Stop when the requested result is verified. Report adjacent issues as deferred; do not fix them.

## Repository hygiene

- Keep one canonical implementation, config, and protocol; use Git history instead of versioned copies.
- Keep runs, generated artifacts, downloads, caches, logs, checkpoints, datasets, and temporary files out of Git and in ignored or temporary roots.
- Preserve unrelated and pre-existing user changes. Never delete user data or runs without approval.
- Before finishing, run focused validation, inspect `git status --short`, and remove only task-created temporary files.
- Commit and push each verified, appropriately scoped change when the branch and GitHub remote are clear.
