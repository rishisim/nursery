# ChildLens calibration recovery v5 — administrative recovery status

The original incident at commit `020fd29` remains unchanged. A separate
metadata-only administrative process created an owner-private mode-0600 input
containing exactly the immutable 18 development participants and zero locked
participants. Its cryptographic allowlist/locked-set overlap check passed, and
the clean scientific process has no input path to the legacy mixed-scope
manifests.

The frozen scientific config hash remains
`b048ca4f4950eaf37d8e751a88ee358d5eabeb83b5187302502d9e08d62b130d`.
No route, model, fold rule, duration/lag scale, architecture, training budget,
positive control, gate, threshold, or terminal state changed.

Clean Stage 0 passed before media access:

- 18 development and zero locked participants;
- 15 minutes per participant and 4.5 core observation hours;
- three participant-disjoint folds of six;
- minimum per-participant support of 60 two-second, 60 six-second, and
  44 eighteen-second windows;
- frozen bounded acquisition intervals and all signed lag positions;
- synthetic held-out learner sensitivity passed.

Selective acquisition has not started because the approved temporary
read-only Keeper credential is absent. In macOS Keychain Access, create a
temporary login-keychain Password Item named
`ChildLens-v1.2-Keeper-Repo-Token`, with account
`childlens-v1.2-read-only`, and use the approved read-only ChildLens Keeper
repository token as its password. Then resume this task. Do not paste the token
into chat or a shell command.

No ChildLens media was acquired or decoded, and no embeddings, learner
training, lag scores, simulator data, or side-cue conditions were produced.
