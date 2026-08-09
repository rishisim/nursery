# ChildLens 30-video extension status

The user-selected extension means 30 videos total: the immutable original 15
plus exactly 15 new participant-distinct videos. The additional selection is
now frozen. It used only release metadata and verified official
speech-presence annotations; it did not open media, lexical content, model
output, confidence, agreement, or any causal endpoint.

The selected extension contributes 900 speech seconds and 135 fixed
referential windows. Together with the original sample, the calibration target
is 30 participant-distinct videos, 1,800 speech seconds, and 272 windows. This
is the only post-failure expansion. No further expansion, third visual model,
Qwen-only substitute, empirical-free fallback, prompt tuning, parser
correction, threshold change, or size-based reselection is permitted.

The secure transfer and offline calibration implementations are ready. The
transfer handles one source at a time, verifies exact remote bytes, creates a
deterministic bounded clip, removes the transient full source before the next
transfer, and preserves the 20-GiB simultaneous raw cap, 73-GiB namespace cap,
and 50-GiB free-space floor. The selected source set and projected retained
clips both fit those limits.

Execution is currently paused before acquisition because the prescribed
one-use read-only Keeper token is absent from macOS Keychain. No new media,
restricted inference, calibration aggregate, learner, or causal endpoint has
been opened. Restore the same already authorized read-only token without
placing it on the command line:

```sh
security add-generic-password -U \
  -s "ChildLens-v1.2-Keeper-Repo-Token" \
  -a "childlens-v1.2-read-only" \
  -w
```

After the credential is restored, the prepared zero-argument acquisition
command is:

```sh
python3 scripts/run_childlens_30_video_extension_acquisition_v1_8.py
```

The acquisition controller deletes the Keychain item only after all 15
selected additions have been transferred, clipped, verified, and checkpointed.
The subsequent offline combined-calibration command is:

```sh
python3 scripts/run_childlens_30_video_calibration_extension_v1_8.py
```
