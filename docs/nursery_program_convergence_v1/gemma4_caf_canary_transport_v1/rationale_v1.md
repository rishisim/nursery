# Gemma 4 public-canary CAF transport amendment rationale

This namespace records one pre-model, public-fixture transport correction. The prior canary failed because macOS `say` could not write the requested little-endian PCM into the frozen AIFF intermediate. No model was loaded, no output existed, no ChildLens content was opened, and the one permitted JSON-fence parser correction was irrelevant and remains unused.

The additive amendment changes only two bound filename occurrences: `german.aiff` becomes `german.caf` in the `say` destination and ffmpeg input. A public-only diagnostic already showed that CAF accepts the same fixed German phrase, `Anna` voice, 170-word-per-minute rate, and `LEI16@22050` data format; the unchanged ffmpeg normalization then produces the frozen final WAV shape—mono signed 16-bit PCM, 16 kHz, exactly 160,000 samples. This amendment records that engineering remedy but does not execute it.

Everything scientifically or semantically relevant remains fixed: five public frames and their offsets, prompt, five-field schema, parser and one-fence allowance, cached Gemma bytes, read-only Qwen3 comparator, 15 items, 900 seconds, 137 windows, K=5 complement protection, outward rounding, coverage and abstention gates, calibration protocol, simulator-oracle truth, and the no-learner/no-outcome boundary. The earlier AIFF failure, validation receipt, terminal technical-revise decision, and disposition remain immutable evidence for the failed attempt.

Any future CAF canary must use the new output namespace, bind the amendment hash, reverify all earlier hashes, and issue fresh canary and activation receipts. This transport amendment alone authorizes neither public model execution nor restricted inference.
