# Gemma 4 restricted-worker subprocess-environment correction v1.6

Status: `FROZEN_PRE_RESTRICTED_RERUN_WORKER_ENVIRONMENT_ONLY`

The v1.5 public seal, canary, activation, quarantine/sample checks, and read-only Qwen inputs passed. The bounded restricted preflight locally opened or read authorized inputs and opened the bounded media file descriptors. No payload was exported. Before the Gemma worker subprocess launched, `_worker_environment` passed four mandatory offline and telemetry keys through the firewall scrubber's restricted `extra` mapping. The firewall correctly rejected them with typed diagnostic `E_SUBPROCESS_ENV`; the aggregate v1.5 failure receipt recorded `E_INTERNAL`.

The v1.6 correction deletes only those four redundant extras. `OMP_NUM_THREADS="2"` and `VECLIB_MAXIMUM_THREADS="2"` remain the only scrubber extras. The firewall's exact 14 mandatory defaults remain unchanged, including the four deleted extras as mandatory value `"1"`. The existing post-scrubber `TMPDIR=str(work)` addition remains confined to the absolute owner-private quarantine-local work directory and may not be printed or exported. The final environment is exactly 14 mandatory keys, two thread controls, and `TMPDIR`.

Worker arguments, media descriptors, subprocess confinement, network denial, checkpointing, model, prompt, schema, parser, sample, thresholds, calibration bounds, no-fallback rule, and simulator-oracle boundary do not change. No parser allowance was spent. This amendment itself launches no worker, runs no inference, creates no model output, and opens no scientific endpoint.
