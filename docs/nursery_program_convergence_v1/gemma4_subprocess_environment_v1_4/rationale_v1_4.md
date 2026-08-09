# Gemma 4 public-canary subprocess-environment correction v1.4

Status: `FROZEN_PRE_CANARY_SUBPROCESS_ENVIRONMENT_ONLY`

The v1.3 public canary stopped with `E_SUBPROCESS_ENV` before child-process launch and before model generation. Its runner passed four offline and telemetry-control variables through the firewall scrubber's restricted `extra` mapping. The firewall correctly rejected those keys because they are not caller-selectable extras; all four already exist with the intended value `"1"` in the scrubber's mandatory minimal environment.

The v1.4 correction removes only that redundant mapping at the single subprocess call site. The runner must call `scrubbed_subprocess_environment()` with no arguments, must not modify its returned mapping, and must fail closed before launch unless `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, and `DO_NOT_TRACK` are each byte-identical to UTF-8 `0x31` (`"1"`). The firewall, its extra whitelist, its mandatory defaults, credential and proxy scrubbing, network isolation, subprocess arguments, file descriptors, and I/O confinement are unchanged.

This is a public, content-independent invocation defect, not semantic evidence or a scientific result. The prior v1.3 failure receipt remains authoritative. All prompt, media, CAF, template-order, model, schema, parser, sample, export, gate, calibration, and simulator-oracle contracts remain frozen. This amendment executes nothing, permits no fallback, authorizes no restricted inference, and does not open the scientific endpoint.
