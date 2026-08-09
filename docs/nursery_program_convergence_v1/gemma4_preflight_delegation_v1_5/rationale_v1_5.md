# Gemma 4 public-canary preflight-delegation correction v1.5

Status: `FROZEN_PRE_CANARY_PREFLIGHT_DELEGATION_ONLY`

The v1.4 public-only attempt failed before fixture creation and model generation. Its sandboxed executor temporarily replaced `v13._public_preflight` with the v1.4 preflight, while a v1.4 helper dynamically called that same module attribute, producing recursion and the recorded `E_SANDBOXED_CANARY` failure. A public synthetic diagnostic established that merely capturing the original v1.3 preflight is insufficient: nested v1.3 execution intentionally overlays the historical worker before v1.2 performs its seal equality check, so rerunning amendment preflight logic then fails with `E_TEMPLATE_BINDING`.

The v1.5 correction transports a value, not executable amendment logic. Before any historical global is overlaid, the sandboxed child computes and validates the complete v1.5 seal exactly once and stores its canonical JSON bytes, digest, and exact parser mode in a local closure. Nested v1.3 and v1.2 seal checks temporarily receive a pure projection callable that can only return a fresh decode of those same bytes for the exact parser mode. It reads no module globals, filesystem, environment, network, content, or model state. Any mismatch fails closed, and both historical callables must be restored on all paths.

This is a public, content-independent preflight-delegation repair. It does not weaken validation: the full preflight still runs once against pristine bound state, and the supplied seal must be byte-identical. It prevents the same validated seal from being recomputed under deliberately mutated compatibility overlays.

The v1.4 environment, CAF, placeholder-order, prompt, model, schema, parser, sample, export, gate, calibration, no-fallback, and simulator-oracle contracts remain unchanged. No parser correction was spent. This amendment executes nothing, accesses no restricted content, authorizes no restricted inference, and does not open the scientific endpoint.
