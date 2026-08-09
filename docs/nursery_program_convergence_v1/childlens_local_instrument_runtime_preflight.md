# Convergence v1 local-instrument installation preflight

Date: 2026-07-22  
Host: Apple M5, 32 GiB unified memory  
Controlling contract: `frozen_childlens_pseudo_calibration_protocol.json`  
Scope: public metadata, existing public/synthetic receipts, and commands not executed in this preflight

## Decision

The Qwen3-ASR-1.7B plus forced-aligner path is **license-ready and runtime-proven on one public synthetic fixture**, but it needs a fresh hash-locked environment canary because its existing receipt omitted Python, PyTorch, and the full dependency lock.

Gemma 4 E4B 4-bit is **runtime-proven but license-blocked for activation**. The exact MLX conversion revision is public and ungated, but its metadata says `license: gemma` and it contains no license file. Its bound Google upstream revision says Apache-2.0 and MLX-VLM code is MIT. That mismatch must be resolved in a written conversion-provenance/license receipt before download or restricted inference. No click-through or license was accepted here.

The prior public runs remain plumbing evidence only. Neither authorizes restricted processing, and neither output is human truth.

## Frozen artifacts and measured fit

| Path | Exact revision(s) | Existing artifact receipt | Public synthetic M5 measurement |
|---|---|---:|---:|
| Qwen3-ASR-1.7B-hf | `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` | 4,087,646,324 bytes; manifest `16d6148c1b3c5ad1e791ec3373d73f18023ad559674b421d54afd68d37665073` | 6.14 GiB peak RSS; 28.458 s |
| Qwen3-ForcedAligner-0.6B-hf | `c07281df297b9905d24a508279258cccf987a064` | 1,847,236,856 bytes; manifest `c1668c6b377fb29288316b6d6036cdbb4c6f15360abe9d8dd0f90a2e0554226a` | 1.072 GiB peak RSS; 3.403 s |
| MLX Gemma 4 E4B IT 4-bit | conversion `475b9088d29754a3379866cf5aeb6b41acd313c2`; upstream `ee0ef6023621cff504d758262d4e04895a5af4a2` | 5,179,241,512 bytes; manifest `34310498dc5b809bf4baa79a5290c9e675022d12b725993317bccbffacf1d3ae` | 6.247 GiB peak MLX memory; 17.135 s |

Qwen model/code metadata is Apache-2.0 and both exact model revisions are public, ungated, and non-private. The exact Google Gemma upstream revision is public, ungated, and Apache-2.0. MLX-VLM 0.6.6 is MIT. The conversion mismatch above is the sole license blocker found in this track.

The E4B measurement used MLX-VLM 0.6.6, MLX 0.32.0, and Transformers 5.14.1. Version 0.5.0 had a KV-shared loading failure. Do not downgrade. The E4B run also emitted audio-preprocessor numerical warnings, so one successful fixture is not adequate semantic validation.

## Disk admission

The exact three model snapshots total 11,114,124,692 bytes, or 10.351 GiB. Reserve an additional 4 GiB for two isolated environments, license/manifest receipts, and small synthetic results, plus 6 GiB transient space for sequential Hugging Face downloads and lock construction. The conservative peak admission is therefore **20.5 GiB**.

Immediately before installation, require at least **70.5 GiB free** on the target volume so the frozen 50-GiB free-space floor survives peak use. Download one snapshot at a time. Do not retain a second cache copy. Abort if the predicted post-step floor is below 50 GiB.

## Environment lock commands

These commands are predeclared and were not run by this preflight. They deliberately create separate Python 3.12 environments. The direct pins are in `runtime_locks/`; `uv pip compile --generate-hashes` must resolve and freeze every transitive wheel before installation. The new lock is the controlling runtime receipt, not an assumption that it reproduces the earlier unrecorded PyTorch environment.

```sh
export CONVERGENCE_ROOT="$HOME/Library/Application Support/Nursery Public Instrument Runtime/convergence-v1"
export CONTRACT_ROOT="$PWD/docs/nursery_program_convergence_v1"
mkdir -p "$CONVERGENCE_ROOT/locks"

/opt/homebrew/bin/uv venv --python 3.12 "$CONVERGENCE_ROOT/qwen-env"
/opt/homebrew/bin/uv pip compile \
  "$CONTRACT_ROOT/runtime_locks/qwen3_asr_mps.in" \
  --python "$CONVERGENCE_ROOT/qwen-env/bin/python" \
  --exclude-newer 2026-07-22T23:59:59Z \
  --generate-hashes \
  --output-file "$CONVERGENCE_ROOT/locks/qwen3_asr_mps.txt"
/opt/homebrew/bin/uv pip sync \
  --python "$CONVERGENCE_ROOT/qwen-env/bin/python" \
  --require-hashes "$CONVERGENCE_ROOT/locks/qwen3_asr_mps.txt"

/opt/homebrew/bin/uv venv --python 3.12 "$CONVERGENCE_ROOT/gemma-env"
/opt/homebrew/bin/uv pip compile \
  "$CONTRACT_ROOT/runtime_locks/gemma4_e4b_mlx.in" \
  --python "$CONVERGENCE_ROOT/gemma-env/bin/python" \
  --exclude-newer 2026-07-22T23:59:59Z \
  --generate-hashes \
  --output-file "$CONVERGENCE_ROOT/locks/gemma4_e4b_mlx.txt"
/opt/homebrew/bin/uv pip sync \
  --python "$CONVERGENCE_ROOT/gemma-env/bin/python" \
  --require-hashes "$CONVERGENCE_ROOT/locks/gemma4_e4b_mlx.txt"
```

The proposed `torch==2.13.0` pin is the current public macOS wheel selected for the new convergence lock, not the unrecorded version behind the earlier receipt. It must pass the synthetic MPS canary. Record the SHA-256 of both compiled lockfiles, `python --version`, `ffmpeg -version`, `sw_vers`, and the output of `importlib.metadata` for every direct package.

## Conditional snapshot commands

Do not run these until the disk gate passes. The Qwen commands also require a saved Apache-2.0 model/code receipt. The Gemma command is additionally blocked until the conversion-license mismatch is resolved in writing. No authentication token is needed for the currently public revisions; explicitly unset token variables to prevent accidental credential use.

```sh
export PUBLIC_ROOT="$HOME/Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"

env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_TELEMETRY=1 \
  /opt/homebrew/bin/hf download Qwen/Qwen3-ASR-1.7B-hf \
  --revision bcd2b5b7f32b480ab5790554cfa8347f246a14f3 \
  --local-dir "$PUBLIC_ROOT/qwen3-asr-1.7b-hf"

env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_TELEMETRY=1 \
  /opt/homebrew/bin/hf download Qwen/Qwen3-ForcedAligner-0.6B-hf \
  --revision c07281df297b9905d24a508279258cccf987a064 \
  --local-dir "$PUBLIC_ROOT/qwen3-forced-aligner-0.6b-hf"

# BLOCKED until the MLX conversion's license/provenance receipt is resolved.
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_TELEMETRY=1 \
  /opt/homebrew/bin/hf download mlx-community/gemma-4-e4b-it-4bit \
  --revision 475b9088d29754a3379866cf5aeb6b41acd313c2 \
  --local-dir "$PUBLIC_ROOT/gemma-4-e4b-it-4bit"
```

After each sequential download, compute the repository's existing relative-path/size/file-SHA manifest algorithm and require exact equality with the receipt above before inference. Also retain the pinned README, upstream license, conversion config, and `hf` revision response. A revision match alone is not a byte-manifest check.

## Network-denied public canaries

These commands use the existing fixed-path, self-generated-fixture runners. `sandbox-exec` denies all network access, and each runner independently attempts an outbound connection and fails closed unless denial is observed. Offline and telemetry flags are defense in depth. Run only one MPS-heavy process at a time.

```sh
export REPO_ROOT="$PWD"
export PUBLIC_ROOT="$HOME/Library/Application Support/ChildLens Public Model Bakeoff/v1.3.1"
export CONVERGENCE_ROOT="$HOME/Library/Application Support/Nursery Public Instrument Runtime/convergence-v1"
export DENY_NETWORK='(version 1) (allow default) (deny network*)'

/usr/bin/sandbox-exec -p "$DENY_NETWORK" \
  /usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 GRADIO_ANALYTICS_ENABLED=False \
  "$CONVERGENCE_ROOT/qwen-env/bin/python" \
  "$REPO_ROOT/scripts/run_qwen3_asr_public_synthetic_smoke_v1_3_1.py"

/usr/bin/sandbox-exec -p "$DENY_NETWORK" \
  /usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 GRADIO_ANALYTICS_ENABLED=False \
  "$CONVERGENCE_ROOT/gemma-env/bin/python" \
  "$REPO_ROOT/scripts/run_gemma4_public_synthetic_bakeoff_v1_3_1.py" \
  --model e4b --output "$PUBLIC_ROOT/results/gemma4-e4b-convergence-v1.json"
```

Canary admission requires: network sentinel pass; exact manifest match; MPS available; schema-valid output; both modalities structurally consumed; no external request; peak memory below 28 GiB; at least 4 GiB live memory headroom; and the 50-GiB disk floor. The E4B runner uses a frozen JSON-only prompt followed by Draft 2020-12 validation; it does **not** exercise MLX-VLM server-side constrained decoding. If the contract interprets “strict schema” as grammar-constrained generation rather than schema-valid output, a separate public loopback-only server canary must be frozen and run before activation.

## Exact residual blockers

1. Resolve the exact E4B conversion's `gemma` metadata/no-license-file discrepancy against the Apache-2.0 upstream and record conversion provenance. Do not infer permission from public download access.
2. Generate and hash the two complete dependency locks; the old Qwen receipt cannot reconstruct its PyTorch environment.
3. Re-run both public synthetic canaries under the locks. The earlier metrics are evidence of fit, not evidence that the proposed new dependency lock behaves identically.
4. Decide explicitly whether the frozen contract requires grammar-constrained JSON. The existing E4B canary proves postvalidated schema conformance only.

No restricted ChildLens inference should start until all four are closed.

## Primary public sources

- [Qwen3-ASR repository and Apache-2.0 license](https://github.com/QwenLM/Qwen3-ASR)
- [Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)
- [Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf)
- [Google Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it)
- [MLX E4B conversion](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit)
- [MLX-VLM v0.6.6](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.6)
- Existing public runtime receipts: `qwen3_asr_public_synthetic_runtime_receipt.json` and `gemma4_public_synthetic_runtime_bakeoff.json`
