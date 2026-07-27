# EgoBabyVLM `triple` public/dummy CUDA preflight

**Status:** PASS

**Decision date:** 2026-07-27

**Scope:** engineering compatibility only. No scientific experiment, score,
restricted media, ChildLens material, or BabyView material was used.

## Decision

Hugging Face Job `6a66ba7e7ef3c0846496a1bf` completed the bounded single-L4
preflight at the immutable upstream and public-weight pins. The reviewed
machine-readable result is
[`results/egobaby_cuda_preflight.json`](../results/egobaby_cuda_preflight.json).
It establishes engineering compatibility only.

Proposed bounded run:

- provider: Hugging Face Jobs;
- image/environment: pinned EgoBabyVLM `pixi.lock` at commit
  `224621caf0628270b6115845ac75a65b984234a3`;
- GPU: one NVIDIA L4 (`l4x1`);
- maximum wall time: 60 minutes;
- current account hardware-list rate: USD 0.80/hour;
- user-approved ceiling: USD 2.50;
- operational upper-bound cost at the listed rate: USD 0.80;
- inputs: public pinned weights and self-authored tensors/text only;
- network destinations: GitHub for the two public repositories and Hugging
  Face for the two public model repositories.

No distributed run is included in that ceiling. Single-GPU compatibility is
required; DDP remains explicitly untested unless separately safe within an
approved budget.

## Remote submission evidence

The user authorized the bounded job. On 2026-07-26, no remote job could be
created:

1. Hugging Face Jobs connector, Docker argv submission on `l4x1`:
   `INVALID_ARGUMENT` before job creation.
2. Hugging Face Jobs connector, Docker string submission on `l4x1`:
   `INVALID_ARGUMENT` before job creation.
3. Authenticated connector, minimal two-minute `cpu-basic` UV validation:
   `INVALID_ARGUMENT` before job creation.
4. Official local `hf jobs` CLI fallback: rejected locally as not logged in;
   no API submission occurred.

After Jobs authentication was restored, two L4 allocations ran:

1. Job `6a66ad8adb23d7a7ec1cf0ba` stopped during bootstrap because Pixi 0.49
   predates the pinned lockfile schema. No learner code executed.
2. Job `6a66ade6db23d7a7ec1cf0e2` used Pixi 0.69 and reached the preflight. It
   failed closed before model initialization because the frozen manifest had
   recorded Hugging Face/Xet ETags rather than the Git LFS file SHA-256 values.
   Immutable LFS pointers at the pinned revisions establish the corrected
   DINO SHA-256 `d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841`
   and BERT SHA-256
   `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.

These are environment/manifest repairs to the canonical preflight, not bridge
attempts and not changes to either public prior.

Two subsequent allocations advanced the runtime:

1. Job `6a66aef4db23d7a7ec1cf0fe` validated the corrected file hashes and
   initialized CUDA/distributed code, then stopped because an unused bundled
   COCO path interpolation required environment values. Disposable empty paths
   now satisfy resolution; no dataset was accessed.
2. Job `6a66afc07ef3c0846496a170` initialized the shared prior and reached the
   first CUDA contrastive operation. PyTorch deterministic mode correctly
   rejected cuBLAS execution without a pre-import
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The canonical entrypoint now sets that
   declared deterministic workspace configuration before importing PyTorch.

Three minimal-Ubuntu linker remedies were capped after the same Triton helper
failure (missing compiler, compiler without a CUDA link path, explicit driver
path). The distinct official
`pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel` environment resolved that
infrastructure failure. Job `6a66b30e7ef3c0846496a184` then completed the
DINO forward path and exposed a pinned-runtime reproducibility invariant:
xFormers 0.0.32 supplies no deterministic backward operator for the
block-diagonal attention used by DINO/iBOT. The preflight now follows its
already frozen CUDA criterion—deterministic cuBLAS workspace plus
fresh-process next-loss comparison at `rtol=1e-5`, `atol=1e-6`—rather than
requiring globally deterministic algorithms that the pinned learner cannot
execute.

Job `6a66b4b3db23d7a7ec1cf152` then completed every objective forward loss,
including DINO and iBOT, but the strict audit detected a non-finite SSL
gradient under FP16 mixed precision. The L4 preflight now resolves all DINO
teacher/student backbone and head mixed-precision dtypes to BF16. BF16 retains
the same two-byte storage/compute class with a wider exponent range; this is a
recorded engineering-smoke precision override, not an initialization,
architecture, objective, or learner substitution.

Job `6a66b5c87ef3c0846496a19a` confirmed finite BF16 SSL gradients but caught
that the SSL student did not update. The pinned `DINOv2SSL` wrapper composes
the raw DINO config without the outer training setup that normally resolves
the scaled learning rate, leaving `optim.lr: 0.0`. The smoke config now
explicitly resolves `lr` and `min_lr` to `1e-6`; this is shared, nonzero, and
recorded, and exists solely to demonstrate the required optimizer update.

The connector identifies the account as `rishisim`, and the official hardware
listing exposes `l4x1`. Across the repair ledger and final pass, L4 jobs
consumed 1,692 running seconds, an upper-bound USD 0.376 at the listed USD
0.80/hour rate. The successful job consumed 150 provider seconds (USD 0.0333
at that rate); its measured preflight entrypoint was 48.72 seconds. No
checkpoint, weight, raw log, environment, dataset, or scientific learner result
was retained.

## Reproduced upstream invariants

The pinned upstream `triple` configuration cannot run as documented:

1. `mode/triple.yaml` requires post-SSL vision synchronization.
2. `dinov2/vitb14_coco.yaml` sets `pretrained_dir: null`.
3. `ContrastiveTrainer._validate_sync_compatibility` rejects synchronized
   training unless the SSL path has a pretrained initialization.
4. The bundled SSL override uses `ffn_layer: swiglufused` and
   `block_chunks: 2`; the public `facebook/dinov2-base` prior is the standard
   MLP ViT-B/14 state, so strict key loading is not possible under the bundled
   architecture.
5. The public prior stores a 518-pixel position grid (1,370 tokens), while the
   upstream training contract uses 224 pixels (257 tokens).

This is a configuration/initialization incompatibility, not a learner failure.

## Selected bridge

The adapter pins and hashes one public `facebook/dinov2-base` revision, maps
every tensor explicitly into the in-tree standard-MLP ViT-B/14 naming scheme,
concatenates query/key/value tensors in declared order, and deterministically
interpolates only the position grid from 518 to 224 pixels. It rejects missing
or unmapped keys. The converted state is loaded strictly into:

- `CustomDINOv2VisionEncoder`;
- the DINO/iBOT student before FSDP wrapping; and
- the DINO/iBOT teacher through upstream's student-to-teacher synchronization.

The CUDA entrypoint then requires byte/numeric equality across all shared
backbone tensors before the first optimizer step.

The isolated attributed upstream patch adds:

- immutable Hugging Face revision forwarding for BERT;
- recognition of an explicitly loaded SSL student prior as pretrained
  initialization;
- RNG and SSL-iteration checkpoint state required for a fresh-process resume,
  including normalization to the CPU `uint8` RNG representation required by
  PyTorch 2.8 restore APIs.

It does not vendor or fork the full upstream learner.

## Attempt ledger

1. **Documented upstream configuration — rejected.** Exact runtime constructor
   invariant: synchronized `triple` requires a pretrained SSL initialization,
   but `pretrained_dir` is null.
2. **Public prior with bundled SSL architecture — rejected.** Exact static
   architecture invariant: standard MLP keys cannot strictly initialize the
   bundled SwiGLU/block-chunk model.
3. **Architecture-matched shared-prior adapter — passed on CUDA.**
   This is the only contract-preserving candidate. It uses the frozen learner,
   one ViT-B/14 prior, 224-pixel input, and no random fallback.

The L4 run `6a66b781db23d7a7ec1cf176` completed the full objective cycle,
post-SSL synchronization, and checkpoint save before exposing a PyTorch 2.8
checkpoint-restore compatibility defect: `torch.set_rng_state` rejected the
deserialized RNG state because it was not a CPU `ByteTensor`.
The isolated resume adapter now normalizes saved CPU and CUDA RNG tensors to
contiguous CPU `uint8` tensors before calling the official restore APIs. This
does not alter learner weights, scheduling, data lineage, or the reproducibility
criterion.

The follow-up L4 run `6a66b8dc7ef3c0846496a1b2` verified that resume repair,
including the next scheduled contrastive objective under the frozen tolerance.
It then exposed a preflight-fixture-only device mismatch: the fabricated
evaluator tensor targeted CUDA while its deterministic generator defaulted to
CPU. The fixture now creates its generator on CUDA; no learner or evaluation
protocol behavior changed.

The final L4 run `6a66ba7e7ef3c0846496a1bf` passed every acceptance invariant.
The bridge candidate is accepted for engineering compatibility; the no-go rule
was not reached.

## Runtime acceptance

The CUDA entrypoint fails closed unless it observes:

- exact upstream, BERT, DINO, weight, environment, config, and data-lineage
  hashes;
- the exact `contrastive ×4, MLM ×1, DINO/iBOT ×1` order;
- finite family-specific losses, finite nonzero gradients, and an optimizer
  update for every objective family;
- strict pre-step shared-backbone equality and exact post-SSL teacher-to-CLIP
  synchronization;
- checkpoint save and fresh-process resume with restored scheduler, RNG, SSL
  iteration, config, and lineage;
- a tolerance-declared next scheduled result;
- the official `ContrastiveFeatureExtractor` interface; and
- fabricated noun, adjective, and lexical macro-aggregation wiring with no
  scientific score retained.

Checkpoint, downloads, caches, logs, and temporary fixtures remained outside
the repository. Only the compact reviewed aggregate is retained.

The disposable runner contract is:

```bash
python scripts/prepare_egobaby_cuda_preflight.py \
  --destination "$OS_TEMP/egobabyvlm"
cd "$OS_TEMP/egobabyvlm"
pixi run pip install -e "$NURSERY_ROOT"
pixi run nursery-egobaby-cuda-preflight \
  --upstream-root "$OS_TEMP/egobabyvlm" \
  --config "$NURSERY_ROOT/configs/egobaby_cuda_preflight.json" \
  --output "$OS_TEMP/aggregate.json" \
  --provider hugging-face-jobs \
  --maximum-wall-time 1h \
  --upper-bound-cost-usd 0.80
```

`$OS_TEMP` must be an OS-managed disposable directory and `$NURSERY_ROOT` the
reviewed checkout; neither is a repository output root.

## Next authorized stage

Phase 2 is complete. The next protocol stage is the separately authorized
governance and preregistration work in the architecture review. Generator
implementation, ChildLens preparation, video generation, TTS, ASR, translation,
and scientific training remain unauthorized.
