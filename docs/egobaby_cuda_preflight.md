# EgoBabyVLM `triple` public/dummy CUDA preflight

**Status:** INFRASTRUCTURE GATE

**Decision date:** 2026-07-26

**Scope:** engineering compatibility only. No scientific experiment, score,
restricted media, ChildLens material, or BabyView material was used.

## Decision

Local inspection and platform-independent packaging are complete. The user
approved the bounded CUDA run and Jobs authentication has been restored. Two
short failed allocations exposed bootstrap/manifest defects before learner
execution; both were repaired in place. A real end-to-end CUDA result is still
mandatory before this record can become PASS.

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

The connector identifies the account as `rishisim`, and the official hardware
listing exposes `l4x1`. The two allocations consumed 137 seconds of L4 runtime,
approximately USD 0.03 at the listed rate. Neither produced a retained artifact
or learner result.

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
- RNG and SSL-iteration checkpoint state required for a fresh-process resume.

It does not vendor or fork the full upstream learner.

## Attempt ledger

1. **Documented upstream configuration — rejected.** Exact runtime constructor
   invariant: synchronized `triple` requires a pretrained SSL initialization,
   but `pretrained_dir` is null.
2. **Public prior with bundled SSL architecture — rejected.** Exact static
   architecture invariant: standard MLP keys cannot strictly initialize the
   bundled SwiGLU/block-chunk model.
3. **Architecture-matched shared-prior adapter — packaged, awaiting CUDA.**
   This is the only contract-preserving candidate. It uses the frozen learner,
   one ViT-B/14 prior, 224-pixel input, and no random fallback.

The third item has not failed. The three-failed-attempt ENGINEERING NO-GO rule
has therefore not been reached.

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

Checkpoint, downloads, caches, logs, and temporary fixtures remain outside the
repository. A successful reviewed aggregate JSON will replace this gate with
PASS. A CUDA failure will be diagnosed within the frozen constraints and
recorded here; it will not silently change the learner or prior.

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
  --upper-bound-cost-usd 2.50
```

`$OS_TEMP` must be an OS-managed disposable directory and `$NURSERY_ROOT` the
reviewed checkout; neither is a repository output root.

## Next authorized stage

Only the bounded public/dummy CUDA job above. Generator implementation,
ChildLens preparation, video generation, TTS, ASR, translation, and scientific
training remain unauthorized.
