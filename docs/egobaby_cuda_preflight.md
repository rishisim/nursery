# EgoBabyVLM `triple` public/dummy CUDA preflight

**Status:** INFRASTRUCTURE GATE

**Decision date:** 2026-07-26

**Scope:** engineering compatibility only. No scientific experiment, score,
restricted media, ChildLens material, or BabyView material was used.

## Decision

Local inspection and platform-independent packaging are complete. A real CUDA
run is mandatory before this record can become PASS, and no already-running or
previously approved CUDA job was available. The configured Hugging Face account
can submit Jobs, but a new GPU Job is billable and therefore awaits explicit
approval.

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

The connector identifies the account as `rishisim`, can list Jobs, and reported
no running Jobs. The official hardware listing exposes `l4x1`, so the requested
flavor exists. Because even the minimal CPU validation fails identically, the
remaining gate is the Jobs connector/account entitlement or its submission
authorization, not the preflight payload.

**Cost incurred: USD 0.00.** No remote job ID, GPU allocation, artifact, or log
was created.

Exact unblock action: restore Jobs submission authorization for the connected
Hugging Face account, or authenticate the local official CLI with an account
that has Hugging Face Jobs access. After that external action, rerun the already
approved single-L4 command below without changing the scientific contract.

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
