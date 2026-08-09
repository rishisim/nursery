# Gemma 4 E4B provenance preflight

Date: 2026-07-22  
Scope: official public metadata and source code only; no weights downloaded  
Decision: **RECOMMEND_PINNED_LOCAL_CONVERSION; REJECT_EXISTING_COMMUNITY_CONVERSION**

## Exact upstream

Use the instruction-tuned Google repository `google/gemma-4-E4B-it` at immutable Hugging Face revision:

```text
ee0ef6023621cff504d758262d4e04895a5af4a2
```

The pinned model card declares `license: apache-2.0`, `library_name: transformers`, and `pipeline_tag: any-to-any`. Hugging Face reports this revision as public, non-private, and ungated. Google's official Gemma 4 license page publishes the Apache License 2.0 and was last updated 2026-04-01.

The pinned repository has nine files totaling **16,024,823,729 bytes**. Its weight file is:

| File | Bytes | Hugging Face LFS SHA-256 |
|---|---:|---|
| `model.safetensors` | 15,992,595,884 | `cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503` |

The remaining 32,227,845 bytes are the tokenizer, processor/configuration, chat template, model card, and repository metadata. The exact config identifies `Gemma4ForConditionalGeneration`, `model_type: gemma4`, and BF16 source dtype.

## Why the existing MLX community conversion still fails provenance

`mlx-community/gemma-4-e4b-it-4bit` at revision `475b9088d29754a3379866cf5aeb6b41acd313c2` is public and ungated, but it is not an acceptable activation artifact:

- its pinned card declares `license: gemma`, conflicting with the current Apache-2.0 upstream;
- it identifies only `base_model: google/gemma-4-E4B-it`, not an immutable upstream revision;
- it says it was converted from a local MLX-VLM checkout without naming a version or commit;
- its ten-file repository contains no `LICENSE` or `NOTICE` file;
- therefore its exact source ancestry, converter code, and license carry-through cannot be established from the repository itself.

Its metadata is useful only as a size/recipe cross-check: ten files total **5,179,241,512 bytes**; `model.safetensors` is 5,146,800,534 bytes with LFS SHA-256 `932b8271fc3fe65adcc78b96c10c6268bbfb13e8f67d1358727c0d6ee97e1eff`. Its config records 4-bit affine quantization with group size 64. These facts do not cure the provenance failure.

## Recommended MLX-compatible path

Convert the pinned Google snapshot locally; do not use or derive from the ambiguous community weights.

Pin MLX-VLM **v0.6.6**, Git commit `c9e27b08311f9c1f38690c46034b04c7598d73ee`. The release source is MIT-licensed. Its converter source blob is `5793d33dec6d982a5e55d2c0b305a925234e0eca`. The v0.6.6 conversion code explicitly supports a Hugging Face revision, and its default affine recipe is 4 bits with group size 64. The converter excludes multimodal modules through its model-aware quantization predicate, writes MLX safetensors/config, and saves processor material.

Freeze these conversion parameters explicitly rather than relying on defaults:

```text
source revision: ee0ef6023621cff504d758262d4e04895a5af4a2
converter: mlx-vlm 0.6.6 / c9e27b08311f9c1f38690c46034b04c7598d73ee
quantization: affine
weight bits: 4
group size: 64
mixed-bit recipe: none
source dtype: bfloat16 from pinned config
trust_remote_code: false
upload: forbidden
```

The wheel `mlx_vlm-0.6.6-py3-none-any.whl` is 1,948,458 bytes with PyPI SHA-256 `c0fcbd9c0297ac94e923772a59610173553a42b36c74370dc65191250b91ef6e`. The already frozen runtime family also pins MLX 0.32.0 and Transformers 5.14.1; the complete transitive environment must be generated with hashes before conversion.

### Predeclared commands—not executed

```sh
export PUBLIC_INSTRUMENT_ROOT="$HOME/Library/Application Support/ChildLens Instruments/provisional-calibration-v1"
export GOOGLE_SOURCE="$PUBLIC_INSTRUMENT_ROOT/sources/google-gemma-4-E4B-it-ee0ef602"
export LOCAL_MLX="$PUBLIC_INSTRUMENT_ROOT/models/gemma-4-E4B-it-4bit-local-ee0ef602-c9e27b08"
export MLX_ENV="$PUBLIC_INSTRUMENT_ROOT/mlx-vlm-venv"

env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN HF_HUB_DISABLE_TELEMETRY=1 \
  /opt/homebrew/bin/hf download google/gemma-4-E4B-it \
  --revision ee0ef6023621cff504d758262d4e04895a5af4a2 \
  --local-dir "$GOOGLE_SOURCE"

test "$(stat -f %z "$GOOGLE_SOURCE/model.safetensors")" = 15992595884
test "$(shasum -a 256 "$GOOGLE_SOURCE/model.safetensors" | cut -d ' ' -f 1)" = \
  cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503

/usr/bin/sandbox-exec -p '(version 1) (allow default) (deny network*)' \
  /usr/bin/env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  "$MLX_ENV/bin/mlx_vlm.convert" \
  --hf-path "$GOOGLE_SOURCE" \
  --mlx-path "$LOCAL_MLX" \
  --quantize --q-mode affine --q-bits 4 --q-group-size 64
```

Do not pass `--trust-remote-code`, `--upload-repo`, a Hub model ID to the offline converter, or the community conversion as input.

## Required conversion receipt

Before any public canary or restricted use, place the following beside the locally converted artifact:

1. a copy of the Apache-2.0 license and the pinned Google model-card metadata;
2. a `PROVENANCE.json` binding the source repository/revision, nine-file source manifest, source weight hash, converter version/commit/blob, complete hashed dependency lock, and the explicit quantization parameters above;
3. a prominent notice that the weights were locally quantized/modified;
4. an exact relative-path/size/file-SHA-256 manifest of the converted output;
5. a record that no remote code, authentication token, upload, or external inference was used;
6. the public mixed audio-image network-denied canary receipt.

The converter's generated model card is not a substitute for this receipt because a local source path does not itself encode the upstream repository revision.

## Storage admission

The exact source snapshot is 16.025 GB decimal (14.924 GiB). The prior untrusted conversion is 5.179 GB decimal and provides a reasonable engineering estimate, not a promised size, for the new 4-bit output. Reserve at least **22 GB decimal** for simultaneous source plus converted artifacts, plus environment and temporary-write headroom. Preserve the convergence protocol's 50-GiB post-peak free-space floor.

After a successful manifest and canary, the source snapshot may be removed only under the project's normal recoverable-storage policy after retaining its immutable manifest, model card, license, and provenance receipt. No deletion is authorized by this preflight.

## Terminal recommendation

The exact acceptable path is:

```text
google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2
  -> local network-denied MLX-VLM v0.6.6 conversion
  -> affine 4-bit / group-size 64 / no mixed recipe
  -> locally manifested Apache-2.0 derivative with modification notice
```

This path has unambiguous source licensing and reproducible conversion ancestry. The existing `mlx-community/gemma-4-e4b-it-4bit@475b9088...` remains **FAIL_CLOSED_PROVENANCE** and must not be activated.

## Primary sources

- [Pinned Google Gemma 4 E4B IT repository](https://huggingface.co/google/gemma-4-E4B-it/tree/ee0ef6023621cff504d758262d4e04895a5af4a2)
- [Google's Gemma 4 Apache-2.0 license](https://ai.google.dev/gemma/apache_2)
- [Ambiguous pinned MLX community conversion](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit/tree/475b9088d29754a3379866cf5aeb6b41acd313c2)
- [MLX-VLM v0.6.6](https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.6)
- [Pinned MLX-VLM converter source](https://github.com/Blaizzy/mlx-vlm/blob/c9e27b08311f9c1f38690c46034b04c7598d73ee/mlx_vlm/convert.py)
- [Pinned MLX-VLM MIT license](https://github.com/Blaizzy/mlx-vlm/blob/c9e27b08311f9c1f38690c46034b04c7598d73ee/LICENSE)

