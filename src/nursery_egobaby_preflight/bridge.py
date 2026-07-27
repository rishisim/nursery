"""Strict Hugging Face DINOv2-base to upstream DINOv2 ViT-B/14 conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConversionRecord:
    source_keys: int
    target_keys: int
    mapped_source_keys: tuple[str, ...]
    target_shapes: dict[str, tuple[int, ...]]
    target_dtypes: dict[str, str]
    positional_interpolation: dict[str, int | str]


def _copy(tensor: Any) -> Any:
    return tensor.detach().cpu().contiguous().clone()


def _interpolate_position_embeddings(tensor: Any, target_tokens: int) -> Any:
    import math

    import torch.nn.functional as functional

    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"unexpected position embedding shape: {tuple(tensor.shape)}")
    source_patch_tokens = tensor.shape[1] - 1
    target_patch_tokens = target_tokens - 1
    source_side = math.isqrt(source_patch_tokens)
    target_side = math.isqrt(target_patch_tokens)
    if source_side**2 != source_patch_tokens or target_side**2 != target_patch_tokens:
        raise ValueError("position embeddings must describe square patch grids plus one CLS token")
    cls_token = tensor[:, :1]
    patches = tensor[:, 1:].reshape(1, source_side, source_side, tensor.shape[-1]).permute(0, 3, 1, 2)
    patches = functional.interpolate(
        patches.float(),
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(tensor.dtype)
    return _copy(
        __import__("torch").cat(
            (cls_token, patches.permute(0, 2, 3, 1).reshape(1, target_patch_tokens, tensor.shape[-1])),
            dim=1,
        )
    )


def convert_hf_dinov2_base(
    source: dict[str, Any],
    *,
    target_position_tokens: int = 257,
) -> tuple[dict[str, Any], ConversionRecord]:
    """Convert every public HF DINOv2-base backbone tensor with no fallback."""
    target: dict[str, Any] = {}
    consumed: set[str] = set()

    def move(source_key: str, target_key: str) -> None:
        if source_key not in source:
            raise KeyError(f"required source key missing: {source_key}")
        if target_key in target:
            raise KeyError(f"duplicate target key: {target_key}")
        target[target_key] = _copy(source[source_key])
        consumed.add(source_key)

    move("embeddings.cls_token", "cls_token")
    move("embeddings.mask_token", "mask_token")
    move("embeddings.patch_embeddings.projection.weight", "patch_embed.proj.weight")
    move("embeddings.patch_embeddings.projection.bias", "patch_embed.proj.bias")
    position_key = "embeddings.position_embeddings"
    if position_key not in source:
        raise KeyError(f"required source key missing: {position_key}")
    source_position_tokens = int(source[position_key].shape[1])
    target["pos_embed"] = _interpolate_position_embeddings(source[position_key], target_position_tokens)
    consumed.add(position_key)

    for layer in range(12):
        prefix = f"encoder.layer.{layer}"
        block = f"blocks.{layer}"
        for suffix in ("weight", "bias"):
            qkv_keys = [
                f"{prefix}.attention.attention.{part}.{suffix}"
                for part in ("query", "key", "value")
            ]
            missing = [key for key in qkv_keys if key not in source]
            if missing:
                raise KeyError(f"required source keys missing: {missing}")
            target[f"{block}.attn.qkv.{suffix}"] = __import__("torch").cat(
                [_copy(source[key]) for key in qkv_keys],
                dim=0,
            )
            consumed.update(qkv_keys)
        simple = {
            f"{prefix}.attention.output.dense.weight": f"{block}.attn.proj.weight",
            f"{prefix}.attention.output.dense.bias": f"{block}.attn.proj.bias",
            f"{prefix}.layer_scale1.lambda1": f"{block}.ls1.gamma",
            f"{prefix}.layer_scale2.lambda1": f"{block}.ls2.gamma",
            f"{prefix}.mlp.fc1.weight": f"{block}.mlp.fc1.weight",
            f"{prefix}.mlp.fc1.bias": f"{block}.mlp.fc1.bias",
            f"{prefix}.mlp.fc2.weight": f"{block}.mlp.fc2.weight",
            f"{prefix}.mlp.fc2.bias": f"{block}.mlp.fc2.bias",
            f"{prefix}.norm1.weight": f"{block}.norm1.weight",
            f"{prefix}.norm1.bias": f"{block}.norm1.bias",
            f"{prefix}.norm2.weight": f"{block}.norm2.weight",
            f"{prefix}.norm2.bias": f"{block}.norm2.bias",
        }
        for source_key, target_key in simple.items():
            move(source_key, target_key)

    move("layernorm.weight", "norm.weight")
    move("layernorm.bias", "norm.bias")
    unexpected = sorted(set(source) - consumed)
    if unexpected:
        raise ValueError(f"unmapped source keys: {unexpected}")
    record = ConversionRecord(
        source_keys=len(source),
        target_keys=len(target),
        mapped_source_keys=tuple(sorted(consumed)),
        target_shapes={key: tuple(value.shape) for key, value in target.items()},
        target_dtypes={key: str(value.dtype) for key, value in target.items()},
        positional_interpolation={
            "mode": "bicubic",
            "antialias": "true",
            "source_tokens": source_position_tokens,
            "target_tokens": target_position_tokens,
        },
    )
    return target, record


def strict_state_equality(left: dict[str, Any], right: dict[str, Any]) -> dict[str, str | int]:
    import hashlib

    import torch

    if set(left) != set(right):
        raise AssertionError(
            f"state keys differ: missing={sorted(set(left) - set(right))}, "
            f"unexpected={sorted(set(right) - set(left))}"
        )
    digest = hashlib.sha256()
    for key in sorted(left):
        a = left[key].detach().cpu().contiguous()
        b = right[key].detach().cpu().contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise AssertionError(
                f"{key}: shape/dtype mismatch {tuple(a.shape)}/{a.dtype} != {tuple(b.shape)}/{b.dtype}"
            )
        if not torch.equal(a, b):
            raise AssertionError(f"{key}: tensors are not byte/numerically equal")
        digest.update(key.encode())
        digest.update(a.numpy().tobytes())
    return {"keys": len(left), "sha256": digest.hexdigest()}
