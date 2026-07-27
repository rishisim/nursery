from __future__ import annotations

import ast
from pathlib import Path


def test_bridge_has_no_random_or_non_strict_fallback() -> None:
    source = Path("src/nursery_egobaby_preflight/bridge.py").read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert "random" not in names
    assert "allclose" not in attributes
    assert "equal" in attributes


def test_upstream_patch_is_narrow_and_attributed() -> None:
    patch = Path("patches/egobabyvlm_shared_prior.patch").read_text()
    changed = [line for line in patch.splitlines() if line.startswith("diff --git")]
    assert changed == [
        "diff --git a/apps/baselines/clip/modeling/text_encoder.py b/apps/baselines/clip/modeling/text_encoder.py",
        "diff --git a/apps/baselines/clip/modeling/dinov2_ssl.py b/apps/baselines/clip/modeling/dinov2_ssl.py",
        "diff --git a/apps/baselines/clip/training/checkpoint.py b/apps/baselines/clip/training/checkpoint.py",
        "diff --git a/apps/baselines/clip/training/loop.py b/apps/baselines/clip/training/loop.py",
    ]
